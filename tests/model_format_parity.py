#!/usr/bin/env python3
"""
Compare YOLO model formats (.pt, .onnx, .tflite, .engine) for detection parity.

The PyTorch model (.pt) is the baseline. Every other format is run on the same
audio file and matched against the baseline detections, so you can confirm that
converting or quantizing a model does not drop performance radically.

Why this file has two roles
---------------------------
Each format needs its own conda environment (birdbox-pt, birdbox-onnx,
birdbox-tflite, birdbox-engine). A single Python process cannot hold all four
dependency stacks at once. So this one file runs in two modes:

  1. Orchestrator (default): discovers models, picks the right conda env per
     format, and launches one worker subprocess per model via ``conda run``.
     It only needs the standard library, so you can start it from any env.
  2. Worker (``--worker``): runs inside a single conda env, loads one model with
     the shared BirdCallDetector pipeline, and writes normalized detections to
     a JSON file for the orchestrator to read back.

Usage
-----
    # Run everything and write the markdown report
    python tests/model_format_parity.py

    # Point at a different audio file or model folder
    python tests/model_format_parity.py --audio tests/test.wav --models-dir tests/models_for_test

    # Ignore cached results and run every model again
    python tests/model_format_parity.py --force

Models that already have a result JSON in the results folder (default:
tests/parity_results) are not run again. Their stored detections feed straight
into the report. Delete a model's JSON or pass ``--force`` to recompute.

Adding or removing models
--------------------------
Drop model files into the models folder (default: tests/models_for_test). Any
file ending in .pt, .onnx, .tflite, or .engine is picked up automatically. The
conda env is chosen from the file extension, so quantized exports (for example a
16-bit Just-Bird_fp16.onnx) work with no code changes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# --------------------------------------------------------------------------- #
# Configuration (edit these defaults or override them on the command line)
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
TESTS_DIR = REPO_ROOT / "tests"

DEFAULT_AUDIO = TESTS_DIR / "test.wav"
DEFAULT_MODELS_DIR = TESTS_DIR / "models_for_test"
DEFAULT_REPORT = TESTS_DIR / "model_format_parity_report.md"
DEFAULT_RESULTS_DIR = TESTS_DIR / "parity_results"

# Species mapping used for every model (they are all the same trained network).
DEFAULT_SPECIES_MAPPING = "Just-Bird"

# The baseline every other format is compared against.
BASELINE_FORMAT = ".pt"

# One conda env per format. Extensions not listed here are skipped.
ENV_FOR_FORMAT: Dict[str, str] = {
    ".pt": "birdbox-pt",
    ".onnx": "birdbox-onnx",
    ".tflite": "birdbox-tflite",
    ".engine": "birdbox-engine",
}
SUPPORTED_FORMATS = tuple(ENV_FOR_FORMAT.keys())

# Detection settings (kept identical across formats for a fair comparison).
DEFAULT_CONF = 0.2
DEFAULT_NMS_IOU = 0.7
DEFAULT_SONG_GAP = 0.1

# A candidate box counts as "the same detection" as a baseline box when their
# 2D (time and frequency) IoU is at least this value and species matches.
MATCH_IOU = 0.3

# Thresholds that decide the PASS / WARN verdict per model.
PASS_MIN_MATCH_RATE = 0.90   # fraction of baseline detections that were matched
PASS_MAX_EXTRA_RATE = 0.10   # extra candidate detections, relative to baseline
PASS_MAX_MEAN_CONF_DELTA = 0.05  # mean absolute confidence difference on matches


# --------------------------------------------------------------------------- #
# Worker mode: runs inside a single conda env and detects on one model
# --------------------------------------------------------------------------- #

def run_worker(args: argparse.Namespace) -> int:
    """Load one model, run detection, and dump normalized detections to JSON."""
    sys.path.insert(0, str(SRC_DIR))
    try:
        from inference.detect_birds import BirdCallDetector
    except Exception as exc:  # pragma: no cover - import failure is env-specific
        _dump_worker_json(
            args.output,
            model=args.model,
            error=f"Failed to import detection pipeline: {exc}",
        )
        return 1

    try:
        load_start = time.perf_counter()
        detector = BirdCallDetector(
            model_path=args.model,
            species_mapping=args.species_mapping,
            conf_threshold=args.conf,
            nms_iou_threshold=args.nms_iou,
            song_gap_threshold=args.song_gap,
            num_workers=1,
        )
        load_seconds = time.perf_counter() - load_start

        detect_start = time.perf_counter()
        detections = detector.detect_single_file(args.audio, no_merge=args.raw)
        detect_seconds = time.perf_counter() - detect_start
    except Exception as exc:
        import traceback

        _dump_worker_json(
            args.output,
            model=args.model,
            error=f"{exc}\n{traceback.format_exc()}",
        )
        return 1

    normalized = [_normalize_detection(det) for det in detections]
    _dump_worker_json(
        args.output,
        model=args.model,
        detections=normalized,
        load_seconds=load_seconds,
        detect_seconds=detect_seconds,
        merged=not args.raw,
    )
    return 0


def _normalize_detection(det: Dict) -> Dict:
    """Flatten merged or raw detections into one schema the orchestrator reads."""
    is_merged = "detections_merged" in det
    confidence = det["avg_confidence"] if is_merged else det["confidence"]
    return {
        "species": det["species"],
        "species_id": int(det["species_id"]),
        "confidence": float(confidence),
        "max_confidence": float(det.get("max_confidence", confidence)),
        "time_start": float(det["time_start"]),
        "time_end": float(det["time_end"]),
        "freq_low_hz": float(det["freq_low_hz"]),
        "freq_high_hz": float(det["freq_high_hz"]),
    }


def _dump_worker_json(output: str, **payload) -> None:
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as handle:
        json.dump(payload, handle, indent=2)


# --------------------------------------------------------------------------- #
# Orchestrator mode: discover models, dispatch workers, compare, report
# --------------------------------------------------------------------------- #

def discover_models(models_dir: Path) -> List[Path]:
    """Return every supported model file in the folder, sorted by name."""
    models: List[Path] = []
    for path in sorted(models_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in ENV_FOR_FORMAT:
            models.append(path)
    return models


def pick_baseline(models: List[Path]) -> Optional[Path]:
    """Choose the .pt baseline (prefers a file named like Just-Bird.pt)."""
    candidates = [m for m in models if m.suffix.lower() == BASELINE_FORMAT]
    if not candidates:
        return None
    for model in candidates:
        if "just-bird" in model.stem.lower():
            return model
    return candidates[0]


def conda_executable() -> str:
    return os.environ.get("CONDA_EXE", "conda")


def load_cached_result(model: Path, args: argparse.Namespace, out_json: Path) -> Optional[Dict]:
    """
    Reuse a previous worker run if its JSON is still usable.

    Returns None when there is no JSON, it is unreadable, it recorded an error,
    or it holds the wrong detection kind (raw vs merged). In those cases the
    model is run again.
    """
    if not out_json.is_file():
        return None

    try:
        with open(out_json) as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None

    if "error" in payload or "detections" not in payload:
        return None
    if payload.get("merged", True) != (not args.raw):
        return None

    return {
        "model": model,
        "name": model.name,
        "format": model.suffix.lower(),
        "env": ENV_FOR_FORMAT[model.suffix.lower()],
        "file_size_bytes": model.stat().st_size,
        "detections": payload["detections"],
        "load_seconds": payload.get("load_seconds"),
        "detect_seconds": payload.get("detect_seconds"),
        "merged": payload.get("merged", True),
        "cached": True,
    }


def run_model_in_env(model: Path, args: argparse.Namespace, out_json: Path) -> Dict:
    """Launch a worker subprocess in the model's conda env and read its output."""
    env_name = ENV_FOR_FORMAT[model.suffix.lower()]
    out_json.parent.mkdir(parents=True, exist_ok=True)
    if out_json.exists():
        out_json.unlink()

    command = [
        conda_executable(), "run", "--no-capture-output", "-n", env_name,
        "python", str(Path(__file__).resolve()),
        "--worker",
        "--model", str(model),
        "--audio", str(args.audio),
        "--output", str(out_json),
        "--species-mapping", args.species_mapping,
        "--conf", str(args.conf),
        "--nms-iou", str(args.nms_iou),
        "--song-gap", str(args.song_gap),
    ]
    if args.raw:
        command.append("--raw")

    print(f"\n{'=' * 70}")
    print(f"Running {model.name} in conda env '{env_name}'")
    print(f"{'=' * 70}")

    wall_start = time.perf_counter()
    completed = subprocess.run(command, cwd=str(REPO_ROOT))
    wall_seconds = time.perf_counter() - wall_start

    result: Dict = {
        "model": model,
        "name": model.name,
        "format": model.suffix.lower(),
        "env": env_name,
        "file_size_bytes": model.stat().st_size,
        "wall_seconds": wall_seconds,
    }

    if completed.returncode != 0 or not out_json.exists():
        result["error"] = (
            f"Worker exited with code {completed.returncode}. "
            "See the log above for the traceback."
        )
        return result

    with open(out_json) as handle:
        payload = json.load(handle)

    if "error" in payload:
        result["error"] = payload["error"]
        return result

    result.update(
        detections=payload.get("detections", []),
        load_seconds=payload.get("load_seconds"),
        detect_seconds=payload.get("detect_seconds"),
        merged=payload.get("merged", True),
    )
    return result


def _box_iou_2d(a: Dict, b: Dict) -> float:
    """IoU of two detections in (time seconds, frequency Hz) space."""
    t_left = max(a["time_start"], b["time_start"])
    t_right = min(a["time_end"], b["time_end"])
    f_low = max(a["freq_low_hz"], b["freq_low_hz"])
    f_high = min(a["freq_high_hz"], b["freq_high_hz"])

    if t_right <= t_left or f_high <= f_low:
        return 0.0

    intersection = (t_right - t_left) * (f_high - f_low)
    area_a = (a["time_end"] - a["time_start"]) * (a["freq_high_hz"] - a["freq_low_hz"])
    area_b = (b["time_end"] - b["time_start"]) * (b["freq_high_hz"] - b["freq_low_hz"])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def compare_to_baseline(baseline: List[Dict], candidate: List[Dict]) -> Dict:
    """Greedy-match candidate detections to baseline and score the agreement."""
    unmatched_baseline = list(range(len(baseline)))
    matched_pairs: List[Dict] = []
    used_candidate = set()

    # For each baseline detection, take the best same-species candidate box.
    for b_index in list(unmatched_baseline):
        b_det = baseline[b_index]
        best_iou = MATCH_IOU
        best_c_index = None
        for c_index, c_det in enumerate(candidate):
            if c_index in used_candidate:
                continue
            if c_det["species_id"] != b_det["species_id"]:
                continue
            iou = _box_iou_2d(b_det, c_det)
            if iou >= best_iou:
                best_iou = iou
                best_c_index = c_index
        if best_c_index is not None:
            used_candidate.add(best_c_index)
            unmatched_baseline.remove(b_index)
            matched_pairs.append({
                "iou": best_iou,
                "conf_delta": candidate[best_c_index]["confidence"] - b_det["confidence"],
                "time_delta": candidate[best_c_index]["time_start"] - b_det["time_start"],
            })

    matched = len(matched_pairs)
    baseline_count = len(baseline)
    candidate_count = len(candidate)
    extra = candidate_count - matched

    mean_iou = sum(p["iou"] for p in matched_pairs) / matched if matched else 0.0
    mean_conf_delta = (
        sum(abs(p["conf_delta"]) for p in matched_pairs) / matched if matched else 0.0
    )
    max_conf_delta = max((abs(p["conf_delta"]) for p in matched_pairs), default=0.0)
    mean_time_delta = (
        sum(abs(p["time_delta"]) for p in matched_pairs) / matched if matched else 0.0
    )

    match_rate = matched / baseline_count if baseline_count else 1.0
    extra_rate = extra / baseline_count if baseline_count else float(extra)

    return {
        "baseline_count": baseline_count,
        "candidate_count": candidate_count,
        "matched": matched,
        "missed": len(unmatched_baseline),
        "extra": extra,
        "match_rate": match_rate,
        "extra_rate": extra_rate,
        "mean_iou": mean_iou,
        "mean_conf_delta": mean_conf_delta,
        "max_conf_delta": max_conf_delta,
        "mean_time_delta": mean_time_delta,
    }


def verdict_for(comparison: Dict) -> str:
    """Turn comparison numbers into a PASS / WARN label."""
    if (
        comparison["match_rate"] >= PASS_MIN_MATCH_RATE
        and comparison["extra_rate"] <= PASS_MAX_EXTRA_RATE
        and comparison["mean_conf_delta"] <= PASS_MAX_MEAN_CONF_DELTA
    ):
        return "PASS"
    return "WARN"


def species_counts(detections: List[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for det in detections:
        counts[det["species"]] = counts.get(det["species"], 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# Markdown report
# --------------------------------------------------------------------------- #

def write_report(report_path: Path, args: argparse.Namespace,
                 baseline_result: Dict, results: List[Dict]) -> None:
    """Write the markdown overview comparing every model to the baseline."""
    detection_kind = "raw detections" if args.raw else "merged song segments"
    baseline_dets = baseline_result.get("detections", [])

    lines: List[str] = []
    lines.append("# Model Format Parity Report")
    lines.append("")
    lines.append(
        f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. "
        f"The PyTorch model is the baseline. Every other format is matched "
        f"against it using {detection_kind}."
    )
    lines.append("")

    # Run settings.
    lines.append("## Run settings")
    lines.append("")
    lines.append("| Setting | Value |")
    lines.append("| :--- | :--- |")
    lines.append(f"| Audio file | `{args.audio}` |")
    lines.append(f"| Species mapping | `{args.species_mapping}` |")
    lines.append(f"| Confidence threshold | `{args.conf}` |")
    lines.append(f"| NMS IoU threshold | `{args.nms_iou}` |")
    lines.append(f"| Song gap threshold | `{args.song_gap}` |")
    lines.append(f"| Detection output | {detection_kind} |")
    lines.append(f"| Match IoU threshold | `{MATCH_IOU}` |")
    lines.append(f"| Baseline model | `{baseline_result['name']}` |")
    lines.append(f"| Baseline detections | {len(baseline_dets)} |")
    lines.append("")

    # Overview table.
    lines.append("## Overview")
    lines.append("")
    lines.append(
        "| Model | Format | Size | Conda Env | Detections | Matched | Missed | Extra "
        "| Match Rate | Mean IoU | Mean Conf Δ | Max Conf Δ | Load (s) | Detect (s) | Verdict |"
    )
    lines.append(
        "| :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
        "| ---: | ---: | ---: | :---: |"
    )

    for result in results:
        size_str = _fmt_file_size(result.get("file_size_bytes"))

        if result.get("error"):
            lines.append(
                f"| `{result['name']}` | {result['format']} | {size_str} | `{result['env']}` "
                f"| - | - | - | - | - | - | - | - | - | - | **FAIL** |"
            )
            continue

        load_str = _fmt_seconds_plain(result.get("load_seconds"))
        detect_str = _fmt_seconds_plain(result.get("detect_seconds"))

        if result["is_baseline"]:
            lines.append(
                f"| `{result['name']}` | {result['format']} | {size_str} | `{result['env']}` "
                f"| {len(result['detections'])} | - | - | - | - | - | - | - "
                f"| {load_str} | {detect_str} | _baseline_ |"
            )
            continue

        comp = result["comparison"]
        lines.append(
            f"| `{result['name']}` | {result['format']} | {size_str} | `{result['env']}` "
            f"| {comp['candidate_count']} | {comp['matched']} | {comp['missed']} "
            f"| {comp['extra']} | {comp['match_rate'] * 100:.1f}% "
            f"| {comp['mean_iou']:.3f} | {comp['mean_conf_delta']:.4f} "
            f"| {comp['max_conf_delta']:.4f} | {load_str} | {detect_str} "
            f"| **{result['verdict']}** |"
        )
    lines.append("")

    # What the metrics mean.
    lines.append("### What the metrics mean")
    lines.append("")
    lines.append(
        "Each converted model is matched against the baseline detection by "
        "detection. A pair counts as the same detection when both share the "
        f"species and their box IoU is at least {MATCH_IOU}."
    )
    lines.append("")
    lines.append("| Metric | What it tells you |")
    lines.append("| :--- | :--- |")
    lines.append(
        "| Mean IoU | Average overlap between each matched box and its baseline "
        "box, measured in time and frequency. `1.000` is a perfect overlap. Lower "
        "values mean the boxes drifted in position or size. |"
    )
    lines.append(
        "| Mean Conf Δ | Average absolute confidence difference across matched "
        "detections. Near `0` means the converted model is about as sure as the "
        "baseline. Larger values mean confidence shifted after conversion. |"
    )
    lines.append(
        "| Max Conf Δ | The single largest confidence difference among matched "
        "detections. It surfaces the worst-case outlier that the mean can hide. |"
    )
    lines.append(
        "| Size | On-disk file size of the model. Useful when comparing full-precision "
        "and quantized exports. |"
    )
    lines.append(
        "| Load (s) | Wall-clock seconds to load the model and build the detector. |"
    )
    lines.append(
        "| Detect (s) | Wall-clock seconds spent running inference on the audio. |"
    )
    lines.append("")

    # How to read the verdict.
    lines.append("### How to read the verdict")
    lines.append("")
    lines.append(
        f"A model earns **PASS** when it matches at least "
        f"{PASS_MIN_MATCH_RATE * 100:.0f}% of the baseline detections, adds no more "
        f"than {PASS_MAX_EXTRA_RATE * 100:.0f}% extra detections, and keeps the mean "
        f"confidence difference at or below {PASS_MAX_MEAN_CONF_DELTA}. **WARN** means "
        "one of those limits was crossed and the export deserves a closer look. "
        "**FAIL** means the model did not run. Check the console log for its traceback."
    )
    lines.append("")

    # Per-model detail with species breakdown.
    lines.append("## Per-model detail")
    lines.append("")
    baseline_species = species_counts(baseline_dets)

    for result in results:
        lines.append(f"### {result['name']}")
        lines.append("")

        if result.get("error"):
            lines.append("This model failed to run. Error reported by the worker.")
            lines.append("")
            lines.append("```text")
            lines.append(result["error"].strip())
            lines.append("```")
            lines.append("")
            continue

        if result["is_baseline"]:
            lines.append("This is the baseline model. All others are compared to it.")
            lines.append("")

        cand_species = species_counts(result["detections"])
        all_species = sorted(set(baseline_species) | set(cand_species))
        lines.append("| Species | Baseline count | This model count |")
        lines.append("| :--- | ---: | ---: |")
        for species in all_species:
            lines.append(
                f"| `{species}` | {baseline_species.get(species, 0)} "
                f"| {cand_species.get(species, 0)} |"
            )
        lines.append("")

        if result["is_baseline"]:
            lines.append(
                f"File size is {_fmt_file_size(result.get('file_size_bytes'))}. "
                f"Load time was {_fmt_seconds(result.get('load_seconds'))} and "
                f"detection took {_fmt_seconds(result.get('detect_seconds'))}."
            )
            lines.append("")
        else:
            comp = result["comparison"]
            lines.append(
                f"File size is {_fmt_file_size(result.get('file_size_bytes'))}. "
                f"Mean start-time shift on matched detections is "
                f"{comp['mean_time_delta']:.3f} s. Load time was "
                f"{_fmt_seconds(result.get('load_seconds'))} and detection took "
                f"{_fmt_seconds(result.get('detect_seconds'))}."
            )
            lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _fmt_seconds(value: Optional[float]) -> str:
    return f"{value:.1f} s" if value is not None else "unknown"


def _fmt_seconds_plain(value: Optional[float]) -> str:
    return f"{value:.1f}" if value is not None else "-"


def _fmt_file_size(size_bytes: Optional[int]) -> str:
    if size_bytes is None:
        return "-"
    if size_bytes >= 1024 ** 2:
        return f"{size_bytes / (1024 ** 2):.1f} MiB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KiB"
    return f"{size_bytes} B"


# --------------------------------------------------------------------------- #
# Orchestrator entry point
# --------------------------------------------------------------------------- #

def run_orchestrator(args: argparse.Namespace) -> int:
    audio = Path(args.audio)
    models_dir = Path(args.models_dir)

    if not audio.exists():
        print(f"Error: audio file not found: {audio}", file=sys.stderr)
        return 1
    if not models_dir.is_dir():
        print(f"Error: models folder not found: {models_dir}", file=sys.stderr)
        return 1

    models = discover_models(models_dir)
    if not models:
        print(f"Error: no {', '.join(SUPPORTED_FORMATS)} models in {models_dir}", file=sys.stderr)
        return 1

    baseline_model = pick_baseline(models)
    if baseline_model is None:
        print(
            f"Error: no {BASELINE_FORMAT} baseline model in {models_dir}. "
            "The comparison needs a PyTorch model as the reference.",
            file=sys.stderr,
        )
        return 1

    print(f"Discovered {len(models)} model(s) in {models_dir}:")
    for model in models:
        tag = " (baseline)" if model == baseline_model else ""
        print(f"  - {model.name} -> {ENV_FOR_FORMAT[model.suffix.lower()]}{tag}")

    results_dir = Path(args.results_dir)

    # Run the baseline first so we can compare everything against it.
    ordered = [baseline_model] + [m for m in models if m != baseline_model]

    raw_results: Dict[Path, Dict] = {}
    for model in ordered:
        out_json = results_dir / (model.name.replace(".", "_") + ".json")

        cached = None if args.force else load_cached_result(model, args, out_json)
        if cached is not None:
            print(f"\nSkipping {model.name}: reusing existing detections from {out_json}")
            raw_results[model] = cached
        else:
            raw_results[model] = run_model_in_env(model, args, out_json)

    baseline_result = raw_results[baseline_model]
    if baseline_result.get("error"):
        print(
            "\nError: the baseline model failed to run, so there is nothing to "
            f"compare against.\n{baseline_result['error']}",
            file=sys.stderr,
        )
        return 1

    baseline_dets = baseline_result["detections"]

    # Build the ordered result list with comparisons attached.
    results: List[Dict] = []
    for model in ordered:
        result = raw_results[model]
        result["is_baseline"] = model == baseline_model
        if not result.get("error") and not result["is_baseline"]:
            result["comparison"] = compare_to_baseline(baseline_dets, result["detections"])
            result["verdict"] = verdict_for(result["comparison"])
        results.append(result)

    report_path = Path(args.report)
    write_report(report_path, args, baseline_result, results)

    _print_console_summary(results, baseline_result)
    print(f"\nMarkdown report written to: {report_path}")
    print(f"Raw per-model detections written to: {results_dir}")
    return 0


def _print_console_summary(results: List[Dict], baseline_result: Dict) -> None:
    print(f"\n{'=' * 70}")
    print("PARITY SUMMARY")
    print(f"{'=' * 70}")
    print(f"Baseline: {baseline_result['name']} "
          f"({len(baseline_result['detections'])} detections)")
    for result in results:
        if result["is_baseline"]:
            continue
        if result.get("error"):
            print(f"  {result['name']:<32} FAIL (did not run)")
            continue
        comp = result["comparison"]
        print(
            f"  {result['name']:<32} {result['verdict']:<5} "
            f"match {comp['match_rate'] * 100:5.1f}%  "
            f"IoU {comp['mean_iou']:.3f}  "
            f"confΔ {comp['mean_conf_delta']:.4f}  "
            f"det {comp['candidate_count']}"
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare YOLO model formats against the .pt baseline for parity.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--audio", type=str, default=str(DEFAULT_AUDIO),
                        help=f"Audio file to run inference on (default: {DEFAULT_AUDIO})")
    parser.add_argument("--models-dir", type=str, default=str(DEFAULT_MODELS_DIR),
                        help=f"Folder holding the models to compare (default: {DEFAULT_MODELS_DIR})")
    parser.add_argument("--species-mapping", type=str, default=DEFAULT_SPECIES_MAPPING,
                        help=f"Species mapping name (default: {DEFAULT_SPECIES_MAPPING})")
    parser.add_argument("--report", type=str, default=str(DEFAULT_REPORT),
                        help=f"Markdown report output path (default: {DEFAULT_REPORT})")
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR),
                        help=f"Where to keep per-model detection JSON (default: {DEFAULT_RESULTS_DIR})")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF,
                        help=f"Confidence threshold (default: {DEFAULT_CONF})")
    parser.add_argument("--nms-iou", type=float, default=DEFAULT_NMS_IOU,
                        help=f"NMS IoU threshold (default: {DEFAULT_NMS_IOU})")
    parser.add_argument("--song-gap", type=float, default=DEFAULT_SONG_GAP,
                        help=f"Song gap merge threshold in seconds (default: {DEFAULT_SONG_GAP})")
    parser.add_argument("--raw", action="store_true",
                        help="Compare raw (unmerged) detections instead of merged songs.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run every model even when a result JSON already exists.")

    # Worker-only arguments (used when this file re-invokes itself in a conda env).
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=str, default=None, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.worker:
        if not args.model or not args.output:
            print("Worker mode requires --model and --output.", file=sys.stderr)
            return 2
        return run_worker(args)
    return run_orchestrator(args)


if __name__ == "__main__":
    sys.exit(main())
