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
    # Run both default suites and write both markdown reports
    python tests/model_format_parity.py

    # Point at a specific model folder (single-suite mode)
    python tests/model_format_parity.py --models-dir tests/models_for_test_just_bird \
        --report docs/models-and-metrics/just-bird-model-types.md \
        --results-dir tests/parity_results_just_bird

    # Ignore cached results and run every model again
    python tests/model_format_parity.py --force

Default suites (run when no ``--models-dir`` is given):
  - tests/models_for_test_just_bird  →  docs/models-and-metrics/just-bird-model-types.md
  - tests/models_for_test_all_in_one →  docs/models-and-metrics/all-in-one-model-types.md

Models that already have a result JSON in the results folder are not run again.
Their stored detections feed straight into the report. Delete a model's JSON or
pass ``--force`` to recompute.

Adding or removing models
--------------------------
Drop model files into the relevant models folder. Any file ending in .pt, .onnx,
.tflite, or .engine is picked up automatically. The conda env is chosen from the
file extension, so quantized exports (for example a 16-bit Just-Bird_fp16.onnx)
work with no code changes.

Species mapping is inferred from model filenames (e.g. ``All-In-One_fp16.pt`` →
``All-In-One``). Pass ``--species-mapping`` to override. After changing the
mapping, delete cached JSON in the results folder or pass ``--force``.
"""

from __future__ import annotations

import argparse
import copy
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

# Fallbacks used only in explicit single-suite mode (--models-dir given on CLI).
_SINGLE_SUITE_DEFAULT_REPORT = REPO_ROOT / "docs" / "data" / "model-types.md"
_SINGLE_SUITE_DEFAULT_RESULTS_DIR = TESTS_DIR / "parity_results"

# Default dual-suite configuration (what runs when no path flags are given).
JUST_BIRD_SUITE: Dict = {
    "models_dir": TESTS_DIR / "models_for_test_just_bird",
    "results_dir": TESTS_DIR / "parity_results_just_bird",
    "report": REPO_ROOT / "docs" / "models-and-metrics" / "just-bird-model-types.md",
}
ALL_IN_ONE_SUITE: Dict = {
    "models_dir": TESTS_DIR / "models_for_test_all_in_one",
    "results_dir": TESTS_DIR / "parity_results_all_in_one",
    "report": REPO_ROOT / "docs" / "models-and-metrics" / "all-in-one-model-types.md",
}
DEFAULT_SUITES = [JUST_BIRD_SUITE, ALL_IN_ONE_SUITE]

# Species mapping fallback when filenames do not match a known model family.
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
        species_mapping=args.species_mapping,
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


def infer_species_mapping_for_model(model: Path) -> str:
    """Resolve the species mapping name from a model filename via config."""
    sys.path.insert(0, str(SRC_DIR))
    import config

    return config.get_species_mapping_for_model(str(model))


def resolve_species_mapping(models: List[Path], explicit: Optional[str]) -> Optional[str]:
    """
    Choose the species mapping for a parity run.

    When ``explicit`` is set, that value is used. Otherwise the mapping is
    inferred from every model filename. All models in the folder must agree.
    """
    inferred_by_model: Dict[str, List[str]] = {}
    unresolved: List[str] = []

    for model in models:
        try:
            mapping = infer_species_mapping_for_model(model)
        except ValueError:
            unresolved.append(model.name)
            continue
        inferred_by_model.setdefault(mapping, []).append(model.name)

    if explicit:
        if inferred_by_model and explicit not in inferred_by_model:
            seen = ", ".join(
                f"{name} ({mapping})"
                for mapping, names in sorted(inferred_by_model.items())
                for name in names
            )
            print(
                f"Warning: --species-mapping {explicit!r} does not match inferred "
                f"mappings from filenames: {seen}",
                file=sys.stderr,
            )
        return explicit

    if unresolved:
        print(
            "Error: could not infer species mapping for: "
            + ", ".join(unresolved)
            + ". Pass --species-mapping explicitly (e.g. All-In-One).",
            file=sys.stderr,
        )
        return None

    if not inferred_by_model:
        print(
            f"Warning: no mapping inferred from filenames. "
            f"Using default {DEFAULT_SPECIES_MAPPING!r}.",
            file=sys.stderr,
        )
        return DEFAULT_SPECIES_MAPPING

    if len(inferred_by_model) > 1:
        details = "; ".join(
            f"{mapping}: {', '.join(names)}"
            for mapping, names in sorted(inferred_by_model.items())
        )
        print(
            "Error: models in the folder map to different species mappings. "
            f"{details}. Use one model family per run or pass --species-mapping.",
            file=sys.stderr,
        )
        return None

    mapping = next(iter(inferred_by_model))
    print(f"Species mapping: {mapping} (inferred from model filenames)")
    return mapping


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
    if payload.get("species_mapping") != args.species_mapping:
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
# Material for MkDocs report
# --------------------------------------------------------------------------- #

def write_report(report_path: Path, args: argparse.Namespace,
                 baseline_result: Dict, results: List[Dict]) -> None:
    """Write a Material for MkDocs page comparing every model to the baseline."""
    detection_kind = "raw detections" if args.raw else "merged song segments"
    baseline_dets = baseline_result.get("detections", [])
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    L: List[str] = []
    a = L.append  # shorthand

    suite_name = args.species_mapping or "Model"
    a(f"# {suite_name} Model Types and Format Parity")
    a("")
    a(
        f"BirdBox ships seven pretrained models. "
        f"Two of them, Just-Bird and All-In-One, are released in multiple runtime formats "
        f"to cover different deployment targets. "
        f"This page covers the **{suite_name}** model. "
        f"It documents every available format and shows how each one compares against "
        f"the PyTorch baseline on an identical audio file, so you can confirm that "
        f"conversion or quantization does not degrade detection quality."
    )
    a("")
    a("---")
    a("")

    # Supported formats overview.
    a("## Supported model formats")
    a("")
    a(
        "Each format targets a different deployment scenario. "
        "Install the matching runtime with `python install.py --model-format <FORMAT>`. "
        "See [Installation](../getting-started/installation.md) for the full setup guide."
    )
    a("")
    a("| Format | Typical use case | Runtime |")
    a("| :--- | :--- | :--- |")
    a("| `.pt` | Default. PyTorch checkpoint, easiest to get started. | PyTorch (CUDA or CPU) |")
    a("| `.onnx` | Cross-platform deployment, quantized variants available. | ONNX Runtime (GPU or CPU) |")
    a("| `.tflite` | Edge devices and mobile targets. | LiteRT / ai-edge-litert (CPU) |")
    a("| `.engine` | Maximum throughput on NVIDIA GPUs. | TensorRT (NVIDIA GPU required) |")
    a("")
    a(
        "!!! warning \"Platform restrictions\""
    )
    a(
        "    `.engine` files are compiled for a specific GPU architecture. "
        "A model built on one card may not run on a different GPU generation."
    )
    a("")
    a(
        "!!! danger \"Each format requires its own Python environment\""
    )
    a(
        "    Do not load a `.tflite`, `.onnx`, or `.engine` model from a `.pt` environment. "
        "The wrong environment will either raise an import error immediately or silently degrade results. "
        "Run `python install.py --model-format <FORMAT>` to install the correct runtime before switching formats. "
        "See [Install Parameters](../getting-started/installation.md#install-parameters) for the full table."
    )
    a("")
    a("---")
    a("")

    # Parity test section.
    a("## Format parity test")
    a("")
    a(
        "The table below is produced automatically by running "
        "`python tests/model_format_parity.py` from the repository root. "
        "The PyTorch model is the baseline. Every other format runs inference on the "
        f"same audio clip and its {detection_kind} are matched against the baseline "
        "box by box."
    )
    a("")
    a(
        "!!! note \"Last run\""
    )
    a(
        f"    Generated on {generated}. "
        f"Audio: `{Path(args.audio).name}`, "
        f"species mapping: `{args.species_mapping}`, "
        f"confidence threshold: `{args.conf}`, "
        f"baseline: `{baseline_result['name']}` ({len(baseline_dets)} detections)."
    )
    a("")

    # --- Table 1: at a glance ---
    a("### At a glance")
    a("")
    a("| Model | Format | Size | Detections | Verdict |")
    a("| :--- | :--- | ---: | ---: | :---: |")

    for result in results:
        size_str = _fmt_file_size(result.get("file_size_bytes"))

        if result.get("error"):
            a(
                f"| `{result['name']}` | `{result['format']}` | {size_str} "
                f"| - | :material-close-circle: FAIL |"
            )
            continue

        det_count = (
            len(result["detections"]) if result["is_baseline"]
            else result["comparison"]["candidate_count"]
        )

        if result["is_baseline"]:
            a(
                f"| `{result['name']}` | `{result['format']}` | {size_str} "
                f"| {det_count} | _baseline_ |"
            )
        else:
            verdict = result["verdict"]
            verdict_cell = (
                ":material-check-circle: PASS" if verdict == "PASS"
                else ":material-alert-circle: WARN"
            )
            a(
                f"| `{result['name']}` | `{result['format']}` | {size_str} "
                f"| {det_count} | {verdict_cell} |"
            )
    a("")

    # --- Table 2: detection matching ---
    a("### Detection matching")
    a("")
    a("| Model | Matched | Missed | Extra | Match Rate | Mean IoU |")
    a("| :--- | ---: | ---: | ---: | ---: | ---: |")

    for result in results:
        if result.get("error"):
            a(f"| `{result['name']}` | - | - | - | - | - |")
            continue
        if result["is_baseline"]:
            a(f"| `{result['name']}` | — | — | — | _baseline_ | — |")
            continue
        comp = result["comparison"]
        a(
            f"| `{result['name']}` | {comp['matched']} | {comp['missed']} "
            f"| {comp['extra']} | {comp['match_rate'] * 100:.1f}% "
            f"| {comp['mean_iou']:.3f} |"
        )
    a("")

    # --- Table 3: confidence and timing ---
    a("### Confidence and timing")
    a("")
    a("| Model | Mean Conf Δ | Max Conf Δ | Load (s) | Detect (s) |")
    a("| :--- | ---: | ---: | ---: | ---: |")

    for result in results:
        load_str = _fmt_seconds_plain(result.get("load_seconds"))
        detect_str = _fmt_seconds_plain(result.get("detect_seconds"))

        if result.get("error"):
            a(f"| `{result['name']}` | - | - | - | - |")
            continue
        if result["is_baseline"]:
            a(f"| `{result['name']}` | — | — | {load_str} | {detect_str} |")
            continue
        comp = result["comparison"]
        a(
            f"| `{result['name']}` | {comp['mean_conf_delta']:.4f} "
            f"| {comp['max_conf_delta']:.4f} | {load_str} | {detect_str} |"
        )
    a("")

    # Verdict explanation.
    a("### Verdict criteria")
    a("")
    a(
        f"!!! success \"PASS\""
    )
    a(
        f"    The model matches at least "
        f"{PASS_MIN_MATCH_RATE * 100:.0f}% of baseline detections, adds no more than "
        f"{PASS_MAX_EXTRA_RATE * 100:.0f}% extra detections, and keeps the mean "
        f"confidence difference at or below {PASS_MAX_MEAN_CONF_DELTA}. "
        f"The export is safe to use in place of the baseline."
    )
    a("")
    a(
        "!!! warning \"WARN\""
    )
    a(
        "    One or more thresholds were exceeded. "
        "The model runs but performance may have degraded after conversion or quantization. "
        "Inspect the per-model detail below and compare detections on your own audio before deploying."
    )
    a("")
    a(
        "!!! failure \"FAIL\""
    )
    a(
        "    The worker subprocess exited with an error. "
        "The model did not produce any detections. "
        "Check the per-model detail below for the full traceback."
    )
    a("")

    # Metric glossary.
    a("### Metric glossary")
    a("")
    a(
        "Matching uses a greedy algorithm: for each baseline detection the candidate "
        "detection with the highest 2D IoU (time and frequency) above "
        f"`{MATCH_IOU}` and the same species is selected."
    )
    a("")
    a("| Metric | What it tells you |")
    a("| :--- | :--- |")
    a(
        "| **Mean IoU** | Average time-and-frequency box overlap between matched pairs. "
        "`1.000` is a perfect overlap. Lower values mean boxes drifted in position or size. |"
    )
    a(
        "| **Mean Conf Δ** | Average absolute confidence difference on matched detections. "
        "Near `0.000` means the converted model is as confident as the baseline. |"
    )
    a(
        "| **Max Conf Δ** | Largest single confidence difference among matched detections. "
        "Surfaces worst-case outliers that the mean hides. |"
    )
    a(
        "| **Load (s)** | Wall-clock seconds to load the model file and build the detector. |"
    )
    a(
        "| **Detect (s)** | Wall-clock seconds spent running inference on the audio clip. |"
    )
    a("")
    a("---")
    a("")

    # Per-model detail.
    a("## Per-model detail")
    a("")
    a("---")
    a("")
    baseline_species = species_counts(baseline_dets)

    for result in results:
        a(f"### {result['name']}")
        a("")

        if result.get("error"):
            a(
                "!!! failure \"This model failed to run\""
            )
            # Indent the traceback inside the admonition.
            for line in result["error"].strip().splitlines():
                a(f"    {line}")
            a("")
            a("---")
            a("")
            continue

        if result["is_baseline"]:
            a(
                "!!! info \"Baseline\""
            )
            a(
                "    All other formats are compared against this model. "
                "Its detections define what a correct result looks like."
            )
            a("")

        cand_species = species_counts(result["detections"])
        all_species = sorted(set(baseline_species) | set(cand_species))
        a("| Species | Baseline | This model |")
        a("| :--- | ---: | ---: |")
        for sp in all_species:
            a(
                f"| `{sp}` | {baseline_species.get(sp, 0)} "
                f"| {cand_species.get(sp, 0)} |"
            )
        a("")

        size_str = _fmt_file_size(result.get("file_size_bytes"))
        load_str = _fmt_seconds(result.get("load_seconds"))
        detect_str = _fmt_seconds(result.get("detect_seconds"))

        if result["is_baseline"]:
            a(
                f"File size: {size_str}. "
                f"Load time: {load_str}. "
                f"Detection time: {detect_str}."
            )
        else:
            comp = result["comparison"]
            verdict = result["verdict"]
            if verdict == "PASS":
                admonition = "success"
                label = "PASS"
            else:
                admonition = "warning"
                label = "WARN"
            a(
                f"!!! {admonition} \"{label}\""
            )
            a(
                f"    Matched {comp['matched']} of {comp['baseline_count']} baseline "
                f"detections ({comp['match_rate'] * 100:.1f}%), with {comp['missed']} missed "
                f"and {comp['extra']} extra. "
                f"Mean IoU: {comp['mean_iou']:.3f}. "
                f"Mean Conf Δ: {comp['mean_conf_delta']:.4f}. "
                f"Mean start-time shift: {comp['mean_time_delta']:.3f} s."
            )
            a("")
            a(
                f"File size: {size_str}. "
                f"Load time: {load_str}. "
                f"Detection time: {detect_str}."
            )
        a("")
        a("---")
        a("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(L), encoding="utf-8")


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
# Multi-suite helpers
# --------------------------------------------------------------------------- #

def run_suite(base_args: argparse.Namespace, suite: Dict) -> int:
    """Run one parity suite by overlaying suite paths onto a copy of base_args."""
    args = copy.copy(base_args)
    args.models_dir = str(suite["models_dir"])
    args.results_dir = str(suite["results_dir"])
    args.report = str(suite["report"])
    return run_orchestrator(args)


def run_all_default_suites(base_args: argparse.Namespace) -> int:
    """Run every suite in DEFAULT_SUITES. Returns 0 only when all suites pass."""
    overall_rc = 0
    for suite in DEFAULT_SUITES:
        print(f"\n{'#' * 70}")
        print(f"Suite: {suite['models_dir'].name}")
        print(f"{'#' * 70}")
        rc = run_suite(base_args, suite)
        if rc != 0:
            overall_rc = rc
    return overall_rc


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

    species_mapping = resolve_species_mapping(models, args.species_mapping)
    if species_mapping is None:
        return 1
    args.species_mapping = species_mapping

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
    parser.add_argument(
        "--models-dir", type=str, default=None,
        help=(
            "Folder holding the models to compare. "
            "When omitted, both default suites "
            f"({JUST_BIRD_SUITE['models_dir'].name}, {ALL_IN_ONE_SUITE['models_dir'].name}) "
            "are run in sequence."
        ),
    )
    parser.add_argument("--species-mapping", type=str, default=None,
                        help=(
                            "Species mapping name (e.g. Just-Bird, All-In-One). "
                            "Inferred from model filenames when omitted."
                        ))
    parser.add_argument(
        "--report", type=str, default=None,
        help=(
            "Markdown report output path. "
            "Defaults per suite in dual mode, or "
            f"{_SINGLE_SUITE_DEFAULT_REPORT} in explicit single-suite mode."
        ),
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help=(
            "Where to keep per-model detection JSON. "
            "Defaults per suite in dual mode, or "
            f"{_SINGLE_SUITE_DEFAULT_RESULTS_DIR} in explicit single-suite mode."
        ),
    )
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

    # Dual-suite mode: no explicit models-dir was provided.
    if args.models_dir is None:
        return run_all_default_suites(args)

    # Single-suite mode: fill in defaults for any omitted path flags.
    if args.results_dir is None:
        args.results_dir = str(_SINGLE_SUITE_DEFAULT_RESULTS_DIR)
    if args.report is None:
        args.report = str(_SINGLE_SUITE_DEFAULT_REPORT)
    return run_orchestrator(args)


if __name__ == "__main__":
    sys.exit(main())
