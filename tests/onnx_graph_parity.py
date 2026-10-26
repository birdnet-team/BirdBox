#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 BirdNET-Team
# SPDX-License-Identifier: MIT
"""
Compare self-contained ONNX graphs against the Python detection pipeline.

``src/inference/onnx_export.py`` bakes PCEN, the network, NMS and the merge
into one graph (the one the browser demo runs). This test checks that such a
graph still reproduces ``detect_birds.py``, the path the CLI uses. ``tests/model_format_parity.py`` covers the plain network exports
(.onnx, .tflite, .engine), which run inside ``detect_birds.py`` instead.

How the comparison is set up
----------------------------
Each graph stores its export settings in ONNX metadata: source model, species
mapping, merge method, NMS IoU, default conf and gap. The reference
``BirdCallDetector`` is built from the same values, so the two sides only
differ in the implementation. Audio is resampled with librosa for both sides,
as ``pcen_inference.py`` does; the graph itself expects 32 kHz input.

Merged boxes are matched one-to-one at IoU >= 0.5 with the matcher of
``model_format_parity.py``, and the PASS / WARN verdict uses its thresholds.

Test audio
----------
tests/test.wav plus the xeno-canto soundscapes in tests/parity_audio (fetch
them once with ``python tests/parity_audio/download.py``).

Environment
-----------
Needs the detection pipeline and ONNX Runtime in one env, for example
birdbox-onnx (the export env).

Usage
-----
    # The browser demo model against models/Just-Bird.pt
    python tests/onnx_graph_parity.py --onnx docs/models/Just-Bird_fp16.onnx

    # Several graphs, other audio, results as JSON
    python tests/onnx_graph_parity.py --onnx private_models/Just-Bird_fp32.onnx \\
        docs/models/Just-Bird_fp16.onnx --audio my_recordings/ --json parity.json

Exit code 0 when every graph passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(TESTS_DIR))

from model_format_parity import (  # noqa: E402
    DEFAULT_AUDIO,
    compare_to_baseline,
    resolve_audio_files,
    verdict_for,
)

GRAPH_SAMPLE_RATE = 32000


def graph_metadata(session) -> Dict[str, str]:
    """ONNX metadata_props written by onnx_export.py."""
    return dict(session.get_modelmeta().custom_metadata_map)


def resolve_settings(meta: Dict[str, str], args: argparse.Namespace) -> Dict:
    """Detection settings for both sides: graph metadata, then command-line overrides."""
    pt_model = args.pt_model or str(REPO_ROOT / "models" / meta["source_model"])
    species_mapping = meta.get("species_mapping")
    if not species_mapping or species_mapping == "unknown":
        sys.exit(f"Graph has no species mapping in its metadata: {meta.get('source_model')}")
    return {
        "pt_model": pt_model,
        "species_mapping": species_mapping,
        "merge_method": meta.get("merge_method", "reconstruct"),
        "conf": args.conf if args.conf is not None else float(meta["default_conf"]),
        "gap": args.gap if args.gap is not None else float(meta["default_song_gap"]),
        "nms_iou": float(meta["nms_iou_threshold"]),
        "freq_iou": float(meta["freq_iou_threshold"]) if "freq_iou_threshold" in meta else None,
    }


def build_detector(settings: Dict):
    """BirdCallDetector configured like the graph."""
    from inference.detect_birds import BirdCallDetector

    fusion = settings["merge_method"] == "fusion"
    return BirdCallDetector(
        model_path=settings["pt_model"],
        species_mapping=settings["species_mapping"],
        conf_threshold=settings["conf"],
        nms_iou_threshold=settings["nms_iou"],
        song_gap_threshold=0.1 if fusion else settings["gap"],
        freq_iou_threshold=settings["freq_iou"],
        merge_method=settings["merge_method"],
        phrase_gap=settings["gap"] if fusion else None,
        verbose=False,
    )


def load_graph_audio(detector, audio_path: Path) -> np.ndarray:
    """Mono float32 at 32 kHz, loaded and resampled as detect_birds.py does."""
    import librosa

    audio, sr = detector.load_audio(str(audio_path))
    if sr != GRAPH_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=GRAPH_SAMPLE_RATE)
    return np.ascontiguousarray(audio, dtype=np.float32)


def graph_rows_to_detections(rows: np.ndarray, file_name: str) -> List[Dict]:
    """Graph output columns -> the schema compare_to_baseline reads."""
    return [
        {
            "file": file_name,
            "species_id": int(round(row[7])),
            "confidence": float(row[4]),
            "time_start": float(row[0]),
            "time_end": float(row[1]),
            "freq_low_hz": float(row[2]),
            "freq_high_hz": float(row[3]),
        }
        for row in rows
    ]


def reference_to_detections(detections: List[Dict], file_name: str) -> List[Dict]:
    return [
        {
            "file": file_name,
            "species_id": int(det["species_id"]),
            "confidence": float(det["avg_confidence"]),
            "time_start": float(det["time_start"]),
            "time_end": float(det["time_end"]),
            "freq_low_hz": float(det["freq_low_hz"]),
            "freq_high_hz": float(det["freq_high_hz"]),
        }
        for det in detections
    ]


def run_graph(graph_path: Path, audio_files: List[Path], args: argparse.Namespace,
              reference_cache: Dict[Tuple, Tuple]) -> Dict:
    """Run one graph and its matching reference on every recording."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(graph_path), providers=["CPUExecutionProvider"])
    settings = resolve_settings(graph_metadata(session), args)
    key = tuple(sorted(settings.items()))
    if key not in reference_cache:
        reference_cache[key] = (build_detector(settings), {})
    detector, reference_by_file = reference_cache[key]

    print(f"\n{graph_path.name}: {settings['species_mapping']}, {settings['merge_method']}, "
          f"conf {settings['conf']}, gap {settings['gap']}, NMS IoU {settings['nms_iou']}, "
          f"reference {Path(settings['pt_model']).name}")
    print(f"  {'recording':32s} {'seconds':>8s} {'ref':>5s} {'onnx':>5s} {'matched':>8s} "
          f"{'mean IoU':>9s} {'onnx s':>7s}")

    reference: List[Dict] = []
    candidate: List[Dict] = []
    graph_seconds = 0.0
    for audio_path in audio_files:
        name = audio_path.name
        if name not in reference_by_file:
            reference_by_file[name] = reference_to_detections(
                detector.detect_single_file(str(audio_path)), name)
        audio = load_graph_audio(detector, audio_path)
        start = time.perf_counter()
        rows = session.run(None, {
            "audio": audio,
            "conf": np.array([settings["conf"]], dtype=np.float32),
            "song_gap": np.array([settings["gap"]], dtype=np.float32),
        })[0]
        elapsed = time.perf_counter() - start
        graph_seconds += elapsed
        ref_file = reference_by_file[name]
        got_file = graph_rows_to_detections(rows, name)
        reference += ref_file
        candidate += got_file

        per_file = compare_to_baseline(ref_file, got_file)
        print(f"  {name[:32]:32s} {len(audio) / GRAPH_SAMPLE_RATE:8.1f} {len(ref_file):5d} "
              f"{len(got_file):5d} {per_file['matched']:8d} {per_file['mean_iou']:9.4f} {elapsed:7.1f}")

    comparison = compare_to_baseline(reference, candidate)
    verdict = verdict_for(comparison)
    print(f"  {'total':32s} {'':8s} {comparison['baseline_count']:5d} "
          f"{comparison['candidate_count']:5d} {comparison['matched']:8d} "
          f"{comparison['mean_iou']:9.4f} {graph_seconds:7.1f}")
    print(f"  {verdict}: {comparison['match_rate']:.1%} of reference boxes matched, "
          f"{comparison['extra']} extra, mean |conf delta| {comparison['mean_conf_delta']:.4f}")
    return {"graph": str(graph_path), "settings": settings, "verdict": verdict,
            "comparison": comparison, "graph_seconds": graph_seconds}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare self-contained ONNX graphs (onnx_export.py) against detect_birds.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--onnx", nargs="+", required=True, type=Path,
                        help="Graph(s) written by src/inference/onnx_export.py")
    parser.add_argument("--audio", nargs="+", default=[str(p) for p in DEFAULT_AUDIO],
                        help="Audio files or folders (default: tests/test.wav and tests/parity_audio)")
    parser.add_argument("--pt-model", default=None,
                        help="Reference .pt model (default: models/<source_model from the graph metadata>)")
    parser.add_argument("--conf", type=float, default=None,
                        help="Confidence threshold for both sides (default: the graph's default_conf)")
    parser.add_argument("--gap", type=float, default=None,
                        help="Phrase gap (fusion) or song gap for both sides (default: the graph's default_song_gap)")
    parser.add_argument("--json", type=Path, default=None, help="Also write the results to this JSON file")
    args = parser.parse_args(argv)

    audio_files = resolve_audio_files(args.audio)
    if not audio_files:
        print("No audio files to test.", file=sys.stderr)
        return 1
    print(f"Recordings: {len(audio_files)}")

    reference_cache: Dict[Tuple, Tuple] = {}
    results = [run_graph(path, audio_files, args, reference_cache) for path in args.onnx]

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2))
    return 0 if all(r["verdict"] == "PASS" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
