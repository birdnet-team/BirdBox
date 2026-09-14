#!/usr/bin/env python3
"""Sweep inference merge hyperparameters on a frozen checkpoint.

``song_gap`` is applied in F-beta (filter-then-merge) and can be searched from
one ``detect_birds --no-merge`` dump (cheap; no new detect). ``nms_iou`` is
applied at detect time, so each NMS value needs a full inference pass
(expensive; wall time is dominated by those detect runs).

Do not mix this search with learning-rate trials. Freeze merge params while
training, then run this on the winning weights.

Usage (from the BirdBox repo root):

    python src/evaluation/hyperopt_merge.py \\
        --raw-detections results/Northeastern-US \\
        --labels datasets/Northeastern-US_testset/annotations.csv \\
        --song-gaps 0.0 0.1 0.2 0.5 1.0 2.0

    python src/evaluation/hyperopt_merge.py \\
        --model models/Northeastern-US.pt \\
        --audio datasets/Northeastern-US_testset/soundscape_data \\
        --species-mapping Northeastern-US \\
        --labels datasets/Northeastern-US_testset/annotations.csv \\
        --nms-ious 0.5 0.6 0.7 0.8 \\
        --song-gaps 0.1 0.2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

BIRDBOX_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.f_beta_score_analysis import FBetaScoreAnalyzer
from inference.utils.output_paths import (
    is_default_results_path,
    resolve_raw_detections_json,
    resolve_results_directory,
)

SPECIES_CHOICES = [
    "Just-Bird",
    "All-In-One",
    "Hawaii",
    "Northeastern-US",
    "Southern-Sierra-Nevada",
    "Western-US",
    "Amazon-Basin",
]


def _conf_thresholds(conf_range: Tuple[float, float, float]) -> List[float]:
    start, stop, step = conf_range
    values = np.arange(start, stop + step, step)
    return [round(float(value), 3) for value in values]


def _best_overall(df: pd.DataFrame, species: str) -> Optional[pd.Series]:
    subset = df[df["species"] == species]
    valid = subset["f_beta_score"].dropna()
    if valid.empty:
        return None
    return subset.loc[valid.idxmax()]


def summarize_results(df: pd.DataFrame, *, beta: float) -> Dict:
    micro = _best_overall(df, "Overall_Micro")
    macro = _best_overall(df, "Overall_Macro")
    if micro is None:
        raise ValueError("F-beta results have no Overall_Micro row")
    return {
        "micro_f1": float(micro["f_beta_score"]),
        "macro_f1": float(macro["f_beta_score"]) if macro is not None else float("nan"),
        "best_conf": float(micro["confidence_threshold"]),
        "precision": float(micro["precision"]),
        "recall": float(micro["recall"]),
        "beta": beta,
    }


def run_detect(
    *,
    audio: Path,
    model: Path,
    species_mapping: str,
    output_dir: Path,
    nms_iou: float,
    conf: float,
    num_workers: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "src/inference/detect_birds.py",
        "--audio", str(audio),
        "--model", str(model),
        "--species-mapping", species_mapping,
        "--output-path", str(output_dir),
        "--conf", str(conf),
        "--nms-iou", str(nms_iou),
        "--no-merge",
        "--num-workers", str(num_workers),
    ]
    print(f"+ {' '.join(cmd)}", flush=True)
    completed = subprocess.run(cmd, cwd=str(BIRDBOX_ROOT), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"detect_birds failed for nms_iou={nms_iou}")
    raw_path = output_dir / "raw_detections.json"
    if not raw_path.is_file():
        raise FileNotFoundError(f"detect_birds did not write {raw_path}")
    return raw_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid-search song_gap and/or nms_iou using 1-minute event F-beta"
    )
    parser.add_argument(
        "--raw-detections",
        type=str,
        default=None,
        help="Existing raw_detections.json or results directory (song_gap-only search; no new detect)",
    )
    parser.add_argument("--labels", type=str, required=True, help="Ground-truth annotations.csv")
    parser.add_argument(
        "--output-path",
        type=str,
        default="results/hyperopt_merge",
        help="Directory for per-setting F-beta runs and summary.csv",
    )
    parser.add_argument(
        "--song-gaps",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.2, 0.5, 1.0, 2.0],
        help="song_gap values in seconds (default: 0 0.1 0.2 0.5 1 2)",
    )
    parser.add_argument(
        "--nms-ious",
        nargs="+",
        type=float,
        default=None,
        help=(
            "If set, re-run detect_birds --no-merge once per NMS IoU "
            "(expensive: one full inference pass per value), then sweep "
            "song_gap on each dump"
        ),
    )
    parser.add_argument("--model", type=str, default=None, help="Checkpoint for --nms-ious")
    parser.add_argument("--audio", type=str, default=None, help="Wav or directory for --nms-ious")
    parser.add_argument(
        "--species-mapping",
        type=str,
        choices=SPECIES_CHOICES,
        default=None,
    )
    parser.add_argument("--detect-conf", type=float, default=0.001)
    parser.add_argument("--detect-workers", type=int, default=8)
    parser.add_argument("--iou-threshold", type=float, default=0.25)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument(
        "--conf-range",
        nargs=3,
        type=float,
        metavar=("MIN", "MAX", "STEP"),
        default=[0.00, 1.0, 0.01],
    )
    parser.add_argument("--num-workers", type=int, default=8, help="F-beta confidence-sweep workers")
    parser.add_argument("--no-plot", action="store_true", help="Skip F-beta plots (faster)")
    parser.add_argument("--single-cls", action="store_true")
    parser.add_argument("--single-cls-name", type=str, default="bird")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_path)
    output_root.mkdir(parents=True, exist_ok=True)
    labels_path = Path(args.labels)
    if not labels_path.is_file():
        raise FileNotFoundError(f"Labels not found: {labels_path}")

    conf_thresholds = _conf_thresholds(tuple(args.conf_range))
    rows: List[Dict] = []

    if args.nms_ious:
        if not args.model or not args.audio or not args.species_mapping:
            raise SystemExit("--nms-ious requires --model, --audio, and --species-mapping")
        model = Path(args.model)
        audio = Path(args.audio)
        if not model.is_file():
            raise FileNotFoundError(f"Model not found: {model}")
        if not audio.exists():
            raise FileNotFoundError(f"Audio not found: {audio}")
        raw_jobs = []
        for nms_iou in args.nms_ious:
            detect_dir = output_root / f"nms_{nms_iou:.2f}" / "detect"
            raw_path = run_detect(
                audio=audio,
                model=model,
                species_mapping=args.species_mapping,
                output_dir=detect_dir,
                nms_iou=nms_iou,
                conf=args.detect_conf,
                num_workers=args.detect_workers,
            )
            raw_jobs.append((nms_iou, raw_path))
    else:
        if not args.raw_detections:
            raise SystemExit("Provide --raw-detections, or --nms-ious with --model/--audio")
        raw_path = args.raw_detections
        if is_default_results_path(raw_path):
            raw_path = resolve_results_directory(raw_path)
        raw_path = Path(resolve_raw_detections_json(raw_path))
        if not raw_path.is_file():
            raise FileNotFoundError(f"Raw detections not found: {raw_path}")
        nms_from_file = None
        with raw_path.open(encoding="utf-8") as handle:
            model_config = json.load(handle).get("model_config", {})
        nms_from_file = model_config.get("nms_iou_threshold")
        raw_jobs = [(nms_from_file, raw_path)]

    probe = FBetaScoreAnalyzer(
        iou_threshold=args.iou_threshold,
        beta=args.beta,
        song_gap=args.song_gaps[0],
        single_cls=args.single_cls,
        single_cls_name=args.single_cls_name,
        log_initialization=False,
    )

    for nms_iou, raw_path in raw_jobs:
        detections_data = probe.load_detections(str(raw_path))
        labels = probe.load_labels(str(labels_path))
        for song_gap in args.song_gaps:
            nms_tag = f"{float(nms_iou):.2f}" if nms_iou is not None else "fixed"
            setting_dir = output_root / f"nms_{nms_tag}" / f"song_gap_{song_gap:.2f}"
            print(f"=== nms_iou={nms_tag} song_gap={song_gap} ===", flush=True)
            analyzer = FBetaScoreAnalyzer(
                iou_threshold=args.iou_threshold,
                beta=args.beta,
                song_gap=song_gap,
                single_cls=args.single_cls,
                single_cls_name=args.single_cls_name,
            )
            results_df = analyzer.analyze_confidence_thresholds(
                str(raw_path),
                str(labels_path),
                conf_thresholds,
                num_workers=args.num_workers,
                detections_data=detections_data,
                labels=list(labels),
            )
            setting_dir.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(setting_dir / f"f{args.beta}_score_analysis.csv", index=False)
            analyzer.find_optimal_thresholds(results_df).to_csv(
                setting_dir / "optimal_thresholds.csv", index=False
            )
            if not args.no_plot:
                analyzer.plot_f_beta_curves(results_df, str(setting_dir), top_classes=12)
            analyzer.print_summary(results_df)
            summary = summarize_results(results_df, beta=args.beta)
            summary.update(
                {
                    "nms_iou": nms_iou,
                    "song_gap": song_gap,
                    "iou_threshold": args.iou_threshold,
                    "raw_detections": str(raw_path),
                    "f_beta_dir": str(setting_dir),
                }
            )
            (setting_dir / "score.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            rows.append(summary)
            print(
                f"micro F{args.beta}={summary['micro_f1']:.4f}  "
                f"conf={summary['best_conf']:.2f}  "
                f"P={summary['precision']:.4f}  R={summary['recall']:.4f}",
                flush=True,
            )

    summary_df = pd.DataFrame(rows)
    summary_csv = output_root / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    if not summary_df.empty:
        best = summary_df.loc[summary_df["micro_f1"].idxmax()]
        print(
            f"\nBest: nms_iou={best.get('nms_iou')} song_gap={best['song_gap']} "
            f"micro F{args.beta}={best['micro_f1']:.4f} conf={best['best_conf']:.2f}",
            flush=True,
        )
        print(f"Wrote {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
