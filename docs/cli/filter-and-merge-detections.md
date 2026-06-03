# `src/evaluation/filter_and_merge_detections.py`

Applies a chosen confidence threshold to raw detections and reconstructs merged song segments.

## Purpose

This tool is the deployment-equivalent post-step after threshold optimization:

- input: raw JSON produced with `detect_birds.py --no-merge`
- operation: filter by confidence, then merge by temporal gap
- output: final analysis/export artifacts without repeating inference

## Usage

```bash
python src/evaluation/filter_and_merge_detections.py \
  --input results/Hawaii/raw_detections.json \
  --output-path results/Hawaii/filtered_detections \
  --conf 0.2 \
  --song-gap 0.1 \
  --output-format all
```

## Parameters

| Flag | Required | Default | Description |
|---|---|---:|---|
| `--input` | yes | - | Raw detections JSON from inference (`--no-merge`). |
| `--output-path` | no | `results/filtered_detections` | Output base path (extension by format). |
| `--conf` | no | `0.2` | Confidence threshold in `[0.0, 1.0]`. |
| `--song-gap` | no | JSON config or `0.1` | Merge gap in seconds. |
| `--output-format` | no | `json-with-algorithm-metadata` | One of `json-with-algorithm-metadata`, `simplified-csv`, `xeno-canto-annota-json`, `raven-selection-table`, `all`. |

## Output Semantics

- JSON retains:
  - `model_config` from raw input
  - `filtering_config` for traceability (`confidence_threshold`, `song_gap_threshold`)
  - counts (`original_detection_count`, merged `detection_count`)
- CSV is compatible with annotation tooling.
- Xeno-Canto and Raven exporters target common annotation ecosystems.

## Files You Interact With

- input JSON (`--input`): raw detections file
- output base (`--output-path`): final filtered/merged artifacts
- optional species mapping context embedded in JSON `model_config` is reused for exporter behavior

## Common Pattern

Use `optimal_thresholds.csv` from F-beta analysis, then run this command once with the selected threshold and the same `song-gap` assumptions used during evaluation.
