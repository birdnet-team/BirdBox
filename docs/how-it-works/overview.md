# Pipeline Overview

BirdBox evaluation is designed around a **raw-detections-first** workflow:

1. run inference once with permissive confidence and `--no-merge`
2. optimize operating threshold with F-beta analysis
3. apply the chosen threshold and reconstruct songs
4. evaluate class confusion on merged detections

This avoids re-running expensive inference while exploring thresholds.

## Stage Graph

```mermaid
flowchart TD
    A["Audio files"] --> B["detect_birds.py --no-merge<br/>raw detections JSON"]
    B --> C["f_beta_score_analysis.py<br/>threshold sweep"]
    B --> D["filter_and_merge_detections.py<br/>choose threshold + merge"]
    D --> E["simplified.csv"]
    E --> F["confusion_matrix_analysis.py"]
```

## Stage Responsibilities

- `src/inference/detect_birds.py`
  - loads audio, computes PCEN clips, runs YOLO, converts boxes to time/frequency
  - with `--no-merge`, preserves raw detections for later threshold sweeps
- `src/evaluation/f_beta_score_analysis.py`
  - for each confidence threshold: filter raw detections, merge songs, match to labels
  - reports per-class and aggregate precision/recall/F-beta
- `src/evaluation/filter_and_merge_detections.py`
  - applies a single chosen threshold and exports standardized artifacts
- `src/evaluation/confusion_matrix_analysis.py`
  - computes class confusion diagnostics (1D or 2D IoU matching)

## Why Filter Then Merge

!!! info "Order Matters: Filter Before Merge"
    Song reconstruction changes object duration and confidence aggregation. If merging happens before threshold exploration, the threshold behavior becomes biased by a specific pre-merge decision.

    BirdBox therefore always evaluates thresholds in the same order used in deployment:

    1. confidence filter
    2. temporal merge (`song_gap`)

## Typical Command Sequence

```bash
python src/inference/detect_birds.py --audio datasets/SET/soundscape_data --model models/SET.pt --species-mapping SET --output-path results/SET --output-format json-with-algorithm-metadata --conf 0.001 --no-merge
python src/evaluation/f_beta_score_analysis.py --raw-detections results/SET --labels datasets/SET/annotations.csv --output-path results/SET/f_1.0_score_analysis --beta 1.0 --iou-threshold 0.25
python src/evaluation/filter_and_merge_detections.py --raw-detections results/SET --output-path results/SET --conf 0.2 --song-gap 0.1 --output-format all
python src/evaluation/confusion_matrix_analysis.py --detections results/SET --labels datasets/SET/annotations.csv --output-path results/SET/confusion_matrix_analysis --iou-threshold 0.25
```

## End-to-End Workflow

```mermaid
flowchart LR
    A["Audio input<br/>WAV/FLAC/OGG/MP3"] --> B["detect_birds.py<br/>raw detections (--no-merge)"]
    B --> C["f_beta_score_analysis.py<br/>threshold sweep + metrics"]
    C --> D["filter_and_merge_detections.py<br/>apply chosen threshold"]
    D --> E["confusion_matrix_analysis.py<br/>species-level diagnostics"]
```