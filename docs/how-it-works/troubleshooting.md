# Troubleshooting Notes

## No Detections

!!! warning "No Detections"
    - Verify that `--model` and `--species-mapping` are compatible (the mapping must match the one used during training).
    - Lower the confidence threshold for exploratory runs (`--conf 0.1` or `--conf 0.001` for raw capture).
    - Check audio format and quality — WAV/FLAC outperform lossy MP3/OGG for faint calls.
    - Confirm that the audio file actually covers the time ranges recorded in your labels (for evaluation runs).

## Too Many False Positives

!!! warning "Too Many False Positives"
    - Increase `--conf` first to raise the detection bar.
    - Reduce `--song-gap` to prevent unrelated events from being merged into one detection.
    - Tune `--nms-iou` to control duplicate suppression.
    - Use `confusion_matrix_analysis.py` and inspect the background row/column to distinguish species confusion from generic noise hits.

## Memory or Runtime Issues

!!! warning "Memory or Runtime Issues"
    - Reduce parallelism (`--workers` for inference, `--num-workers` for F-beta analysis).
    - Process subsets of files and merge reports afterwards.
    - Prefer GPU-backed runs for large jobs when available.
    - Generate the raw detections JSON once with `--no-merge` — avoid re-running inference during threshold experiments.

## Filename Mismatch in Evaluation

!!! warning "Filename Mismatch"
    Evaluation tools match filenames by their **normalized stem** (extension stripped):

    - `recording_01.wav` and `recording_01.flac` **match** ✓
    - `siteA_recording_01.wav` and `recording_01.flac` **do not match** ✗

    If matching fails, inspect both the detection CSV/JSON filename fields and the labels CSV, then unify base names before re-running metrics.
