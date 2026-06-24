# Troubleshooting

This section provides information for typical mistakes one can make when interacting with BirdBox.

## No Detections

If BirdBox returns an empty results file, the most common cause is a model/mapping mismatch or a threshold that is set too high for exploratory work.

- Verify that `--model` and `--species-mapping` are compatible. The mapping file must be the one used during training.
- Lower the confidence threshold for exploratory runs. Try `--conf 0.1` as a starting point, or `--conf 0.001` to capture every raw hit.
- Check audio format and quality. WAV and FLAC outperform lossy formats such as MP3 and OGG for faint or distant calls.
- Confirm that the audio file actually covers the time ranges recorded in your labels when running an evaluation.

!!! tip "Quick sanity check"
    Run BirdBox on a short clip you know contains a loud, clear call. If that also returns nothing, the issue is almost certainly the model/mapping pair, not the audio.

## Too Many False Positives

False positives usually indicate that the confidence threshold is too permissive or that overlapping detections are being merged incorrectly.

- Increase `--conf` to raise the detection bar. Even a small step (e.g., `0.3` to `0.5`) can cut noise significantly.
- Reduce `--song-gap` to prevent unrelated events from being merged into a single detection.
- Tune `--nms-iou` to control how aggressively duplicate bounding boxes are suppressed.
- Run `confusion_matrix_analysis.py` and inspect the background row and column to distinguish species confusion from generic noise hits.

## Memory or Runtime Issues

Large audio collections can put pressure on RAM or stretch runtimes. Process smarter before scaling up hardware.

- Reduce parallelism by lowering `--workers` for inference and `--num-workers` for F-beta analysis.
- Process subsets of files and merge the resulting reports afterwards rather than running everything in one pass.
- Prefer GPU-backed runs for large jobs when a compatible device is available.

!!! warning "Avoid redundant inference"
    Generate the raw detections JSON once using `--no-merge`. Re-running full inference just to experiment with confidence thresholds wastes significant time on large datasets.

## Filename Mismatch in Evaluation

Evaluation tools match audio files against labels using the **normalized stem**, which is the base filename with the extension stripped. If stems do not match, the file is silently skipped.

| Audio file | Label file | Result |
| :--- | :--- | :--- |
| `recording_01.wav` | `recording_01.flac` | Match |
| `siteA_recording_01.wav` | `recording_01.flac` | No match |

!!! tip "How to fix a mismatch"
    Inspect both the `filename` fields in your detection CSV or JSON and the filenames in your labels CSV. Unify the base names in one of the two sources, then re-run metrics.
