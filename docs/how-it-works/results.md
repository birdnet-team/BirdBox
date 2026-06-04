# Results

By default, BirdBox writes outputs under `results/`.

## Recommended Layout

```text
results/
  raw_detections.json              ← detect_birds --no-merge
  with_algorithm_metadata.json     ← filter_and_merge (optional)
  simplified.csv                   ← filter_and_merge (confusion matrix input)
  f_1.0_score_analysis/            ← f_beta_score_analysis outputs
  confusion_matrix_analysis/       ← confusion matrix outputs

results/run_2/                     ← created automatically if you re-run the pipeline
  raw_detections.json
  simplified.csv
  ...
```

When you run the default pipeline again while `results/` already contains detection outputs, `detect_birds.py` writes to `results/run_2/` (then `run_3/`, …) and stores the active path in `results/.active_run`. Later steps with default paths follow that marker.

Per-dataset runs (e.g. `results/Hawaii/`) use a single flat folder without auto-versioning.

## Reproducibility Tips

!!! tip "Keep Your Results Reproducible"
    - Keep one folder per dataset/model configuration when comparing runs.
    - Always preserve `raw_detections.json` — it is the source for all threshold experiments and can be re-filtered without re-running inference.
    - Archive the exact `--conf` threshold and `--song-gap` value used for any final exports.
