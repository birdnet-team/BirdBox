# Results

By default, BirdBox writes outputs under `results/`.

## Recommended Layout

```text
results/
  <dataset_name>/
    raw_detections.json
    f_1.0_score_analysis/
      f1.0_score_analysis.csv
      optimal_thresholds.csv
      ...
    merged_detections.json
    merged_detections.csv
    confusion_matrix_analysis/
      confusion_matrix.csv
      confusion_matrix_normalized.png
      metadata.txt
```

## Reproducibility Tips

!!! tip "Keep Your Results Reproducible"
    - Keep one subfolder per dataset/model configuration.
    - Always preserve the raw detections JSON — it is the source for all threshold experiments and can be re-filtered without re-running inference.
    - Archive the exact `--conf` threshold and `--song-gap` value used for any final exports.
