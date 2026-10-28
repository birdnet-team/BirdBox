
Sweep a range of confidence thresholds on raw (unmerged) detections and compute precision, recall, and F-beta scores for every species class at every threshold. Generates performance curves, a species-vs-threshold heatmap, and an `optimal_thresholds.csv` table identifying the best operating point per species.

---

## Usage Synopsis

=== "Linux / macOS"
    ```bash
    python src/evaluation/f_beta_score_analysis.py \
        --raw-detections results \
        --labels path/to/labels.csv
    ```
=== "Windows (PowerShell)"
    ```powershell
    python src/evaluation/f_beta_score_analysis.py `
        --raw-detections results `
        --labels path/to/labels.csv
    ```
=== "Windows (CMD)"
    ```cmd
    python src/evaluation/f_beta_score_analysis.py ^
        --raw-detections results ^
        --labels path/to/labels.csv
    ```

!!! warning "Input Must Be Raw (Unmerged) Detections"
    This script expects the JSON output of `detect_birds.py --no-merge`. It re-applies filtering and merging internally at each threshold so that the sweep exactly matches the app's filter-then-merge workflow. Passing pre-merged detections will produce incorrect results.

## Parameters

| Parameter | Type / Default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `--raw-detections` | `PATH` / `results` | No | Raw detections **file** or `results/` directory from [`detect_birds.py --no-merge`](detect-birds.md). Default resolves to `raw_detections.json` (follows `results/.active_run` when re-running). Must contain per-clip `confidence` scores. |
| `--labels` | `PATH` / — | **Yes** | Path to the ground truth labels CSV file. Filenames are matched without extensions. |
| `--conf-range` | `MIN MAX STEP` / `0.00 1.0 0.01` | No | Confidence threshold range to test. Three space-separated floats: minimum, maximum, and step size. The default tests 101 thresholds from 0.00 to 1.00 in steps of 0.01. |
| `--beta` | `FLOAT` / `1.0` | No | Beta parameter for the F-beta score formula. Controls the precision/recall trade-off. See [choosing beta](#choosing-beta) below. |
| `--iou-threshold` | `FLOAT` / `0.25` | No | IoU threshold for matching a detection to a ground truth label (0.0–1.0). |
| `--song-gap` | `FLOAT` / from JSON or `0.1` | No | Max temporal gap in seconds for merging surviving detections into song segments at each threshold step. Defaults to the value stored in the JSON's `model_config.song_gap_threshold`. |
| `--freq-iou` | `FLOAT` / `0.6` for Just-Bird, off otherwise | No | Just-Bird only. Minimum frequency IoU required before two bird detections merge into one song at each confidence step. Omit to use the JSON value or `0.6`. Pass a negative value for the time-only merger. Other species mappings ignore this flag. |
| `--merge` | `CHOICE` / `auto` | No | `fusion` (window ownership + weighted box fusion) or `reconstruct` (song merge with `--song-gap` / `--freq-iou`). `auto` uses `model_config.merge_method` from the JSON, else `fusion` for Just-Bird and `reconstruct` for All-In-One. See [Window fusion](detect-birds.md#window-fusion-just-bird-default). |
| `--phrase-gap` | `FLOAT` / model default | No | Fusion only. Join same-band boxes closer than this many seconds into one phrase box, `0` = off. Omit to use `model_config.phrase_gap`, else the model default (0.05 for Just-Bird, 0 for All-In-One). |
| `--single-cls` | flag / off | No | Collapse all species into one class for binary bird-detection evaluation. |
| `--single-cls-name` | `STR` / `bird` | No | Class name to use when `--single-cls` is active. |
| `--no-optimal-matching` | flag / off | No | Use greedy matching instead of the Hungarian algorithm. Not recommended: greedy matching is order-dependent and produces suboptimal, less reproducible results. |
| `--no-plot` | flag / off | No | Skip generating plot files. Useful for fast sweeps in CI or scripted pipelines. |
| `--output-path` | `PATH` / `results/f_beta_score_analysis` | No | Output directory for all result files. Created automatically if it does not exist. |
| `--num-workers` | `INT` / `1` | No | Number of worker processes for parallelising the confidence-threshold sweep. Each worker handles a subset of thresholds independently. |

## Parameter Deep-Dives

### Choosing `--beta`

The beta parameter weights recall relative to precision in the F-beta formula [[7]](#ref-powers):

$$
F_\beta = \frac{(1 + \beta^2) \cdot \text{precision} \cdot \text{recall}}{\beta^2 \cdot \text{precision} + \text{recall}}
$$

where

$$
\text{precision} = \frac{TP}{TP + FP} \qquad \text{recall} = \frac{TP}{TP + FN}
$$

| Beta | Score name | Emphasis | Recommended when |
| :--- | :--- | :--- | :--- |
| `0.5` | F0.5 | Precision × 2 over recall | False positives are very costly |
| `1.0` (default) | F1 | Equal weight | General evaluation baseline |
| `2.0` | F2 | Recall × 2 over precision | Missing a bird is worse than a false alarm |

!!! info "Bird Detection Recommendation"
    For bird call detection, **F2-score (`--beta 2.0`)** is generally preferred. Missing a detection (False Negative) is typically worse than reporting a spurious one (False Positive). The F2-score penalises missed birds more heavily than false alarms.

### `--conf-range MIN MAX STEP`

Defines the grid of confidence thresholds to evaluate. Finer steps give smoother curves and more precise optimal thresholds at the cost of longer runtime.

| Example | Thresholds tested | Runtime (approx.) |
| :--- | :--- | :--- |
| `0.00 1.0 0.01` (default) | 101 | ~1–2 min (single worker) |
| `0.05 0.95 0.05` | 19 | ~15 s |
| `0.10 0.90 0.01` | 81 | ~1 min |

!!! info "Speed Tip"
    Use `--num-workers` to parallelise the sweep across CPU cores. Each worker handles a disjoint set of thresholds independently. Combine with `--no-plot` for the fastest possible sweep.

### `--iou-threshold` — Detection Matching

Controls how strictly a detection must overlap a ground truth label to count as a True Positive. The same value used here should be used later in `confusion_matrix_analysis.py` for a consistent evaluation.

### `--no-optimal-matching`

By default, the Hungarian algorithm is used to find the globally optimal assignment of detections to labels at each threshold. This is order-independent and reproducible. The `--no-optimal-matching` flag switches to greedy matching (first-come, first-served), which is faster for very large datasets but produces order-dependent, suboptimal results.

!!! warning "Not Recommended"
    Only use `--no-optimal-matching` if runtime is critical and you understand the trade-offs. For final published metrics, always use the default optimal matching.

---


## Output Files

All files are written to `--output-path`. The beta value is embedded in filenames (e.g. `f1.0_score_analysis.csv` for the default `--beta 1.0`).

| File | Description |
| :--- | :--- |
| `f{beta}_score_analysis.csv` | Complete results table: one row per (species, confidence threshold) combination with TP, FP, FN, precision, recall, and F-beta score. |
| `f{beta}_score_analysis.json` | Same data in JSON format. |
| `optimal_thresholds.csv` | Best confidence threshold per species (maximising F-beta score). |
| `overall_micro_f{beta}_curve.png` | F-beta, precision, and recall curves for the micro-average across all species. |
| `overall_macro_f{beta}_curve.png` | F-beta, precision, and recall curves for the macro-average across all species. |
| `micro_vs_macro_f{beta}_comparison.png` | Side-by-side comparison of micro and macro averaging methods. |
| `top_species_f{beta}_curves.png` | Per-species curves for the top 12 performing species. |
| `all_species_f{beta}_curves.png` | Per-species curves for all species. |
| `f{beta}_score_heatmap.png` | Heatmap of F-beta score per species (rows) vs confidence threshold (columns). Species are sorted by their best F-beta; gray cells mean the species has no labels and no detections at that threshold. |
| `args.yaml` | Every CLI argument used for this run, including defaults. Written so F-beta runs can be compared later. |

---

## Understanding the Results

### Micro vs Macro Average

Two summary rows are included in the results for each threshold:

| Type of Average | Method | What it tells you |
| :--- | :--- | :--- |
| `Overall_Micro` | Sum all TP, FP, FN across species, then compute F-beta | Overall model performance. Dominated by frequent species. |
| `Overall_Macro` | Compute F-beta per species, then average | Per-class fairness. Gives equal weight to rare and common species. |

Check both: a high micro score with a low macro score means the model performs well on common species but poorly on rare ones.

A species with errors but no true positive at a threshold (only missed calls, or only false detections) scores 0 for precision, recall and F-beta, as with scikit-learn's `zero_division=0`. It therefore pulls the macro average down instead of dropping out of it. Precision, recall and F-beta are only left empty (`NaN`) for a species with no labels and no detections at that threshold.

### Reading `optimal_thresholds.csv`

The file lists, for each species, the full result row (same columns as `f{beta}_score_analysis.csv`) at the confidence threshold where that species achieves its best F-beta score:

```csv
species,confidence_threshold,tp,fp,fn,precision,recall,f_beta_score
amerob,0.35,42.0,6.0,8.0,0.8750,0.8400,0.8571
herthr,0.28,30.0,10.0,5.0,0.7500,0.8571,0.8000
yelwar,0.42,46.0,4.0,5.0,0.9200,0.9020,0.9109
Overall_Micro,0.33,118.0,20.0,18.0,0.8551,0.8676,0.8613
Overall_Macro,0.30,,,,0.8216,0.8540,0.8671
```

In the `Overall_Macro` row, precision, recall and F-beta are each averaged over species separately, so its F-beta is not computed from the precision and recall shown and can lie outside them. Its `tp` / `fp` / `fn` are empty, which is also why the counts in the other rows are written as decimals (`42.0`).

Use the `Overall_Micro` or `Overall_Macro` row to pick a single system-wide threshold when per-species thresholds are impractical.

---

## Examples

### F1-score sweep (default)

=== "Command"
    ```bash
    python src/evaluation/f_beta_score_analysis.py \
        --raw-detections results \
        --labels data/ground_truth.csv
    ```
=== "Expected Output"
    ```text
    Will analyze 101 confidence thresholds: [0.0, 0.01, 0.02, ...]
    Analyzing threshold 0.00: 4823 raw → 312 merged
    Analyzing threshold 0.01: 4823 raw → 298 merged
    ...
    Saved results to CSV: results/f_beta_score_analysis/f1.0_score_analysis.csv
    Saved optimal thresholds to: results/f_beta_score_analysis/optimal_thresholds.csv

    F-BETA SCORE ANALYSIS SUMMARY
    Best Overall Performance (Micro-Average):
      Confidence Threshold: 0.33
      F1-Score: 0.8512
      Precision: 0.8654
      Recall: 0.8421
    ```

### F2-score (recall-focused)

=== "Command"
    ```bash
    python src/evaluation/f_beta_score_analysis.py \
        --raw-detections results \
        --labels data/ground_truth.csv \
        --beta 2.0 \
        --iou-threshold 0.5 \
        --output-path results/f2_analysis
    ```
=== "Expected Output"
    ```text
    Will analyze 101 confidence thresholds
    ...
    Best Overall Performance (Micro-Average):
      Confidence Threshold: 0.22
      F2-Score: 0.8701
      Precision: 0.7943
      Recall: 0.9012
    ```

### Fast coarse sweep

=== "Command"
    ```bash
    python src/evaluation/f_beta_score_analysis.py \
        --raw-detections results \
        --labels data/ground_truth.csv \
        --beta 2.0 \
        --conf-range 0.05 0.95 0.05 \
        --no-plot \
        --output-path results/f2_coarse
    ```
=== "Expected Output"
    ```text
    Will analyze 19 confidence thresholds: [0.05, 0.1, 0.15, ...]
    ...
    FILTERING COMPLETED SUCCESSFULLY
    ```

### Parallel sweep

=== "Command"
    ```bash
    python src/evaluation/f_beta_score_analysis.py \
        --raw-detections results \
        --labels data/ground_truth.csv \
        --beta 2.0 \
        --num-workers 8 \
        --output-path results/f2_parallel
    ```
=== "Expected Output"
    ```text
    Will analyze 101 confidence thresholds
    Threshold sweep: 100%|████████████| 101/101 [00:18<00:00, 5.6thresholds/s]
    ...
    ```

---

## Typical Workflow

This script sits at **Step 2** of the standard evaluation pipeline:

```
Step 1  detect_birds.py --conf 0.001 --no-merge    →  raw_detections.json
Step 2  f_beta_score_analysis.py                   →  optimal_thresholds.csv
Step 3  filter_and_merge_detections.py --conf 0.35 →  simplified.csv
Step 4  confusion_matrix_analysis.py               →  confusion_matrix/
```

After running this script:

1. Open `optimal_thresholds.csv` and read the `Overall_Micro` row to find the recommended system-wide threshold.
2. Pass that threshold to `filter_and_merge_detections.py --conf <threshold>`.
3. Feed the resulting CSV into `confusion_matrix_analysis.py` for species-level error analysis.

---

## References

<a id="ref-powers"></a>
[7] Powers, D. M. (2020). "Evaluation: from precision, recall and F-measure to ROC, informedness, markedness and correlation." arXiv preprint arXiv:2010.16061.