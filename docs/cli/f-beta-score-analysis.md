# F-Beta Score Analysis

Sweep a range of confidence thresholds on raw (unmerged) detections and compute precision, recall, and F-beta scores for every species class at every threshold. Generates performance curves, a species-vs-threshold heatmap, and an `optimal_thresholds.csv` table identifying the best operating point per species.

---

## Usage Synopsis

=== "Linux / macOS"
    ```bash
    python src/evaluation/f_beta_score_analysis.py \
        --raw-detections results \
        --labels path/to/annotations.csv \
        --single-cls
    ```
=== "Windows (PowerShell)"
    ```powershell
    python src/evaluation/f_beta_score_analysis.py `
        --raw-detections results `
        --labels path/to/annotations.csv `
        --single-cls
    ```
=== "Windows (CMD)"
    ```cmd
    python src/evaluation/f_beta_score_analysis.py ^
        --raw-detections results ^
        --labels path/to/annotations.csv ^
        --single-cls
    ```

!!! warning "Input Must Be Raw (Unmerged) Detections"
    This script expects the JSON output of `detect_birds.py --no-merge`. At each threshold it filters the raw boxes first and merges the survivors afterwards, as `detect_birds.py` does. Passing merged detections gives wrong results.

The synopsis uses `--single-cls` for Just-Bird. Drop it for Birds-With-Labels to score each species.

## Parameters

| Parameter | Type / Default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `--raw-detections` | `PATH` / `results` | No | Raw detections **file** or `results/` directory from [`detect_birds.py --no-merge`](detect-birds.md). Default resolves to `raw_detections.json` (follows `results/.active_run` when re-running). Must contain per-clip `confidence` scores. |
| `--labels` | `PATH` / none | **Yes** | Path to the ground truth labels CSV file. Filenames are matched without extensions. |
| `--conf-range` | `MIN MAX STEP` / `0.00 1.0 0.01` | No | Confidence threshold range to test. Three space-separated floats: minimum, maximum, and step size. The default tests 101 thresholds from 0.00 to 1.00 in steps of 0.01. |
| `--beta` | `FLOAT` / `1.0` | No | Beta parameter for the F-beta score formula. Controls the precision/recall trade-off. See [choosing beta](#choosing-beta) below. |
| `--iou-threshold` | `FLOAT` / `0.25` | No | IoU threshold for matching a detection to a ground truth label (0.0 to 1.0). |
| `--merge` | `CHOICE` / `auto` | No | `fusion` (window ownership + weighted box fusion) or `reconstruct` (song merge with `--song-gap` and `--freq-iou`). `auto` uses `model_config.merge_method` from the JSON, which is `fusion` for both models. See [Window fusion](../how-it-works/overview.md#window-fusion). |
| `--phrase-gap` | `FLOAT` / from JSON or `0.05` | No | Fusion only. Join same-band boxes closer than this many seconds into one phrase box. `0` turns it off. Omit to use `model_config.phrase_gap` from the JSON, else the model default of 0.05 s. |
| `--song-gap` | `FLOAT` / from JSON or `0.1` | No | `reconstruct` merge only. Max gap in seconds for merging surviving detections into song boxes at each threshold. Defaults to `model_config.song_gap_threshold` from the JSON. |
| `--freq-iou` | `FLOAT` / `0.6` for Just-Bird, off otherwise | No | `reconstruct` merge with Just-Bird only. Minimum frequency IoU required before two bird detections merge into one song at each threshold. Omit to use the JSON value or `0.6`. Pass a negative value for the time-only merger. |
| `--single-cls` | flag / off | No | Collapse all species into one class for binary bird-detection evaluation. Use it for Just-Bird. |
| `--single-cls-name` | `STR` / `bird` | No | Class name to use when `--single-cls` is active. |
| `--no-optimal-matching` | flag / off | No | Use greedy matching instead of the Hungarian algorithm. Not recommended: greedy matching is order-dependent and produces suboptimal, less reproducible results. |
| `--no-plot` | flag / off | No | Skip generating plot files. Useful for fast sweeps in CI or scripted pipelines. |
| `--output-path` | `PATH` / `results/f_beta_score_analysis` | No | Output directory for all result files. Created automatically if it does not exist. |
| `--num-workers` | `INT` / `1` | No | Number of worker processes for parallelising the confidence-threshold sweep. Each worker handles a subset of thresholds independently. |

## Parameter Deep-Dives

### Choosing Beta

The beta parameter (`--beta`) weights recall relative to precision in the F-beta formula [[25]](#ref-powers), [[27]](#ref-vanrijsbergen):

$$
F_\beta = \frac{(1 + \beta^2) \cdot \text{precision} \cdot \text{recall}}{\beta^2 \cdot \text{precision} + \text{recall}}
$$

where

$$
\text{precision} = \frac{TP}{TP + FP} \qquad \text{recall} = \frac{TP}{TP + FN}
$$

| Beta | Score name | Emphasis | Recommended when |
| :--- | :--- | :--- | :--- |
| `0.5` | F0.5 | Precision counts more than recall | False positives are very costly |
| `1.0` (default) | F1 | Equal weight | General evaluation baseline |
| `2.0` | F2 | Recall counts more than precision | Missing a bird is worse than a false alarm |

!!! info "F1 or F2"
    The pipeline scripts and the models' default thresholds use F1. If missing a bird is worse than a false alarm in your project, use F2 (`--beta 2.0`). It penalizes missed calls more than false detections and usually picks a lower threshold.

### Confidence Range

`--conf-range MIN MAX STEP` defines the grid of confidence thresholds to evaluate. Finer steps give smoother curves and more precise best thresholds. Runtime grows with the number of thresholds.

| Example | Thresholds tested |
| :--- | :--- |
| `0.00 1.0 0.01` (default) | 101 |
| `0.05 0.95 0.05` | 19 |
| `0.10 0.90 0.01` | 81 |

!!! info "Speed Tip"
    Use `--num-workers` to parallelise the sweep across CPU cores. Each worker handles a disjoint set of thresholds independently. Combine with `--no-plot` for the fastest possible sweep.

### Detection Matching

`--iou-threshold` controls how strictly a detection must overlap a ground truth label to count as a True Positive. Each detection and each label counts as one event, as in the event-based metrics for sound event detection [[24]](#ref-mesaros). The same value used here should be used later in `confusion_matrix_analysis.py` for a consistent evaluation.

### Optimal Matching

By default, the Hungarian algorithm [[15]](#ref-kuhn) is used to find the globally optimal assignment of detections to labels at each threshold. This is order-independent and reproducible. The `--no-optimal-matching` flag switches to greedy matching (first-come, first-served), which is faster for very large datasets but produces order-dependent, suboptimal results.

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
| `f{beta}_score_heatmap.png` | Heatmap of F-beta score per species (rows) vs confidence threshold (columns). Species are sorted by their best F-beta. Gray cells mean the species has no labels and no detections at that threshold. |
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

### Reading the Optimal Thresholds File

`optimal_thresholds.csv` lists, for each species, the full result row (same columns as `f{beta}_score_analysis.csv`) at the confidence threshold where that species achieves its best F-beta score:

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

The examples evaluate Just-Bird, so they pass `--single-cls`. For Birds-With-Labels, drop it.

### F1 Sweep (Default)

=== "Command"
    ```bash
    python src/evaluation/f_beta_score_analysis.py \
        --raw-detections results \
        --labels data/annotations.csv \
        --single-cls
    ```
=== "Expected Output (excerpt)"
    ```text
    Initialized F-beta analyzer with IoU threshold: 0.25
    Using F-1.0 score
    Matching method: Optimal (Hungarian)
    Merge: from detections JSON model_config or model defaults
    Single-class mode: True (class='bird')
    Saved run arguments to: results/f_beta_score_analysis/args.yaml
    Will analyze 101 confidence thresholds: [0.0, 0.01, 0.02, ...]
    Loading detections from: results/raw_detections.json
    Loaded 15980 total detections
    ...
    Raw detections: 15980 detections, merge: fusion, phrase_gap=0.05s
    Analyzing F-beta scores (beta=1.0) for 101 confidence thresholds...
    Processing confidence thresholds: 100%|██████████| 101/101 [00:03<00:00, 30.12it/s]
    ...
    Saved optimal thresholds to: results/f_beta_score_analysis/optimal_thresholds.csv
    ...
    F-BETA SCORE ANALYSIS SUMMARY
    ================================================================================
    Best Overall Performance (Micro-Average):
      Confidence Threshold: 0.15
      F1.0-Score: 0.5029
      Precision: 0.3359
      Recall: 1.0000
    ...
    F-BETA SCORE ANALYSIS COMPLETED SUCCESSFULLY
    ```

### F2 Sweep (Recall-Focused)

=== "Command"
    ```bash
    python src/evaluation/f_beta_score_analysis.py \
        --raw-detections results \
        --labels data/annotations.csv \
        --single-cls \
        --beta 2.0 \
        --iou-threshold 0.5 \
        --output-path results/f2_analysis
    ```

The summary then reports `F2.0-Score`, and the files are named `f2.0_score_analysis.csv` and so on.

### Fast Coarse Sweep

=== "Command"
    ```bash
    python src/evaluation/f_beta_score_analysis.py \
        --raw-detections results \
        --labels data/annotations.csv \
        --single-cls \
        --conf-range 0.05 0.95 0.05 \
        --no-plot \
        --output-path results/f1_coarse
    ```
=== "Expected Output (excerpt)"
    ```text
    Will analyze 19 confidence thresholds: [0.05, 0.1, 0.15, ...]
    ...
    F-BETA SCORE ANALYSIS COMPLETED SUCCESSFULLY
    ```

### Parallel Sweep

=== "Command"
    ```bash
    python src/evaluation/f_beta_score_analysis.py \
        --raw-detections results \
        --labels data/annotations.csv \
        --single-cls \
        --num-workers 8 \
        --output-path results/f1_parallel
    ```
=== "Expected Output (excerpt)"
    ```text
    Analyzing F-beta scores (beta=1.0) for 101 confidence thresholds...
    Using multiprocessing with 8 workers
    Processing confidence thresholds: 100%|██████████| 101/101 [00:01<00:00, 80.47it/s]
    ...
    ```

---

## Typical Workflow

This script sits at **Step 2** of the standard evaluation pipeline:

```
Step 1  detect_birds.py --conf 0.001 --no-merge    ->  raw_detections.json
Step 2  f_beta_score_analysis.py                   ->  f_beta_score_analysis/optimal_thresholds.csv
Step 3  filter_and_merge_detections.py --conf 0.15 ->  simplified.csv
Step 4  confusion_matrix_analysis.py               ->  confusion_matrix_analysis/
```

After running this script:

1. Open `optimal_thresholds.csv` and read the `Overall_Micro` row to find the best system-wide threshold.
2. Pass that threshold to `filter_and_merge_detections.py --conf <threshold>`.
3. Feed the resulting CSV into `confusion_matrix_analysis.py` for species-level error analysis.

---

## References

<a id="ref-kuhn"></a>
**[15]** Kuhn, H. W. (1955). "The Hungarian method for the assignment problem." *Naval Research Logistics Quarterly*, 2(1–2):83–97.

<a id="ref-mesaros"></a>
**[24]** Mesaros, A., Heittola, T., & Virtanen, T. (2016). "Metrics for polyphonic sound event detection." *Applied Sciences*, 6(6):162.

<a id="ref-powers"></a>
**[25]** Powers, D. M. (2020). "Evaluation: from precision, recall and F-measure to ROC, informedness, markedness and correlation." arXiv preprint arXiv:2010.16061.

<a id="ref-vanrijsbergen"></a>
**[27]** van Rijsbergen, C. J. (1979). *Information Retrieval* (2nd ed.). Butterworths, London.
