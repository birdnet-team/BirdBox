# F-Beta Score Analysis

Sweep a range of confidence thresholds on raw (unmerged) detections and compute precision, recall, and F-beta scores for every species class at every threshold. The class returns DataFrames. It writes files only through `plot_f_beta_curves`, or when you save the DataFrames yourself.

---

## Import

```python
from evaluation.f_beta_score_analysis import FBetaScoreAnalyzer
```

---

## FBetaScoreAnalyzer

### Constructor

```python
FBetaScoreAnalyzer(
    iou_threshold=0.25,
    beta=1.0,
    use_optimal_matching=True,
    song_gap=None,
    single_cls=False,
    single_cls_name="bird",
    freq_iou_threshold=None,
    log_initialization=True,
    merge_method=None,
    phrase_gap=None,
)
```

| Parameter | Type / Default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `iou_threshold` | `float` / `0.25` | No | IoU threshold for matching a detection to a ground truth label (0.0 to 1.0). Use the same value later in [`ConfusionMatrixAnalyzer`](confusion-matrix-analysis.md) for consistent evaluation. |
| `beta` | `float` / `1.0` | No | Beta parameter for the F-beta score. Controls the weighting of precision and recall. See [Choosing beta](#choosing-beta) below. |
| `use_optimal_matching` | `bool` / `True` | No | If `True`, uses the Hungarian algorithm for globally optimal assignment. If `False`, uses greedy matching. Greedy matching is faster on very large datasets but order-dependent and not recommended for published metrics. |
| `song_gap` | `float` / `None` | No | `reconstruct` merge only. Max gap in seconds to merge detections into song boxes at each threshold. When `None`, reads `model_config.song_gap_threshold` from the JSON, falling back to `0.1`. |
| `single_cls` | `bool` / `False` | No | Collapse all species into one class for binary bird-detection evaluation. Use it for Just-Bird. |
| `single_cls_name` | `str` / `'bird'` | No | Class name to use when `single_cls=True`. |
| `freq_iou_threshold` | `float` / `None` | No | `reconstruct` merge with Just-Bird only. Frequency IoU gate. `None` uses the JSON value or `0.6`. |
| `log_initialization` | `bool` / `True` | No | If `True`, print the settings when the analyzer is created. |
| `merge_method` | `str` / `None` | No | `'fusion'` or `'reconstruct'`. `None` uses `model_config.merge_method` from the JSON, which is `'fusion'` for both models. |
| `phrase_gap` | `float` / `None` | No | Fusion only. `None` uses `model_config.phrase_gap` from the JSON, else the model default of 0.05 s. |

---

### Methods

| Method | Returns | Description |
| :--- | :--- | :--- |
| `analyze_confidence_thresholds(detections_path, labels_path, confidence_thresholds, num_workers=1, detections_data=None, labels=None)` | `pd.DataFrame` | Run the full threshold sweep. Primary entry point. |
| `find_optimal_thresholds(df)` | `pd.DataFrame` | Best threshold per species from the sweep results. |
| `plot_f_beta_curves(df, output_dir, top_classes=10)` | `None` | Write the curve plots and the heatmap to `output_dir`, which must exist. |
| `print_summary(df)` | `None` | Print the best micro and macro scores and the top species. |
| `load_detections(detections_path)` | `Dict` | Load raw detections JSON. |
| `load_labels(labels_path)` | `List[Dict]` | Load ground truth labels CSV. |
| `filter_detections_by_confidence(detections_data, conf_threshold, verbose=False)` | `List[Dict]` | Filter and merge detections at a single threshold. |

---

### The analyze_confidence_thresholds Method

```python
results_df = analyzer.analyze_confidence_thresholds(
    detections_path,
    labels_path,
    confidence_thresholds,
    num_workers=1,
    detections_data=None,
    labels=None,
)
```

Run the confidence-threshold sweep. At each threshold, raw detections are filtered by confidence and then merged. Then precision, recall, and F-beta scores are computed per species.

!!! warning "Input Must Be Raw (Unmerged) Detections"
    Pass the output of `BirdCallDetector.detect(..., no_merge=True)`. Merged detections give wrong results, because the sweep filters and merges by itself at every threshold.

| Parameter | Type / Default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `detections_path` | `str` / none | **Yes** | Path to `raw_detections.json` from [`BirdCallDetector`](detect-birds.md) with `no_merge=True`. |
| `labels_path` | `str` / none | **Yes** | Path to the ground truth labels CSV file. |
| `confidence_thresholds` | `List[float]` / none | **Yes** | Ordered list of confidence thresholds to evaluate. Build it with `np.round(np.arange(0.0, 1.01, 0.01), 2).tolist()` for the default 101-step sweep. |
| `num_workers` | `int` / `1` | No | Number of worker processes. Each handles a disjoint subset of thresholds. |
| `detections_data` | `dict` / `None` | No | Preloaded raw-detections dict (same structure as `load_detections`). Skips re-reading the JSON when the same dump is scored more than once. |
| `labels` | `list[dict]` / `None` | No | Preloaded ground-truth rows (same structure as `load_labels`). Skips re-reading the CSV. |

**Returns:** `pd.DataFrame` with one row per `(species, confidence_threshold)` combination, plus `Overall_Micro` and `Overall_Macro` summary rows per threshold. Columns: `species`, `confidence_threshold`, `tp`, `fp`, `fn`, `precision`, `recall`, `f_beta_score`.

---

### The find_optimal_thresholds Method

```python
optimal = analyzer.find_optimal_thresholds(results_df)
```

Return one row per species, plus `Overall_Micro` and `Overall_Macro`, at the threshold where it reaches its best F-beta score. The columns are the same as in the sweep results. See [Reading optimal_thresholds.csv](#reading-optimal_thresholdscsv).

---

### The load_detections Method

```python
data = analyzer.load_detections(detections_path)
```

Load a `raw_detections.json` file produced by `BirdCallDetector(..., no_merge=True)`. Returns the full JSON as a dict, including `model_config` metadata.

---

### The load_labels Method

```python
labels = analyzer.load_labels(labels_path)
```

Load a ground truth labels CSV. Returns a list of dicts with keys `filename`, `start_time`, `end_time`, `freq_low`, `freq_high`, `species`. Label dicts use different key names than detection dicts. Anything passed via the `labels=` parameter of `analyze_confidence_thresholds()` must use these keys, or matching fails with a `KeyError`.

---

### The filter_detections_by_confidence Method

```python
merged = analyzer.filter_detections_by_confidence(detections_data, conf_threshold, verbose=False)
```

Apply a single confidence threshold to raw detections, then merge the surviving boxes with the analyzer's merge settings. Used internally by `analyze_confidence_thresholds()` but also callable directly.

| Parameter | Type / Default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `detections_data` | `Dict` / none | **Yes** | Dict returned by `load_detections()`. |
| `conf_threshold` | `float` / none | **Yes** | Confidence threshold to apply. |
| `verbose` | `bool` / `False` | No | If `True`, print the filter details. |

**Returns:** `List[Dict]`. The merged detections that survive the threshold.

---

## Parameter Deep-Dives

### Choosing Beta

The beta parameter weights recall relative to precision in the F-beta formula [[25]](#ref-powers), [[27]](#ref-vanrijsbergen):

$$
F_\beta = \frac{(1 + \beta^2) \cdot \text{precision} \cdot \text{recall}}{\beta^2 \cdot \text{precision} + \text{recall}}
$$

| `beta` | Score name | Emphasis | Recommended when |
| :--- | :--- | :--- | :--- |
| `0.5` | F0.5 | Precision counts more than recall | False positives are very costly |
| `1.0` (default) | F1 | Equal weight | General evaluation baseline |
| `2.0` | F2 | Recall counts more than precision | Missing a bird is worse than a false alarm |

!!! info "F1 or F2"
    The pipeline scripts and the models' default thresholds use F1. If missing a bird is worse than a false alarm in your project, use F2 (`beta=2.0`). It penalizes missed calls more than false detections.

### Building the Sweep Range

Use `numpy.arange` to generate the threshold list. Rounding keeps the thresholds readable in the results:

```python
import numpy as np

# Default sweep: 101 thresholds from 0.00 to 1.00
thresholds = np.round(np.arange(0.00, 1.01, 0.01), 3).tolist()

# Coarse sweep: 19 thresholds (faster)
thresholds = np.round(np.arange(0.05, 1.00, 0.05), 3).tolist()

# Fine sweep around a known region
thresholds = np.round(np.arange(0.20, 0.50, 0.005), 3).tolist()
```

Runtime grows with the number of thresholds.

### Optimal Matching

By default, the Hungarian algorithm [[15]](#ref-kuhn) finds the globally optimal assignment of detections to labels at each threshold. This is order-independent and fully reproducible.

Setting `use_optimal_matching=False` switches to greedy matching (first-come, first-served), which is faster for very large datasets but produces order-dependent results.

!!! warning "Not Recommended"
    Only use `use_optimal_matching=False` if runtime is critical and you understand the trade-offs. Always use the default for final published metrics.

---

## Output

`analyze_confidence_thresholds()` and `find_optimal_thresholds()` return DataFrames and write nothing. Save them with `DataFrame.to_csv`, and call `plot_f_beta_curves(df, output_dir)` for the plots. The CLI script [`f_beta_score_analysis.py`](../cli/f-beta-score-analysis.md) writes all of the files below. The filenames embed the beta value (e.g. `f1.0_score_analysis.csv`).

| File | Description |
| :--- | :--- |
| `f{beta}_score_analysis.csv` | Full results table: one row per (species, threshold). |
| `f{beta}_score_analysis.json` | Same data in JSON format. |
| `optimal_thresholds.csv` | Best threshold per species (maximising F-beta). Written from `find_optimal_thresholds()`. |
| `overall_micro_f{beta}_curve.png` | F-beta, precision, recall for micro-average across all species. |
| `overall_macro_f{beta}_curve.png` | F-beta, precision, recall for macro-average across all species. |
| `micro_vs_macro_f{beta}_comparison.png` | Side-by-side micro vs macro comparison. |
| `top_species_f{beta}_curves.png` | Per-species curves for the top performing species (`top_classes`, 12 in the CLI). |
| `all_species_f{beta}_curves.png` | Per-species curves for all species. |
| `f{beta}_score_heatmap.png` | Heatmap of F-beta per species (rows) vs threshold (columns). Species are sorted by their best F-beta. Gray cells mean the species has no labels and no detections at that threshold. |
| `args.yaml` | Every CLI argument used for this run. Written by the CLI script only. |

### Reading optimal_thresholds.csv

`find_optimal_thresholds()` keeps the full result row at each species' best threshold:

```csv
species,confidence_threshold,tp,fp,fn,precision,recall,f_beta_score
amerob,0.35,42.0,6.0,8.0,0.8750,0.8400,0.8571
herthr,0.28,30.0,10.0,5.0,0.7500,0.8571,0.8000
Overall_Micro,0.33,72.0,16.0,13.0,0.8182,0.8471,0.8324
Overall_Macro,0.30,,,,0.8125,0.8486,0.8286
```

Use the `Overall_Micro` row to pick a single system-wide threshold when per-species thresholds are impractical.

---

## Examples

The examples evaluate Just-Bird, so they pass `single_cls=True`. For Birds-With-Labels, drop it.

### Default F1 Sweep

```python
import os

import numpy as np
from evaluation.f_beta_score_analysis import FBetaScoreAnalyzer

analyzer = FBetaScoreAnalyzer(beta=1.0, iou_threshold=0.25, single_cls=True)

thresholds = np.round(np.arange(0.00, 1.01, 0.01), 2).tolist()

results = analyzer.analyze_confidence_thresholds(
    detections_path="results/raw_detections.json",
    labels_path="data/annotations.csv",
    confidence_thresholds=thresholds,
)

optimal = analyzer.find_optimal_thresholds(results)
best = optimal.loc[optimal["species"] == "Overall_Micro"].iloc[0]
print(f"Best system-wide threshold: {best['confidence_threshold']:.2f} (F1 {best['f_beta_score']:.3f})")

# Optional: save the tables and plots
os.makedirs("results/f_beta_score_analysis", exist_ok=True)
results.to_csv("results/f_beta_score_analysis/f1.0_score_analysis.csv", index=False)
optimal.to_csv("results/f_beta_score_analysis/optimal_thresholds.csv", index=False)
analyzer.plot_f_beta_curves(results, "results/f_beta_score_analysis")
```

### F2 Sweep (Recall-Focused)

```python
import numpy as np
from evaluation.f_beta_score_analysis import FBetaScoreAnalyzer

analyzer = FBetaScoreAnalyzer(
    beta=2.0,
    iou_threshold=0.5,
    single_cls=True,
)

results = analyzer.analyze_confidence_thresholds(
    detections_path="results/raw_detections.json",
    labels_path="data/annotations.csv",
    confidence_thresholds=np.round(np.arange(0.00, 1.01, 0.01), 2).tolist(),
)
```

### Parallel Sweep (Fast)

```python
import numpy as np
from evaluation.f_beta_score_analysis import FBetaScoreAnalyzer

analyzer = FBetaScoreAnalyzer(single_cls=True)

results = analyzer.analyze_confidence_thresholds(
    detections_path="results/raw_detections.json",
    labels_path="data/annotations.csv",
    confidence_thresholds=np.round(np.arange(0.00, 1.01, 0.01), 2).tolist(),
    num_workers=8,
)
```

### Coarse Sweep, Then Fine Refinement

```python
import numpy as np
from evaluation.f_beta_score_analysis import FBetaScoreAnalyzer

analyzer = FBetaScoreAnalyzer(single_cls=True)

# Fast coarse pass
coarse = analyzer.analyze_confidence_thresholds(
    detections_path="results/raw_detections.json",
    labels_path="data/annotations.csv",
    confidence_thresholds=np.round(np.arange(0.05, 0.95, 0.05), 3).tolist(),
)

best = coarse[coarse["species"] == "Overall_Micro"]
peak = best.loc[best["f_beta_score"].idxmax(), "confidence_threshold"]

# Fine refinement around the peak
fine = analyzer.analyze_confidence_thresholds(
    detections_path="results/raw_detections.json",
    labels_path="data/annotations.csv",
    confidence_thresholds=np.round(np.arange(max(0, peak - 0.1), min(1, peak + 0.1), 0.005), 3).tolist(),
)
```

---

## Typical Workflow

This class sits at **Step 2** of the standard evaluation pipeline:

```
Step 1  BirdCallDetector(conf_threshold=0.001).detect(..., no_merge=True)      ->  raw_detections.json
Step 2  FBetaScoreAnalyzer().analyze_confidence_thresholds(...)                 ->  results DataFrame
        FBetaScoreAnalyzer().find_optimal_thresholds(results)                   ->  best thresholds
Step 3  DetectionFilter().filter_detections(data, conf_threshold=best_conf)     ->  merged detections
        DetectionFilter().save_results(data, merged, output_path, best_conf, None)  ->  simplified.csv
Step 4  ConfusionMatrixAnalyzer().analyze(..., output_path=...)                 ->  confusion_matrix_analysis/
```

After running the sweep:

1. Read the `Overall_Micro` row of `find_optimal_thresholds()` to get the best system-wide threshold.
2. Pass that threshold to [`DetectionFilter.filter_detections()`](filter-and-merge-detections.md) and save the result with `save_results()`.
3. Feed the resulting `simplified.csv` to [`ConfusionMatrixAnalyzer`](confusion-matrix-analysis.md) for species-level error analysis.

---

## References

<a id="ref-kuhn"></a>
**[15]** Kuhn, H. W. (1955). "The Hungarian method for the assignment problem." *Naval Research Logistics Quarterly*, 2(1–2):83–97.

<a id="ref-powers"></a>
**[25]** Powers, D. M. (2020). "Evaluation: from precision, recall and F-measure to ROC, informedness, markedness and correlation." arXiv preprint arXiv:2010.16061.

<a id="ref-vanrijsbergen"></a>
**[27]** van Rijsbergen, C. J. (1979). *Information Retrieval* (2nd ed.). Butterworths, London.
