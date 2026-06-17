
Sweep a range of confidence thresholds on raw (unmerged) detections and compute precision, recall, and F-beta scores for every species class at every threshold. Returns a DataFrame of results and writes performance curves, a heatmap, and an `optimal_thresholds.csv` table.

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
)
```

| Parameter | Type / Default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `iou_threshold` | `float` / `0.25` | No | IoU threshold for matching a detection to a ground truth label (0.0–1.0). Use the same value later in [`ConfusionMatrixAnalyzer`](confusion-matrix-analysis.md) for consistent evaluation. |
| `beta` | `float` / `1.0` | No | Beta parameter for the F-beta score. Controls precision/recall weighting. See [Choosing beta](#choosing-beta) below. |
| `use_optimal_matching` | `bool` / `True` | No | If `True`, uses the Hungarian algorithm for globally optimal assignment. If `False`, uses greedy matching. Greedy matching is faster on very large datasets but order-dependent and not recommended for published metrics. |
| `song_gap` | `float` / `None` | No | Max gap in seconds to merge detections into song segments at each threshold step. When `None`, reads `model_config.song_gap_threshold` from the JSON, falling back to `0.1`. |
| `single_cls` | `bool` / `False` | No | Collapse all species into one class for binary bird-detection evaluation. |
| `single_cls_name` | `str` / `'bird'` | No | Class name to use when `single_cls=True`. |

---

### Methods

| Method | Returns | Description |
| :--- | :--- | :--- |
| `analyze_confidence_thresholds(detections_path, labels_path, confidence_thresholds, num_workers)` | `pd.DataFrame` | Run the full threshold sweep. Primary entry point. |
| `load_detections(detections_path)` | `Dict` | Load raw detections JSON. |
| `load_labels(labels_path)` | `List[Dict]` | Load ground truth labels CSV. |
| `filter_detections_by_confidence(detections_data, conf_threshold)` | `List[Dict]` | Filter and merge detections at a single threshold. |

---

### analyze_confidence_thresholds

```python
results_df = analyzer.analyze_confidence_thresholds(
    detections_path,
    labels_path,
    confidence_thresholds,
    num_workers=1,
)
```

Run the confidence-threshold sweep. At each threshold, raw detections are filtered by confidence then merged into song segments before computing precision, recall, and F-beta scores per species.

!!! warning "Input Must Be Raw (Unmerged) Detections"
    Pass the output of `BirdCallDetector.detect(..., no_merge=True)`. Pre-merged detections produce incorrect results because the sweep re-applies filter-then-merge at every threshold step.

| Parameter | Type / Default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `detections_path` | `str` / — | **Yes** | Path to `raw_detections.json` from [`BirdCallDetector`](detect-birds.md) with `no_merge=True`. |
| `labels_path` | `str` / — | **Yes** | Path to the ground truth labels CSV file. |
| `confidence_thresholds` | `List[float]` / — | **Yes** | Ordered list of confidence thresholds to evaluate. Build with `numpy.arange(0.0, 1.01, 0.01).tolist()` for the default 101-step sweep. |
| `num_workers` | `int` / `1` | No | Number of worker processes. Each handles a disjoint subset of thresholds. Combine with skipping plots for the fastest sweep. |

**Returns:** `pd.DataFrame` with one row per `(species, confidence_threshold)` combination. Columns: `species`, `confidence_threshold`, `TP`, `FP`, `FN`, `precision`, `recall`, `f_beta`.

---

### load_detections

```python
data = analyzer.load_detections(detections_path)
```

Load a `raw_detections.json` file produced by `BirdCallDetector(..., no_merge=True)`. Returns the full JSON as a dict, including `model_config` metadata.

---

### load_labels

```python
labels = analyzer.load_labels(labels_path)
```

Load a ground truth labels CSV. Returns a list of dicts with keys `filename`, `time_start`, `time_end`, `freq_low_hz`, `freq_high_hz`, `species`.

---

### filter_detections_by_confidence

```python
merged = analyzer.filter_detections_by_confidence(detections_data, conf_threshold)
```

Apply a single confidence threshold to raw detections, then merge surviving clips into song segments. Used internally by `analyze_confidence_thresholds()` but also callable directly.

| Parameter | Type / Default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `detections_data` | `Dict` / — | **Yes** | Dict returned by `load_detections()`. |
| `conf_threshold` | `float` / — | **Yes** | Confidence threshold to apply. |

**Returns:** `List[Dict]` — merged song segments that survive the threshold.

---

## Parameter Deep-Dives

### Choosing beta

The beta parameter weights recall relative to precision in the F-beta formula:

$$
F_\beta = \frac{(1 + \beta^2) \cdot \text{precision} \cdot \text{recall}}{\beta^2 \cdot \text{precision} + \text{recall}}
$$

| `beta` | Score name | Emphasis | Recommended when |
| :--- | :--- | :--- | :--- |
| `0.5` | F0.5 | Precision × 2 over recall | False positives are very costly |
| `1.0` (default) | F1 | Equal weight | General evaluation baseline |
| `2.0` | F2 | Recall × 2 over precision | Missing a bird is worse than a false alarm |

!!! info "Bird Detection Recommendation"
    For bird call detection, **F2-score (`beta=2.0`)** is generally preferred. Missing a detection (False Negative) is typically worse than reporting a spurious one (False Positive).

### confidence_thresholds — Building the Sweep Range

Use `numpy.arange` to generate the threshold list:

```python
import numpy as np

# Default sweep: 101 thresholds from 0.00 to 1.00
thresholds = np.arange(0.00, 1.01, 0.01).tolist()

# Coarse sweep: 19 thresholds (faster)
thresholds = np.arange(0.05, 1.00, 0.05).tolist()

# Fine sweep around a known region
thresholds = np.arange(0.20, 0.50, 0.005).tolist()
```

| Sweep | Thresholds tested | Approx. runtime (1 worker) |
| :--- | :--- | :--- |
| `0.00` to `1.00`, step `0.01` (default) | 101 | ~1–2 min |
| `0.05` to `0.95`, step `0.05` | 19 | ~15 s |
| `0.10` to `0.90`, step `0.01` | 81 | ~1 min |

### use_optimal_matching

By default, the Hungarian algorithm finds the globally optimal assignment of detections to labels at each threshold. This is order-independent and fully reproducible.

Setting `use_optimal_matching=False` switches to greedy matching (first-come, first-served), which is faster for very large datasets but produces order-dependent results.

!!! warning "Not Recommended"
    Only use `use_optimal_matching=False` if runtime is critical and you understand the trade-offs. Always use the default for final published metrics.

---

## Output

`analyze_confidence_thresholds()` returns a DataFrame. Call the script-level `main()` to also write files to disk. The filenames embed the beta value (e.g. `f1.0_score_analysis.csv`).

| File | Description |
| :--- | :--- |
| `f{beta}_score_analysis.csv` | Full results table: one row per (species, threshold). |
| `f{beta}_score_analysis.json` | Same data in JSON format. |
| `optimal_thresholds.csv` | Best threshold per species (maximising F-beta). |
| `overall_micro_f{beta}_curve.png` | F-beta, precision, recall for micro-average across all species. |
| `overall_macro_f{beta}_curve.png` | F-beta, precision, recall for macro-average across all species. |
| `micro_vs_macro_f{beta}_comparison.png` | Side-by-side micro vs macro comparison. |
| `top_species_f{beta}_curves.png` | Per-species curves for the top 12 performing species. |
| `all_species_f{beta}_curves.png` | Per-species curves for all species. |
| `f{beta}_score_heatmap.png` | Heatmap of F-beta per species (rows) vs threshold (columns). |

### Reading optimal_thresholds.csv

```csv
species,optimal_threshold,best_f_beta,precision_at_optimal,recall_at_optimal
amerob,0.35,0.8421,0.8750,0.8400
herthr,0.28,0.7933,0.7500,0.8571
Overall_Micro,0.33,0.8512,0.8654,0.8421
```

Use the `Overall_Micro` row to pick a single system-wide threshold when per-species thresholds are impractical.

---

## Examples

### Default F1 sweep

```python
import numpy as np
from evaluation.f_beta_score_analysis import FBetaScoreAnalyzer

analyzer = FBetaScoreAnalyzer(beta=1.0, iou_threshold=0.25)

thresholds = np.arange(0.00, 1.01, 0.01).tolist()

results = analyzer.analyze_confidence_thresholds(
    detections_path="results/raw_detections.json",
    labels_path="data/ground_truth.csv",
    confidence_thresholds=thresholds,
)

best_row = results[results["species"] == "Overall_Micro"]
best_threshold = best_row.loc[best_row["f_beta"].idxmax(), "confidence_threshold"]
print(f"Best system-wide threshold: {best_threshold:.2f}")
```

### F2 sweep (recall-focused)

```python
import numpy as np
from evaluation.f_beta_score_analysis import FBetaScoreAnalyzer

analyzer = FBetaScoreAnalyzer(
    beta=2.0,
    iou_threshold=0.5,
)

results = analyzer.analyze_confidence_thresholds(
    detections_path="results/raw_detections.json",
    labels_path="data/ground_truth.csv",
    confidence_thresholds=np.arange(0.00, 1.01, 0.01).tolist(),
)
```

### Parallel sweep (fast)

```python
import numpy as np
from evaluation.f_beta_score_analysis import FBetaScoreAnalyzer

analyzer = FBetaScoreAnalyzer(beta=2.0)

results = analyzer.analyze_confidence_thresholds(
    detections_path="results/raw_detections.json",
    labels_path="data/ground_truth.csv",
    confidence_thresholds=np.arange(0.00, 1.01, 0.01).tolist(),
    num_workers=8,
)
```

### Coarse sweep then fine-grained refinement

```python
import numpy as np
from evaluation.f_beta_score_analysis import FBetaScoreAnalyzer

analyzer = FBetaScoreAnalyzer(beta=2.0)

# Fast coarse pass
coarse = analyzer.analyze_confidence_thresholds(
    detections_path="results/raw_detections.json",
    labels_path="data/ground_truth.csv",
    confidence_thresholds=np.arange(0.05, 0.95, 0.05).tolist(),
)

best = coarse[coarse["species"] == "Overall_Micro"]
peak = best.loc[best["f_beta"].idxmax(), "confidence_threshold"]

# Fine-grained refinement around the peak
fine = analyzer.analyze_confidence_thresholds(
    detections_path="results/raw_detections.json",
    labels_path="data/ground_truth.csv",
    confidence_thresholds=np.arange(max(0, peak - 0.1), min(1, peak + 0.1), 0.005).tolist(),
)
```

---

## Typical Workflow

This class sits at **Step 2** of the standard evaluation pipeline:

```
Step 1  BirdCallDetector(conf_threshold=0.001).detect(..., no_merge=True)  →  raw_detections.json
Step 2  FBetaScoreAnalyzer().analyze_confidence_thresholds(...)             →  optimal_thresholds.csv
Step 3  DetectionFilter().filter_detections(data, conf=0.35)               →  simplified.csv
Step 4  ConfusionMatrixAnalyzer().analyze(...)                              →  confusion_matrix/
```

After running the sweep:

1. Read `Overall_Micro` from `optimal_thresholds.csv` to get the recommended threshold.
2. Pass that threshold to [`DetectionFilter.filter_detections()`](filter-and-merge-detections.md).
3. Feed the resulting CSV to [`ConfusionMatrixAnalyzer`](confusion-matrix-analysis.md) for species-level error analysis.

---

## References

Powers, D. M. (2020). "Evaluation: from precision, recall and F-measure to ROC, informedness, markedness and correlation." arXiv preprint arXiv:2010.16061.
