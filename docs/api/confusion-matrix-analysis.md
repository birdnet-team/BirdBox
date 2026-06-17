
Compute a confusion matrix comparing filtered detection results against ground truth labels. Matches detections to labels using IoU with the Hungarian (optimal) algorithm, then builds a per-species confusion matrix showing correct classifications, misclassifications, false positives, and false negatives.

---

## Import

```python
from evaluation.confusion_matrix_analysis import ConfusionMatrixAnalyzer
```

---

## ConfusionMatrixAnalyzer

### Constructor

```python
ConfusionMatrixAnalyzer(
    iou_threshold=0.25,
    use_2d_iou=True,
    include_background=True,
    single_cls=False,
    single_cls_name="bird",
)
```

| Parameter | Type / Default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `iou_threshold` | `float` / `0.25` | No | IoU threshold for matching a detection to a ground truth label. Use the same value as in [`FBetaScoreAnalyzer`](f-beta-score-analysis.md) for consistent evaluation. |
| `use_2d_iou` | `bool` / `True` | No | If `True`, uses 2D IoU (time × frequency). If `False`, uses 1D IoU (temporal overlap only). 2D IoU is more accurate. Use 1D only when frequency annotations are unreliable or absent. |
| `include_background` | `bool` / `True` | No | If `True`, adds a `background` class for unmatched detections (FP) and unmatched labels (FN). Pass `False` to see only species-vs-species misclassifications. |
| `single_cls` | `bool` / `False` | No | Collapse all species into one class for binary bird-detection evaluation. Reduces the matrix to 2×2. |
| `single_cls_name` | `str` / `'bird'` | No | Class name to use when `single_cls=True`. |

!!! info "Mutually Exclusive Interpretation"
    `single_cls=True` and multi-species evaluation are conceptually mutually exclusive. When active, the confusion matrix reduces to a 2×2 TP/FP/TN/FN table (with or without background).

---

### Methods

| Method | Returns | Description |
| :--- | :--- | :--- |
| `analyze(detections_path, labels_path, output_path)` | `Tuple[np.ndarray, List[str]]` | Run the full analysis. Primary entry point. |
| `load_detections_csv(detections_path)` | `List[Dict]` | Load merged detections CSV. |
| `load_labels_csv(labels_path)` | `List[Dict]` | Load ground truth labels CSV. |
| `compute_confusion_matrix(detections, labels)` | `Tuple[np.ndarray, List[str]]` | Build the matrix from loaded data. |
| `save_results(confusion_matrix, species_list, output_path, detections_path, labels_path)` | `None` | Write CSV, images, and metadata. |

---

### analyze

```python
matrix, species = analyzer.analyze(
    detections_path,
    labels_path,
    output_path=None,
)
```

Run the complete confusion matrix pipeline: load, match, compute, optionally save.

| Parameter | Type / Default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `detections_path` | `str` / — | **Yes** | Path to `simplified.csv` from [`DetectionFilter`](filter-and-merge-detections.md), or a `results/` directory (resolves `simplified.csv` via `.active_run`). |
| `labels_path` | `str` / — | **Yes** | Path to the ground truth labels CSV. Must use the same six-column schema as the detections file. |
| `output_path` | `str` / `None` | No | Directory for output files. When provided, writes the confusion matrix CSV, heatmap images, and metadata. |

**Returns:** `(np.ndarray, List[str])` — the confusion matrix and ordered species list.

---

### load_detections_csv

```python
detections = analyzer.load_detections_csv(detections_path)
```

Load a `simplified.csv` produced by `DetectionFilter`. Returns a list of dicts with keys `filename`, `time_start`, `time_end`, `freq_low_hz`, `freq_high_hz`, `species`.

---

### load_labels_csv

```python
labels = analyzer.load_labels_csv(labels_path)
```

Load a ground truth labels CSV. Returns a list of dicts with the same schema as `load_detections_csv()`.

---

### compute_confusion_matrix

```python
matrix, species_list = analyzer.compute_confusion_matrix(detections, labels)
```

Build the confusion matrix from pre-loaded data. The Hungarian algorithm assigns detections to labels globally at IoU ≥ `iou_threshold`.

| Parameter | Type / Default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `detections` | `List[Dict]` / — | **Yes** | Detections from `load_detections_csv()`. |
| `labels` | `List[Dict]` / — | **Yes** | Labels from `load_labels_csv()`. |

**Returns:** `(np.ndarray, List[str])` — confusion matrix and species list. Matrix shape is `(N+1, N+1)` when `include_background=True` (last row/column is `background`).

---

## Parameter Deep-Dives

### iou_threshold — IoU Matching

Determines how strictly a detection must overlap a ground truth label to count as a True Positive.

| Value | Strictness | Typical use |
| :--- | :--- | :--- |
| `0.1` | Very lenient | When annotations are coarse |
| `0.25` (default) | Moderate | Good starting point for most datasets |
| `0.5` | Strict | When annotations are precise and detailed |

### 2D vs 1D IoU

**2D IoU** (default) measures overlap in both time and frequency:

$$\text{IoU}_{2D} = \frac{t_{\text{overlap}} \cdot f_{\text{overlap}}}{t_{\text{union}} \cdot f_{\text{union}}}$$

**1D IoU** (`use_2d_iou=False`) measures temporal overlap only:

$$\text{IoU}_{1D} = \frac{t_{\text{overlap}}}{t_{\text{union}}}$$

Use `use_2d_iou=False` only when frequency annotations are unreliable or absent.

### include_background — Background Class

When `include_background=True`:

- **Background column** — ground truth labels not matched by any detection. These are **False Negatives**.
- **Background row** — detections that did not match any ground truth label. These are **False Positives**.

Set `include_background=False` to see only the species-level confusion (correct vs wrong species label), ignoring FP/FN counts.

---

## Output Files

All files are written to the `output_path` directory.

| File | Description |
| :--- | :--- |
| `confusion_matrix.csv` | Raw confusion matrix as CSV. |
| `confusion_matrix_detailed.csv` | Same matrix with `pred_<class>` rows and `true_<class>` columns. |
| `confusion_matrix_normalized.png` | Heatmap showing row-normalized percentages. |
| `confusion_matrix_raw.png` | Heatmap showing raw counts. |
| `metadata.txt` | Analysis parameters, file paths, species list, detection and label counts. |

### Reading the Confusion Matrix

The matrix uses **predicted class (rows) × true class (columns)** convention.

| Cell | Meaning |
| :--- | :--- |
| Diagonal cell `[i, i]` | True Positives for species `i` |
| Off-diagonal `[i, j]` (i ≠ j) | Predicted species `i`, actually species `j` (misclassification) |
| `background` column `[i, bg]` | Labels for species `i` with no matching detection (False Negatives) |
| `background` row `[bg, j]` | Detections that matched no label for species `j` (False Positives) |

---

## Examples

### Basic confusion matrix

```python
from evaluation.confusion_matrix_analysis import ConfusionMatrixAnalyzer

analyzer = ConfusionMatrixAnalyzer(iou_threshold=0.25)

matrix, species = analyzer.analyze(
    detections_path="results/simplified.csv",
    labels_path="data/ground_truth.csv",
    output_path="results/confusion_matrix",
)

print(f"Species: {species}")
print(matrix)
```

### Strict 2D IoU matching

```python
from evaluation.confusion_matrix_analysis import ConfusionMatrixAnalyzer

analyzer = ConfusionMatrixAnalyzer(
    iou_threshold=0.5,
    use_2d_iou=True,
)

matrix, species = analyzer.analyze(
    detections_path="results/simplified.csv",
    labels_path="data/ground_truth.csv",
    output_path="results/confusion_strict",
)
```

### 1D IoU (temporal only)

```python
from evaluation.confusion_matrix_analysis import ConfusionMatrixAnalyzer

analyzer = ConfusionMatrixAnalyzer(
    iou_threshold=0.5,
    use_2d_iou=False,
)

matrix, species = analyzer.analyze(
    detections_path="results/simplified.csv",
    labels_path="data/ground_truth.csv",
    output_path="results/confusion_1d",
)
```

### Without background class

```python
from evaluation.confusion_matrix_analysis import ConfusionMatrixAnalyzer

analyzer = ConfusionMatrixAnalyzer(include_background=False)

matrix, species = analyzer.analyze(
    detections_path="results/simplified.csv",
    labels_path="data/ground_truth.csv",
    output_path="results/confusion_no_bg",
)
```

### Binary bird detection

```python
from evaluation.confusion_matrix_analysis import ConfusionMatrixAnalyzer

analyzer = ConfusionMatrixAnalyzer(
    single_cls=True,
    single_cls_name="bird",
)

matrix, species = analyzer.analyze(
    detections_path="results/simplified.csv",
    labels_path="data/ground_truth.csv",
    output_path="results/confusion_binary",
)
```

### Load, compute, and inspect programmatically

```python
from evaluation.confusion_matrix_analysis import ConfusionMatrixAnalyzer
import numpy as np

analyzer = ConfusionMatrixAnalyzer(iou_threshold=0.25)

detections = analyzer.load_detections_csv("results/simplified.csv")
labels = analyzer.load_labels_csv("data/ground_truth.csv")

matrix, species = analyzer.compute_confusion_matrix(detections, labels)

for i, sp in enumerate(species):
    tp = matrix[i, i]
    fn = matrix[i, species.index("background")] if "background" in species else 0
    print(f"{sp}: TP={tp}, FN={fn}")
```

---

## Typical Workflow

This class sits at **Step 4** of the standard evaluation pipeline:

```
Step 1  BirdCallDetector(conf_threshold=0.001).detect(..., no_merge=True)  →  raw_detections.json
Step 2  FBetaScoreAnalyzer().analyze_confidence_thresholds(...)             →  optimal_thresholds.csv
Step 3  DetectionFilter().filter_detections(data, conf=0.35)               →  simplified.csv
Step 4  ConfusionMatrixAnalyzer().analyze(...)                              →  confusion_matrix/
```

!!! info "Input File Format"
    Both `detections_path` and `labels_path` must use the six-column schema: `Filename`, `Start Time (s)`, `End Time (s)`, `Low Freq (Hz)`, `High Freq (Hz)`, `Species eBird Code`. Filenames are matched by stem only (extension-agnostic). See [Ground-Truth Labels — CSV schema](../data/annotations.md#csv-schema) for the full definition.
