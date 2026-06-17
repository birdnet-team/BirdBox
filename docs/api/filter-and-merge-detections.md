
Filter raw bird call detections by confidence threshold and merge them into song segments, without re-running inference. The post-processing companion to `BirdCallDetector(..., no_merge=True)`, letting you test different thresholds instantly on the same raw detection file.

---

## Import

```python
from evaluation.filter_and_merge_detections import DetectionFilter
```

---

## DetectionFilter

### Constructor

```python
DetectionFilter(use_max_confidence=True)
```

| Parameter | Type / Default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `use_max_confidence` | `bool` / `True` | No | Kept for API compatibility with `FBetaScoreAnalyzer`. Has no effect on raw-detection filtering, which always uses the per-clip `confidence` field. |

---

### Methods

| Method | Returns | Description |
| :--- | :--- | :--- |
| `load_detections(input_path)` | `Dict` | Load `raw_detections.json`. |
| `filter_detections(data, conf_threshold, song_gap)` | `List[Dict]` | Filter by confidence then merge. Primary entry point. |
| `save_results(data, filtered_detections, output_path, conf_threshold, song_gap, output_formats)` | `None` | Write filtered results in one or more formats. |
| `save_filtered_json(data, filtered_detections, output_path, conf_threshold, song_gap)` | `None` | Write JSON with algorithm metadata. |
| `save_filtered_csv(filtered_detections, output_path)` | `None` | Write simplified CSV. |
| `save_filtered_xc_json(data, filtered_detections, output_path)` | `None` | Write Xeno-Canto Annota-JSON. |
| `save_filtered_raven_txt(filtered_detections, output_path)` | `None` | Write Raven Selection Table. |

---

### load_detections

```python
data = df.load_detections(input_path)
```

Load a `raw_detections.json` file produced by [`BirdCallDetector`](detect-birds.md) with `no_merge=True`.

| Parameter | Type / Default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `input_path` | `str` / — | **Yes** | Path to the raw detections JSON file or a `results/` directory. When a directory is passed, `raw_detections.json` is resolved automatically (follows `results/.active_run`). |

**Returns:** `Dict` with keys `detections` (list of raw clip dicts), `model_config`, and metadata fields.

---

### filter_detections

```python
merged = df.filter_detections(data, conf_threshold, song_gap=None)
```

Apply a confidence threshold to raw detections, then merge surviving clips into song segments using `reconstruct_songs`. This is the same filter-then-merge order used by `BirdCallDetector` internally.

| Parameter | Type / Default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `data` | `Dict` / — | **Yes** | Dict returned by `load_detections()`. |
| `conf_threshold` | `float` / — | **Yes** | Confidence threshold (0.0–1.0). All detections with `confidence < threshold` are discarded before merging. |
| `song_gap` | `float` / `None` | No | Max gap in seconds between two detections of the same species to merge into one song. When `None`, reads `model_config.song_gap_threshold` from the JSON, falling back to `0.1`. |

**Returns:** `List[Dict]` — merged song segments.

!!! info "Why filter raw clips and not merged songs?"
    The `filter-then-merge` order matches the app's internal workflow. Filtering already-merged songs by average or max confidence gives different (and generally worse) results.

---

### save_results

```python
df.save_results(
    data,
    filtered_detections,
    output_path,
    conf_threshold,
    song_gap,
    output_formats=None,
)
```

Write filtered detections to one or more output formats under `output_path`.

| Parameter | Type / Default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `data` | `Dict` / — | **Yes** | Original data dict from `load_detections()`. Used to preserve `model_config` in JSON output. |
| `filtered_detections` | `List[Dict]` / — | **Yes** | Merged detections returned by `filter_detections()`. |
| `output_path` | `str` / — | **Yes** | Output directory. Created automatically if it does not exist. |
| `conf_threshold` | `float` / — | **Yes** | Threshold applied (recorded in output metadata). |
| `song_gap` | `float` / — | **Yes** | Song gap used (recorded in output metadata). |
| `output_formats` | `List[str]` / `['json-with-algorithm-metadata', 'simplified-csv']` | No | One or more format keys: `json-with-algorithm-metadata`, `simplified-csv`, `xeno-canto-annota-json`, `raven-selection-table`, or `all`. |

---

## Parameter Deep-Dives

### conf_threshold — Confidence Threshold

The threshold is applied to raw per-clip detections before song merging. It is directly comparable to the `conf_threshold` you would pass to `BirdCallDetector`.

### song_gap — Song Gap

Controls how aggressively surviving detections are fused into continuous song segments. Omitting it reuses the `song_gap_threshold` from the JSON's `model_config`, ensuring consistency with the original detection run.

| `song_gap` | Effect |
| :--- | :--- |
| `None` (recommended) | Reads from JSON `model_config.song_gap_threshold`, falls back to `0.1` |
| `0.1` | Merge detections ≤ 0.1 s apart |
| `0.5` | Merge detections ≤ 0.5 s apart (moderate) |
| `2.0` | Merge detections ≤ 2 s apart (aggressive, may over-merge) |

---

## Output Files

Files written by `save_results()` to `output_path`:

```text
output_path/with_algorithm_metadata.json   ← json-with-algorithm-metadata
output_path/simplified.csv                 ← simplified-csv
output_path/xeno-canto-annota.json         ← xeno-canto-annota-json
output_path/raven_selection_table.txt      ← raven-selection-table
```

For the schema of each format see [Detection Output Formats](../data/outputs.md).

---

## Examples

### Basic filter and merge

```python
from evaluation.filter_and_merge_detections import DetectionFilter

df = DetectionFilter()
data = df.load_detections("results/raw_detections.json")

merged = df.filter_detections(data, conf_threshold=0.35)
print(f"Merged: {len(merged)} song segments")

df.save_results(
    data,
    merged,
    output_path="results/filtered_0.35",
    conf_threshold=0.35,
    song_gap=0.1,
)
```

### All formats

```python
from evaluation.filter_and_merge_detections import DetectionFilter

df = DetectionFilter()
data = df.load_detections("results/raw_detections.json")
merged = df.filter_detections(data, conf_threshold=0.25)

df.save_results(
    data,
    merged,
    output_path="results/filtered_0.25",
    conf_threshold=0.25,
    song_gap=0.1,
    output_formats=["all"],
)
```

### Rapid threshold comparison

```python
from evaluation.filter_and_merge_detections import DetectionFilter

df = DetectionFilter()
data = df.load_detections("results/raw_detections.json")

for threshold in [0.20, 0.30, 0.40, 0.50]:
    merged = df.filter_detections(data, conf_threshold=threshold)
    print(f"conf={threshold:.2f} → {len(merged)} segments")
```

### Apply threshold from F-beta analysis

```python
import pandas as pd
from evaluation.filter_and_merge_detections import DetectionFilter

optimal = pd.read_csv("results/f_beta_score_analysis/optimal_thresholds.csv")
best_threshold = optimal[optimal["species"] == "Overall_Micro"]["optimal_threshold"].iloc[0]

df = DetectionFilter()
data = df.load_detections("results/raw_detections.json")
merged = df.filter_detections(data, conf_threshold=best_threshold)

df.save_results(
    data,
    merged,
    output_path="results/final",
    conf_threshold=best_threshold,
    song_gap=data["model_config"]["song_gap_threshold"],
    output_formats=["simplified-csv", "json-with-algorithm-metadata"],
)
```

---

## Typical Workflow

This class sits at **Step 3** of the standard evaluation pipeline:

```
Step 1  BirdCallDetector(conf_threshold=0.001).detect(..., no_merge=True)  →  raw_detections.json
Step 2  FBetaScoreAnalyzer().analyze_confidence_thresholds(...)             →  optimal_thresholds.csv
Step 3  DetectionFilter().filter_detections(data, conf=0.35)               →  simplified.csv
Step 4  ConfusionMatrixAnalyzer().analyze(...)                              →  confusion_matrix/
```

!!! info "When to Use This Class"
    - **After F-beta analysis**: Apply the optimal threshold without re-running inference.
    - **For rapid threshold experiments**: Test multiple thresholds on the same raw file in seconds.
    - **To generate CSV files**: Produce CSV files compatible with `ConfusionMatrixAnalyzer` or external tools.
    - **For custom thresholds**: Apply any threshold outside your original sweep range.
