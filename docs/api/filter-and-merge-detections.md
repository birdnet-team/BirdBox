# Filter and Merge Detections

Filter raw bird call detections by confidence and merge them, without re-running inference. `DetectionFilter` is the companion to `BirdCallDetector(..., no_merge=True)`. It lets you try different thresholds in seconds on the same raw detection file.

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
| `filter_detections(data, conf_threshold, song_gap, freq_iou_threshold, merge_method, phrase_gap)` | `List[Dict]` | Filter by confidence, then merge. Primary entry point. |
| `save_results(data, filtered_detections, output_path, conf_threshold, song_gap, output_formats, merge_settings)` | `None` | Write filtered results in one or more formats. |
| `save_filtered_json(data, filtered_detections, output_path, conf_threshold, song_gap, merge_settings)` | `None` | Write JSON with algorithm metadata. |
| `save_filtered_csv(filtered_detections, output_path)` | `None` | Write simplified CSV. |
| `save_filtered_xc_json(data, filtered_detections, output_path)` | `None` | Write Xeno-Canto Annota-JSON. |
| `save_filtered_raven_txt(filtered_detections, output_path, multi_file=None)` | `None` | Write a Raven Selection Table to `output_path`, or for a multi-file run one table per recording into `raven/` next to it. `multi_file=None` treats the run as multi-file when any detection carries a `filename`. |

---

### The load_detections Method

```python
data = df.load_detections(input_path)
```

Load a `raw_detections.json` file produced by [`BirdCallDetector`](detect-birds.md) with `no_merge=True`.

| Parameter | Type / Default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `input_path` | `str` / none | **Yes** | Path to the raw detections JSON **file**. Only the CLI resolves a `results/` directory to `raw_detections.json` and follows `results/.active_run`. Passing a directory here raises `IsADirectoryError`. |

**Returns:** `Dict` with keys `detections` (list of raw clip dicts), `model_config`, and metadata fields.

---

### The filter_detections Method

```python
merged = df.filter_detections(
    data,
    conf_threshold,
    song_gap=None,
    freq_iou_threshold=None,
    merge_method=None,
    phrase_gap=None,
)
```

Apply a confidence threshold to raw detections, then merge the surviving boxes with [`merge_detections`](detect-birds.md#the-merge_detections-function). Unset merge settings come from the JSON's `model_config`, so the merge is the same as in the detection run. For both models that is window fusion with a 0.05 s phrase gap.

| Parameter | Type / Default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `data` | `Dict` / none | **Yes** | Dict returned by `load_detections()`. |
| `conf_threshold` | `float` / none | **Yes** | Confidence threshold (0.0 to 1.0). All detections with `confidence < threshold` are discarded before merging. |
| `song_gap` | `float` / `None` | No | `reconstruct` merge only. Max gap in seconds between two detections of the same species to merge into one song. `None` reads `model_config.song_gap_threshold`, falling back to `0.1`. |
| `freq_iou_threshold` | `float` / `None` | No | `reconstruct` merge with Just-Bird only. Frequency IoU gate. `None` uses the JSON value or `0.6`. |
| `merge_method` | `str` / `None` | No | `'fusion'` or `'reconstruct'`. `None` reads `model_config.merge_method`, which is `'fusion'` for both models. |
| `phrase_gap` | `float` / `None` | No | Fusion only. Join same-band boxes closer than this many seconds. `None` reads `model_config.phrase_gap`, else 0.05 s. |

**Returns:** `List[Dict]`. The merged detections. For single-file input each one gets the recording's `filename`.

!!! info "Why filter raw clips and not merged boxes?"
    `BirdCallDetector` also filters first and merges afterwards. Filtering merged boxes by their average or maximum confidence gives different and generally worse results.

---

### The save_results Method

```python
df.save_results(
    data,
    filtered_detections,
    output_path,
    conf_threshold,
    song_gap,
    output_formats=None,
    merge_settings=None,
)
```

Write filtered detections to one or more output formats under `output_path`.

| Parameter | Type / Default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `data` | `Dict` / none | **Yes** | Original data dict from `load_detections()`. Used to preserve `model_config` in JSON output. |
| `filtered_detections` | `List[Dict]` / none | **Yes** | Merged detections returned by `filter_detections()`. |
| `output_path` | `str` / none | **Yes** | Output directory. It is created if it does not exist. |
| `conf_threshold` | `float` / none | **Yes** | Threshold applied. Recorded in `filtering_config` of the JSON output. |
| `song_gap` | `float` or `None` / none | **Yes** | Song gap recorded in `filtering_config`. Pass `None` for window fusion. |
| `output_formats` | `List[str]` / `['json-with-algorithm-metadata', 'simplified-csv']` | No | One or more format keys: `json-with-algorithm-metadata`, `simplified-csv`, `xeno-canto-annota-json`, `raven-selection-table`, or `all`. |
| `merge_settings` | `Dict` / `None` | No | Merge settings to record in `filtering_config`, such as the output of `resolve_merge_settings()`. Without it, the JSON records only the threshold and the song gap. |

---

## Parameter Deep-Dives

### Confidence Threshold

The threshold is applied to the raw per-clip detections before merging. So it means the same as the `conf_threshold` you would pass to `BirdCallDetector`.

### Merge Settings

Leave the merge settings at `None` to reuse the values from the JSON's `model_config`. Then the merge matches the original detection run.

| Arguments | Effect |
| :--- | :--- |
| none (recommended) | Window fusion with the phrase gap from the JSON (0.05 s) |
| `phrase_gap=0` | One box per call or syllable |
| `phrase_gap=0.3` | Longer phrase boxes |
| `merge_method='reconstruct', song_gap=0.2` | Song merge. Wide song boxes joined across pauses up to 0.2 s. |

`song_gap` alone has no effect, because the default merge is window fusion. See [Merge Method](../cli/detect-birds.md#merge-method).

---

## Output Files

Files written by `save_results()` to `output_path`:

```text
output_path/with_algorithm_metadata.json   <- json-with-algorithm-metadata
output_path/simplified.csv                 <- simplified-csv
output_path/xeno-canto-annota.json         <- xeno-canto-annota-json
output_path/raven_selection_table.txt      <- raven-selection-table (single-file input)
output_path/raven/<Filename>.txt           <- raven-selection-table (multi-file input, one per recording)
```

For the schema of each format, see [Detection Output Formats](../data/outputs.md).

---

## Examples

### Basic Filter and Merge

```python
from evaluation.filter_and_merge_detections import DetectionFilter

df = DetectionFilter()
data = df.load_detections("results/raw_detections.json")

merged = df.filter_detections(data, conf_threshold=0.35)
print(f"Merged: {len(merged)} detections")

df.save_results(
    data,
    merged,
    output_path="results/filtered_0.35",
    conf_threshold=0.35,
    song_gap=None,
)
```

### All Formats, With the Merge Settings Recorded

```python
from evaluation.filter_and_merge_detections import DetectionFilter
from inference.detect_birds import resolve_merge_settings

df = DetectionFilter()
data = df.load_detections("results/raw_detections.json")
merged = df.filter_detections(data, conf_threshold=0.25, phrase_gap=0.1)
settings = resolve_merge_settings(data["model_config"], phrase_gap=0.1)

df.save_results(
    data,
    merged,
    output_path="results/filtered_0.25",
    conf_threshold=0.25,
    song_gap=settings["song_gap_threshold"],
    output_formats=["all"],
    merge_settings=settings,
)
```

### Rapid Threshold Comparison

```python
from evaluation.filter_and_merge_detections import DetectionFilter

df = DetectionFilter()
data = df.load_detections("results/raw_detections.json")

for threshold in [0.20, 0.30, 0.40, 0.50]:
    merged = df.filter_detections(data, conf_threshold=threshold)
    print(f"conf={threshold:.2f} -> {len(merged)} detections")
```

### Apply the Threshold From F-Beta Analysis

```python
import pandas as pd
from evaluation.filter_and_merge_detections import DetectionFilter

optimal = pd.read_csv("results/f_beta_score_analysis/optimal_thresholds.csv")
best_threshold = float(optimal.loc[optimal["species"] == "Overall_Micro", "confidence_threshold"].iloc[0])

df = DetectionFilter()
data = df.load_detections("results/raw_detections.json")
merged = df.filter_detections(data, conf_threshold=best_threshold)

df.save_results(
    data,
    merged,
    output_path="results/final",
    conf_threshold=best_threshold,
    song_gap=None,
    output_formats=["simplified-csv", "json-with-algorithm-metadata"],
)
```

The CSV comes from the CLI script [`f_beta_score_analysis.py`](../cli/f-beta-score-analysis.md). In Python, take the threshold from [`FBetaScoreAnalyzer.find_optimal_thresholds()`](f-beta-score-analysis.md#the-find_optimal_thresholds-method) instead.

---

## Typical Workflow

This class sits at **Step 3** of the standard evaluation pipeline:

```
Step 1  BirdCallDetector(conf_threshold=0.001).detect(..., no_merge=True)      ->  raw_detections.json
Step 2  FBetaScoreAnalyzer().analyze_confidence_thresholds(...)                 ->  results DataFrame
        FBetaScoreAnalyzer().find_optimal_thresholds(results)                   ->  best thresholds
Step 3  DetectionFilter().filter_detections(data, conf_threshold=best_conf)     ->  merged detections
        DetectionFilter().save_results(data, merged, output_path, best_conf, None)  ->  simplified.csv
Step 4  ConfusionMatrixAnalyzer().analyze(..., output_path=...)                 ->  confusion_matrix_analysis/
```

!!! info "When to Use This Class"
    - **After F-beta analysis**: Apply the best threshold without re-running inference.
    - **For rapid threshold experiments**: Test multiple thresholds on the same raw file in seconds.
    - **To generate CSV files**: Produce CSV files compatible with `ConfusionMatrixAnalyzer` or external tools.
    - **For custom thresholds**: Apply any threshold outside your original sweep range.
