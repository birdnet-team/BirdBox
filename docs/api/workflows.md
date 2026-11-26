# Typical Workflows

This chapter shows how to assemble the API classes into end-to-end Python scripts. For a description of every parameter, see the dedicated API pages. For the equivalent CLI commands, see [CLI Workflows](../cli/workflows.md).

For the data formats of detections and ground truth annotations, see [Data In/Out](../data/fileflows.md).

The examples run from the project root with `src/` on the import path. See [Setup](api-type.md#setup).

---

## Simple Workflow for Detection Only

Run inference on a single file or a directory of audio files. Both examples rely on the defaults for all other parameters. The confidence threshold defaults to the model's value (0.15 for Just-Bird, 0.21 for Birds-With-Labels).

### Single File

```python
from inference.detect_birds import BirdCallDetector

detector = BirdCallDetector(
    model_path="models/Just-Bird_fp32.pt",
    species_mapping="Just-Bird",
)

detections = detector.detect(
    "path/to/recording.wav",
    output_path="results",
)
# Writes: results/with_algorithm_metadata.json
```

### Directory Batch

```python
from inference.detect_birds import BirdCallDetector

detector = BirdCallDetector(
    model_path="models/Birds-With-Labels_fp32.pt",
    species_mapping="Birds-With-Labels",
)

detections = detector.detect(
    "path/to/audio/folder",
    output_path="results",
    output_formats=["simplified-csv"],
)
# Writes: results/simplified.csv
```

!!! info "Species Mapping Values"
    `species_mapping` is `"Just-Bird"` or `"Birds-With-Labels"` and must match the model. See [Audio and Model Inputs: Species mapping](../data/audio-and-model.md#species-mapping). `config.get_species_mapping_for_model(model_path)` infers it from the model filename.

---

## Simple Workflow for Detection and Evaluation

The following code runs inference and evaluates the results against ground truth annotations. Steps 2 to 4 build on the output of step 1. The example uses Just-Bird, so steps 2 and 4 pass `single_cls=True`. For Birds-With-Labels, change the model and mapping and drop `single_cls=True`.

!!! info "Why run detection at low confidence?"
    `FBetaScoreAnalyzer` filters and merges the raw detections by itself at each threshold. Merged or already filtered detections would give wrong scores. Always run step 1 with `conf_threshold=0.001` and `no_merge=True`.

```python
import numpy as np
from inference.detect_birds import BirdCallDetector
from evaluation.f_beta_score_analysis import FBetaScoreAnalyzer
from evaluation.filter_and_merge_detections import DetectionFilter
from evaluation.confusion_matrix_analysis import ConfusionMatrixAnalyzer

# Step 1: Run inference at very low confidence, keep raw detections
detector = BirdCallDetector(
    model_path="models/Just-Bird_fp32.pt",
    species_mapping="Just-Bird",
    conf_threshold=0.001,
)
detector.detect(
    "path/to/audio/folder",
    output_path="results",
    no_merge=True,
)
# Writes: results/raw_detections.json

# Step 2: Sweep confidence thresholds to find the best one
analyzer = FBetaScoreAnalyzer(single_cls=True)
results = analyzer.analyze_confidence_thresholds(
    detections_path="results/raw_detections.json",
    labels_path="path/to/annotations.csv",
    confidence_thresholds=np.round(np.arange(0.0, 1.01, 0.01), 2).tolist(),
)
optimal = analyzer.find_optimal_thresholds(results)
best_conf = float(optimal.loc[optimal["species"] == "Overall_Micro", "confidence_threshold"].iloc[0])
# Returns DataFrames only. Nothing is written in this step.

# Step 3: Filter at the best threshold and merge
df = DetectionFilter()
data = df.load_detections("results/raw_detections.json")
merged = df.filter_detections(data, conf_threshold=best_conf)
df.save_results(data, merged, "results/filtered", best_conf, song_gap=None)
# Writes: results/filtered/with_algorithm_metadata.json and results/filtered/simplified.csv

# Step 4: Compute the confusion matrix
cm_analyzer = ConfusionMatrixAnalyzer(iou_threshold=0.25, single_cls=True)
matrix, species = cm_analyzer.analyze(
    detections_path="results/filtered/simplified.csv",
    labels_path="path/to/annotations.csv",
    output_path="results/confusion_matrix_analysis",
)
# Writes: results/confusion_matrix_analysis/confusion_matrix.csv, *.png, metadata.txt
```

Unlike the CLI, the API takes exact file paths. It does not follow `results/.active_run`.

---

## Complete Workflow Script

The following script mirrors the pipeline from [CLI Workflows: Complete Workflow Script](../cli/workflows.md#complete-workflow-script). Every parameter is set explicitly for full reproducibility.

```python
import os

import numpy as np

from inference.detect_birds import BirdCallDetector, resolve_merge_settings
from evaluation.f_beta_score_analysis import FBetaScoreAnalyzer
from evaluation.filter_and_merge_detections import DetectionFilter
from evaluation.confusion_matrix_analysis import ConfusionMatrixAnalyzer

# Configuration
MODEL_PATH          = "models/Just-Bird_fp32.pt"
SPECIES_MAPPING     = "Just-Bird"
SINGLE_CLS          = True         # True for Just-Bird, False for Birds-With-Labels
AUDIO_PATH          = "data/audio/"
LABELS_PATH         = "data/annotations.csv"
RESULTS_DIR         = "results"

CONF_LOW            = 0.001        # Step 1: capture all raw detections
BETA                = 1.0          # Step 2: F1, as in the pipeline scripts
CONF_RANGE_MIN      = 0.00
CONF_RANGE_MAX      = 1.00
CONF_RANGE_STEP     = 0.01
IOU_THRESHOLD       = 0.25         # Steps 2 and 4: matching strictness
PHRASE_GAP          = 0.05         # Steps 2 and 3: window fusion phrase gap (seconds)
NMS_IOU             = 0.7          # Step 1: NMS inside each clip
NUM_WORKERS_DETECT  = 1
NUM_WORKERS_SWEEP   = 8
OUTPUT_FORMATS      = ["simplified-csv", "json-with-algorithm-metadata"]

# Step 1: Inference
print("=== Step 1: Inference ===")
detector = BirdCallDetector(
    model_path=MODEL_PATH,
    species_mapping=SPECIES_MAPPING,
    conf_threshold=CONF_LOW,
    nms_iou_threshold=NMS_IOU,
    phrase_gap=PHRASE_GAP,
    num_workers=NUM_WORKERS_DETECT,
)
detector.detect(
    audio_path=AUDIO_PATH,
    output_path=RESULTS_DIR,
    no_merge=True,
)

# Step 2: F-beta sweep
print("\n=== Step 2: F-beta score analysis ===")
sweep_output = f"{RESULTS_DIR}/f_beta_score_analysis"
os.makedirs(sweep_output, exist_ok=True)

fbeta = FBetaScoreAnalyzer(
    iou_threshold=IOU_THRESHOLD,
    beta=BETA,
    phrase_gap=PHRASE_GAP,
    single_cls=SINGLE_CLS,
)
thresholds = np.round(
    np.arange(CONF_RANGE_MIN, CONF_RANGE_MAX + CONF_RANGE_STEP / 2, CONF_RANGE_STEP), 3
).tolist()
results_df = fbeta.analyze_confidence_thresholds(
    detections_path=f"{RESULTS_DIR}/raw_detections.json",
    labels_path=LABELS_PATH,
    confidence_thresholds=thresholds,
    num_workers=NUM_WORKERS_SWEEP,
)
optimal = fbeta.find_optimal_thresholds(results_df)
results_df.to_csv(f"{sweep_output}/f{BETA}_score_analysis.csv", index=False)
optimal.to_csv(f"{sweep_output}/optimal_thresholds.csv", index=False)
fbeta.plot_f_beta_curves(results_df, sweep_output)

best_conf = float(optimal.loc[optimal["species"] == "Overall_Micro", "confidence_threshold"].iloc[0])
print(f"\nBest threshold (micro-average): {best_conf:.2f}")

# Step 3: Filter and merge
print("\n=== Step 3: Filter and merge ===")
merge_output = f"{RESULTS_DIR}/filtered_{best_conf:.2f}"

df_filter = DetectionFilter()
raw_data = df_filter.load_detections(f"{RESULTS_DIR}/raw_detections.json")
merged = df_filter.filter_detections(raw_data, conf_threshold=best_conf, phrase_gap=PHRASE_GAP)
merge_settings = resolve_merge_settings(raw_data["model_config"], phrase_gap=PHRASE_GAP)
df_filter.save_results(
    raw_data,
    merged,
    output_path=merge_output,
    conf_threshold=best_conf,
    song_gap=merge_settings["song_gap_threshold"],
    output_formats=OUTPUT_FORMATS,
    merge_settings=merge_settings,  # recorded in filtering_config of the JSON
)
print(f"Kept {len(merged)} merged detections at conf >= {best_conf:.2f}")

# Step 4: Confusion matrix
print("\n=== Step 4: Confusion matrix ===")
cm_analyzer = ConfusionMatrixAnalyzer(
    iou_threshold=IOU_THRESHOLD,
    use_2d_iou=True,
    include_background=True,
    single_cls=SINGLE_CLS,
)
matrix, species = cm_analyzer.analyze(
    detections_path=f"{merge_output}/simplified.csv",
    labels_path=LABELS_PATH,
    output_path=f"{RESULTS_DIR}/confusion_matrix_analysis",
)

print(f"\nClasses evaluated: {[s for s in species if s != 'background']}")
print("Done. Results written to:", RESULTS_DIR)
```
