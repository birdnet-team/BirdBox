
This chapter shows how to assemble the individual API classes into end-to-end Python scripts. For a description of every parameter see the dedicated API pages. For the equivalent CLI commands see the [CLI Workflows](../cli/workflows.md).

Regarding any data format including detections and ground truth annotations see [Data In/Out](../data/overview.md).

---

## Simple Workflow — Detection Only

Run inference on a single file or a directory of audio files. Both approaches rely on sensible defaults for all parameters.

### Single file

```python
from inference.detect_birds import BirdCallDetector

detector = BirdCallDetector(
    model_path="models/best.pt",
    species_mapping="Hawaii",
)

detections = detector.detect(
    "path/to/recording.wav",
    output_path="results",
)
```

### Directory batch

```python
from inference.detect_birds import BirdCallDetector

detector = BirdCallDetector(
    model_path="models/best.pt",
    species_mapping="Hawaii",
)

detections = detector.detect(
    "path/to/audio/folder",
    output_path="results",
)
```

!!! info "Possible Species Mapping Values"
    To see the full list of selectable species mappings see [Audio and Model Inputs — Species mapping](../data/audio-and-model.md#species-mapping-species-mapping), or call `config.get_species_mapping_for_model(model_path)` to infer the name from your model filename.

---

## Simple Workflow — Detection & Evaluation

The following code shows how to run inference and evaluate the results against ground truth annotation data. Steps 2–4 build directly on the output of Step 1.

!!! info "Why run detection at low confidence?"
    `FBetaScoreAnalyzer` re-applies filtering and merging internally at each threshold. Passing high-confidence pre-merged detections would corrupt the sweep. Always run Step 1 with `conf_threshold=0.001` and `no_merge=True`.

```python
import numpy as np
from inference.detect_birds import BirdCallDetector
from evaluation.f_beta_score_analysis import FBetaScoreAnalyzer
from evaluation.filter_and_merge_detections import DetectionFilter
from evaluation.confusion_matrix_analysis import ConfusionMatrixAnalyzer

# Step 1: Run inference at very low confidence, keep raw detections
detector = BirdCallDetector(
    model_path="models/model_name.pt",
    species_mapping="mapping_name",
    conf_threshold=0.001,
)
detector.detect(
    "path/to/audio/folder",
    output_path="results",
    no_merge=True,
)
# Writes: results/raw_detections.json

# Step 2: Sweep confidence thresholds to find the optimal one
analyzer = FBetaScoreAnalyzer(beta=2.0)
results = analyzer.analyze_confidence_thresholds(
    detections_path="results/raw_detections.json",
    labels_path="path/to/labels.csv",
    confidence_thresholds=np.arange(0.00, 1.01, 0.01).tolist(),
)
# Writes: results/f_beta_score_analysis/optimal_thresholds.csv

# Step 3: Apply the optimal threshold and produce merged detections
import pandas as pd
optimal = pd.read_csv("results/f_beta_score_analysis/optimal_thresholds.csv")
best_conf = optimal[optimal["species"] == "Overall_Micro"]["optimal_threshold"].iloc[0]

df = DetectionFilter()
data = df.load_detections("results/raw_detections.json")
merged = df.filter_detections(data, conf_threshold=best_conf)
df.save_results(data, merged, "results/filtered", best_conf, song_gap=0.1)
# Writes: results/filtered/simplified.csv

# Step 4: Compute the confusion matrix
cm_analyzer = ConfusionMatrixAnalyzer(iou_threshold=0.25)
matrix, species = cm_analyzer.analyze(
    detections_path="results/filtered/simplified.csv",
    labels_path="path/to/labels.csv",
    output_path="results/confusion_matrix",
)
# Writes: results/confusion_matrix/confusion_matrix.csv, *.png, metadata.txt
```

---

## Complete Workflow Script

The following script mirrors the pipeline from the [CLI Workflows — Complete Workflow Script](../cli/workflows.md#complete-workflow-script). Every parameter is set explicitly for full reproducibility.

```python
import numpy as np
import pandas as pd

from inference.detect_birds import BirdCallDetector
from evaluation.f_beta_score_analysis import FBetaScoreAnalyzer
from evaluation.filter_and_merge_detections import DetectionFilter
from evaluation.confusion_matrix_analysis import ConfusionMatrixAnalyzer

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH          = "models/Hawaii.pt"
SPECIES_MAPPING     = "Hawaii"
AUDIO_PATH          = "data/audio/"
LABELS_PATH         = "data/labels.csv"
RESULTS_DIR         = "results"

CONF_LOW            = 0.001        # Step 1: capture all raw detections
BETA                = 2.0          # Step 2: F2 weights recall over precision
CONF_RANGE_MIN      = 0.00
CONF_RANGE_MAX      = 1.00
CONF_RANGE_STEP     = 0.01
IOU_THRESHOLD       = 0.25         # Steps 2 & 4: matching strictness
SONG_GAP            = 0.1          # Steps 1, 3: merge gap (seconds)
NMS_IOU             = 0.7          # Step 1: per-clip NMS
NUM_WORKERS_DETECT  = 1
NUM_WORKERS_SWEEP   = 8
OUTPUT_FORMATS      = ["simplified-csv", "json-with-algorithm-metadata"]

# ── Step 1: Inference ──────────────────────────────────────────────────────────
print("=== Step 1: Inference ===")
detector = BirdCallDetector(
    model_path=MODEL_PATH,
    species_mapping=SPECIES_MAPPING,
    conf_threshold=CONF_LOW,
    nms_iou_threshold=NMS_IOU,
    song_gap_threshold=SONG_GAP,
    num_workers=NUM_WORKERS_DETECT,
)
detector.detect(
    audio_path=AUDIO_PATH,
    output_path=RESULTS_DIR,
    no_merge=True,
)

# ── Step 2: F-beta sweep ───────────────────────────────────────────────────────
print("\n=== Step 2: F-beta score analysis ===")
sweep_output = f"{RESULTS_DIR}/f_beta_score_analysis"

fbeta = FBetaScoreAnalyzer(
    iou_threshold=IOU_THRESHOLD,
    beta=BETA,
    song_gap=SONG_GAP,
)
thresholds = np.arange(CONF_RANGE_MIN, CONF_RANGE_MAX + CONF_RANGE_STEP / 2, CONF_RANGE_STEP).tolist()
results_df = fbeta.analyze_confidence_thresholds(
    detections_path=f"{RESULTS_DIR}/raw_detections.json",
    labels_path=LABELS_PATH,
    confidence_thresholds=thresholds,
    num_workers=NUM_WORKERS_SWEEP,
)

optimal = pd.read_csv(f"{sweep_output}/optimal_thresholds.csv")
best_conf = float(optimal[optimal["species"] == "Overall_Micro"]["optimal_threshold"].iloc[0])
print(f"\nOptimal threshold (micro-average): {best_conf:.2f}")

# ── Step 3: Filter and merge ───────────────────────────────────────────────────
print("\n=== Step 3: Filter and merge ===")
merge_output = f"{RESULTS_DIR}/filtered_{best_conf:.2f}"

df_filter = DetectionFilter()
raw_data = df_filter.load_detections(f"{RESULTS_DIR}/raw_detections.json")
merged = df_filter.filter_detections(raw_data, conf_threshold=best_conf, song_gap=SONG_GAP)
df_filter.save_results(
    raw_data,
    merged,
    output_path=merge_output,
    conf_threshold=best_conf,
    song_gap=SONG_GAP,
    output_formats=OUTPUT_FORMATS,
)
print(f"Kept {len(merged)} song segments at conf≥{best_conf:.2f}")

# ── Step 4: Confusion matrix ───────────────────────────────────────────────────
print("\n=== Step 4: Confusion matrix ===")
cm_analyzer = ConfusionMatrixAnalyzer(
    iou_threshold=IOU_THRESHOLD,
    use_2d_iou=True,
    include_background=True,
)
matrix, species = cm_analyzer.analyze(
    detections_path=f"{merge_output}/simplified.csv",
    labels_path=LABELS_PATH,
    output_path=f"{RESULTS_DIR}/confusion_matrix",
)

print(f"\nSpecies evaluated: {[s for s in species if s != 'background']}")
print("Done. Results written to:", RESULTS_DIR)
```
