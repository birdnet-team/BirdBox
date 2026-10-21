#Requires -Version 5.1
# SPDX-FileCopyrightText: 2026 BirdNET-Team
# SPDX-License-Identifier: MIT
#
# Default BirdBox workflow: detect birds and evaluate against labels.
# Same as run_pipeline.sh / run_pipeline.bat; pick your shell.
# Skip run_hpo.sh unless you need the advanced merge-parameter search.

$ErrorActionPreference = "Stop"

# Optional: activate local virtual environment if present.
# If .venv is missing, script uses python from PATH.
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "Activating .venv"
    . ".venv\Scripts\Activate.ps1"
}


#### select the dataset on which inference shall be performed #####
# Just-Bird trained on the train and test splits of every regional dataset,
# so its scores on these test sets are in-domain.
# $DATASET_NAME = "All-In-One_testset"
# $DATASET_NAME = "Western-US"
# $DATASET_NAME = "Hawaii_testset"
# $DATASET_NAME = "Northeastern-US_testset-subset"
$DATASET_NAME = "Northeastern-US_testset"


#### audio folder and labels inside datasets\$DATASET_NAME #####
$AUDIO_DIR = "datasets\$DATASET_NAME\soundscape_data"
$LABELS_CSV = "datasets\$DATASET_NAME\annotations.csv"


#### select model #####
$DATASET_BASE = $DATASET_NAME -replace "_testset-subset", "" -replace "_testset", ""
$MODEL_PATH = "models\Just-Bird.pt"
# $MODEL_PATH = "models\$DATASET_BASE.pt"
# $MODEL_PATH = "models\All-In-One-Transfer.pt"


#### select the species mapping (according to dataset and model) #####
$SPECIES_MAPPING = "Just-Bird"
# $SPECIES_MAPPING = $DATASET_BASE
# $SPECIES_MAPPING = "All-In-One"


#### toggle single class mode ($true for Just-Bird) #####
$USE_SINGLE_CLS = $true
# $USE_SINGLE_CLS = $false


#### select output path #####
$OUTPUT_PATH = "results\Just-Bird"
# $OUTPUT_PATH = "results\$DATASET_BASE"
# $OUTPUT_PATH = "results\All-In-One-Transfer"


#########################################################################
# The most important parameters are already set via the script variables.
# But details like IoU threshold,
# merge settings, etc. can be changed below this heading.
# To skip entire steps (for instance the confusion matrix)
# just uncomment the respective lines.
#########################################################################


$SINGLE_CLS_FLAG = @()
if ($USE_SINGLE_CLS) { $SINGLE_CLS_FLAG = @("--single-cls") }

# Merge settings for steps 2 and 3. Empty keeps the model's defaults:
# window fusion for Just-Bird, Hawaii and Southern-Sierra-Nevada (phrase gap
# 0.05 s for Just-Bird, 0 for the others), the song merge with a 0.1 s gap
# for the other models.
$MERGE_FLAGS = @()
# $MERGE_FLAGS = @("--phrase-gap", "0")    # window fusion: one box per call
# $MERGE_FLAGS = @("--phrase-gap", "0.3")  # window fusion: longer phrase boxes
# $MERGE_FLAGS = @("--song-gap", "0.2")    # song merge: join songs across longer pauses

# Step 1: Run inference with low confidence and
# --no-merge to get raw (unmerged) detections.
# This matches the filter-then-merge policy when
# later filtering at each confidence threshold.
Write-Host "Running inference (raw detections, no merge)..."
python src\inference\detect_birds.py `
    --audio "$AUDIO_DIR" `
    --model "$MODEL_PATH" `
    --species-mapping "$SPECIES_MAPPING" `
    --output-path "$OUTPUT_PATH" `
    --output-format json-with-algorithm-metadata `
    --conf 0.001 `
    --no-merge `
    --nms-iou 0.8 `
    --num-workers 4


# Step 2: F-beta analysis on raw detections:
# at each confidence threshold we filter then merge.
Write-Host "Running F-beta score analysis (filter-then-merge per threshold)..."
python src\evaluation\f_beta_score_analysis.py `
    --raw-detections "$OUTPUT_PATH" `
    --labels "$LABELS_CSV" `
    --output-path "$OUTPUT_PATH\f_1.0_score_analysis" `
    --beta 1.0 `
    --iou-threshold 0.25 `
    @MERGE_FLAGS `
    --num-workers 8 `
    @SINGLE_CLS_FLAG


# Step 3: From raw detections, filter at conf=0.25 and merge.
Write-Host "Filtering raw detections at conf=0.25 and merging for confusion matrix..."
python src\evaluation\filter_and_merge_detections.py `
    --raw-detections "$OUTPUT_PATH" `
    --output-path "$OUTPUT_PATH" `
    --output-format all `
    --conf 0.2 `
    @MERGE_FLAGS


# # Step 4: Run confusion matrix analysis.
Write-Host "Running confusion matrix analysis..."
python src\evaluation\confusion_matrix_analysis.py `
    --detections "$OUTPUT_PATH" `
    --labels "$LABELS_CSV" `
    --output-path "$OUTPUT_PATH\confusion_matrix_analysis" `
    --iou-threshold 0.25 `
    @SINGLE_CLS_FLAG


# Step 5: Examine results in results\ directory
Write-Host ""
Write-Host "All tasks completed!"
Write-Host "Results can now be examined in the $OUTPUT_PATH directory."
