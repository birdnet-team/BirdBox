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
# $DATASET_NAME = "Birds-With-Labels_testset"
# $DATASET_NAME = "Western-US"
# $DATASET_NAME = "Hawaii_testset"
# $DATASET_NAME = "Northeastern-US_testset-subset"
$DATASET_NAME = "Northeastern-US_testset"


#### audio folder and labels inside datasets\$DATASET_NAME #####
$AUDIO_DIR = "datasets\$DATASET_NAME\soundscape_data"
$LABELS_CSV = "datasets\$DATASET_NAME\annotations.csv"


#### select model #####
$MODEL_PATH = "models\Just-Bird.pt"
# $MODEL_PATH = "models\Birds-With-Labels-Transfer.pt"


#### select the species mapping (according to dataset and model) #####
$SPECIES_MAPPING = "Just-Bird"
# $SPECIES_MAPPING = "Birds-With-Labels"


#### toggle single class mode ($true for Just-Bird) #####
$USE_SINGLE_CLS = $true
# $USE_SINGLE_CLS = $false


#### select output path #####
$OUTPUT_PATH = "results\Just-Bird"
# $OUTPUT_PATH = "results\Birds-With-Labels-Transfer"


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
# window fusion with a 0.05 s phrase gap for Just-Bird, the song merge with a
# 0.1 s gap for Birds-With-Labels.
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


# Step 3: From raw detections, filter at the model's default confidence
# (0.15 for Just-Bird, 0.2 for Birds-With-Labels; add --conf to change it) and merge.
Write-Host "Filtering raw detections at the model's default confidence and merging for confusion matrix..."
python src\evaluation\filter_and_merge_detections.py `
    --raw-detections "$OUTPUT_PATH" `
    --output-path "$OUTPUT_PATH" `
    --output-format all `
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
