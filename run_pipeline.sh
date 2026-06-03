#!/bin/bash

# Exit immediately if a command fails
set -e

# Optional: activate local virtual environment if present.
# If .venv does not exist, the script uses the current Python on PATH.
if [ -f ".venv/bin/activate" ]; then
    echo "Activating .venv"
    # shellcheck source=/dev/null
    source ".venv/bin/activate"
fi


#### select the dataset on which inference shall be performed #####
# DATASET_NAME="All-In-One_testset"
# DATASET_NAME="Western-US"
# DATASET_NAME="Hawaii_testset"
DATASET_NAME="Northeastern-US_testset-subset"


#### select model #####
MODEL_PATH="models/${DATASET_NAME/_testset-subset/}.pt"
# MODEL_PATH="models/Just-Bird.pt"
# MODEL_PATH="models/All-In-One-Transfer.pt"


#### select the species mapping (according to dataset and model) #####
SPECIES_MAPPING="${DATASET_NAME/_testset-subset/}"
# SPECIES_MAPPING="Just-Bird"
# SPECIES_MAPPING="All-In-One"


#### toggle single class mode #####
# USE_SINGLE_CLS=true
USE_SINGLE_CLS=false


#### select output path #####
OUTPUT_PATH="results/${DATASET_NAME/_testset-subset/}"
# OUTPUT_PATH="results/Just-Bird"
# OUTPUT_PATH="results/All-In-One-Transfer"


#########################################################################
# The most important parameters are already set via the script variables.
# But details like IoU threshold, 
# song gap threshold, etc. can be changed below this heading.
# To skip entire steps (for instance the confusion matrix) 
# just uncomment the respective lines.
#########################################################################


RAW_DETECTIONS_BASE="${OUTPUT_PATH}/raw_detections"
MERGED_DETECTIONS_BASE="${OUTPUT_PATH}/merged_detections"

SINGLE_CLS_FLAG=()
if [ "${USE_SINGLE_CLS}" = true ]; then
    SINGLE_CLS_FLAG+=(--single-cls)
fi

# Step 1: Run inference with low confidence and 
# --no-merge to get raw (unmerged) detections.
# This matches the filter-then-merge policy when 
# later filtering at each confidence threshold.
echo "Running inference (raw detections, no merge)..."
python src/inference/detect_birds.py \
    --audio "datasets/${DATASET_NAME}/soundscape_data" \
    --model "${MODEL_PATH}" \
    --species-mapping "${SPECIES_MAPPING}" \
    --output-path "${RAW_DETECTIONS_BASE}" \
    --output-format json-with-algorithm-metadata \
    --conf 0.001 \
    --no-merge \
    --nms-iou 0.8 \
    --workers 4


# Step 2: F-beta analysis on raw detections: 
# at each confidence threshold we filter then merge.
echo "Running F-beta score analysis (filter-then-merge per threshold)..."
python src/evaluation/f_beta_score_analysis.py \
    --detections "${RAW_DETECTIONS_BASE}.json" \
    --labels "datasets/${DATASET_NAME}/annotations.csv" \
    --output-path "${OUTPUT_PATH}/f_1.0_score_analysis" \
    --beta 1.0 \
    --iou-threshold 0.25 \
    --song-gap 0.1 \
    --num-workers 8 \
    "${SINGLE_CLS_FLAG[@]}"


# Step 3: From raw detections, filter at conf=0.25 and merge.
echo "Filtering raw detections at conf=0.25 and merging for confusion matrix..."
python src/evaluation/filter_and_merge_detections.py \
    --input "${RAW_DETECTIONS_BASE}.json" \
    --output-path "${MERGED_DETECTIONS_BASE}" \
    --output-format all \
    --conf 0.2 \
    --song-gap 0.1


# # Step 4: Run confusion matrix analysis.
echo "Running confusion matrix analysis..."
python src/evaluation/confusion_matrix_analysis.py \
    --detections "${MERGED_DETECTIONS_BASE}.csv" \
    --labels "datasets/${DATASET_NAME}/annotations.csv" \
    --output-path "${OUTPUT_PATH}/confusion_matrix_analysis" \
    --iou-threshold 0.25 \
    "${SINGLE_CLS_FLAG[@]}"


# Step 5: Examine results in results/ directory
echo
echo "All tasks completed!"
echo "Results can now be examined in the ${OUTPUT_PATH} directory."
