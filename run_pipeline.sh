#!/bin/bash

# Exit immediately if a command fails
set -e


DATASET_NAME="Northeastern-US_testset"
SPECIES_MAPPING="Northeastern-US"
USE_SINGLE_CLS=false

MODEL_PATH="models/${DATASET_NAME/_testset/}.pt"
# MODEL_PATH="models/Just-Bird.pt"
OUTPUT_PATH="results/${DATASET_NAME/_testset/}"
# OUTPUT_PATH="results/Just-Bird"
RAW_DETECTIONS_BASE="${OUTPUT_PATH}/raw_detections"
MERGED_DETECTIONS_BASE="${OUTPUT_PATH}/merged_detections"

SINGLE_CLS_FLAG=()
if [ "${USE_SINGLE_CLS}" = true ]; then
    SINGLE_CLS_FLAG+=(--single-cls)
fi

# Activate your conda environment (assumes conda is installed & initialized)
source /home/mi/conda/miniconda3_8/etc/profile.d/conda.sh
conda activate birdbox

# Step 1: Run inference with low confidence and --no-merge to get raw (unmerged) detections.
# This matches the filter-then-merge policy when later filtering at each confidence threshold.
echo "Running inference (raw detections, no merge)..."
python src/inference/detect_birds.py \
    --audio "datasets/${DATASET_NAME}/soundscape_data" \
    --model "${MODEL_PATH}" \
    --species-mapping "${SPECIES_MAPPING}" \
    --output-path "${RAW_DETECTIONS_BASE}" \
    --output-format json \
    --conf 0.001 \
    --no-merge \
    --nms-iou 0.8 \
    --workers 4


# Step 2: F-beta analysis on raw detections: at each confidence threshold we filter then merge.
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
