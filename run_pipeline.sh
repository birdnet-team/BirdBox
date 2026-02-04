#!/bin/bash

# Exit immediately if a command fails
set -e

# Activate your conda environment (assumes conda is installed & initialized)
source /home/mi/conda/miniconda3_8/etc/profile.d/conda.sh
conda activate birdbox

# Step 1: Run inference with low confidence and --no-merge to get raw (unmerged) detections.
# This matches the filter-then-merge policy when later filtering at each confidence threshold.
echo "Running inference (raw detections, no merge)..."
python src/inference/detect_birds.py \
    --audio datasets/Hawaii_subset/soundscape_data_wav \
    --model models/Hawaii.pt \
    --species-mapping Hawaii \
    --output-path results/raw_detections_hawaii_subset/raw_detections \
    --output-format json \
    --conf 0.001 \
    --no-merge

# Step 2: F-beta analysis on raw detections: at each confidence threshold we filter then merge.
echo "Running F-beta score analysis (filter-then-merge per threshold)..."
python src/evaluation/f_beta_score_analysis.py \
    --detections results/raw_detections_hawaii_subset/raw_detections.json \
    --labels datasets/Hawaii_subset/annotations.csv \
    --output-path results/f_1.0_score_analysis \
    --beta 1.0 \
    --iou-threshold 0.25 \
    --song-gap 0.1

# Step 3: From raw detections, filter at conf=0.25 and merge.
echo "Filtering raw detections at conf=0.25 and merging for confusion matrix..."
python src/evaluation/filter_and_merge_detections.py \
    --input results/raw_detections_hawaii_subset/raw_detections.json \
    --output-path results/merged_detections_hawaii_subset/merged_detections \
    --output-format all \
    --conf 0.2 \
    --song-gap 0.1

# Step 4: Run confusion matrix analysis.
echo "Running confusion matrix analysis..."
python src/evaluation/confusion_matrix_analysis.py \
    --detections results/merged_detections_hawaii_subset/merged_detections.csv \
    --labels datasets/Hawaii_subset/annotations.csv \
    --output-path results/confusion_matrix_analysis \
    --iou-threshold 0.25

# Step 5: Examine results in results/ directory

echo
echo "All tasks completed!"
echo "The detections can now be exported and used for further analysis."
