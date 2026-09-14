#!/bin/bash
#
# Advanced / optional: search merge hyperparameters (song_gap, nms_iou)
# on a frozen model. Everyday detect + evaluate is run_pipeline.sh
# (or .bat / .ps1). Skip unless you have labels and want to retune merge.

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
# DATASET_NAME="Northeastern-US_testset-subset"
DATASET_NAME="Northeastern-US_testset"


#### derive base name (strip dataset suffix for model, mapping, and results) #####
DATASET_BASE="${DATASET_NAME/_testset-subset/}"
DATASET_BASE="${DATASET_BASE/_testset/}"


#### select model #####
MODEL_PATH="models/${DATASET_BASE}.pt"
# MODEL_PATH="models/Just-Bird.pt"
# MODEL_PATH="models/All-In-One-Transfer.pt"


#### select the species mapping (according to dataset and model) #####
SPECIES_MAPPING="${DATASET_BASE}"
# SPECIES_MAPPING="Just-Bird"
# SPECIES_MAPPING="All-In-One"


#### toggle single class mode #####
# USE_SINGLE_CLS=true
USE_SINGLE_CLS=false


#### select output path #####
OUTPUT_PATH="results/${DATASET_BASE}/hyperopt_merge"
# OUTPUT_PATH="results/Just-Bird/hyperopt_merge"


#### merge-parameter grid #####
SONG_GAPS=(0.0 0.1 0.2 0.5 1.0 2.0)
NMS_IOUS=(0.5 0.6 0.7 0.8)

IOU_THRESHOLD=0.25
BETA=1.0
DETECT_CONF=0.001
DETECT_WORKERS=18
FBETA_WORKERS=9


#########################################################################
# Choose exactly one of the two options below (they are alternatives,
# not sequential steps). Detect + single F-beta stay in run_pipeline.sh.
# Freeze these merge params while training (BirdBox-Train/run_hpo.sh).
#
# Runtime: Option A reuses one raw_detections.json (no new detect).
# Option B re-runs detect_birds --no-merge once per NMS_IOUS value,
# which dominates wall time. Prefer A unless you need to search nms_iou.
#########################################################################


SINGLE_CLS_FLAG=()
if [ "${USE_SINGLE_CLS}" = true ]; then
    SINGLE_CLS_FLAG+=(--single-cls)
fi


### only enable one of the two ###
# Option A (cheap): sweep song_gap only. Reuses the existing
# detect_birds --no-merge dump (results/<DATASET>/raw_detections.json).
# Does not run detection again. nms_iou is whatever that dump was built with.
echo "Option A: sweeping song_gap on existing raw detections (no new detect)..."
python src/evaluation/hyperopt_merge.py \
    --raw-detections "results/${DATASET_BASE}" \
    --labels "datasets/${DATASET_NAME}/annotations.csv" \
    --output-path "${OUTPUT_PATH}" \
    --song-gaps "${SONG_GAPS[@]}" \
    --iou-threshold "${IOU_THRESHOLD}" \
    --beta "${BETA}" \
    --num-workers "${FBETA_WORKERS}" \
    --no-plot \
    "${SINGLE_CLS_FLAG[@]}"

# Option B (expensive): sweep nms_iou and song_gap.
# nms_iou is applied inside YOLO, so detect_birds --no-merge runs once per
# NMS_IOUS value (len(NMS_IOUS) full inference passes), then song_gap is
# swept on each dump. Comment Option A if you enable this.
# echo "Option B: re-running detect once per nms_iou, then sweeping song_gap..."
# python src/evaluation/hyperopt_merge.py \
#     --model "${MODEL_PATH}" \
#     --audio "datasets/${DATASET_NAME}/soundscape_data" \
#     --species-mapping "${SPECIES_MAPPING}" \
#     --labels "datasets/${DATASET_NAME}/annotations.csv" \
#     --output-path "${OUTPUT_PATH}" \
#     --nms-ious "${NMS_IOUS[@]}" \
#     --song-gaps "${SONG_GAPS[@]}" \
#     --detect-conf "${DETECT_CONF}" \
#     --detect-workers "${DETECT_WORKERS}" \
#     --iou-threshold "${IOU_THRESHOLD}" \
#     --beta "${BETA}" \
#     --num-workers "${FBETA_WORKERS}" \
#     --no-plot \
#     "${SINGLE_CLS_FLAG[@]}"
# ### only enable one of the two ###


echo
echo "All tasks completed!"
echo "Merge HPO summary is in ${OUTPUT_PATH}/summary.csv"
