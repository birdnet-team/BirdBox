@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Exit if any command fails
set "FAILED=0"

REM Optional: activate local virtual environment if present.
REM If .venv is missing, script uses python from PATH.
if exist ".venv\Scripts\activate.bat" (
    echo Activating .venv
    call ".venv\Scripts\activate.bat"
    if errorlevel 1 goto :fail
)


REM #### select the dataset on which inference shall be performed #####
REM set "DATASET_NAME=All-In-One_testset"
REM set "DATASET_NAME=Western-US"
REM set "DATASET_NAME=Hawaii_testset"
set "DATASET_NAME=Northeastern-US_testset-subset"


REM #### select model #####
set "DATASET_BASE=%DATASET_NAME:_testset-subset=%"
set "MODEL_PATH=models\%DATASET_BASE%.pt"
REM set "MODEL_PATH=models\Just-Bird.pt"
REM set "MODEL_PATH=models\All-In-One-Transfer.pt"


REM #### select the species mapping (according to dataset and model) #####
set "SPECIES_MAPPING=%DATASET_BASE%"
REM set "SPECIES_MAPPING=Just-Bird"
REM set "SPECIES_MAPPING=All-In-One"


REM #### toggle single class mode #####
REM set "USE_SINGLE_CLS=true"
set "USE_SINGLE_CLS=false"


REM #### select output path #####
set "OUTPUT_PATH=results\%DATASET_BASE%"
REM set "OUTPUT_PATH=results\Just-Bird"
REM set "OUTPUT_PATH=results\All-In-One-Transfer"


REM #########################################################################
REM # The most important parameters are already set via the script variables.
REM # But details like IoU threshold,
REM # song gap threshold, etc. can be changed below this heading.
REM # To skip entire steps (for instance the confusion matrix)
REM # just uncomment the respective lines.
REM #########################################################################


set "RAW_DETECTIONS_BASE=%OUTPUT_PATH%\raw_detections"
set "MERGED_DETECTIONS_BASE=%OUTPUT_PATH%\merged_detections"

set "SINGLE_CLS_FLAG="
if /I "%USE_SINGLE_CLS%"=="true" set "SINGLE_CLS_FLAG=--single-cls"

REM Step 1: Run inference with low confidence and
REM --no-merge to get raw (unmerged) detections.
REM This matches the filter-then-merge policy when
REM later filtering at each confidence threshold.
echo Running inference (raw detections, no merge)...
python src\inference\detect_birds.py ^
    --audio "datasets\%DATASET_NAME%\soundscape_data" ^
    --model "%MODEL_PATH%" ^
    --species-mapping "%SPECIES_MAPPING%" ^
    --output-path "%RAW_DETECTIONS_BASE%" ^
    --output-format json-with-algorithm-metadata ^
    --conf 0.001 ^
    --no-merge ^
    --nms-iou 0.8 ^
    --workers 4
if errorlevel 1 goto :fail


REM Step 2: F-beta analysis on raw detections:
REM at each confidence threshold we filter then merge.
echo Running F-beta score analysis (filter-then-merge per threshold)...
python src\evaluation\f_beta_score_analysis.py ^
    --raw-detections "%RAW_DETECTIONS_BASE%.json" ^
    --labels "datasets\%DATASET_NAME%\annotations.csv" ^
    --output-path "%OUTPUT_PATH%\f_1.0_score_analysis" ^
    --beta 1.0 ^
    --iou-threshold 0.25 ^
    --song-gap 0.1 ^
    --num-workers 8 ^
    %SINGLE_CLS_FLAG%
if errorlevel 1 goto :fail


REM Step 3: From raw detections, filter at conf=0.25 and merge.
echo Filtering raw detections at conf=0.25 and merging for confusion matrix...
python src\evaluation\filter_and_merge_detections.py ^
    --raw-detections "%RAW_DETECTIONS_BASE%.json" ^
    --output-path "%MERGED_DETECTIONS_BASE%" ^
    --output-format all ^
    --conf 0.2 ^
    --song-gap 0.1
if errorlevel 1 goto :fail


REM REM Step 4: Run confusion matrix analysis.
echo Running confusion matrix analysis...
python src\evaluation\confusion_matrix_analysis.py ^
    --detections "%MERGED_DETECTIONS_BASE%.csv" ^
    --labels "datasets\%DATASET_NAME%\annotations.csv" ^
    --output-path "%OUTPUT_PATH%\confusion_matrix_analysis" ^
    --iou-threshold 0.25 ^
    %SINGLE_CLS_FLAG%
if errorlevel 1 goto :fail


REM Step 5: Examine results in results\ directory
echo.
echo All tasks completed!
echo Results can now be examined in the %OUTPUT_PATH% directory.
exit /b 0

:fail
echo.
echo Pipeline failed. See the error above.
exit /b 1
