# Typical Workflows

This chapter documents all user-facing parameters for the executable scripts.
Most CLI parameters have defaults that work well in many cases.
Sometimes it helps to tune them for your dataset.

For the data formats of detections and ground truth annotations, see [Data In/Out](../data/fileflows.md).

---

## Simple Workflow for Detection Only

The detection script runs on a single audio file or on a whole directory.
Both examples rely on the default values of the CLI parameters.
For full control, see [Complete Workflow Script](#complete-workflow-script).

The examples use Just-Bird. For Birds-With-Labels, pass `--model models/Birds-With-Labels_fp32.pt --species-mapping Birds-With-Labels`.

Select the tab below that matches your operating system.

### Single File Processing

=== "Linux / macOS"

    ```bash
    python src/inference/detect_birds.py \
        --audio path/to/recording.wav \
        --model models/Just-Bird_fp32.pt \
        --species-mapping Just-Bird
    ```

=== "Windows (PowerShell)"

    ```powershell
    python src/inference/detect_birds.py `
        --audio path/to/recording.wav `
        --model models/Just-Bird_fp32.pt `
        --species-mapping Just-Bird
    ```

=== "Windows (CMD)"

    ```cmd
    python src/inference/detect_birds.py ^
        --audio path/to/recording.wav ^
        --model models/Just-Bird_fp32.pt ^
        --species-mapping Just-Bird
    ```

### Batch Directory Processing

=== "Linux / macOS"

    ```bash
    python src/inference/detect_birds.py \
        --audio path/to/audio/folder \
        --model models/Just-Bird_fp32.pt \
        --species-mapping Just-Bird
    ```

=== "Windows (PowerShell)"

    ```powershell
    python src/inference/detect_birds.py `
        --audio path/to/audio/folder `
        --model models/Just-Bird_fp32.pt `
        --species-mapping Just-Bird
    ```

=== "Windows (CMD)"

    ```cmd
    python src/inference/detect_birds.py ^
        --audio path/to/audio/folder ^
        --model models/Just-Bird_fp32.pt ^
        --species-mapping Just-Bird
    ```

!!! info "Species Mapping Values"
    `--species-mapping` must match the model. See [Audio and Model Inputs: Species mapping](../data/audio-and-model.md#species-mapping).

---

## Simple Workflow for Detection and Evaluation

The following scripts run inference and evaluate the results against ground truth annotations.
They rely on the default values of the CLI parameters.
For full control, see [Complete Workflow Script](#complete-workflow-script).

Run detection with `--no-merge`, before any merging. The F-beta analysis merges by itself. At each confidence threshold it filters the raw boxes first and merges the survivors afterwards.

Steps 2 to 4 use default paths under `results/`. Step 1 writes `results/raw_detections.json`, or `results/run_N/raw_detections.json` if `results/` already has outputs. Later steps follow `results/.active_run` and find `raw_detections.json` and `simplified.csv` by themselves.

The example uses Just-Bird, so steps 2 and 4 pass `--single-cls`. For Birds-With-Labels, change the model and mapping in step 1 and drop `--single-cls`.

Step 3 filters at the model's default threshold (0.15 for Just-Bird, 0.21 for Birds-With-Labels). To use the best threshold from step 2 instead, pass it with `--conf`. Step 2 prints it as the best **Overall_Micro** threshold.

=== "Linux / macOS"

    ```bash
    # Step 1: Run inference with low confidence and --no-merge to get raw detections
    python src/inference/detect_birds.py \
        --audio path/to/audio/folder \
        --model models/Just-Bird_fp32.pt \
        --species-mapping Just-Bird \
        --output-path results \
        --conf 0.001 \
        --no-merge

    # Step 2: Analyze F-beta scores to find the best threshold
    python src/evaluation/f_beta_score_analysis.py \
        --labels path/to/annotations.csv \
        --single-cls

    # Step 3: Filter raw detections at the model's default threshold and merge
    # (add --conf <threshold> to use the best threshold from step 2)
    python src/evaluation/filter_and_merge_detections.py

    # Step 4: Generate the confusion matrix
    python src/evaluation/confusion_matrix_analysis.py \
        --labels path/to/annotations.csv \
        --single-cls

    # Step 5: Examine the results in the results/ directory
    ```

=== "Windows (PowerShell)"

    ```powershell
    # Step 1: Run inference with low confidence and --no-merge to get raw detections
    python src/inference/detect_birds.py `
        --audio path/to/audio/folder `
        --model models/Just-Bird_fp32.pt `
        --species-mapping Just-Bird `
        --output-path results `
        --conf 0.001 `
        --no-merge

    # Step 2: Analyze F-beta scores to find the best threshold
    python src/evaluation/f_beta_score_analysis.py `
        --labels path/to/annotations.csv `
        --single-cls

    # Step 3: Filter raw detections at the model's default threshold and merge
    # (add --conf <threshold> to use the best threshold from step 2)
    python src/evaluation/filter_and_merge_detections.py

    # Step 4: Generate the confusion matrix
    python src/evaluation/confusion_matrix_analysis.py `
        --labels path/to/annotations.csv `
        --single-cls

    # Step 5: Examine the results in the results/ directory
    ```

=== "Windows (CMD)"

    ```cmd
    rem Step 1: Run inference with low confidence and --no-merge to get raw detections
    python src/inference/detect_birds.py ^
        --audio path/to/audio/folder ^
        --model models/Just-Bird_fp32.pt ^
        --species-mapping Just-Bird ^
        --output-path results ^
        --conf 0.001 ^
        --no-merge

    rem Step 2: Analyze F-beta scores to find the best threshold
    python src/evaluation/f_beta_score_analysis.py ^
        --labels path/to/annotations.csv ^
        --single-cls

    rem Step 3: Filter raw detections at the model's default threshold and merge
    rem (add --conf <threshold> to use the best threshold from step 2)
    python src/evaluation/filter_and_merge_detections.py

    rem Step 4: Generate the confusion matrix
    python src/evaluation/confusion_matrix_analysis.py ^
        --labels path/to/annotations.csv ^
        --single-cls

    rem Step 5: Examine the results in the results/ directory
    ```

---

## Complete Workflow Script

The following scripts give the most control over BirdBox.
They are the [pipeline scripts](https://github.com/birdnet-team/BirdBox/blob/main/run_pipeline.sh) at the repository root.

Adapt any parameter to your use case.
Each Python script has its own page in this chapter that describes its CLI parameters.

=== "Linux / macOS"

    ```bash
    --8<-- "run_pipeline.sh"
    ```

=== "Windows (PowerShell)"

    If PowerShell refuses to run the script, see the execution policy note in the [Installation Commands](../installation.md#installation-commands).

    ```powershell
    --8<-- "run_pipeline.ps1"
    ```

=== "Windows (CMD)"

    ```bat
    --8<-- "run_pipeline.bat"
    ```