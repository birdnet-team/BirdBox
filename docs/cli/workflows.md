
This chapter documents all user-facing parameters for the executable scripts.
Most of the CLI parameters have default values that work just fine in many cases.
However, sometimes it can be beneficial to tune them by hand according to the used dataset.

Regarding any data format including detections and grund truth annotations see [Data In/Out](../data/overview.md).

---

## Simple Workflow for Detection Only

The bird detection script can be run on individual audio files or in batch mode across an entire directory. 
Both approaches highly depend on default values for CLI parameters.
For full control see [Complete Workflow Script](#complete-workflow-script).

Select the tab below that matches your operating system.

### Single File Processing

=== "Linux / macOS"
    ```bash
    python src/inference/detect_birds.py \
        --audio path/to/recording.wav \
        --model models/best.pt \
        --species-mapping species_mapping
    ```    

=== "Windows (PowerShell)"

    ```powershell
    python src/inference/detect_birds.py `
        --audio path/to/recording.wav `
        --model models/best.pt `
        --species-mapping species_mapping
    ```

=== "Windows (CMD)"

    ```cmd
    python src/inference/detect_birds.py ^
        --audio path/to/recording.wav ^
        --model models/best.pt ^
        --species-mapping species_mapping
    ```

### Batch Directory Processing

=== "Linux / macOS"

    ```bash
    python src/inference/detect_birds.py \
        --audio path/to/audio/folder \
        --model models/best.pt \
        --species-mapping species_mapping
    ```

=== "Windows (PowerShell)"

    ```powershell
    python src/inference/detect_birds.py `
        --audio path/to/audio/folder `
        --model models/best.pt `
        --species-mapping species_mapping
    ```

=== "Windows (CMD)"

    ```cmd
    python src/inference/detect_birds.py ^
        --audio path/to/audio/folder ^
        --model models/best.pt ^
        --species-mapping species_mapping
    ```

!!! info "Possible Species Mapping Values"
    To see the full list of selectable species mappings see the [overview](detect-birds.md#allowed-species-mapping-values).

---

## Simple Workflow for Detection & Evaluation

The following scripts show how to run inference and evaluate the results on given ground truth annoation data.
They highly depend on default values for the CLI parameters.
For full controll see [Complete Workflow Script](#complete-workflow-script).

Note: The F-beta score analysis has to be run before any merging because it handles merging itself.
At each confidence threshold it computes the corresponding merging individually.

=== "Linux / macOS"

    ```bash
    # Step 1: Run inference with low confidence and --no-merge to get raw detections
    python src/inference/detect_birds.py \
        --audio path/to/audio/folder \
        --model models/model_name.pt \
        --species-mapping mapping_name \
        --output-path results/raw_detections \
        --conf 0.001 \
        --no-merge \

    # Step 2: Analyze F-beta scores to find optimal threshold
    python src/evaluation/f_beta_score_analysis.py \
        --raw-detections results/raw_detections.json \
        --labels path/to/labels.csv \

    # Step 3: Filter raw detections to optimal threshold and merge
    python src/evaluation/filter_and_merge_detections.py \
        --raw-detections results/raw_detections.json \

    # Step 4: Generate confusion matrix
    python src/evaluation/confusion_matrix_analysis.py \
        --labels path/to/labels.csv \

    # Step 5: Examine results in results/ directory
    ```

=== "Windows (PowerShell)"

    ```powershell
    # Step 1: Run inference with low confidence and --no-merge to get raw detections
    python src/inference/detect_birds.py `
        --audio path/to/audio/folder `
        --model models/model_name.pt `
        --species-mapping mapping_name `
        --output-path results/raw_detections `
        --conf 0.001 `
        --no-merge `

    # Step 2: Analyze F-beta scores to find optimal threshold
    python src/evaluation/f_beta_score_analysis.py `
        --raw-detections results/raw_detections.json `
        --labels path/to/labels.csv `

    # Step 3: Filter raw detections to optimal threshold and merge
    python src/evaluation/filter_and_merge_detections.py `
        --raw-detections results/raw_detections.json `

    # Step 4: Generate confusion matrix
    python src/evaluation/confusion_matrix_analysis.py `
        --labels path/to/labels.csv `

    # Step 5: Examine results in results/ directory
    ```

=== "Windows (CMD)"

    ```cmd
    rem Step 1: Run inference with low confidence and --no-merge to get raw detections
    python src/inference/detect_birds.py ^
        --audio path/to/audio/folder ^
        --model models/model_name.pt ^
        --species-mapping mapping_name ^
        --output-path results/raw_detections ^
        --conf 0.001 ^
        --no-merge ^

    rem Step 2: Analyze F-beta scores to find optimal threshold
    python src/evaluation/f_beta_score_analysis.py ^
        --raw-detections results/raw_detections.json ^
        --labels path/to/labels.csv ^

    rem Step 3: Filter raw detections to optimal threshold and merge
    python src/evaluation/filter_and_merge_detections.py ^
        --raw-detections results/raw_detections.json ^

    rem Step 4: Generate confusion matrix
    python src/evaluation/confusion_matrix_analysis.py ^
        --labels path/to/labels.csv ^

    rem Step 5: Examine results in results/ directory
    ```

## Complete Workflow Script

The following scripts offer the most elaborate and precise way of interacting with BirdBox.
They mirror the already present [scripts](https://github.com/birdnet-team/BirdBox/blob/main/run_pipeline.sh) at the repository root.

Feel free to adapt any parameter to your use-case.
For a detailed description of each CLI parameter see the other sections in this chapter.
There exists one for each Python file.

=== "Linux / macOS"

    ```bash
    --8<-- "run_pipeline.sh"
    ```

=== "Windows (PowerShell)"

    ```powershell
    --8<-- "run_pipeline.ps1"
    ```

=== "Windows (CMD)"

    ```bat
    --8<-- "run_pipeline.bat"
    ```