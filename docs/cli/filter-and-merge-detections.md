# Filter and Merge Detections

Filter raw bird call detections by confidence and merge them, without re-running inference. This script is the companion to `detect_birds.py --no-merge`. It lets you try different confidence thresholds on the same raw detection file in seconds.

---

## Usage Synopsis

=== "Linux / macOS"
    ```bash
    python src/evaluation/filter_and_merge_detections.py \
        --raw-detections results \
        --conf 0.15
    ```
=== "Windows (PowerShell)"
    ```powershell
    python src/evaluation/filter_and_merge_detections.py `
        --raw-detections results `
        --conf 0.15
    ```
=== "Windows (CMD)"
    ```cmd
    python src/evaluation/filter_and_merge_detections.py ^
        --raw-detections results ^
        --conf 0.15
    ```

## Parameters

| Parameter | Type / Default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `--raw-detections` | `PATH` / `results` | No | Raw detections **file** or `results/` directory from [`detect_birds.py --no-merge`](detect-birds.md). Default resolves to `raw_detections.json` (follows `results/.active_run`). |
| `--conf` | `FLOAT` / model default | No | Confidence threshold for filtering (0.0 to 1.0). All raw detections with `confidence < threshold` are discarded before merging. Omit to use the default for `model_config.species_mapping`: `0.15` for Just-Bird, `0.21` for Birds-With-Labels. |
| `--merge` | `CHOICE` / `auto` | No | `fusion` (window ownership + weighted box fusion) or `reconstruct` (song merge with `--song-gap` and `--freq-iou`). `auto` uses `model_config.merge_method` from the JSON, which is `fusion` for both models. See [Window fusion](../how-it-works/overview.md#window-fusion). |
| `--phrase-gap` | `FLOAT` / from JSON or `0.05` | No | Fusion only. Join same-band boxes closer than this many seconds into one phrase box. `0` turns it off. Omit to use `model_config.phrase_gap` from the JSON, else the model default of 0.05 s. |
| `--song-gap` | `FLOAT` / from JSON or `0.1` | No | `reconstruct` merge only. Maximum gap in seconds between two detections of the same species that are still merged into one song box. Omit to use `model_config.song_gap_threshold` from the JSON, else `0.1`. |
| `--freq-iou` | `FLOAT` / `0.6` for Just-Bird, off otherwise | No | `reconstruct` merge with Just-Bird only. Minimum frequency IoU required before two bird detections merge into one song. Omit to use `model_config.freq_iou_threshold` when present, otherwise `0.6`. Pass a negative value for the time-only merger. |
| `--output-path` | `PATH` / `results` | No | Output directory for result files. Each format writes a fixed descriptive filename inside this directory (see [Output Files](#output-files)). The directory is created automatically if it does not exist. |
| `--output-format` | `CHOICE [...]` / `json-with-algorithm-metadata simplified-csv` | No | One or more output formats (space-separated). Accepts `json-with-algorithm-metadata`, `simplified-csv`, `xeno-canto-annota-json`, `raven-selection-table`, or `all`. |

## Parameter Deep-Dives

### Confidence Threshold

The `--conf` threshold is applied to the raw per-clip detections **before** merging. So it means the same as the `--conf` value you would pass to `detect_birds.py`.

!!! info "Why filter raw clips and not merged boxes?"
    `detect_birds.py` also filters first and merges afterwards. Filtering merged boxes by their average or maximum confidence gives different and generally worse results.

### Merge Settings

If you leave the merge flags unset, the script reuses the values from the JSON's `model_config` block, which `detect_birds.py` wrote. So detection and post-processing use the same merge. For both models that is window fusion with a 0.05 s phrase gap.

| Flag | Effect |
| :--- | :--- |
| `--phrase-gap 0` | One box per call or syllable |
| `--phrase-gap 0.05` | Default. Joins only boxes that nearly touch. |
| `--phrase-gap 0.3` | Longer phrase boxes |
| `--merge reconstruct --song-gap 0.2` | Song merge. Wide song boxes joined across pauses up to 0.2 s. |

`--song-gap` alone has no effect, because the default merge is window fusion. See [Merge Method](detect-birds.md#merge-method).

### Choosing Output Formats

`--output-format` accepts **one or more** format names separated by spaces. Specify `all` to write every format in one run. For full schema documentation of each format see [Detection Output Formats](../data/outputs.md).

## Output Files

When run with `--output-path results/merged`, the following files are written depending on `--output-format`:

```text
results/merged/with_algorithm_metadata.json   <- json-with-algorithm-metadata
results/merged/simplified.csv                 <- simplified-csv
results/merged/xeno-canto-annota.json         <- xeno-canto-annota-json
results/merged/raven_selection_table.txt      <- raven-selection-table (single-file input)
results/merged/raven/<Filename>.txt           <- raven-selection-table (multi-file input, one per recording)
```

For the structure of each output file see [Detection Output Formats](../data/outputs.md).

---

## Examples

### Basic Filter and Merge

=== "Command"
    ```bash
    python src/evaluation/filter_and_merge_detections.py \
        --raw-detections results \
        --conf 0.35 \
        --output-path results/filtered_0.35
    ```
=== "Expected Output"
    ```text
    ✓ Created output directory: results/filtered_0.35

    Loading detections from: results/raw_detections.json
    Loaded 15980 total detections
    Confidence range: 0.001 - 0.808
    Mean confidence: 0.030
    Filtered at conf>=0.35, merged (fusion, phrase_gap=0.05s) -> 72 detections
    Saved filtered detections to JSON: results/filtered_0.35/with_algorithm_metadata.json
    Saved filtered detections to CSV: results/filtered_0.35/simplified.csv

    ================================================================================
    FILTERING SUMMARY
    ================================================================================
    Confidence threshold: 0.35
    Original detections: 15980
    Merged detections: 72
    Detections after filter (before merge): 160

    Species detected: 1

    bird: 72 merged detections
      Confidence stats: avg=0.590, min=0.364, max=0.808
    ...
    FILTERING COMPLETED SUCCESSFULLY
    ```

### All Formats and an Explicit Phrase Gap

=== "Command"
    ```bash
    python src/evaluation/filter_and_merge_detections.py \
        --raw-detections results \
        --conf 0.25 \
        --phrase-gap 0.1 \
        --output-format all \
        --output-path results/filtered_0.25
    ```
=== "Expected Output (excerpt)"
    ```text
    Filtered at conf>=0.25, merged (fusion, phrase_gap=0.1s) -> 124 detections
    Saved filtered detections to JSON: results/filtered_0.25/with_algorithm_metadata.json
    Saved filtered detections to CSV: results/filtered_0.25/simplified.csv
    Saved filtered detections to Xeno-Canto Annota-JSON: results/filtered_0.25/xeno-canto-annota.json
    Saved filtered detections to Raven Selection Tables in: results/filtered_0.25/raven
    ```

With raw detections from a single recording, the Raven line is `Saved filtered detections to Raven Selection Table: results/filtered_0.25/raven_selection_table.txt`.

### Multiple Formats

=== "Command"
    ```bash
    python src/evaluation/filter_and_merge_detections.py \
        --raw-detections results \
        --conf 0.4 \
        --output-format json-with-algorithm-metadata simplified-csv \
        --output-path results/filtered_0.4
    ```
=== "Expected Output (excerpt)"
    ```text
    Filtered at conf>=0.4, merged (fusion, phrase_gap=0.05s) -> 64 detections
    Saved filtered detections to JSON: results/filtered_0.4/with_algorithm_metadata.json
    Saved filtered detections to CSV: results/filtered_0.4/simplified.csv
    ```

## Typical Workflow

This script sits at **Step 3** of the standard evaluation pipeline:

```
Step 1  detect_birds.py --conf 0.001 --no-merge    ->  raw_detections.json
Step 2  f_beta_score_analysis.py                   ->  f_beta_score_analysis/optimal_thresholds.csv
Step 3  filter_and_merge_detections.py --conf 0.15 ->  simplified.csv
Step 4  confusion_matrix_analysis.py               ->  confusion_matrix_analysis/
```

!!! info "When to Use This Tool"
    - **After F-beta analysis**: Apply the best threshold found by `f_beta_score_analysis.py` without re-running inference.
    - **For rapid threshold experiments**: Test multiple thresholds on the same raw file in seconds.
    - **To generate CSV files**: Produce ground-truth-compatible CSV files for external evaluation tools or `confusion_matrix_analysis.py`.
    - **For custom thresholds**: Apply any threshold that falls outside your original `--conf-range`.
