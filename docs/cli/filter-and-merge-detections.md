
Filter raw bird call detections by confidence threshold and merge them into song segments — without re-running inference. This script is the post-processing companion to `detect-birds --no-merge`, allowing you to test different confidence thresholds instantly on the same raw detection file.

---

## Usage Synopsis

=== "Linux / macOS"
    ```bash
    python src/evaluation/filter_and_merge_detections.py \
        --raw-detections results/raw_detections.json \
        --conf 0.25
    ```
=== "Windows (PowerShell)"
    ```powershell
    python src/evaluation/filter_and_merge_detections.py `
        --raw-detections results/raw_detections.json `
        --conf 0.25
    ```
=== "Windows (CMD)"
    ```cmd
    python src/evaluation/filter_and_merge_detections.py ^
        --raw-detections results/raw_detections.json ^
        --conf 0.25
    ```

## Parameters

| Parameter | Type / Default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `--raw-detections` | `PATH` / — | **Yes** | Path to the raw detections JSON file produced by `detect_birds.py --no-merge`. |
| `--conf` | `FLOAT` / `0.2` | No | Confidence threshold for filtering (0.0–1.0). All raw detections with `confidence < threshold` are discarded before merging. |
| `--song-gap` | `FLOAT` / from JSON or `0.1` | No | Maximum temporal gap in seconds between two detections of the same species that are still merged into one song segment. When omitted, the value stored in the JSON's `model_config.song_gap_threshold` is used; falls back to `0.1` if not present. |
| `--output-path` | `PATH` / `results/merged_detections` | No | Base output path for result files (without extension). The correct extension is appended per format. The parent directory is created automatically if it does not exist. |
| `--output-format` | `CHOICE [...]` / `json-with-algorithm-metadata simplified-csv` | No | One or more output formats (space-separated). Accepts `json-with-algorithm-metadata`, `simplified-csv`, `xeno-canto-annota-json`, `raven-selection-table`, or `all`. |

## Parameter Deep-Dives

### `--conf` — Confidence Threshold

This threshold is applied to the raw per-clip detections **before** song merging. Because it operates on raw clips (not merged songs), it is directly comparable to the `--conf` value you would have passed to `detect_birds.py`.

!!! info "Why filter raw clips and not merged songs?"
    The `filter-then-merge` order is the same workflow used internally by the app. Filtering merged songs by their average or max confidence gives different — and generally worse — results than filtering the underlying raw clips first and then merging the survivors.

### `--song-gap` — Song Gap Threshold

Controls how aggressively surviving detections are fused into continuous song segments after filtering. If you leave it unset, the script reuses the `song_gap_threshold` value from the JSON's `model_config` block (set when running `detect_birds.py`), ensuring consistency between detection and post-processing runs.

| Value | Effect |
| :--- | :--- |
| omitted | Reads from JSON `model_config.song_gap_threshold`, falls back to `0.1` |
| `0.1` | Merge detections ≤ 0.1 s apart (recommended default) |
| `0.5` | Merge detections ≤ 0.5 s apart (moderate) |
| `2.0` | Merge detections ≤ 2 s apart (aggressive, may over-merge) |

### `--output-format` — Output Formats

Accepts **one or more** format names separated by spaces. Specify `all` to write every format in one run.

| Format | Extension | Description |
| :--- | :--- | :--- |
| `json-with-algorithm-metadata` | `.json` | Filtered results with model config, filtering config, and detection metadata preserved. |
| `simplified-csv` | `.csv` | Flat CSV matching the `annotations.csv` training format. Ready for direct comparison with ground truth. |
| `xeno-canto-annota-json` | `.xc.json` | Xeno-Canto Annota-JSON export. |
| `raven-selection-table` | `.txt` | Raven Pro Selection Table (tab-separated). |
| `all` | all of the above | Writes all four formats simultaneously. |

## Output Files

When run with `--output-path results/merged`, the following files are written depending on `--output-format`:

```
results/merged.json        ← json-with-algorithm-metadata
results/merged.csv         ← simplified-csv
results/merged.xc.json     ← xeno-canto-annota-json
results/merged.txt         ← raven-selection-table
```

### JSON output structure

```json
{
  "audio_files": ["recording1.wav", "recording2.wav"],
  "file_count": 2,
  "model_config": {
    "model": "models/Hawaii.pt",
    "confidence_threshold": 0.001,
    "song_gap_threshold": 0.1,
    "species_mapping": "Hawaii"
  },
  "filtering_config": {
    "confidence_threshold": 0.35,
    "song_gap_threshold": 0.1
  },
  "detection_count": 23,
  "original_detection_count": 4823,
  "detections": [...]
}
```

### CSV output structure

```csv
Filename,Start Time (s),End Time (s),Low Freq (Hz),High Freq (Hz),Species eBird Code
recording.wav,12.5,14.2,2151,5820,amerob
recording.wav,25.3,27.8,1890,4560,herthr
```

The CSV format is identical to the ground truth `annotations.csv`, making it suitable for direct evaluation with `confusion_matrix_analysis.py`.

---

## Examples

### Basic filter and merge

=== "Command"
    ```bash
    python src/evaluation/filter_and_merge_detections.py \
        --raw-detections results/raw_detections.json \
        --conf 0.35 \
        --output-path results/filtered_0.35
    ```
=== "Expected Output"
    ```text
    Loading detections from: results/raw_detections.json
    Loaded 4823 total detections
    Confidence range: 0.001 - 0.998
    Mean confidence: 0.183

    Filtered at conf>=0.35, merged (song_gap=0.1s) -> 23 segments

    ================================================================================
    FILTERING SUMMARY
    ================================================================================
    Confidence threshold: 0.35
    Original detections: 4823
    Merged segments: 23
    ```

### All formats + explicit song-gap

=== "Command"
    ```bash
    python src/evaluation/filter_and_merge_detections.py \
        --raw-detections results/raw_detections.json \
        --conf 0.25 \
        --song-gap 0.1 \
        --output-format all \
        --output-path results/filtered_0.25
    ```
=== "Expected Output"
    ```text
    Filtered at conf>=0.25, merged (song_gap=0.1s) -> 31 segments
    Saved filtered detections to JSON: results/filtered_0.25.json
    Saved filtered detections to CSV: results/filtered_0.25.csv
    Saved filtered detections to Xeno-Canto Annota-JSON: results/filtered_0.25.xc.json
    Saved filtered detections to Raven Selection Table: results/filtered_0.25.txt
    ```

### Multiple formats

=== "Command"
    ```bash
    python src/evaluation/filter_and_merge_detections.py \
        --raw-detections results/raw_detections.json \
        --conf 0.4 \
        --output-format json-with-algorithm-metadata simplified-csv \
        --output-path results/filtered_0.4
    ```
=== "Expected Output"
    ```text
    Filtered at conf>=0.40, merged (song_gap=0.1s) -> 18 segments
    Saved filtered detections to JSON: results/filtered_0.4.json
    Saved filtered detections to CSV: results/filtered_0.4.csv
    ```

## Typical Workflow

This script sits at **Step 3** of the standard evaluation pipeline:

```
Step 1  detect_birds.py --conf 0.001 --no-merge    →  raw_detections.json
Step 2  f_beta_score_analysis.py                   →  optimal_thresholds.csv
Step 3  filter_and_merge_detections.py --conf 0.35 →  filtered_detections.csv
Step 4  confusion_matrix_analysis.py               →  confusion_matrix/
```

!!! info "When to Use This Tool"
    - **After F-beta analysis**: Apply the optimal threshold found by `f_beta_score_analysis.py` without re-running inference.
    - **For rapid threshold experiments**: Test multiple thresholds on the same raw file in seconds.
    - **To generate CSV files**: Produce ground-truth-compatible CSV files for external evaluation tools or `confusion_matrix_analysis.py`.
    - **For custom thresholds**: Apply any threshold that falls outside your original `--conf-range`.
