
Detect bird calls in arbitrary-length audio files using a trained YOLO model. Processes WAV, FLAC, OGG, and MP3 files through the same PCEN spectrogram pipeline used during training, and returns timestamped song segments with species labels and confidence scores.

---

## Usage Synopsis

=== "Linux / macOS"
    ```bash
    python src/inference/detect_birds.py \
        --audio path/to/audio.wav \
        --model models/best.pt \
        --species-mapping species_mapping
    ```
=== "Windows (PowerShell)"
    ```powershell
    python src/inference/detect_birds.py `
        --audio path/to/audio.wav `
        --model models/best.pt `
        --species-mapping species_mapping
    ```
=== "Windows (CMD)"
    ```cmd
    python src/inference/detect_birds.py ^
        --audio path/to/audio.wav ^
        --model models/best.pt ^
        --species-mapping species_mapping
    ```

## Parameters

| Parameter | Type / Default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `--audio` | `PATH` / — | **Yes** | Path to an audio file (WAV, FLAC, OGG, MP3) or a directory. Directories are searched recursively for all supported audio files. |
| `--model` | `PATH` / — | **Yes** | Path to the trained YOLO model file (`.pt`, `.onnx`, `.engine`, etc.). |
| `--species-mapping` | `CHOICE` / — | **Yes** | Dataset key used to map class IDs to species eBird codes. Must match the mapping the model was trained with. See [allowed values](../data/audio-and-model.md#species-mapping-species-mapping) in the Data section. |
| `--output-path` | `PATH` / `results` | No | Output directory for result files (default: `results/`, auto-versioned to `results/run_N/` when outputs already exist). See [Output Formats](#output-formats). |
| `--output-format` | `CHOICE [...]` / `json-with-algorithm-metadata` | No | One or more output formats (space-separated). Accepts `json-with-algorithm-metadata`, `simplified-csv`, `xeno-canto-annota-json`, `raven-selection-table`, or `all`. Ignored when `--no-merge` is set (only `raw_detections.json` is written). |
| `--conf` | `FLOAT` / `0.2` | No | Confidence threshold (0.0–1.0). Detections below this value are discarded. The default of `0.2` works well for direct use. For evaluation workflows, use `0.001` together with `--no-merge` to retain all raw detections. |
| `--nms-iou` | `FLOAT` / `0.7` | No | IoU threshold for Non-Maximum Suppression applied both per-clip and across overlapping time windows. Higher values keep more overlapping detections. Lower values suppress more aggressively. |
| `--song-gap` | `FLOAT` / `0.1` | No | Maximum temporal gap in seconds between two detections of the same species that are still merged into one continuous song segment. Increase for species with long pauses between phrases. Decrease to keep phrases separate. |
| `--workers` | `INT` / `1` | No | Number of parallel inference workers. Each worker loads its own copy of the model. Increase on multi-core systems with a GPU to speed up batch processing of long files. |
| `--no-merge` | flag / off | No | Evaluation mode: clip-level detections only, writes **`raw_detections.json`** and ignores `--output-format`. Use with low `--conf` (e.g. `0.001`) for `f_beta_score_analysis.py` / `filter_and_merge_detections.py`. |

## Parameter Deep-Dives

### `--conf` — Confidence Threshold

The confidence threshold is the single most important tuning parameter. It controls how many detections reach the output.

| Use-case | Recommended value |
| :--- | :--- |
| Quick field recording scan | `0.2` (default) |
| High-precision output (few false positives) | `0.4`–`0.6` |
| Comprehensive evaluation (feed into F-beta sweep) | `0.001` with `--no-merge` |

!!! info "Evaluation Workflow Tip"
    For evaluation, run detection once at a very low confidence (`--conf 0.001 --no-merge`) to capture all candidate detections as raw JSON. Then use `f_beta_score_analysis.py` to find the optimal threshold, and apply it cheaply with `filter_and_merge_detections.py`—without re-running inference.

### `--song-gap` — Song Gap Threshold

After detection, temporally adjacent detections of the same species are merged into continuous song segments. Two detections are merged when the gap between them is ≤ `--song-gap` seconds.

```
Raw detections (same species):
  71.80s – 72.11s
  72.50s – 73.20s   ← gap = 0.39 s  (merged if song-gap ≥ 0.39)
  73.50s – 75.24s   ← gap = 0.30 s  (merged if song-gap ≥ 0.30)

Result with --song-gap 0.5:
  71.80s – 75.24s   (3 clips merged, avg_conf reported)
```

| Value | Effect |
| :--- | :--- |
| `0.05` | Very conservative — only clips nearly touching are merged |
| `0.1` (default) | Good balance for most species |
| `0.5` | Moderate — merges phrases separated by short pauses |
| `2.0` | Aggressive — may over-merge distinct song bouts |

### `--nms-iou` — NMS IoU Threshold

Applied inside each 3-second spectrogram clip and again across overlapping time windows. It removes duplicate bounding boxes that exceed the IoU overlap threshold, keeping only the highest-confidence box.

!!! info "Relationship to `--song-gap`"
    `--nms-iou` removes duplicates *within* and *across* overlapping clips. `--song-gap` then merges the surviving detections into song segments. They operate at different stages of the pipeline and do not conflict.

### `--workers` — Parallel Workers

Each additional worker loads a full copy of the model into memory. On GPU systems, multiple workers share the same GPU but run in separate threads, each owning its model copy to avoid thread-safety issues.

!!! warning "Memory Usage"
    With `--workers 4` and a 100 MB model, approximately 400 MB of model memory is allocated (plus VRAM per worker). Monitor memory usage when increasing workers significantly.

### `--no-merge` — Evaluation mode { #--no-merge--evaluation-mode }

When set, `detect_birds.py` enters evaluation mode:

- Song merging is skipped (clip-level detections kept).
- Only **`raw_detections.json`** is written under `--output-path` (default `results/`).
- `--output-format` is ignored; a note is printed if other formats were requested.

Use this for the [detection & evaluation workflow](workflows.md#simple-workflow-for-detection-evaluation). For normal field use, omit `--no-merge` and pick formats with `--output-format`.

### `--output-format` — Output Formats

Accepts **one or more** format names separated by spaces. Specify `all` to write every format in one run. For full schema documentation of each format see [Detection Output Formats](../data/outputs.md).

### `--species-mapping` - Interpretation of Output Labels

The mapping name must match the label space the model was trained on. It is **not** inferred from the weights filename. You pass it explicitly.
For details see [Data-Input/Species-Mapping](../data/audio-and-model.md#species-mapping-species-mapping).

---

## Output Formats

The `--output-format` flag controls which file(s) are written under `--output-path` (unless `--no-merge` is set). Full schema documentation for every format, including JSON field tables and CSV column definitions, is in [Detection Output Formats](../data/outputs.md).

---

## Examples

### Single file

=== "Command"
    ```bash
    python src/inference/detect_birds.py \
        --audio recording.wav \
        --model models/Hawaii.pt \
        --species-mapping Hawaii
    ```
=== "Expected Output"
    ```text
    Loading audio: recording.wav
    Duration: 120.00 seconds
    Sample rate: 32000 Hz

    Processing audio with PCEN...
    Detecting: 100%|████████████| 79/79 [00:12<00:00]

    Found 47 raw detections
    Reconstructing continuous bird songs from detections...
    Final count: 12 song segments
    ```

### Directory batch

=== "Command"
    ```bash
    python src/inference/detect_birds.py \
        --audio /path/to/audio/folder \
        --model models/Western-US.pt \
        --species-mapping Western-US \
        --output-path results \
        --output-format all
    ```
=== "Expected Output"
    ```text
    Found 8 audio files in directory: /path/to/audio/folder

    ============================================================
    Processing file 1/8: dawn_chorus.wav
    ============================================================
    ...
    TOTAL DETECTIONS ACROSS ALL FILES: 94
    ```

### Evaluation workflow

=== "Command"
    ```bash
    python src/inference/detect_birds.py \
        --audio data/test_audio/ \
        --model models/Hawaii.pt \
        --species-mapping Hawaii \
        --conf 0.001 \
        --output-path results \
        --output-format json-with-algorithm-metadata \
        --no-merge
    ```
=== "Expected Output"
    ```text
    Found 12 audio files in directory: data/test_audio/
    ...
    TOTAL DETECTIONS ACROSS ALL FILES: 4823
    Saved detections to: results/raw_detections.json
    ```

### Parallel inference

=== "Command"
    ```bash
    python src/inference/detect_birds.py \
        --audio long_recording.flac \
        --model models/All-In-One.pt \
        --species-mapping All-In-One \
        --workers 4 \
        --output-path results \
        --output-format simplified-csv
    ```
=== "Expected Output"
    ```text
    Loading 4 model copies for parallel inference...
    Pipeline (4 workers): 100%|████| 240/240 [00:18<00:00]
    Final count: 31 song segments
    Saved detections to CSV: results/simplified.csv
    ```

!!! warning "Lossy Audio Formats"
    The model was trained on lossless WAV files. When processing MP3 or OGG input, detection performance may degrade — especially for faint calls and high-frequency species. Use WAV or FLAC whenever possible. If you must use MP3, ensure a bitrate of ≥ 256 kbps.
