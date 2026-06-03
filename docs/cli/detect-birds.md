
Detect bird calls in arbitrary-length audio files using a trained YOLO model. Processes WAV, FLAC, OGG, and MP3 files through the same PCEN spectrogram pipeline used during training, and returns timestamped song segments with species labels and confidence scores.

## Usage Synopsis

```bash
python src/inference/detect_birds.py --audio <path> --model <path> --species-mapping <key>
```

## Parameters

| Parameter | Type / Default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `--audio` | `PATH` / — | **Yes** | Path to an audio file (WAV, FLAC, OGG, MP3) or a directory. Directories are searched recursively for all supported audio files. |
| `--model` | `PATH` / — | **Yes** | Path to the trained YOLO model file (`.pt`, `.onnx`, `.engine`, etc.). |
| `--species-mapping` | `CHOICE` / — | **Yes** | Dataset key used to map class IDs to species eBird codes. Must match the mapping the model was trained with. See [allowed values](#allowed---species-mapping-values) below. |
| `--output-path` | `PATH` / `results/all_detections` | No | Base output path for results files. The file extension is appended automatically depending on `--output-format`. The parent directory is created automatically if it does not exist. |
| `--output-format` | `CHOICE` / `json-with-algorithm-metadata` | No | Output format for results. See [output formats](#output-formats) below. |
| `--conf` | `FLOAT` / `0.2` | No | Confidence threshold (0.0–1.0). Detections below this value are discarded. The default of `0.2` works well for direct use. For evaluation workflows, use `0.001` together with `--no-merge` to retain all raw detections. |
| `--nms-iou` | `FLOAT` / `0.7` | No | IoU threshold for Non-Maximum Suppression applied both per-clip and across overlapping time windows. Higher values keep more overlapping detections; lower values suppress more aggressively. |
| `--song-gap` | `FLOAT` / `0.1` | No | Maximum temporal gap in seconds between two detections of the same species that are still merged into one continuous song segment. Increase for species with long pauses between phrases; decrease to keep phrases separate. |
| `--workers` | `INT` / `1` | No | Number of parallel inference workers. Each worker loads its own copy of the model. Increase on multi-core systems with a GPU to speed up batch processing of long files. |
| `--no-merge` | flag / off | No | Output raw, unmerged detections instead of reconstructed song segments. Use together with a very low `--conf` (e.g. `0.001`) when generating input for `filter_and_merge_detections.py` or `f_beta_score_analysis.py`. |

### Allowed `--species-mapping` values

| Value | Description |
| :--- | :--- |
| `Just-Bird` | Binary bird / no-bird detector |
| `All-In-One` | Combined multi-region model |
| `Hawaii` | Hawaiian species |
| `Northeastern-US` | Northeastern United States species |
| `Southern-Sierra-Nevada` | Southern Sierra Nevada species |
| `Western-US` | Western United States species |
| `Amazon-Basin` | Amazon Basin species |

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

## Output Formats

The `--output-format` flag controls which file(s) are written. The `--output-path` value is used as the base name; the correct extension is appended automatically.

| Format | Extension | Description |
| :--- | :--- | :--- |
| `json-with-algorithm-metadata` | `.json` | Full detection JSON including model config, confidence scores, and all detection fields. |
| `simplified-csv` | `.csv` | Flat CSV matching the `annotations.csv` training format. Includes a `Confidence` column. |
| `xeno-canto-annota-json` | `.xc.json` | Xeno-Canto Annota-JSON format for use with the Xeno-Canto platform. |
| `raven-selection-table` | `.txt` / `_raven/` | Raven Pro Selection Table (tab-separated). Single-file input → one `.txt`; directory input → one `.txt` per source file inside a `_raven/` subdirectory. |
| `all` | all of the above | Writes all four formats in one run. |

### JSON output structure

```json
{
  "audio_file": "recording.wav",
  "model_config": {
    "model": "models/Hawaii.pt",
    "confidence_threshold": 0.2,
    "nms_iou_threshold": 0.7,
    "song_gap_threshold": 0.1,
    "species_mapping": "Hawaii"
  },
  "detection_count": 3,
  "detections": [
    {
      "species": "amerob",
      "species_id": 2,
      "time_start": 12.5,
      "time_end": 14.2,
      "avg_confidence": 0.47,
      "max_confidence": 0.80,
      "detections_merged": 6,
      "freq_low_hz": 2151,
      "freq_high_hz": 5820,
      "filename": "recording.wav",
      "file_path": "data/recording.wav"
    }
  ]
}
```

### CSV output structure

```csv
Filename,Start Time (s),End Time (s),Low Freq (Hz),High Freq (Hz),Species eBird Code,Confidence
recording.wav,12.5,14.2,2151,5820,amerob,0.470
recording.wav,25.3,27.8,1890,4560,herthr,0.612
```

## Examples

=== "Single file"
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

=== "Directory batch"
    ```bash
    python src/inference/detect_birds.py \
        --audio /path/to/audio/folder \
        --model models/Western-US.pt \
        --species-mapping Western-US \
        --output-path results/detections \
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

=== "Evaluation workflow"
    ```bash
    python src/inference/detect_birds.py \
        --audio data/test_audio/ \
        --model models/Hawaii.pt \
        --species-mapping Hawaii \
        --conf 0.001 \
        --output-path results/raw_detections \
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

=== "Parallel inference"
    ```bash
    python src/inference/detect_birds.py \
        --audio long_recording.flac \
        --model models/All-In-One.pt \
        --species-mapping All-In-One \
        --workers 4 \
        --output-path results/detections \
        --output-format simplified-csv
    ```
=== "Expected Output"
    ```text
    Loading 4 model copies for parallel inference...
    Pipeline (4 workers): 100%|████| 240/240 [00:18<00:00]
    Final count: 31 song segments
    Saved detections to CSV: results/detections.csv
    ```

!!! warning "Lossy Audio Formats"
    The model was trained on lossless WAV files. When processing MP3 or OGG input, detection performance may degrade — especially for faint calls and high-frequency species. Use WAV or FLAC whenever possible. If you must use MP3, ensure a bitrate of ≥ 256 kbps.
