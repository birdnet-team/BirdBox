# Detecting Birds

Detect bird calls in audio files of any length with a trained YOLO model. `detect_birds.py` processes WAV, FLAC, OGG, and MP3 files through the same log-floor spectrogram pipeline used during training. It returns time and frequency boxes with species labels and confidence scores.

---

## Usage Synopsis

=== "Linux / macOS"
    ```bash
    python src/inference/detect_birds.py \
        --audio path/to/audio.wav \
        --model models/Just-Bird_fp32.pt \
        --species-mapping Just-Bird
    ```
=== "Windows (PowerShell)"
    ```powershell
    python src/inference/detect_birds.py `
        --audio path/to/audio.wav `
        --model models/Just-Bird_fp32.pt `
        --species-mapping Just-Bird
    ```
=== "Windows (CMD)"
    ```cmd
    python src/inference/detect_birds.py ^
        --audio path/to/audio.wav ^
        --model models/Just-Bird_fp32.pt ^
        --species-mapping Just-Bird
    ```

## Parameters

| Parameter | Type / Default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `--audio` | `PATH` / none | **Yes** | Path to an audio file (WAV, FLAC, OGG, MP3) or a directory. Directories are searched recursively for all supported audio files. |
| `--model` | `PATH` / none | **Yes** | Path to the trained YOLO model file (`.pt`, `.onnx` or `.tflite`). |
| `--species-mapping` | `CHOICE` / none | **Yes** | `Just-Bird` or `Birds-With-Labels`, the model the class IDs come from. It maps them to eBird codes and sets the default threshold. Must match the model. See [Species mapping](../data/audio-and-model.md#species-mapping) in the Data section. |
| `--output-path` | `PATH` / `results` | No | Output directory for result files (default: `results/`, auto-versioned to `results/run_N/` when outputs already exist). See [Output Formats](#output-formats). |
| `--output-format` | `CHOICE [...]` / `json-with-algorithm-metadata` | No | One or more output formats (space-separated). Accepts `json-with-algorithm-metadata`, `simplified-csv`, `xeno-canto-annota-json`, `raven-selection-table`, or `all`. Ignored when `--no-merge` is set (only `raw_detections.json` is written). |
| `--conf` | `FLOAT` / model default | No | Confidence threshold (0.0–1.0). Detections below this value are discarded. The default is `0.15` for Just-Bird and `0.21` for Birds-With-Labels, for direct use. For evaluation workflows, use `0.001` together with `--no-merge` to retain all raw detections. |
| `--nms-iou` | `FLOAT` / `0.7` | No | IoU threshold for Non-Maximum Suppression inside each 3-second clip. Higher values keep more overlapping detections. Lower values suppress more aggressively. |
| `--merge` | `CHOICE` / `auto` | No | How detections from the 50%-overlap windows are merged: `fusion` (window ownership + weighted box fusion, tight call-level boxes), `reconstruct` (song merge with `--song-gap` and `--freq-iou`), or `auto`, which is `fusion` for both models. See [Merging the Windows](../how-it-works/overview.md#merging-the-windows). |
| `--phrase-gap` | `FLOAT` / `0.05` | No | Fusion only. Join boxes in the same frequency band that are closer than this many seconds into one phrase box. `0` keeps one box per call or syllable. See [Phrase Gap](../how-it-works/overview.md#phrase-gap). |
| `--label-conf` | `FLOAT` / `0.25` | No | Birds-With-Labels only. A box is named after its top species when that species' score reaches this value, and `bird` otherwise. See [Species Names](../how-it-works/overview.md#species-names). |
| `--song-gap` | `FLOAT` / `0.1` | No | `reconstruct` merge only. Maximum gap in seconds between two detections of the same species that are still merged into one song box. Increase for species with long pauses between phrases. Decrease to keep phrases separate. |
| `--freq-iou` | `FLOAT` / `0.6` for Just-Bird, off otherwise | No | `reconstruct` merge with Just-Bird only. Minimum frequency IoU required before two `bird` detections are merged into one song. Birds-With-Labels ignores this flag and keeps the time-only merger. Pass a negative value to force the time-only merger for Just-Bird. |
| `--num-workers` | `INT` / `1` | No | Number of CPU processes for spectrogram features, rendering, and multi-file preprocess. YOLO inference stays in the parent and runs in batches. |
| `--no-merge` | flag / off | No | Evaluation mode: clip-level detections only, writes **`raw_detections.json`** and ignores `--output-format`. Use it with a low `--conf` (e.g. `0.001`) for `f_beta_score_analysis.py` and `filter_and_merge_detections.py`. |
| `--verbose` | flag / off | No | Print per-file processing details, clip-level progress bars, and a detection summary. Default is a single file-level progress bar. |

!!! danger "Environment must match the model format"
    Each model format (`.pt`, `.onnx`, `.tflite`) requires a different Python environment. Using the wrong environment raises an import error or gives worse results without a warning. Run `python install.py --model-format <FORMAT>` once for each format you want to use. See [Install Parameters](../installation.md#install-parameters) for the full table.

## Parameter Deep-Dives

### Confidence Threshold

The confidence threshold (`--conf`) is the single most important tuning parameter. It controls how many detections reach the output.

| Use case | Recommended value |
| :--- | :--- |
| Quick field recording scan | `0.15` for Just-Bird, `0.21` for Birds-With-Labels (default) |
| High-precision output (few false positives) | `0.4` to `0.6` |
| Comprehensive evaluation (feed into F-beta sweep) | `0.001` with `--no-merge` |

!!! info "Evaluation Workflow Tip"
    For evaluation, run detection once at a very low confidence (`--conf 0.001 --no-merge`) to capture all candidate detections as raw JSON. Then use `f_beta_score_analysis.py` to find the best threshold. Apply it with `filter_and_merge_detections.py`, which needs no new inference run.

### Merge Method

`--merge` picks how the detections of the overlapping windows become the final boxes.

| Value | Output |
| :--- | :--- |
| `fusion` (default for both models) | One tight box per call, or per phrase with `--phrase-gap` |
| `reconstruct` | Wide song boxes, joined with `--song-gap` |

`auto` picks `fusion`. `--song-gap` and `--freq-iou` only apply to `reconstruct`, and `--phrase-gap` only to `fusion`. [Merging the Windows](../how-it-works/overview.md#merging-the-windows) explains both merges.

### Phrase Gap

With `fusion`, `--phrase-gap` joins boxes in the same frequency band that are less than this many seconds apart.

| Value | Effect |
| :--- | :--- |
| `0` | One box per call or syllable. Use this if your annotations box single calls. |
| `0.05` (default) | Joins only boxes that nearly touch, for recordings of any annotation convention. |
| `0.1` to `0.5` | Longer phrase boxes, if your annotations box phrases or songs. |

[Phrase Gap](../how-it-works/overview.md#phrase-gap) shows how the gap changes the scores.

### Song Gap Threshold

This applies only to the `reconstruct` merge (`--merge reconstruct`). With window fusion, the default, `--song-gap` has no effect. The song merge joins detections of the same species that follow each other in time into one song box. Two detections are merged when the gap between them is at most `--song-gap` seconds.

```
Raw detections (same species):
  71.80s - 72.11s
  72.50s - 73.20s   <- gap = 0.39 s  (merged if song-gap >= 0.39)
  73.50s - 75.24s   <- gap = 0.30 s  (merged if song-gap >= 0.30)

Result with --song-gap 0.5:
  71.80s - 75.24s   (3 detections merged, avg_confidence reported)
```

| Value | Effect |
| :--- | :--- |
| `0.05` | Very conservative. Only detections that nearly touch are merged. |
| `0.1` (default) | Good balance for most species |
| `0.5` | Moderate. Merges phrases separated by short pauses. |
| `2.0` | Aggressive. May merge distinct song bouts. |

### Frequency IoU Gate for Just-Bird

This also applies only to `--merge reconstruct`. Just-Bird predicts a single class. Two songs that overlap in time would otherwise become one box, and the frequency span would grow to cover both. When `--species-mapping` is `Just-Bird`, a detection joins an open song only if the time gap is within `--song-gap` and the frequency IoU is at least `--freq-iou` (default `0.6`). Several songs can stay open at once.

Birds-With-Labels does not use this gate. Different species already stay in separate groups.

### NMS IoU Threshold

`--nms-iou` is applied by YOLO inside each 3-second spectrogram clip. Of two boxes that overlap by more than this IoU, it keeps only the one with the higher confidence. Duplicates from overlapping windows are handled later by the merge.

!!! info "Relationship to the merge"
    `--nms-iou` removes duplicates *within* each clip. The merge (window fusion, or `--song-gap` with `reconstruct`) then combines the detections of overlapping clips. They operate at different stages of the pipeline and do not conflict.

### Parallel Workers

With `--num-workers`, CPU work (spectrogram features, rendering, and whole-file preprocess on folders) runs in a process pool so each worker has its own GIL and a single BLAS thread. YOLO inference stays in the parent process and is batched, which avoids loading extra model copies and is safer on a single GPU. Matches the `num_workers` constructor argument.

!!! warning "Memory Usage"
    `--num-workers` does not load extra YOLO weights. Workers add RAM for matplotlib and librosa only. The parent holds one model (and one GPU context when using CUDA).

### Noise Floor

Each clip's noise floor is taken over the 60 s of audio up to the clip's end ([rolling floor](../how-it-works/spectrogram.md#stage-5-log-floor-normalization)), so nothing after a clip changes its boxes. The JSON output records the floor window in `model_config.floor_window_seconds`.

### Evaluation Mode

With `--no-merge`, `detect_birds.py` enters evaluation mode:

- The merge is skipped, so the clip-level detections are kept.
- Only **`raw_detections.json`** is written under `--output-path` (default `results/`).
- `--output-format` is ignored. A note is printed if other formats were requested.

Use this for the [detection and evaluation workflow](workflows.md#simple-workflow-for-detection-and-evaluation). For normal field use, omit `--no-merge` and pick formats with `--output-format`.

### Choosing Output Formats

`--output-format` accepts **one or more** format names separated by spaces. Specify `all` to write every format in one run. For full schema documentation of each format see [Detection Output Formats](../data/outputs.md).

### Species Mapping

The `--species-mapping` name must match the label space the model was trained on. It is **not** inferred from the weights filename. You pass it explicitly.
For details, see [Species mapping](../data/audio-and-model.md#species-mapping) in the Data section.

---

## Output Formats

The `--output-format` flag controls which file(s) are written under `--output-path` (unless `--no-merge` is set). Full schema documentation for every format, including JSON field tables and CSV column definitions, is in [Detection Output Formats](../data/outputs.md).

Every run also writes `args.yaml` in the same output directory. That file records every CLI argument used for the run, including defaults, so you can compare settings such as `--nms-iou` and `--phrase-gap` across result folders.

---

## Examples

The examples use [Just-Bird](../models-and-metrics/just-bird.md), the recommended model. It detects bird vocalizations with a single `bird` class. For species-level output, use the Birds-With-Labels model with its species mapping, as in the verbose logging example.

### Single File

=== "Command"
    ```bash
    python src/inference/detect_birds.py \
        --audio recording.wav \
        --model models/Just-Bird_fp32.pt \
        --species-mapping Just-Bird
    ```
=== "Expected Output"
    ```text
    ✓ Created output directory: results
    Saved run arguments to: results/args.yaml
    Processing files: 100%|██████████| 1/1 [00:14<00:00, 14.12s/file]
    Saved detections to: results/with_algorithm_metadata.json
    ```

The first line appears only when the output directory is new.

### Directory Batch

=== "Command"
    ```bash
    python src/inference/detect_birds.py \
        --audio /path/to/audio/folder \
        --model models/Just-Bird_fp32.pt \
        --species-mapping Just-Bird \
        --output-path results \
        --output-format all
    ```
=== "Expected Output"
    ```text
    Saved run arguments to: results/args.yaml
    Processing files: 100%|██████████| 8/8 [01:52<00:00, 14.03s/file]
    Saved detections to: results/with_algorithm_metadata.json
    Saved detections to CSV: results/simplified.csv
    Saved detections to Xeno-Canto Annota-JSON: results/xeno-canto-annota.json
    Saved Raven Selection Tables to directory: results/raven
    ```

### Evaluation Workflow

=== "Command"
    ```bash
    python src/inference/detect_birds.py \
        --audio data/test_audio/ \
        --model models/Just-Bird_fp32.pt \
        --species-mapping Just-Bird \
        --conf 0.001 \
        --output-path results \
        --no-merge
    ```
=== "Expected Output"
    ```text
    Saved run arguments to: results/args.yaml
    Processing files: 100%|██████████| 12/12 [02:52<00:00, 14.38s/file]
    Saved detections to: results/raw_detections.json
    ```

### Parallel Inference

=== "Command"
    ```bash
    python src/inference/detect_birds.py \
        --audio long_recording.flac \
        --model models/Just-Bird_fp32.pt \
        --species-mapping Just-Bird \
        --num-workers 4 \
        --output-path results \
        --output-format simplified-csv
    ```
=== "Expected Output"
    ```text
    Saved run arguments to: results/args.yaml
    Processing files: 100%|██████████| 1/1 [00:18<00:00, 18.00s/file]
    Saved detections to CSV: results/simplified.csv
    ```

### Verbose Logging

=== "Command"
    ```bash
    python src/inference/detect_birds.py \
        --audio /path/to/audio/folder \
        --model models/Birds-With-Labels_fp32.pt \
        --species-mapping Birds-With-Labels \
        --verbose
    ```
=== "Expected Output"
    ```text
    Saved run arguments to: results/args.yaml
    Loaded model: models/Birds-With-Labels_fp32.pt
    Species mapping: Birds-With-Labels
    Species count: 634
    Confidence threshold: 0.21
    NMS IoU threshold: 0.7
    Bird class 633: boxes from the bird score, named by species (species score >= 0.25, else "bird")
    Merge method: fusion
    Phrase gap: 0.05s
    Found 8 audio files in directory: /path/to/audio/folder
    Processing 8 audio files...

    ============================================================
    Processing file 1/8: dawn_chorus.wav
    ============================================================
    Loading audio: /path/to/audio/folder/dawn_chorus.wav
    Duration: 300.00 seconds
    Sample rate: 32000 Hz
    Computing log-floor spectrograms (rolling 60 s floor)...
    Audio is 300.00s. Padding 1.50s of silence so the final 3s clip is complete.
    Planning to extract 200 clips from 301.5s audio
    Successfully extracted 200 clips
    Running detection on 200 clips...
    Detecting: 100%|██████████| 200/200 [00:12<00:00]
    Found 490 raw detections
    Merging detections (fusion)...
    Final count: 130 merged detections
    Found 130 detections in this file
    ...
    TOTAL DETECTIONS ACROSS ALL FILES: 1040
    ...
    DETECTION SUMMARY
    ...
    hawama: 92 merged detections
        6.94s -   8.18s ( 1.25s duration,  1 raw boxes merged, avg conf: 0.735, max conf: 0.735)
       20.84s -  20.98s ( 0.14s duration,  3 raw boxes merged, avg conf: 0.301, max conf: 0.301)
      ... and 90 more
      Stats: avg duration 0.58s, avg raw boxes merged 1.2
    ```

!!! warning "Lossy Audio Formats"
    MP3 and OGG compression can hide faint calls and high frequencies, so detection may get worse. In a study of avian recordings, MP3 compression reduced the detections of human listeners and automated recognisers [[22]](#ref-macphail). Use WAV or FLAC whenever possible. If you must use MP3, use a bitrate of at least 256 kbps.

---

## References

<a id="ref-macphail"></a>
**[22]** MacPhail, A. G., Yip, D. A., Knight, E. C., Hedley, R., Knaggs, M., Shonfield, J., Upham-Mills, E., & Bayne, E. M. (2024). "Audio data compression affects acoustic indices and reduces detections of birds by human listening and automated recognisers." *Bioacoustics*, 33(1):74–90.
