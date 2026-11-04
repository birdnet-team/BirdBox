
Detect bird calls in arbitrary-length audio files using a trained YOLO model. Processes WAV, FLAC, OGG, and MP3 files through the same log-floor spectrogram pipeline used during training, and returns timestamped song segments with species labels and confidence scores.

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
| `--species-mapping` | `CHOICE` / — | **Yes** | `Just-Bird` or `Birds-With-Labels`, the model the class IDs come from. It maps them to eBird codes and sets the default merge. Must match the model. See [Species mapping](../data/audio-and-model.md#species-mapping-species-mapping) in the Data section. |
| `--output-path` | `PATH` / `results` | No | Output directory for result files (default: `results/`, auto-versioned to `results/run_N/` when outputs already exist). See [Output Formats](#output-formats). |
| `--output-format` | `CHOICE [...]` / `json-with-algorithm-metadata` | No | One or more output formats (space-separated). Accepts `json-with-algorithm-metadata`, `simplified-csv`, `xeno-canto-annota-json`, `raven-selection-table`, or `all`. Ignored when `--no-merge` is set (only `raw_detections.json` is written). |
| `--conf` | `FLOAT` / model default | No | Confidence threshold (0.0–1.0). Detections below this value are discarded. The default is `0.15` for Just-Bird and `0.2` for the other models, for direct use. For evaluation workflows, use `0.001` together with `--no-merge` to retain all raw detections. |
| `--nms-iou` | `FLOAT` / `0.7` | No | IoU threshold for Non-Maximum Suppression applied both per-clip and across overlapping time windows. Higher values keep more overlapping detections. Lower values suppress more aggressively. |
| `--merge` | `CHOICE` / `auto` | No | How detections from the 50%-overlap windows are merged: `fusion` (window ownership + weighted box fusion, tight call-level boxes), `reconstruct` (song merge with `--song-gap` / `--freq-iou`), or `auto`: `fusion` for Just-Bird, `reconstruct` for Birds-With-Labels. See [Window fusion](#window-fusion-just-bird-default). |
| `--phrase-gap` | `FLOAT` / `0.05` (Just-Bird), `0` (Birds-With-Labels) | No | Fusion only. Join boxes in the same frequency band that are closer than this many seconds into one phrase box. `0` keeps one box per call or syllable. See [phrase boxes](#window-fusion-just-bird-default). |
| `--song-gap` | `FLOAT` / `0.1` | No | `reconstruct` merge only. Maximum temporal gap in seconds between two detections of the same species that are still merged into one continuous song segment. Increase for species with long pauses between phrases. Decrease to keep phrases separate. |
| `--freq-iou` | `FLOAT` / `0.6` for Just-Bird, off otherwise | No | `reconstruct` merge with Just-Bird only. Minimum frequency IoU required before two `bird` detections are merged into one song. Other species mappings ignore this flag and keep the time-only merger. Pass a negative value to force the time-only merger for Just-Bird. |
| `--num-workers` | `INT` / `1` | No | Number of CPU processes for spectrogram features, rendering, and multi-file preprocess. YOLO inference stays in the parent and runs in batches. |
| `--no-merge` | flag / off | No | Evaluation mode: clip-level detections only, writes **`raw_detections.json`** and ignores `--output-format`. Use with low `--conf` (e.g. `0.001`) for `f_beta_score_analysis.py` / `filter_and_merge_detections.py`. |
| `--verbose` | flag / off | No | Print per-file processing details, clip-level progress bars, and a detection summary. Default is a single file-level progress bar. |

!!! danger "Environment must match the model format"
    Each model format (`.pt`, `.onnx`, `.tflite`, `.engine`) requires a different Python environment. Using the wrong environment raises an import error or silently degrades results. Run `python install.py --model-format <FORMAT>` once for each format you want to use. See [Install Parameters](../installation.md#install-parameters) for the full table.

## Parameter Deep-Dives

### `--conf` — Confidence Threshold

The confidence threshold is the single most important tuning parameter. It controls how many detections reach the output.

| Use-case | Recommended value |
| :--- | :--- |
| Quick field recording scan | `0.15` for Just-Bird, `0.2` for the other models (default) |
| High-precision output (few false positives) | `0.4`–`0.6` |
| Comprehensive evaluation (feed into F-beta sweep) | `0.001` with `--no-merge` |

!!! info "Evaluation Workflow Tip"
    For evaluation, run detection once at a very low confidence (`--conf 0.001 --no-merge`) to capture all candidate detections as raw JSON. Then use `f_beta_score_analysis.py` to find the optimal threshold, and apply it cheaply with `filter_and_merge_detections.py`—without re-running inference.

### Window fusion (Just-Bird default) { #window-fusion-just-bird-default }

Detection runs on 3 s windows with a 1.5 s hop, so almost every call is seen by two windows. The song merge (`reconstruct`) unions both views, which widens boxes and chains fast calls into multi-second events. Window fusion (`--merge fusion`, the default for Just-Bird) keeps one tight box per call instead, in four steps:

1. **Ownership.** Keep a box only from the window whose middle 1.5 s holds its centre, so each call is kept once.
2. **Weighted box fusion.** Average near-duplicate boxes of one window (2D IoU ≥ 0.4), weighted by confidence.
3. **Drop nested boxes.** Remove a box with at least 80% of its area inside a larger box of the same species, such as a syllable inside the box of its call. This removes 10–25% of the boxes without lowering F1.
4. **Join.** Link boxes in the same frequency band (frequency IoU ≥ 0.8) that overlap in time by at least 0.05 s, so a long call split across windows becomes one box. With a phrase gap, boxes closer than the gap are linked instead.

[How It Works](../how-it-works/overview.md#window-fusion-just-bird) shows each step in a figure. In the output, `detections_merged` counts the raw detections behind each box. `--song-gap` and `--freq-iou` do not apply to fusion.

**Phrase boxes (`--phrase-gap`).** Some datasets box every syllable, others one box per phrase or song. A phrase gap joins boxes in the same band that are less than the gap apart. Only this time rule changes with the gap, so the boxes change gradually as it grows. The default is **0.05 s for Just-Bird**, which is run on recordings of any convention. If your annotations box single calls, pass `--phrase-gap 0`.

<img src="../../img/metrics/just_bird_phrase_gap.png#only-light" width="1000" alt="Best F1 at IoU 0.5 over the phrase gap on Hawai'i val, Southern Sierra Nevada and Western-US val, October against November model. F1 falls with the gap on Hawai'i val and changes little on the other two sets">
<img src="../../img/metrics/just_bird_phrase_gap_dark.png#only-dark" width="1000" alt="Best F1 at IoU 0.5 over the phrase gap on Hawai'i val, Southern Sierra Nevada and Western-US val, October against November model. F1 falls with the gap on Hawai'i val and changes little on the other two sets">

Each point is the best F1 over all thresholds on full recordings, and the y-axes differ between panels. The first two panels use the Just-Bird recipe trained without Southern Sierra Nevada, and the third the released models on validation minutes they did not train on.

On call labels the score falls steadily as the gap grows, while on phrase and song labels it barely moves. The 0.05 s default costs 0.005 on Hawai'i and adds at most 0.005 on the other two sets. The current model (November 2026) is well ahead on song labels at every gap, because it draws phrase boxes where its training labels did.

**Fusion or song merge.** Fusion suits models that box single calls. On test sets labelled per call it raised micro F1 at IoU 0.5 over the song merge, from 0.515 to 0.582 on Hawai'i and from 0.615 to 0.637 on Southern Sierra Nevada (models trained on each dataset alone). Most Birds-With-Labels training labels box whole songs and match the song merge's wide boxes better (fusion scored 0.02–0.05 lower), so that model keeps `reconstruct`. It only uses a phrase gap with `--merge fusion`, where the default is 0. Pass `--merge reconstruct` to use the song merge with Just-Bird.

### `--song-gap` — Song Gap Threshold

This applies to the `reconstruct` merge (the default for Birds-With-Labels). After detection, temporally adjacent detections of the same species are merged into continuous song segments. Two detections are merged when the gap between them is ≤ `--song-gap` seconds.

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

### Frequency IoU gate for Just-Bird

Just-Bird predicts a single class. Two songs that overlap in time would otherwise become one box, and the frequency span would grow to cover both. When `--species-mapping` is `Just-Bird`, a detection joins an open song only if the time gap is within `--song-gap` and the frequency IoU is at least `--freq-iou` (default `0.6`). Several songs can stay open at once.

Other species mappings do not use this gate. Different species already stay in separate groups.

### `--nms-iou` — NMS IoU Threshold

Applied inside each 3-second spectrogram clip and again across overlapping time windows. It removes duplicate bounding boxes that exceed the IoU overlap threshold, keeping only the highest-confidence box.

!!! info "Relationship to the merge"
    `--nms-iou` removes duplicates *within* each clip. The merge (window fusion, or `--song-gap` with `reconstruct`) then combines the detections of overlapping clips. They operate at different stages of the pipeline and do not conflict.

### `--num-workers` — Parallel Workers

CPU work (spectrogram features, rendering, and whole-file preprocess on folders) runs in a process pool so each worker has its own GIL and a single BLAS thread. YOLO inference stays in the parent process and is batched, which avoids loading extra model copies and is safer on a single GPU. Matches the `num_workers` constructor argument.

!!! warning "Memory Usage"
    `--num-workers` does not load extra YOLO weights. Workers add RAM for matplotlib/librosa only. The parent holds one model (and one GPU context when using CUDA).

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

Every run also writes `args.yaml` in the same output directory. That file records every CLI argument used for the run, including defaults, so you can compare settings such as `--nms-iou` and `--song-gap` across result folders.

---

## Examples

The examples use [Just-Bird](../models-and-metrics/just-bird.md), the recommended model. It detects bird vocalizations with a single `bird` class. For species-level output, use the Birds-With-Labels model with its species mapping, as in the verbose logging example.

### Single file

=== "Command"
    ```bash
    python src/inference/detect_birds.py \
        --audio recording.wav \
        --model models/Just-Bird_fp32.pt \
        --species-mapping Just-Bird
    ```
=== "Expected Output"
    ```text
    Saved run arguments to: results/args.yaml
    Processing files: 100%|████████████| 1/1 [00:12<00:00, 12.00s/file]

    Saved detections to: results/with_algorithm_metadata.json
    ```

### Directory batch

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
    Processing files: 100%|████████████| 8/8 [00:24<00:00,  3.00s/file]

    Saved detections to: results/with_algorithm_metadata.json
    Saved detections to CSV: results/simplified.csv
    Saved detections to Xeno-Canto Annota-JSON: results/xeno-canto-annota.json
    Saved Raven Selection Tables to directory: results/raven
    ```

### Evaluation workflow

=== "Command"
    ```bash
    python src/inference/detect_birds.py \
        --audio data/test_audio/ \
        --model models/Just-Bird_fp32.pt \
        --species-mapping Just-Bird \
        --conf 0.001 \
        --output-path results \
        --output-format json-with-algorithm-metadata \
        --no-merge
    ```
=== "Expected Output"
    ```text
    Saved run arguments to: results/args.yaml
    Processing files: 100%|████████████| 12/12 [00:36<00:00,  3.00s/file]

    Saved detections to: results/raw_detections.json
    ```

### Parallel inference

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
    Processing files: 100%|████████████| 1/1 [00:18<00:00, 18.00s/file]

    Saved detections to CSV: results/simplified.csv
    ```

### Verbose logging

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
    Found 8 audio files in directory: /path/to/audio/folder

    ============================================================
    Processing file 1/8: dawn_chorus.wav
    ============================================================

    Loading audio: /path/to/audio/folder/dawn_chorus.wav
    Duration: 60.00 seconds
    Sample rate: 32000 Hz

    Computing log-floor spectrograms...
    Planning to extract 39 clips from 60.0s audio
    Successfully extracted 39 clips

    Running detection on 39 clips...
    Detecting: 100%|████████████| 39/39 [00:03<00:00]
    Found 47 raw detections
    Reconstructing continuous bird songs from detections...
    Final count: 12 song segments
    Found 12 detections in this file
    ...
    TOTAL DETECTIONS ACROSS ALL FILES: 94
    ```

!!! warning "Lossy Audio Formats"
    The model was trained on lossless WAV files. When processing MP3 or OGG input, detection performance may degrade — especially for faint calls and high-frequency species. In a study of avian recordings, MP3 compression reduced the detections of human listeners and automated recognisers [[19]](#ref-macphail). Use WAV or FLAC whenever possible. If you must use MP3, ensure a bitrate of ≥ 256 kbps.

---

## References

<a id="ref-macphail"></a>
**[19]** MacPhail, A. G., Yip, D. A., Knight, E. C., Hedley, R., Knaggs, M., Shonfield, J., Upham-Mills, E., & Bayne, E. M. (2024). "Audio data compression affects acoustic indices and reduces detections of birds by human listening and automated recognisers." *Bioacoustics*, 33(1):74–90.
