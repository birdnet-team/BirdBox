# Detecting Birds

Detect bird calls in audio files with a trained YOLO model. `BirdCallDetector` processes WAV, FLAC, OGG, and MP3 files through the log-floor spectrogram pipeline. It returns time and frequency boxes with species labels and confidence scores.

---

## Import

```python
from inference.detect_birds import BirdCallDetector, merge_detections, reconstruct_songs, find_audio_files
```

---

## BirdCallDetector

The main entry point for all inference workflows. Instantiate once per model, then call `detect()` for any number of files.

### Constructor

```python
BirdCallDetector(
    model_path,
    species_mapping,
    conf_threshold=None,
    nms_iou_threshold=0.7,
    song_gap_threshold=0.1,
    num_workers=1,
    verbose=True,
    freq_iou_threshold=None,
    merge_method=None,
    phrase_gap=None,
    label_conf=None,
)
```

| Parameter | Type / Default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `model_path` | `str` / none | **Yes** | Path to the trained YOLO model file (`.pt`, `.onnx` or `.tflite`). |
| `species_mapping` | `str` / none | **Yes** | `'Just-Bird'` or `'Birds-With-Labels'`, the model the class IDs come from. It maps them to eBird codes and sets the default threshold. Must match the model. See [Species mapping](../data/audio-and-model.md#species-mapping). |
| `conf_threshold` | `float` / `None` | No | Confidence threshold (0.0 to 1.0). Detections below this value are discarded. `None` uses the model default: `0.15` for Just-Bird and `0.21` for Birds-With-Labels. Use `0.001` to keep all raw detections for evaluation. |
| `nms_iou_threshold` | `float` / `0.7` | No | IoU threshold for Non-Maximum Suppression inside each 3-second clip. |
| `song_gap_threshold` | `float` / `0.1` | No | `reconstruct` merge only. Maximum gap in seconds between two detections of the same species that are still merged into one song box. |
| `num_workers` | `int` / `1` | No | Number of CPU processes for spectrogram features, rendering, and multi-file preprocess. YOLO inference stays in the parent and runs in batches. |
| `verbose` | `bool` / `True` | No | If `True`, print per-file processing details and clip-level progress bars. If `False`, show a single file-level progress bar. The CLI passes `False` unless `--verbose` is set. |
| `freq_iou_threshold` | `float` / `None` | No | `reconstruct` merge with Just-Bird only: frequency IoU gate (`None` = `0.6`, negative = time-only). |
| `merge_method` | `str` / `None` | No | `'fusion'` (window ownership + weighted box fusion) or `'reconstruct'` (song merge). `None` or `'auto'` picks `'fusion'` for both models. See [Window fusion](../how-it-works/overview.md#window-fusion). |
| `phrase_gap` | `float` / `None` | No | Fusion only. Join same-band boxes closer than this many seconds into one phrase box. `0` keeps call-level boxes. `None` uses the model default of 0.05 s. |
| `label_conf` | `float` / `None` | No | Birds-With-Labels only. A box is named after its top species when that species' score reaches this, and `bird` otherwise. `None` uses 0.25. See [Species Names](../how-it-works/overview.md#species-names). |

!!! warning "Memory Usage"
    Extra workers are render processes, not extra model copies. The parent holds one YOLO model. Workers use additional RAM for matplotlib and librosa, not another set of weights.

---

### Methods

| Method | Returns | Description |
| :--- | :--- | :--- |
| `detect(audio_path, output_path, output_formats, no_merge)` | `List[Dict]` | Detect in a file or directory. Main entry point. |
| `detect_single_file(audio_path, progress_callback, no_merge)` | `List[Dict]` | Detect in a single audio file. |
| `detect_multiple_files(audio_paths, output_path, output_formats, no_merge)` | `List[Dict]` | Detect across a list of audio files. |
| `save_results(detections, output_path, audio_path, output_formats, no_merge)` | `None` | Write detections to one or more output formats. |
| `merge_overlapping_detections(detections, merge_mode, file_end)` | `List[Dict]` | Merge raw clip detections. |
| `load_audio(audio_path)` | `Tuple[np.ndarray, int]` | Load an audio file, returns `(signal, sample_rate)`. |

---

### The detect Method

```python
detector.detect(
    audio_path,
    output_path=None,
    output_formats=None,
    no_merge=False,
)
```

Detect bird calls in an audio file or directory. Automatically routes to `detect_single_file()` or `detect_multiple_files()` based on whether `audio_path` is a file or a directory.

| Parameter | Type / Default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `audio_path` | `str` / none | **Yes** | Path to a single audio file (WAV, FLAC, OGG, MP3) or a directory. Directories are searched recursively. |
| `output_path` | `str` / `None` | No | Output directory for result files. It is created if it does not exist. If `None`, results are only returned, not written. |
| `output_formats` | `List[str]` / `['json-with-algorithm-metadata']` | No | One or more output format keys. Accepts `json-with-algorithm-metadata`, `simplified-csv`, `xeno-canto-annota-json`, `raven-selection-table`, or `all`. Ignored when `no_merge=True`. |
| `no_merge` | `bool` / `False` | No | Evaluation mode. Returns raw per-clip detections and writes only `raw_detections.json`. Use with `conf_threshold=0.001` for F-beta sweep workflows. |

**Returns:** `List[Dict]`. The merged detections, or the raw clip detections when `no_merge=True`.

---

### The detect_single_file Method

```python
detector.detect_single_file(
    audio_path,
    progress_callback=None,
    no_merge=False,
)
```

Detect bird calls in a single audio file. Use this method directly when you need a `progress_callback` (e.g. in a GUI or web app).

| Parameter | Type / Default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `audio_path` | `str` / none | **Yes** | Path to the audio file. |
| `progress_callback` | `Callable[[int, int, str], None]` / `None` | No | Called as `callback(current, total, message)` after each clip. Useful for progress bars. |
| `no_merge` | `bool` / `False` | No | If `True`, return raw clip-level detections without merging. |

**Returns:** `List[Dict]`. The detections for this file.

---

### The save_results Method

```python
detector.save_results(
    detections,
    output_path,
    audio_path=None,
    output_formats=None,
    no_merge=False,
)
```

Write detections to one or more output formats under `output_path`.

| Parameter | Type / Default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `detections` | `List[Dict]` / none | **Yes** | Detections returned by `detect()` or `detect_single_file()`. |
| `output_path` | `str` / none | **Yes** | Directory to write output files into. It must exist. |
| `audio_path` | `str` / `None` | No | Source audio path, used for metadata in JSON output. |
| `output_formats` | `List[str]` / `['json-with-algorithm-metadata']` | No | One or more format keys (same values as `detect()`). |
| `no_merge` | `bool` / `False` | No | If `True`, writes only `raw_detections.json` and ignores `output_formats`. |

---

### The merge_overlapping_detections Method

```python
detector.merge_overlapping_detections(detections, merge_mode='reconstruct', file_end=None)
```

Merge raw clip detections with the detector's merge method, or remove duplicates with NMS.

| Parameter | Type / Default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `detections` | `List[Dict]` / none | **Yes** | Raw per-clip detections. |
| `merge_mode` | `str` / `'reconstruct'` | No | `'reconstruct'` (default) applies the detector's `merge_method`, which is window fusion unless you chose the song merge. `detect()` always uses it. `'nms'` removes duplicates by IoU, keeping the highest-confidence box. |
| `file_end` | `float` / `None` | No | End of the file's last window, used by window-fusion ownership. `None` takes the latest `clip_end` among the detections. |

**Returns:** `List[Dict]`. The merged or deduplicated detections.

---

## The merge_detections Function

Standalone version of the detector's merge, for raw detections saved with `--no-merge`. Filter by confidence first, then merge. The evaluation scripts use it, so their scores match detection.

```python
from inference.detect_birds import file_ends_from_raw, merge_detections, resolve_merge_settings

settings = resolve_merge_settings(raw_json["model_config"])  # fusion for both models
filtered = [d for d in raw_json["detections"] if d["confidence"] >= 0.15]
merged = merge_detections(filtered, **settings, file_end=file_ends_from_raw(raw_json["detections"]))
```

| Parameter | Type / Default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `detections` | `List[Dict]` / none | **Yes** | Raw detections with `time_start`, `time_end`, `freq_low_hz`, `freq_high_hz`, `confidence`, `species_id`, `species`, and for fusion `clip_start` and `clip_end`. Optional `filename` for multi-file inputs. |
| `merge_method` | `str` / none | **Yes** | `'fusion'` or `'reconstruct'`. |
| `song_gap_threshold` | `float` / none | **Yes** | `reconstruct` only. Max gap in seconds. |
| `freq_iou_threshold` | `float` / `None` | No | `reconstruct` only. Frequency IoU gate. |
| `phrase_gap` | `float` / `None` | No | Fusion only. Join same-band boxes closer than this into phrase boxes. `None` or `0` keeps call-level boxes. This function has no model default. `resolve_merge_settings` fills it in. |
| `file_end` | `float`, `Dict[str, float]` or `None` | No | Fusion only. End of each file's last window. `file_ends_from_raw` builds the per-file dict from the unfiltered raw list, which keeps ownership identical to live detection. |
| `label_conf` | `float` / `None` | No | Birds-With-Labels only. Species score a box needs to be named after its top species. `None` uses 0.25. |

`resolve_merge_settings(model_config, song_gap, freq_iou, merge_method, phrase_gap)` returns these keyword arguments from a raw JSON's `model_config`. Explicit arguments win, then the JSON, then the model defaults (window fusion with a 0.05 s phrase gap).

**Returns:** `List[Dict]`. The merged detections, with the fields below.

| Field | Type | Description |
| :--- | :--- | :--- |
| `species` | `str` | eBird species code, or `bird` |
| `species_id` | `int` | Model class ID |
| `time_start` | `float` | Start time in seconds |
| `time_end` | `float` | End time in seconds |
| `avg_confidence` | `float` | Average confidence of the merged raw boxes |
| `max_confidence` | `float` | Highest confidence among the merged raw boxes |
| `detections_merged` | `int` | Number of raw boxes that were merged |
| `freq_low_hz` | `float` | Lower frequency edge in Hz |
| `freq_high_hz` | `float` | Upper frequency edge in Hz |
| `filename` | `str` | Source filename (multi-file inputs only) |

Birds-With-Labels adds `bird_id`, `species_confidence` and `alternatives`. See [Merged detection objects](../data/outputs.md#merged-detection-objects).

---

## The reconstruct_songs Function

The song merge on its own. `merge_detections` calls it for `merge_method='reconstruct'`. The default merge of both models is window fusion, so use `merge_detections` to merge saved raw detections as detection does.

```python
from inference.detect_birds import reconstruct_songs

merged = reconstruct_songs(detections, song_gap_threshold)
merged = reconstruct_songs(detections, song_gap_threshold, freq_iou_threshold=0.6)
```

| Parameter | Type / Default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `detections` | `List[Dict]` / none | **Yes** | Raw detections, each with `time_start`, `time_end`, `species_id`, `species`, `confidence`, `freq_low_hz`, `freq_high_hz`. Optional `filename` key for multi-file inputs. |
| `song_gap_threshold` | `float` / none | **Yes** | Max gap in seconds between two detections of the same species to merge them into one song box. |
| `freq_iou_threshold` | `float` / `None` | No | If set, also require this frequency IoU. `None` keeps the time-only merger. With Just-Bird, `merge_detections` passes `0.6` unless overridden. |

**Returns:** `List[Dict]`. The song boxes, sorted by `(filename, time_start)`, with the same fields as `merge_detections`.

---

## The find_audio_files Function

Find all supported audio files in a path.

```python
from inference.detect_birds import find_audio_files

paths = find_audio_files(audio_path)
```

| Parameter | Type / Default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `audio_path` | `str` / none | **Yes** | Path to a single audio file or a directory. Directories are searched recursively. |
| `verbose` | `bool` / `True` | No | If `True`, print how many audio files were found in a directory. |

**Returns:** `List[str]`. A sorted list of file paths, relative if `audio_path` is relative. Supported extensions are `.wav`, `.flac`, `.ogg` and `.mp3`, in any letter case.

---

## Examples

The examples use [Just-Bird](../models-and-metrics/just-bird.md), the recommended model. It detects bird vocalizations with a single `bird` class. For species-level output, use the Birds-With-Labels model with its species mapping, as in the directory batch example.

### Basic Detection

=== "Single file"
    ```python
    from inference.detect_birds import BirdCallDetector

    detector = BirdCallDetector(
        model_path="models/Just-Bird_fp32.pt",
        species_mapping="Just-Bird",
    )  # conf_threshold defaults to 0.15 for Just-Bird

    detections = detector.detect(
        "recording.wav",
        output_path="results",
        output_formats=["simplified-csv"],
    )

    print(f"Found {len(detections)} detections")
    ```

=== "Directory batch"
    ```python
    from inference.detect_birds import BirdCallDetector

    detector = BirdCallDetector(
        model_path="models/Birds-With-Labels_fp32.pt",
        species_mapping="Birds-With-Labels",
    )  # conf_threshold defaults to 0.21 for Birds-With-Labels

    detections = detector.detect(
        "/path/to/audio/folder",
        output_path="results",
        output_formats=["all"],
    )
    ```

### Evaluation Workflow (Raw Detections)

```python
from inference.detect_birds import BirdCallDetector

detector = BirdCallDetector(
    model_path="models/Just-Bird_fp32.pt",
    species_mapping="Just-Bird",
    conf_threshold=0.001,
)

raw_detections = detector.detect(
    "data/test_audio/",
    output_path="results",
    no_merge=True,
)
```

This writes `results/raw_detections.json`. Feed it into [`FBetaScoreAnalyzer`](f-beta-score-analysis.md) next.

### With Progress Callback

```python
from inference.detect_birds import BirdCallDetector

detector = BirdCallDetector(
    model_path="models/Just-Bird_fp32.pt",
    species_mapping="Just-Bird",
)

def on_progress(current, total, message):
    print(f"[{current}/{total}] {message}")

detections = detector.detect_single_file(
    "long_recording.flac",
    progress_callback=on_progress,
)
```

### Merging Saved Raw Detections

```python
import json
from inference.detect_birds import file_ends_from_raw, merge_detections, resolve_merge_settings

with open("results/raw_detections.json") as f:
    data = json.load(f)

raw = data["detections"]
settings = resolve_merge_settings(data["model_config"])
kept = [d for d in raw if d["confidence"] >= 0.15]

merged = merge_detections(kept, **settings, file_end=file_ends_from_raw(raw))
print(f"{len(kept)} raw detections -> {len(merged)} merged detections")
```

### Parallel Inference

```python
from inference.detect_birds import BirdCallDetector

detector = BirdCallDetector(
    model_path="models/Just-Bird_fp32.pt",
    species_mapping="Just-Bird",
    num_workers=4,
)

detections = detector.detect(
    "long_recording.wav",
    output_path="results",
    output_formats=["simplified-csv"],
)
```

!!! warning "Lossy Audio Formats"
    MP3 and OGG compression can hide faint calls and high frequencies, so detection may get worse. In a study of avian recordings, MP3 compression reduced the detections of human listeners and automated recognisers [[22]](#ref-macphail). Use WAV or FLAC whenever possible. If you must use MP3, use a bitrate of at least 256 kbps.

---

## References

<a id="ref-macphail"></a>
**[22]** MacPhail, A. G., Yip, D. A., Knight, E. C., Hedley, R., Knaggs, M., Shonfield, J., Upham-Mills, E., & Bayne, E. M. (2024). "Audio data compression affects acoustic indices and reduces detections of birds by human listening and automated recognisers." *Bioacoustics*, 33(1):74–90.
