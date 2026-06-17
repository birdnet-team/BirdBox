
Detect bird calls in audio files using a trained YOLO model. Processes WAV, FLAC, OGG, and MP3 files through the PCEN spectrogram pipeline and returns timestamped song segments with species labels and confidence scores.

---

## Import

```python
from inference.detect_birds import BirdCallDetector, reconstruct_songs, find_audio_files
```

---

## BirdCallDetector

The main entry point for all inference workflows. Instantiate once per model, then call `detect()` for any number of files.

### Constructor

```python
BirdCallDetector(
    model_path,
    species_mapping,
    conf_threshold=0.001,
    nms_iou_threshold=0.7,
    song_gap_threshold=0.1,
    num_workers=1,
)
```

| Parameter | Type / Default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `model_path` | `str` / — | **Yes** | Path to the trained YOLO model file (`.pt`, `.onnx`, `.engine`, etc.). |
| `species_mapping` | `str` / — | **Yes** | Dataset key for class-ID-to-species mapping. Must match the mapping the model was trained with. See [allowed values](../data/audio-and-model.md#species-mapping-species-mapping). |
| `conf_threshold` | `float` / `0.001` | No | Confidence threshold (0.0–1.0). Detections below this value are discarded. Use `0.2` for direct field use. Use `0.001` to retain all raw detections for evaluation. |
| `nms_iou_threshold` | `float` / `0.7` | No | IoU threshold for Non-Maximum Suppression applied per-clip and across overlapping time windows. |
| `song_gap_threshold` | `float` / `0.1` | No | Maximum temporal gap in seconds between two detections of the same species that are still merged into one continuous song segment. |
| `num_workers` | `int` / `1` | No | Number of parallel inference workers. Each worker loads its own model copy. Increase on multi-core GPU systems for batch processing. |

!!! warning "Memory Usage"
    Each additional worker loads a full copy of the model. With `num_workers=4` and a 100 MB model, approximately 400 MB of model memory is allocated. Monitor memory when increasing workers significantly.

---

### Methods

| Method | Returns | Description |
| :--- | :--- | :--- |
| `detect(audio_path, output_path, output_formats, no_merge)` | `List[Dict]` | Detect in a file or directory. Main entry point. |
| `detect_single_file(audio_path, progress_callback, no_merge)` | `List[Dict]` | Detect in a single audio file. |
| `detect_multiple_files(audio_paths, output_path, output_formats, no_merge)` | `List[Dict]` | Detect across a list of audio files. |
| `save_results(detections, output_path, audio_path, output_formats, no_merge)` | `None` | Write detections to one or more output formats. |
| `merge_overlapping_detections(detections, merge_mode)` | `List[Dict]` | Merge raw clip detections. |
| `load_audio(audio_path)` | `Tuple[np.ndarray, int]` | Load an audio file, returns `(signal, sample_rate)`. |

---

### detect

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
| `audio_path` | `str` / — | **Yes** | Path to a single audio file (WAV, FLAC, OGG, MP3) or a directory. Directories are searched recursively. |
| `output_path` | `str` / `None` | No | Output directory for result files. If `None`, results are only returned, not written. |
| `output_formats` | `List[str]` / `['json-with-algorithm-metadata']` | No | One or more output format keys. Accepts `json-with-algorithm-metadata`, `simplified-csv`, `xeno-canto-annota-json`, `raven-selection-table`, or `all`. Ignored when `no_merge=True`. |
| `no_merge` | `bool` / `False` | No | Evaluation mode. Returns raw per-clip detections and writes only `raw_detections.json`. Use with `conf_threshold=0.001` for F-beta sweep workflows. |

**Returns:** `List[Dict]` — merged song segments (or raw clip detections when `no_merge=True`).

---

### detect_single_file

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
| `audio_path` | `str` / — | **Yes** | Path to the audio file. |
| `progress_callback` | `Callable[[int, int, str], None]` / `None` | No | Called as `callback(current, total, message)` after each clip. Useful for progress bars. |
| `no_merge` | `bool` / `False` | No | If `True`, return raw clip-level detections without merging. |

**Returns:** `List[Dict]` — detections for this file.

---

### save_results

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
| `detections` | `List[Dict]` / — | **Yes** | Detections returned by `detect()` or `detect_single_file()`. |
| `output_path` | `str` / — | **Yes** | Directory to write output files into. |
| `audio_path` | `str` / `None` | No | Source audio path, used for metadata in JSON output. |
| `output_formats` | `List[str]` / `['json-with-algorithm-metadata']` | No | One or more format keys (same values as `detect()`). |
| `no_merge` | `bool` / `False` | No | If `True`, writes only `raw_detections.json` and ignores `output_formats`. |

---

### merge_overlapping_detections

```python
detector.merge_overlapping_detections(detections, merge_mode='reconstruct')
```

Merge raw clip detections into song segments or deduplicate with NMS.

| Parameter | Type / Default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `detections` | `List[Dict]` / — | **Yes** | Raw per-clip detections. |
| `merge_mode` | `str` / `'reconstruct'` | No | `'reconstruct'` merges temporally adjacent detections into song segments (default). `'nms'` removes duplicates by IoU, keeping the highest-confidence box. |

**Returns:** `List[Dict]` — merged or deduplicated detections.

---

## reconstruct_songs

Standalone version of the song-merging logic. Use this outside a `BirdCallDetector` instance, for example in evaluation scripts that operate on previously saved raw detections.

```python
from inference.detect_birds import reconstruct_songs

merged = reconstruct_songs(detections, song_gap_threshold)
```

| Parameter | Type / Default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `detections` | `List[Dict]` / — | **Yes** | Raw detections, each with `time_start`, `time_end`, `species_id`, `species`, `confidence`, `freq_low_hz`, `freq_high_hz`. Optional `filename` key for multi-file inputs. |
| `song_gap_threshold` | `float` / — | **Yes** | Max gap in seconds between two detections of the same species to merge them into one song segment. |

**Returns:** `List[Dict]` — merged song segments, sorted by `(filename, time_start)`.

Each merged segment contains:

| Field | Type | Description |
| :--- | :--- | :--- |
| `species` | `str` | eBird species code |
| `species_id` | `int` | Model class ID |
| `time_start` | `float` | Start time in seconds |
| `time_end` | `float` | End time in seconds |
| `avg_confidence` | `float` | Running average confidence across merged clips |
| `max_confidence` | `float` | Highest confidence among merged clips |
| `detections_merged` | `int` | Number of raw clips that were merged |
| `freq_low_hz` | `int` | Lowest frequency of merged bounding boxes |
| `freq_high_hz` | `int` | Highest frequency of merged bounding boxes |
| `filename` | `str` | Source filename (multi-file inputs only) |

---

## find_audio_files

Discover all supported audio files in a path.

```python
from inference.detect_birds import find_audio_files

paths = find_audio_files(audio_path)
```

| Parameter | Type / Default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `audio_path` | `str` / — | **Yes** | Path to a single audio file or a directory. Directories are searched recursively. |

**Returns:** `List[str]` — sorted list of absolute file paths. Supported extensions: `.wav`, `.flac`, `.ogg`, `.mp3`.

---

## Examples

### Basic detection

=== "Single file"
    ```python
    from inference.detect_birds import BirdCallDetector

    detector = BirdCallDetector(
        model_path="models/Hawaii.pt",
        species_mapping="Hawaii",
        conf_threshold=0.2,
    )

    detections = detector.detect(
        "recording.wav",
        output_path="results",
        output_formats=["simplified-csv"],
    )

    print(f"Found {len(detections)} song segments")
    ```

=== "Directory batch"
    ```python
    from inference.detect_birds import BirdCallDetector

    detector = BirdCallDetector(
        model_path="models/Western-US.pt",
        species_mapping="Western-US",
    )

    detections = detector.detect(
        "/path/to/audio/folder",
        output_path="results",
        output_formats=["all"],
    )
    ```

### Evaluation workflow (raw detections)

```python
from inference.detect_birds import BirdCallDetector

detector = BirdCallDetector(
    model_path="models/Hawaii.pt",
    species_mapping="Hawaii",
    conf_threshold=0.001,
)

raw_detections = detector.detect(
    "data/test_audio/",
    output_path="results",
    no_merge=True,
)
```

This writes `results/raw_detections.json`. Feed it into [`FBetaScoreAnalyzer`](f-beta-score-analysis.md) next.

### With progress callback

```python
from inference.detect_birds import BirdCallDetector

detector = BirdCallDetector(
    model_path="models/All-In-One.pt",
    species_mapping="All-In-One",
)

def on_progress(current, total, message):
    print(f"[{current}/{total}] {message}")

detections = detector.detect_single_file(
    "long_recording.flac",
    progress_callback=on_progress,
)
```

### Using reconstruct_songs independently

```python
import json
from inference.detect_birds import reconstruct_songs

with open("results/raw_detections.json") as f:
    data = json.load(f)

raw = data["detections"]
gap = data["model_config"]["song_gap_threshold"]

merged = reconstruct_songs(raw, song_gap_threshold=gap)
print(f"{len(raw)} raw detections → {len(merged)} song segments")
```

### Parallel inference

```python
from inference.detect_birds import BirdCallDetector

detector = BirdCallDetector(
    model_path="models/All-In-One.pt",
    species_mapping="All-In-One",
    num_workers=4,
)

detections = detector.detect(
    "long_recording.wav",
    output_path="results",
    output_formats=["simplified-csv"],
)
```

!!! warning "Lossy Audio Formats"
    The model was trained on lossless WAV files. When processing MP3 or OGG input, detection performance may degrade, especially for faint calls and high-frequency species. Use WAV or FLAC whenever possible. If you must use MP3, ensure a bitrate of ≥ 256 kbps.
