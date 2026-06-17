# Detection Output Formats

BirdBox can write inference results in four interchange formats. The same choices apply to [detect_birds](../cli/detect-birds.md) and [filter_and_merge_detections](../cli/filter-and-merge-detections.md). Both accept `--output-format` as a space-separated list or `all`.

Both commands take `--output-path` as an **output directory**. Each format writes a fixed, descriptive filename inside that directory.

---

## Available Formats

| Value | File(s) written | Primary use |
| :--- | :--- | :--- |
| `json-with-algorithm-metadata` | `with_algorithm_metadata.json` (merged) or `raw_detections.json` (`--no-merge`) | F-beta threshold sweeps via [`detect_birds --no-merge`](../cli/detect-birds.md#--no-merge--evaluation-mode). Re-threshold without re-inference via [`filter_and_merge_detections`](../cli/filter-and-merge-detections.md). Archival output with full `model_config`. The evaluation pipeline reads `raw_detections.json` only. |
| `simplified-csv` | `simplified.csv` | Confusion matrix analysis. Same six columns as [ground-truth CSV](annotations.md). |
| `xeno-canto-annota-json` | `xeno-canto-annota.json` | [Annota-JSON](https://xeno-canto.org/article/321){ target="_blank" rel="noopener noreferrer" } export for Xeno-Canto submission prep |
| `raven-selection-table` | `raven_selection_table.txt` or `raven/*.txt` | Raven manual review |
| `all` | All of the above | One-shot export of every format |

## 1. json-with-algorithm-metadata

**Filename:** `with_algorithm_metadata.json` or `raw_detections.json` depending on `--no-merge` parameter.

Includes the detections as well as algorithm specific metadata such as nms iou threshold, average confidence, max confidence and so on.

### Top-level structure

```json
{
  "audio_file": "/path/to/recording.wav",
  "model_config": {
    "model": "models/Hawaii.pt",
    "confidence_threshold": 0.2,
    "nms_iou_threshold": 0.7,
    "song_gap_threshold": 0.1,
    "species_mapping": "Hawaii"
  },
  "detection_count": 12,
  "detections": [ /* see field tables below */ ]
}
```

In batch mode (directory input), `audio_file` is replaced by `"audio_files": [...]` and `"file_count": N`. After [`filter_and_merge_detections`](../cli/filter-and-merge-detections.md), `model_config` is kept and a `filtering_config` block is added:

```json
{
  "filtering_config": { "confidence_threshold": 0.25, "song_gap_threshold": 0.1 },
  "original_detection_count": 1842,
  "detection_count": 87,
  "detections": [ /* merged song segments */ ]
}
```

### Detection objects — raw (unmerged)

Produced by inference when `--no-merge` is set:

| Field | Type | Description |
| :--- | :--- | :--- |
| `species` | string | eBird code from species mapping |
| `species_id` | int | YOLO class index |
| `confidence` | float | Box confidence (0–1) |
| `time_start` | float | Start time in seconds (file timeline) |
| `time_end` | float | End time in seconds |
| `freq_low_hz` | number | Low frequency bound (Hz) |
| `freq_high_hz` | number | High frequency bound (Hz) |
| `clip_start` | float | Start of the 3 s spectrogram clip (s) |
| `clip_end` | float | End of that clip (s) |
| `filename` | string | Present for batch / multi-file runs |
| `file_path` | string | Full path to source audio (batch mode) |

### Detection objects — merged

Song reconstruction merges adjacent same-species detections when the gap ≤ `song_gap_threshold`:

| Field | Type | Description |
| :--- | :--- | :--- |
| `species` | string | eBird code |
| `species_id` | int | Class index |
| `time_start`, `time_end` | float | Merged interval |
| `freq_low_hz`, `freq_high_hz` | number | Min/max over merged boxes |
| `avg_confidence` | float | Mean confidence of merged clips |
| `max_confidence` | float | Max confidence in the segment |
| `detections_merged` | int | Number of raw boxes combined |
| `filename` | string | Optional. Present per source file in batch mode. |

!!! info "Raw vs merged in evaluation"
    [`f_beta_score_analysis.py`](../cli/f-beta-score-analysis.md) reads **raw** JSON and re-applies filter-then-merge per confidence threshold. [`confusion_matrix_analysis.py`](../cli/confusion-matrix-analysis.md) expects **merged** intervals in simplified CSV, not this JSON.

---

## 2. simplified-csv

**Filename:** `simplified.csv`

Flat table with the same six geometry/species columns as the [ground-truth `annotations.csv`](annotations.md#csv-schema), plus a `Confidence` column added by `detect_birds.py` (absent after filtering).

```text
Filename,Start Time (s),End Time (s),Low Freq (Hz),High Freq (Hz),Species eBird Code[,Confidence]
```

### Column reference

| Column | Description |
| :--- | :--- |
| `Filename` | Basename of the audio file (or per-detection `filename` in batch mode) |
| `Start Time (s)` | `time_start`, formatted to **one** decimal place |
| `End Time (s)` | `time_end`, one decimal place |
| `Low Freq (Hz)` | `freq_low_hz` (integer Hz in practice) |
| `High Freq (Hz)` | `freq_high_hz` |
| `Species eBird Code` | `species` |
| `Confidence` | `confidence` for raw detections. Uses `avg_confidence` when `detections_merged` is present. |

### Example

```csv
Filename,Start Time (s),End Time (s),Low Freq (Hz),High Freq (Hz),Species eBird Code,Confidence
SNE_001_17.wav,12.5,14.2,2151,5820,amerob,0.470
SNE_001_17.wav,25.3,27.8,1890,4560,herthr,0.612
```

Merged simplified CSV (no `Confidence` required) is the usual input to [`confusion_matrix_analysis.py`](../cli/confusion-matrix-analysis.md) via `--detections`.

---

## 3. xeno-canto-annota-json

**Filename:** `xeno-canto-annota.json`

Exports a lean **[Annota-JSON](https://xeno-canto.org/article/321)** payload for Xeno-Canto. BirdBox uses **Cornell/Clements eBird codes** internally. Xeno-Canto expects **AviList** scientific names in this format, see [Taxonomy conversion](#taxonomy-conversion-ebird-avilist).

The set-level envelope contains provenance fields and fixed BirdBox identifiers. Fields like `set_uri`, `set_creator_id`, `set_owner`, `set_license`, `funding`, and `project_uri` are left empty. Export-only XC fields (`original_set_metadata`, `annotation_xc_id`, etc.) are stripped before write.

### Per-annotation fields

Each detection becomes one object in `annotations`:

| Field | Source | Description |
| :--- | :--- | :--- |
| `annotation_source_id` | generated | `birdbox-000001`, … |
| `sound_file` | `filename` / `file_path` / `audio_path` | Recording basename |
| `xc_nr` | parsed or `""` | XC number if filename contains `XC123456` pattern |
| `annotator` | `"BirdBox"` | Fixed |
| `annotator_xc_id` | `""` | User XC id (not set) |
| `frequency_low` | `freq_low_hz` | Rounded float |
| `frequency_high` | `freq_high_hz` | Rounded float |
| `start_time` | `time_start` | Rounded float (seconds) |
| `end_time` | `time_end` | Rounded float |
| `scientific_name` | mapped | **AviList** scientific name (see below) |
| `sound_type` | `"call"` | Fixed default |
| `date_identified` | today's ISO date | Date of export |
| `annotation_remarks` | built string | Includes eBird code and confidence |

Confidence in remarks uses `avg_confidence` when present, else `confidence`.

### Example (truncated)

```json
{
  "set_source": "BirdBox detection results",
  "set_name": "BirdBox detection results",
  "annotation_software_name_and_version": "BirdBox",
  "scope": [{ "taxon_coverage": "Turdus migratorius, Catharus guttatus", "completeness": "part" }],
  "annotations": [
    {
      "annotation_source_id": "birdbox-000001",
      "sound_file": "recording.wav",
      "xc_nr": "",
      "scientific_name": "Turdus migratorius",
      "frequency_low": 2151.0,
      "frequency_high": 5820.0,
      "start_time": 12.5,
      "end_time": 14.2,
      "sound_type": "call",
      "annotation_remarks": "Detected by BirdBox; eBird code: amerob; confidence: 0.470"
    }
  ]
}
```

### Taxonomy conversion (eBird → AviList)

BirdNET-family training data uses **eBird/Clements 2021** codes. Annota-JSON expects **AviList** scientific names. BirdBox bridges this with:

1. `taxonomies/Cornell-to-AviList-mapping.json` — lowercase eBird code → `scientific_name` and `english_name`.
2. **Fallback order** in `build_xeno_canto_json`:
   - AviList mapping entry
   - detection `scientific_name` if already set
   - `ebird_to_name` from the active species mapping (scientific part before `_`)
   - raw eBird code string

Split/merge caveats and how the mapping file is generated are documented in `taxonomies/README.md` in the repository.

Filter-and-merge export reads `model_config.species_mapping` from the input JSON when present.

---

## 4. raven-selection-table

**Filename:** `raven_selection_table.txt` (tab-separated, Raven Selection Table layout)

Species appear in the `Annotation` column as **eBird codes** (not common names). `View` and `Channel` are fixed as `Spectrogram 1` / `1`.

### Columns

| Column | Value |
| :--- | :--- |
| `Selection` | 1-based index, sorted by `time_start` per file |
| `Begin Time (S)` | `time_start` (one decimal) |
| `End Time (S)` | `time_end` (one decimal) |
| `Low Freq (Hz)` | `freq_low_hz` |
| `High Freq (Hz)` | `freq_high_hz` |
| `Annotation` | `species` (eBird code) |

### Single file vs batch

| Input mode | Output path behavior |
| :--- | :--- |
| Single `--audio` file | `{output-path}/raven_selection_table.txt` |
| Directory / multi-file batch | `{output-path}/raven/{Filename}.txt` per source file |

Example: `--output-path results/run` with multiple inputs results in `results/run/raven/SNE_001_17.wav.txt`, etc.

Filter-and-merge always writes a **single** `raven_selection_table.txt` for the merged detection list.

### Example

```text
Selection	View	Channel	Begin Time (S)	End Time (S)	Low Freq (Hz)	High Freq (Hz)	Annotation
1	Spectrogram 1	1	12.5	14.2	2151	5820	amerob
2	Spectrogram 1	1	25.3	27.8	1890	4560	herthr
```
