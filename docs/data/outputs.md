# Detection Output Formats

BirdBox can write inference results in four interchange formats (plus `all`). The same choices apply to `src/inference/detect_birds.py` (`--output-format`, single value) and `src/evaluation/filter_and_merge_detections.py` (`--output-format`, space-separated list or `all`).

Both commands take `--output-path` as an **output directory**. Each format writes a fixed, descriptive filename inside that directory.

## `--output-format` summary

| Value | File(s) written | Primary use |
| :--- | :--- | :--- |
| `json-with-algorithm-metadata` | `with_algorithm_metadata.json` (merged) or `raw_detections.json` (`--no-merge`) | Full structured results; evaluation pipeline uses `raw_detections.json` only |
| `simplified-csv` | `simplified.csv` | Spreadsheets, confusion matrix; same six columns as ground-truth CSV |
| `xeno-canto-annota-json` | `xeno-canto-annota.json` | [Annota-JSON](https://xeno-canto.org/article/321) for Xeno-Canto |
| `raven-selection-table` | `raven_selection_table.txt` or `raven/*.txt` | Raven Pro selection tables (tab-separated) |
| `all` | All of the above | One-shot export |

!!! info "Default formats differ by command"
    `detect_birds.py` defaults to `json-with-algorithm-metadata`.  
    `filter_and_merge_detections.py` defaults to **both** `json-with-algorithm-metadata` and `simplified-csv` when `--output-format` is omitted.

For `--output-format` usage and invocation examples see the [detect-birds](../cli/detect-birds.md) and [filter-and-merge-detections](../cli/filter-and-merge-detections.md) CLI references.

---

## `json-with-algorithm-metadata`

**Filenames:**

| Source | File |
| :--- | :--- |
| `detect_birds.py` (merged, default) | `with_algorithm_metadata.json` |
| `detect_birds.py --no-merge` | `raw_detections.json` only (other `--output-format` values ignored) |
| `filter_and_merge_detections.py` | `with_algorithm_metadata.json` |

**Written by:** `save_detections` / `save_filtered_json`

Canonical machine-readable output. For F-beta threshold sweeps, use **raw** clip-level detections from `detect_birds.py --no-merge` with a low `--conf`.

### Top-level structure (single audio file)

```json
{
  "audio_file": "/absolute/or/relative/path/to/recording.wav",
  "model_config": {
    "model": "models/Hawaii.pt",
    "confidence_threshold": 0.2,
    "nms_iou_threshold": 0.7,
    "song_gap_threshold": 0.1,
    "species_mapping": "Hawaii"
  },
  "detection_count": 12,
  "detections": [ /* see below */ ]
}
```

### Top-level structure (directory / multi-file batch)

When any detection includes a `filename` field:

```json
{
  "audio_files": ["/path/to/a.wav", "/path/to/b.wav"],
  "file_count": 2,
  "model_config": { /* same keys as above */ },
  "detection_count": 340,
  "detections": [ /* each may include filename, file_path */ ]
}
```

### After `filter_and_merge_detections.py`

The JSON keeps `model_config` and adds:

```json
{
  "filtering_config": {
    "confidence_threshold": 0.25,
    "song_gap_threshold": 0.1
  },
  "original_detection_count": 1842,
  "detection_count": 87,
  "detections": [ /* merged song segments */ ]
}
```

Optional keys preserved from input: `audio_files`, `file_count`, `audio_file`.

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

### Detection objects — merged (default inference or after filter-and-merge)

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
| `filename` | string | Optional; per-source file in batch mode |

!!! info "Raw vs merged in evaluation"
    `f_beta_score_analysis.py` reads **raw** JSON and re-applies filter-then-merge per confidence threshold. `confusion_matrix_analysis.py` expects **merged** intervals in simplified CSV, not this JSON.

---

## `simplified-csv`

**Filename:** `simplified.csv`  
**Written by:** `save_detections_csv` / `save_filtered_csv`

Flat table with the same six geometry/species columns as ground-truth `annotations.csv`, plus an optional confidence column on inference export.

### Header

**From `detect_birds.py`:**

```text
Filename,Start Time (s),End Time (s),Low Freq (Hz),High Freq (Hz),Species eBird Code,Confidence
```

**From `filter_and_merge_detections.py`:**

```text
Filename,Start Time (s),End Time (s),Low Freq (Hz),High Freq (Hz),Species eBird Code
```

(No `Confidence` column after filtering.)

### Column reference (inference CSV)

| Column | Description |
| :--- | :--- |
| `Filename` | Basename of the audio file (or per-detection `filename` in batch mode) |
| `Start Time (s)` | `time_start`, formatted to **one** decimal place |
| `End Time (s)` | `time_end`, one decimal place |
| `Low Freq (Hz)` | `freq_low_hz` (integer Hz in practice) |
| `High Freq (Hz)` | `freq_high_hz` |
| `Species eBird Code` | `species` |
| `Confidence` | `confidence` for raw detections; `avg_confidence` when `detections_merged` is present |

### Example

```csv
Filename,Start Time (s),End Time (s),Low Freq (Hz),High Freq (Hz),Species eBird Code,Confidence
SNE_001_17.wav,12.5,14.2,2151,5820,amerob,0.470
SNE_001_17.wav,25.3,27.8,1890,4560,herthr,0.612
```

Merged simplified CSV (no `Confidence` required) is the usual input to `confusion_matrix_analysis.py --detections`.

---

## `xeno-canto-annota-json`

**Filename:** `xeno-canto-annota.json`  
**Written by:** `save_detections_xc_json` / `save_filtered_xc_json` via `src/inference/utils/xeno_canto_export.py`

Exports a lean **Annota-JSON** payload for Xeno-Canto. BirdBox uses **Cornell/Clements eBird codes** internally; Xeno-Canto expects **AviList** scientific names in this format—see [Taxonomy conversion](#taxonomy-conversion-ebird-avilist).

### Set-level fields

| Field | Typical value | Description |
| :--- | :--- | :--- |
| `set_source` | `"BirdBox detection results"` | Provenance string |
| `set_name` | `"BirdBox detection results"` or filter message | Human-readable set title |
| `set_uri` | `""` | Optional URI (not filled by BirdBox) |
| `annotation_software_name_and_version` | `"BirdBox"` | Software identifier |
| `set_creator` | `"BirdBox"` | Creator name |
| `set_creator_id`, `set_owner`, `set_license`, `funding`, `project_uri` | often `""` | Reserved; left empty |
| `project_name` | `"BirdBox"` | Project label |
| `set_remarks` | Generated text | Notes that file came from BirdBox |
| `scope` | array | `taxon_coverage` (comma-separated scientific names), `completeness`: `"part"` |

Export-only XC fields (`original_set_metadata`, `annotation_xc_id`, etc.) are **stripped** before write.

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

## `raven-selection-table`

**Filename:** `raven_selection_table.txt` (tab-separated, Raven Selection Table layout)  
**Written by:** `save_detections_raven_txt` / `save_filtered_raven_txt`

Species appear in the `Annotation` column as **eBird codes** (not common names).

### Columns

| Column | Value |
| :--- | :--- |
| `Selection` | 1-based index, sorted by `time_start` per file |
| `View` | `Spectrogram 1` |
| `Channel` | `1` |
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

Example: `--output-path results/run` with multiple inputs → `results/run/raven/SNE_001_17.wav.txt`, etc.

Filter-and-merge always writes a **single** `raven_selection_table.txt` for the merged detection list.

### Example (TSV)

```text
Selection	View	Channel	Begin Time (S)	End Time (S)	Low Freq (Hz)	High Freq (Hz)	Annotation
1	Spectrogram 1	1	12.5	14.2	2151	5820	amerob
2	Spectrogram 1	1	25.3	27.8	1890	4560	herthr
```

---

## `all`

Sets every format flag above. One `--output-path` directory produces:

| Artifact |
| :--- |
| `with_algorithm_metadata.json` |
| `simplified.csv` |
| `xeno-canto-annota.json` |
| `raven_selection_table.txt` **or** `raven/` (see Raven section) |

For an invocation example see [detect-birds → Directory batch](../cli/detect-birds.md#directory-batch).

---

## Choosing a format

| Goal | Recommended format |
| :--- | :--- |
| F-beta threshold sweep | `json-with-algorithm-metadata` from `detect_birds --no-merge` |
| Cheap re-threshold without re-inference | Same JSON → `filter_and_merge_detections` |
| Confusion matrix / spreadsheet QA | `simplified-csv` (merged) |
| Raven manual review | `raven-selection-table` |
| Xeno-Canto submission prep | `xeno-canto-annota-json` |
| Archival / provenance | `json-with-algorithm-metadata` (includes `model_config`) |

---

## Evaluation artifacts (not `--output-format`)

Separate evaluation scripts write their own outputs (for example `f{beta}_score_analysis.csv`, `optimal_thresholds.csv`, `confusion_matrix.csv`, and plot images). Those formats are not controlled by `--output-format` on detection commands.
