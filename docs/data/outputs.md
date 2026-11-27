# Detection Output Formats

BirdBox can write inference results in four interchange formats. The same choices apply to [detect_birds](../cli/detect-birds.md) and [filter_and_merge_detections](../cli/filter-and-merge-detections.md). Both accept `--output-format` as a space-separated list or `all`.

Both commands take `--output-path` as an **output directory**. Each format writes a fixed, descriptive filename inside that directory.

`detect_birds.py` also writes `args.yaml` in that directory. It stores every CLI argument for the run so later inference folders can be compared. The evaluation scripts write their own `args.yaml` inside their result folders (for example `results/f_beta_score_analysis/` and `results/confusion_matrix_analysis/`).

---

## Available Formats

| Value | File(s) written | Primary use |
| :--- | :--- | :--- |
| `json-with-algorithm-metadata` | `with_algorithm_metadata.json` (merged) or `raw_detections.json` (`--no-merge`) | F-beta threshold sweeps via [`detect_birds --no-merge`](../cli/detect-birds.md#evaluation-mode). Re-threshold without re-inference via [`filter_and_merge_detections`](../cli/filter-and-merge-detections.md). Archival output with full `model_config`. The evaluation pipeline reads `raw_detections.json` only. |
| `simplified-csv` | `simplified.csv` | Confusion matrix analysis. The six columns of the [ground-truth CSV](annotations.md), plus `Confidence` from `detect_birds.py`. |
| `xeno-canto-annota-json` | `xeno-canto-annota.json` | [Annota-JSON](https://xeno-canto.org/article/321){ target="_blank" rel="noopener noreferrer" } export for Xeno-Canto submission prep |
| `raven-selection-table` | `raven_selection_table.txt` (one recording) or `raven/<Filename>.txt` (several) | Raven manual review |
| `all` | All of the above | One-shot export of every format |

## 1. json-with-algorithm-metadata

**Filename:** `with_algorithm_metadata.json` or `raw_detections.json` depending on `--no-merge` parameter.

Holds the detections with all their fields and the settings of the run in `model_config`.

### Top-level structure

A Birds-With-Labels run with the default settings writes:

```json
{
  "audio_file": "/path/to/recording.wav",
  "model_config": {
    "model": "models/Birds-With-Labels_fp32.pt",
    "confidence_threshold": 0.21,
    "nms_iou_threshold": 0.7,
    "song_gap_threshold": 0.1,
    "species_mapping": "Birds-With-Labels",
    "merge_method": "fusion",
    "floor_window_seconds": 60.0,
    "phrase_gap": 0.05,
    "bird_class_id": 633,
    "label_conf": 0.25
  },
  "detection_count": 130,
  "detections": [ /* see field tables below */ ]
}
```

| `model_config` key | Meaning |
| :--- | :--- |
| `model` | Path of the model file |
| `confidence_threshold` | `--conf` of the run |
| `nms_iou_threshold` | `--nms-iou` of the run |
| `song_gap_threshold` | `--song-gap`. Used by `--merge reconstruct` only. |
| `species_mapping` | `--species-mapping` of the run |
| `merge_method` | `fusion` or `reconstruct` (see [Merging the Windows](../how-it-works/overview.md#merging-the-windows)) |
| `floor_window_seconds` | [Noise-floor window](../how-it-works/spectrogram.md#stage-5-log-floor-normalization) of the spectrograms, `60.0` s |
| `phrase_gap` | `--phrase-gap`. Written with `fusion` only. |
| `freq_iou_threshold` | Just-Bird only. `--freq-iou`, used by `--merge reconstruct`. |
| `bird_class_id`, `label_conf` | Birds-With-Labels only. Index of the `bird` class and `--label-conf` (see [Species Names](../how-it-works/overview.md#species-names)). |

In batch mode (directory input), `audio_file` is replaced by `"audio_files": [...]` and `"file_count": N`. After [`filter_and_merge_detections`](../cli/filter-and-merge-detections.md), `model_config` stays as in the raw file and a `filtering_config` block holds the settings of the filter step:

```json
{
  "filtering_config": {
    "confidence_threshold": 0.21,
    "song_gap_threshold": 0.1,
    "merge_method": "fusion",
    "freq_iou_threshold": null,
    "phrase_gap": 0.05,
    "label_conf": 0.25
  },
  "detection_count": 260,
  "original_detection_count": 14324,
  "detections": [ /* merged detections */ ]
}
```

### Raw detection objects

Produced by inference when `--no-merge` is set. Each object is one box of one 3 s window:

| Field | Type | Description |
| :--- | :--- | :--- |
| `species` | string | eBird code from species mapping |
| `species_id` | int | YOLO class index |
| `confidence` | float | Box confidence (0–1) |
| `time_start` | float | Start time in seconds (file timeline) |
| `time_end` | float | End time in seconds |
| `freq_low_hz` | number | Low frequency bound (Hz) |
| `freq_high_hz` | number | High frequency bound (Hz) |
| `clip_start` | float | Start of the 3 s window (s) |
| `clip_end` | float | End of that window (s) |
| `filename` | string | Batch mode only. Basename of the source audio. |
| `file_path` | string | Batch mode only. Full path to the source audio. |

### Merged detection objects

Without `--no-merge`, the boxes of the overlapping windows are merged (see [Merging the Windows](../how-it-works/overview.md#merging-the-windows)). Each object is one merged box:

| Field | Type | Description |
| :--- | :--- | :--- |
| `species` | string | eBird code |
| `species_id` | int | Class index |
| `time_start`, `time_end` | float | Time bounds of the merged box (s) |
| `freq_low_hz`, `freq_high_hz` | number | Frequency bounds of the merged box (Hz) |
| `avg_confidence` | float | Mean confidence of the joined boxes, weighted by their `detections_merged`. A box that was not joined keeps its own confidence. |
| `max_confidence` | float | Highest confidence among the joined boxes |
| `detections_merged` | int | Number of window boxes behind the merged box |
| `filename` | string | Batch mode only. Basename of the source audio. |
| `file_path` | string | Batch mode with `detect_birds.py` only. Full path to the source audio. |

Window fusion averages overlapping boxes weighted by confidence, so its bounds are fractional. The phrase join and the song merge take the outer bounds of the joined boxes.

Birds-With-Labels boxes carry three more fields, raw and merged (see [Species Names](../how-it-works/overview.md#species-names)). Their `species` is `bird` when no species reaches the label threshold, and their confidence is the bird score. The `species` value can also be `unknown`, the class for calls of unknown species. Both names carry over to every other format. Annotation files and other tools may write `unknown` as `????`. The evaluation tools read `????` as `unknown`.

| Field | Type | Description |
| :--- | :--- | :--- |
| `species_confidence` | float | Score of the top species |
| `alternatives` | list | The top three species, best first, each with `species`, `species_id` and `confidence` |
| `bird_id` | int | Class index of `bird` (633) |

!!! info "Raw vs merged in evaluation"
    [`f_beta_score_analysis.py`](../cli/f-beta-score-analysis.md) reads **raw** JSON and re-applies filter-then-merge per confidence threshold. [`confusion_matrix_analysis.py`](../cli/confusion-matrix-analysis.md) expects **merged** intervals in simplified CSV, not this JSON.

---

## 2. simplified-csv { #simplified-csv }

**Filename:** `simplified.csv`

Flat table with the six columns of the [ground-truth `annotations.csv`](annotations.md#csv-schema). `detect_birds.py` adds a seventh `Confidence` column. `filter_and_merge_detections.py` writes the six columns only.

```text
Filename,Start Time (s),End Time (s),Low Freq (Hz),High Freq (Hz),Species eBird Code[,Confidence]
```

### Column reference

| Column | Description |
| :--- | :--- |
| `Filename` | Basename of the audio file (or per-detection `filename` in batch mode) |
| `Start Time (s)` | `time_start`, rounded to **one** decimal place |
| `End Time (s)` | `time_end`, one decimal place |
| `Low Freq (Hz)` | `freq_low_hz`, rounded to whole Hz |
| `High Freq (Hz)` | `freq_high_hz`, whole Hz |
| `Species eBird Code` | `species` |
| `Confidence` | `detect_birds.py` only. `avg_confidence` for merged boxes, `confidence` for raw boxes with `--no-merge`. Three decimals. |

### Example

```csv
Filename,Start Time (s),End Time (s),Low Freq (Hz),High Freq (Hz),Species eBird Code,Confidence
SNE_001_17.wav,12.5,14.2,2151,5820,amerob,0.470
SNE_001_17.wav,25.3,27.8,1890,4560,herthr,0.612
```

The merged CSV is the usual input to [`confusion_matrix_analysis.py`](../cli/confusion-matrix-analysis.md) via `--detections`. It reads the six columns and ignores `Confidence`.

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
| `frequency_low` | `freq_low_hz` | Hz, three decimals |
| `frequency_high` | `freq_high_hz` | Hz, three decimals |
| `start_time` | `time_start` | Seconds, three decimals |
| `end_time` | `time_end` | Seconds, three decimals |
| `scientific_name` | mapped | **AviList** scientific name, or `Aves` for `bird` and `unknown` (see below) |
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

BirdBox labels use **eBird/Clements** codes. Annota-JSON expects **AviList** scientific names. BirdBox bridges this with:

1. `taxonomies/Cornell-to-AviList-mapping.json` maps each lowercase eBird code to `scientific_name` and `english_name`.
2. **Order of lookups** in `build_xeno_canto_json`:
   - `Aves` for boxes without a species (`bird` and `unknown`)
   - AviList mapping entry
   - detection `scientific_name` if already set
   - `ebird_to_name` from the active species mapping (scientific part before `_`)
   - raw eBird code string

19 Birds-With-Labels codes have no AviList entry and take the scientific name from the species mapping. Split/merge caveats and how the mapping file is generated are documented in `taxonomies/README.md` in the repository.

Filter-and-merge export reads `model_config.species_mapping` from the input JSON when present.

---

## 4. raven-selection-table

**Filename:** `raven_selection_table.txt` (tab-separated, Raven Selection Table layout)

Species appear in the `Annotation` column as **eBird codes** (not common names). The table has no confidence column.

### Columns

| Column | Value |
| :--- | :--- |
| `Selection` | 1-based index, sorted by `time_start` per file |
| `View` | `Spectrogram 1` |
| `Channel` | `1` |
| `Begin Time (S)` | `time_start` (three decimals) |
| `End Time (S)` | `time_end` (three decimals) |
| `Low Freq (Hz)` | `freq_low_hz`, whole Hz |
| `High Freq (Hz)` | `freq_high_hz`, whole Hz |
| `Annotation` | `species` (eBird code) |

### Single file vs batch

| Input mode | Output path behavior |
| :--- | :--- |
| Single `--audio` file | `{output-path}/raven_selection_table.txt` |
| Directory / multi-file batch | `{output-path}/raven/{Filename}.txt` per source file |

Example: `--output-path results/run` with multiple inputs results in `results/run/raven/SNE_001_17.wav.txt`, etc. Two recordings with the same name in different subfolders would write to the same table, so give each recording a unique name.

Filter-and-merge follows the same rule: a raw JSON from one recording gives one `raven_selection_table.txt`, a multi-file raw JSON gives `raven/{Filename}.txt` per recording. A Raven table has no filename column, so one table per recording keeps selections from other files off the spectrogram.

### Example

```text
Selection	View	Channel	Begin Time (S)	End Time (S)	Low Freq (Hz)	High Freq (Hz)	Annotation
1	Spectrogram 1	1	12.512	14.236	2151	5820	amerob
2	Spectrogram 1	1	25.304	27.817	1890	4560	herthr
```
