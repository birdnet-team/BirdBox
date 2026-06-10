# Ground-Truth Labels

Evaluation expects ground truth as a **comma-separated CSV** with a fixed header row. The layout matches the bundled test-set files `datasets/Western-US_testset/annotations.csv` and `datasets/Northeastern-US_testset/annotations.csv`.

`f_beta_score_analysis.py` and `confusion_matrix_analysis.py` read this format with `pandas.read_csv` and **require the exact column names** below.

## CSV schema

### Header (required, exact spelling)

```text
Filename,Start Time (s),End Time (s),Low Freq (Hz),High Freq (Hz),Species eBird Code
```

### Column reference

| Column | Type | Required? | Description |
| :--- | :--- | :--- | :--- |
| `Filename` | string | **Yes** | Base recording name as stored on disk (e.g. `SNE_001_17.wav`). Used to match detections; see [Filename matching](#filename-matching). |
| `Start Time (s)` | float | **Yes** | Interval start in **seconds** from the beginning of that file. |
| `End Time (s)` | float | **Yes** | Interval end in seconds (must be ≥ start). |
| `Low Freq (Hz)` | number | **Yes** | Lower frequency bound of the annotation box in **Hz**. |
| `High Freq (Hz)` | number | **Yes** | Upper frequency bound in **Hz**. |
| `Species eBird Code` | string | **Yes** | Cornell/Clements **eBird species code** (e.g. `herthr`, `amerob`, `cangoo`). |

There is **no** confidence column in ground truth. Inference CSV may add a seventh `Confidence` column; see **Detection outputs** for that format.

### Example rows

From the Western-US test set:

```csv
Filename,Start Time (s),End Time (s),Low Freq (Hz),High Freq (Hz),Species eBird Code
SNE_001_17.wav,3.8,5.9,3219,4804,herthr
SNE_001_17.wav,0.0,7.6,2666,5405,herthr
SNE_001_17.wav,0.0,7.7,1297,4564,amerob
```

From the Northeastern-US test set:

```csv
Filename,Start Time (s),End Time (s),Low Freq (Hz),High Freq (Hz),Species eBird Code
SSW_001_0.wav,47.4,51.3,0,2647,cangoo
SSW_001_2.wav,0.0,0.6,0,4509,cangoo
```

!!! info "Multiple rows per file and species"
    One recording may have many rows (overlapping time/frequency boxes and multiple species). That is normal for strong multi-species choruses and dense manual labeling.

!!! info "Unknown or placeholder species codes"
    Some rows use placeholder codes such as `????` when the annotator could not assign a species. They are loaded as literal strings; matching still runs, but metrics for that code may be meaningless unless detections use the same token.

## How evaluation loads labels

Both scripts map each CSV row to an internal dictionary:

| CSV column | Internal key |
| :--- | :--- |
| `Filename` | `filename` |
| `Start Time (s)` | `start_time` |
| `End Time (s)` | `end_time` |
| `Low Freq (Hz)` | `freq_low` |
| `High Freq (Hz)` | `freq_high` |
| `Species eBird Code` | `species` |

**F-beta analysis** (`load_labels`): loads every row.

**Confusion matrix analysis** (`load_labels_csv`): skips rows where `Species eBird Code` is missing/NaN.

For confusion matrix, detection CSV must use the same six columns (optional `Confidence` is ignored).

## Filename matching

Matching pairs detections to labels **per recording**, using only the filename **stem** (path and extension stripped):

| Label file | Detection file | Matched key |
| :--- | :--- | :--- |
| `SNE_001_17.wav` | `SNE_001_17.flac` | `SNE_001_17` |
| `SSW_001_0.wav` | `SSW_001_0.wav` | `SSW_001_0` |

- The **basename** in `Filename` must agree between labels and detections after normalization.
- Different extensions between label CSV and detection export are fine.
- Different **paths** on disk do not matter—only the `Filename` column value.

If no files overlap, F-beta analysis warns and may show example stems; confusion matrix analysis warns and can proceed with unfiltered lists.

## Spatial matching (IoU)

Once rows share a normalized filename, detections and labels are matched in **time × frequency** using 2D IoU. A pair counts as a match when IoU ≥ `--iou-threshold` (default `0.25` in the repository pipeline scripts).

Detection fields: `time_start`, `time_end`, `freq_low_hz`, `freq_high_hz`, `species`.

Label fields: `start_time`, `end_time`, `freq_low`, `freq_high`, `species`.

Species must **agree** for a match (same eBird code).

## Which detection input each tool expects

| Tool | Detection input | Labels input |
| :--- | :--- | :--- |
| `f_beta_score_analysis.py` | Raw JSON from `detect_birds.py --no-merge` | This CSV |
| `confusion_matrix_analysis.py` | Merged simplified CSV (e.g. after `filter_and_merge_detections.py`) | This CSV |

F-beta sweeps confidence on raw clip detections, then merges with the same `song_gap` logic as inference before matching. Confusion matrix expects **already merged** segments in CSV form.

## Creating your own labels

1. Copy the header line exactly (including spaces and parentheses).
2. Use seconds and Hz consistent with your audio length and spectrogram view.
3. Put the same `Filename` values you expect in detection outputs (typically the audio basename).
4. Use eBird codes that exist in your model's species mapping.

Raven or other tools can be adapted if you export or convert to this six-column layout.

For CLI usage and invocation examples for each tool that reads this file, see [f-beta-score-analysis](../cli/f-beta-score-analysis.md) and [confusion-matrix-analysis](../cli/confusion-matrix-analysis.md).
