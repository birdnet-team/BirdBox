# Ground-Truth Labels

Evaluation expects ground truth as a **comma-separated CSV** with a fixed header row. The layout matches the `annotations.csv` files from the utilized Zenodo Datasets listed in [Available Models and Respective Datasets](../models-and-metrics/overview.md#available-models-and-respective-datasets).

[`f_beta_score_analysis.py`](../cli/f-beta-score-analysis.md) and [`confusion_matrix_analysis.py`](../cli/confusion-matrix-analysis.md) read this format with `pandas.read_csv` and **require the exact column names** below.

---

## CSV schema

### Header (required, exact spelling)

```text
Filename,Start Time (s),End Time (s),Low Freq (Hz),High Freq (Hz),Species eBird Code
```

### Column Reference

| Column | Type | Required? | Description |
| :--- | :--- | :--- | :--- |
| `Filename` | string | **Yes** | Base recording name as stored on disk (e.g. `SNE_001_17.wav`). Used to match detections. See [Filename matching](#filename-matching). |
| `Start Time (s)` | float | **Yes** | Interval start in **seconds** from the beginning of that file. |
| `End Time (s)` | float | **Yes** | Interval end in seconds (must be ≥ start). |
| `Low Freq (Hz)` | integer | **Yes** | Lower frequency bound of the annotation box in **Hz**. |
| `High Freq (Hz)` | integer | **Yes** | Upper frequency bound in **Hz**. |
| `Species eBird Code` | string | **Yes** | Cornell/Clements **eBird species code** (e.g. `herthr`, `amerob`, `cangoo`). |

There is **no** confidence column in ground truth. Inference CSV may add a seventh `Confidence` column. See [Detection Output Formats](outputs.md#simplified-csv) for that format.

### Example Rows

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

!!! info "Unknown or placeholder species codes"
    Some rows use placeholder codes such as `????` when the annotator could not assign a species. They are loaded as literal strings. Matching still runs, but metrics for that code may be meaningless unless detections use the same token.

---

## Filename matching

Matching pairs detections to labels **per recording**, using only the filename **stem** (path and extension stripped):

| Label file | Detection file | Matched key |
| :--- | :--- | :--- |
| `SNE_001_17.wav` | `SNE_001_17.flac` | `SNE_001_17` |
| `SSW_001_0.wav` | `SSW_001_0.wav` | `SSW_001_0` |

!!! info "Filename matching rules"
    - The **basename** in `Filename` must agree between labels and detections after normalization.
    - Different extensions between label CSV and detection export are fine.
    - Different **paths** on disk do not matter—only the `Filename` column value.

---

## Creating your own labels

1. Copy the header line exactly (including spaces and parentheses).
2. Use seconds and Hz consistent with your audio length and spectrogram view.
3. Put the same `Filename` values you expect in detection outputs (typically the audio basename).
4. Use eBird codes that exist in your model's species mapping.

Raven or other tools can be adapted if you export or convert to this six-column layout.

For CLI usage and invocation examples for each tool that reads this file, see [f-beta-score-analysis](../cli/f-beta-score-analysis.md) and [confusion-matrix-analysis](../cli/confusion-matrix-analysis.md).
