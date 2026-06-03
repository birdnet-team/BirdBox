
Compute a confusion matrix comparing filtered detection results against ground truth labels. Matches detections to labels using IoU with the Hungarian (optimal) algorithm, then builds a per-species confusion matrix showing correct classifications, misclassifications, false positives, and false negatives.

---

## Usage Synopsis

=== "Linux / macOS"
    ```bash
    python src/evaluation/confusion_matrix_analysis.py \
        --detections results/merged_detections.csv \
        --labels path/to/labels.csv
    ```
=== "Windows (PowerShell)"
    ```powershell
    python src/evaluation/confusion_matrix_analysis.py `
        --detections results/merged_detections.csv `
        --labels path/to/labels.csv
    ```
=== "Windows (CMD)"
    ```cmd
    python src/evaluation/confusion_matrix_analysis.py ^
        --detections results/merged_detections.csv ^
        --labels path/to/labels.csv
    ```

## Parameters

| Parameter | Type / Default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `--detections` | `PATH` / `results/merged_detections.csv` | No | Path to the detections CSV file (output of `filter_and_merge_detections.py` or `detect_birds.py --output-format simplified-csv`). |
| `--labels` | `PATH` / — | **Yes** | Path to the ground truth labels CSV file. Must use the same column format as the detections file. |
| `--iou-threshold` | `FLOAT` / `0.25` | No | IoU threshold for matching a detection to a ground truth label. A detection is a True Positive only if its IoU with a matched label meets or exceeds this value. |
| `--use-1d-iou` | flag / off | No | Use 1D IoU (temporal overlap only) instead of the default 2D IoU (time × frequency). Faster but less accurate. |
| `--no-background` | flag / off | No | Omit the background class from the confusion matrix. By default, unmatched detections and unmatched labels are recorded in a `background` row/column. |
| `--single-cls` | flag / off | No | Collapse all species into a single class for binary bird-detection evaluation. |
| `--single-cls-name` | `STR` / `bird` | No | Class name to use when `--single-cls` is active. |
| `--output-path` | `PATH` / `results/confusion_matrix_analysis` | No | Output directory for all result files. Created automatically if it does not exist. |

## Parameter Deep-Dives

### `--iou-threshold` — IoU Matching Threshold

Determines how strictly a detection must overlap a ground truth label to count as a True Positive. The optimal value depends on how precisely your annotations define the temporal and spectral extent of each call.

| Value | Strictness | Typical use |
| :--- | :--- | :--- |
| `0.1` | Very lenient | Useful when annotations are coarse |
| `0.25` (default) | Moderate | Good starting point for most datasets |
| `0.5` | Strict | When annotations are precise and detailed |

!!! info "Matching Algorithm"
    The script always uses the **Hungarian algorithm** (optimal matching). This means the globally best assignment of detections to labels is found regardless of their order in the file. Results are fully reproducible.

### `--use-1d-iou` vs 2D IoU (default)

**2D IoU** (default) measures overlap in both time and frequency:

$$
\text{IoU}_{2D} = \frac{t_{\text{overlap}} \cdot f_{\text{overlap}}}{t_{\text{union}} \cdot f_{\text{union}}}
$$

**1D IoU** measures overlap in time only:

$$
\text{IoU}_{1D} = \frac{t_{\text{overlap}}}{t_{\text{union}}}
$$

!!! info "Which to use?"
    2D IoU is more accurate because it accounts for both *when* and *at what frequency* a call occurs. Use `--use-1d-iou` only if your frequency annotations are unreliable or absent.

### `--no-background` — Background Class

By default, the confusion matrix includes a `background` class:

- **Background column (last column)** — Ground truth labels that were not matched by any detection. These are **False Negatives**.
- **Background row (last row)** — Detections that did not match any ground truth label. These are **False Positives**.

Use `--no-background` if you only want to see the species-level confusion (correct vs wrong species label), ignoring FP/FN counts.

### `--single-cls` — Single-Class Mode

Remaps all species labels in both detections and ground truth to the same class name (default: `bird`). Useful for evaluating whether the model finds bird calls at all, independent of species classification accuracy.

!!! info "Mutually Exclusive Interpretation"
    `--single-cls` and multi-species evaluation are conceptually mutually exclusive. When `--single-cls` is active, the confusion matrix reduces to a 2×2 TP/FP/TN/FN table (with or without background).

## Output Files

All files are written to the `--output-path` directory.

<table>
  <thead>
    <tr><th>File</th><th>Description</th></tr>
  </thead>
  <tbody>
    <tr>
      <td style="white-space:nowrap"><code>confusion_matrix.csv</code></td>
      <td>Raw confusion matrix as a CSV.</td>
    </tr>
    <tr>
      <td style="white-space:nowrap"><code>confusion_matrix_detailed.csv</code></td>
      <td>Same matrix with labeled <code>pred_&lt;class&gt;</code> rows and <code>true_&lt;class&gt;</code> columns.</td>
    </tr>
    <tr>
      <td style="white-space:nowrap"><code>confusion_matrix_normalized.png</code></td>
      <td>Heatmap visualization showing row-normalized percentages.</td>
    </tr>
    <tr>
      <td style="white-space:nowrap"><code>confusion_matrix_raw.png</code></td>
      <td>Heatmap visualization showing raw counts.</td>
    </tr>
    <tr>
      <td style="white-space:nowrap"><code>metadata.txt</code></td>
      <td>Analysis parameters, file paths, species list, and detection/label counts.</td>
    </tr>
  </tbody>
</table>

### Reading the Confusion Matrix

The matrix uses **predicted class** (rows) × **true class** (columns) convention.

$$
\begin{array}{r|cccc}
\text{Pred} \setminus \text{True} & \text{amerob} & \text{herthr} & \text{yelwar} & \text{bg} \\
\hline
\text{amerob}     & \mathbf{45} & 2           & 1           & 3           \\
\text{herthr}     & 3           & \mathbf{38} & 0           & 2           \\
\text{yelwar}     & 1           & 0           & \mathbf{52} & 1           \\
\text{background} & 4           & 2           & 1           & \mathbf{0}  \\
\end{array}
$$

| Cell | Meaning |
| :--- | :--- |
| Diagonal cell `[i, i]` | True Positives for species `i` |
| Off-diagonal cell `[i, j]` (i ≠ j) | Species `i` detected but actually species `j` (misclassification) |
| `background` column cell `[i, bg]` | Ground truth labels for species `i` that had no matching detection (False Negatives) |
| `background` row cell `[bg, j]` | Detections that did not match any ground truth label for species `j` (False Positives) |

## Examples

### Basic confusion matrix

=== "Command"
    ```bash
    python src/evaluation/confusion_matrix_analysis.py \
        --detections results/filtered_detections.csv \
        --labels data/ground_truth.csv
    ```
=== "Expected Output"
    ```text
    ================================================================================
    CONFUSION MATRIX ANALYSIS
    ================================================================================
    Detections: results/filtered_detections.csv
    Labels: data/ground_truth.csv
    IoU threshold: 0.25
    IoU type: 2D (time-frequency)
    Matching method: Optimal (Hungarian)
    Include background: True

    Loading detections from: results/filtered_detections.csv
    Loaded 23 detections

    Loading ground truth labels from: data/ground_truth.csv
    Loaded 31 ground truth labels

    Found 3 unique species: amerob, herthr, yelwar
    Building confusion matrix...
    ```

### Strict 1D IoU matching

=== "Command"
    ```bash
    python src/evaluation/confusion_matrix_analysis.py \
        --detections results/filtered_detections.csv \
        --labels data/ground_truth.csv \
        --iou-threshold 0.5 \
        --use-1d-iou \
        --output-path results/confusion_strict
    ```
=== "Expected Output"
    ```text
    IoU type: 1D (time only)
    IoU threshold: 0.5
    ...
    Saved confusion matrix results to: results/confusion_strict/
    ```

### Without background class

=== "Command"
    ```bash
    python src/evaluation/confusion_matrix_analysis.py \
        --detections results/filtered_detections.csv \
        --labels data/ground_truth.csv \
        --no-background \
        --output-path results/confusion_no_bg
    ```
=== "Expected Output"
    ```text
    Include background: False
    ...
    Species-only matrix saved (no FP/FN rows/columns).
    ```

### Binary bird detection

=== "Command"
    ```bash
    python src/evaluation/confusion_matrix_analysis.py \
        --detections results/filtered_detections.csv \
        --labels data/ground_truth.csv \
        --single-cls \
        --single-cls-name bird \
        --output-path results/confusion_binary
    ```
=== "Expected Output"
    ```text
    Single-class mode: True (class='bird')
    Found 1 unique species: bird
    Building confusion matrix...
    ```

## Typical Workflow

This script sits at **Step 4** of the standard evaluation pipeline:

```
Step 1  detect_birds.py --conf 0.001 --no-merge    →  raw_detections.json
Step 2  f_beta_score_analysis.py                   →  optimal_thresholds.csv
Step 3  filter_and_merge_detections.py --conf 0.35 →  filtered_detections.csv
Step 4  confusion_matrix_analysis.py               →  confusion_matrix/
```

!!! info "Input File Format"
    Both `--detections` and `--labels` must be CSV files with these exact column headers:

    ```
    Filename, Start Time (s), End Time (s), Low Freq (Hz), High Freq (Hz), Species eBird Code
    ```

    Filenames are matched without their extension (`recording.wav` = `recording.flac`), so mismatched audio formats between detection and label files are handled automatically.
