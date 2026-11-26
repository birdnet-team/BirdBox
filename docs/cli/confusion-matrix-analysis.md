# Confusion Matrix Analysis

Compute a confusion matrix comparing filtered detection results against ground truth labels. Matches detections to labels using IoU with the Hungarian (optimal) algorithm, then builds a per-species confusion matrix showing correct classifications, misclassifications, false positives, and false negatives.

---

## Usage Synopsis

=== "Linux / macOS"
    ```bash
    python src/evaluation/confusion_matrix_analysis.py \
        --detections results \
        --labels path/to/annotations.csv
    ```
=== "Windows (PowerShell)"
    ```powershell
    python src/evaluation/confusion_matrix_analysis.py `
        --detections results `
        --labels path/to/annotations.csv
    ```
=== "Windows (CMD)"
    ```cmd
    python src/evaluation/confusion_matrix_analysis.py ^
        --detections results ^
        --labels path/to/annotations.csv
    ```

## Parameters

| Parameter | Type / Default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `--detections` | `PATH` / `results` | No | Detections **CSV file** or `results/` directory from `filter_and_merge_detections.py`. Default resolves to `simplified.csv` (follows `results/.active_run`). |
| `--labels` | `PATH` / none | **Yes** | Path to the ground truth labels CSV file. Must use the same column format as the detections file. |
| `--iou-threshold` | `FLOAT` / `0.25` | No | IoU threshold for matching a detection to a ground truth label. A detection is a True Positive only if its IoU with a matched label meets or exceeds this value. |
| `--use-1d-iou` | flag / off | No | Use 1D IoU (time overlap only) instead of the default 2D IoU (time and frequency). Faster but less accurate. |
| `--no-background` | flag / off | No | Omit the background class from the confusion matrix. By default, unmatched detections and unmatched labels are recorded in a `background` row and column. |
| `--single-cls` | flag / off | No | Collapse all species into a single class for binary bird-detection evaluation. Use it for Just-Bird. |
| `--single-cls-name` | `STR` / `bird` | No | Class name to use when `--single-cls` is active. |
| `--output-path` | `PATH` / `results/confusion_matrix_analysis` | No | Output directory for all result files. Created automatically if it does not exist. |

## Parameter Deep-Dives

### IoU Matching Threshold

`--iou-threshold` determines how strictly a detection must overlap a ground truth label to count as a True Positive. The best value depends on how precisely your annotations define the temporal and spectral extent of each call.

| Value | Strictness | Typical use |
| :--- | :--- | :--- |
| `0.1` | Very lenient | Useful when annotations are coarse |
| `0.25` (default) | Moderate | Good starting point for most datasets |
| `0.5` | Strict | When annotations are precise and detailed |

!!! info "Matching Algorithm"
    The script always uses the **Hungarian algorithm** [[15]](#ref-kuhn) (optimal matching). This means the globally best assignment of detections to labels is found regardless of their order in the file. Results are fully reproducible.

### 1D IoU vs 2D IoU

**2D IoU** (default) measures the overlap of the two time-frequency boxes. The intersection area is divided by the union area:

$$
\text{IoU}_{2D} = \frac{A_{\cap}}{A_1 + A_2 - A_{\cap}}, \qquad A_{\cap} = t_{\text{overlap}} \cdot f_{\text{overlap}}
$$

Here $A_1$ and $A_2$ are the areas (duration × bandwidth) of the detection and the label.

**1D IoU** (`--use-1d-iou`) measures overlap in time only:

$$
\text{IoU}_{1D} = \frac{t_{\text{overlap}}}{t_{\text{union}}}
$$

!!! info "Which to use?"
    2D IoU is more accurate because it accounts for both *when* and *at what frequency* a call occurs. Use `--use-1d-iou` only if your frequency annotations are unreliable or absent.

### Background Class

By default, the confusion matrix includes a `background` class. Rows are the predicted class, columns are the true (annotated) class:

- **Background column (last column)**: detections that did not match any ground truth label. These are **False Positives**.
- **Background row (last row)**: ground truth labels that were not matched by any detection. These are **False Negatives**.

Use `--no-background` if you only want to see the species-level confusion (correct or wrong species label), without the FP and FN counts.

### Single-Class Mode

`--single-cls` remaps all species labels in both detections and ground truth to the same class name (default: `bird`). It evaluates whether the model finds bird calls at all, independent of species classification. Use it for Just-Bird, which has only the `bird` class.

!!! info "The single-class matrix"
    With the background class, the matrix is 2×2. It holds the True Positives, the False Positives and the False Negatives. The background-background cell is always 0, because there are no true negatives in event detection:

    | Pred \ True | `bird` | `background` |
    | :--- | :--- | :--- |
    | `bird` | TP | FP |
    | `background` | FN | 0 |

    With `--no-background`, only the TP cell remains.

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
      <td>Heatmap visualization, column-normalized: each true-class column sums to 1, showing how that class's instances were distributed across predictions.</td>
    </tr>
    <tr>
      <td style="white-space:nowrap"><code>confusion_matrix_raw.png</code></td>
      <td>Heatmap visualization showing raw counts.</td>
    </tr>
    <tr>
      <td style="white-space:nowrap"><code>metadata.txt</code></td>
      <td>Analysis parameters, file paths, species list, and detection/label counts.</td>
    </tr>
    <tr>
      <td style="white-space:nowrap"><code>args.yaml</code></td>
      <td>Every CLI argument used for this run, including defaults. Written so confusion-matrix runs can be compared later.</td>
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

`bg` stands for the `background` row and column, the last one. It exists only without `--no-background`.

| Cell | Meaning |
| :--- | :--- |
| Diagonal cell `[i, i]` | True Positives for species `i` |
| Off-diagonal cell `[i, j]` (i ≠ j) | Species `i` detected but actually species `j` (misclassification) |
| `background` column cell `[i, bg]` | Detections of species `i` that did not match any ground truth label (False Positives) |
| `background` row cell `[bg, j]` | Ground truth labels for species `j` that had no matching detection (False Negatives) |

## Examples

### Species Confusion Matrix (Birds-With-Labels)

=== "Command"
    ```bash
    python src/evaluation/confusion_matrix_analysis.py \
        --detections results \
        --labels data/annotations.csv
    ```
=== "Expected Output (excerpt)"
    ```text
    ================================================================================
    CONFUSION MATRIX ANALYSIS
    ================================================================================
    Detections: results/simplified.csv
    Labels: data/annotations.csv
    IoU threshold: 0.25
    IoU type: 2D (time-frequency)
    Matching method: Optimal (Hungarian)
    Include background: True
    Single-class mode: False (class='bird')
    Output directory: results/confusion_matrix_analysis
    ================================================================================
    ...
    Loading detections from: results/simplified.csv
    Loaded 72 detections
    ...
    Building confusion matrix...
    ...
    Confusion matrix saved to: results/confusion_matrix_analysis/confusion_matrix.csv
    Detailed confusion matrix saved to: results/confusion_matrix_analysis/confusion_matrix_detailed.csv
    ...
    ANALYSIS COMPLETE
    ```

### Strict 1D IoU Matching

=== "Command"
    ```bash
    python src/evaluation/confusion_matrix_analysis.py \
        --detections results \
        --labels data/annotations.csv \
        --iou-threshold 0.5 \
        --use-1d-iou \
        --output-path results/confusion_strict
    ```

The header then reads `IoU threshold: 0.5` and `IoU type: 1D (time only)`.

### Without Background Class

=== "Command"
    ```bash
    python src/evaluation/confusion_matrix_analysis.py \
        --detections results \
        --labels data/annotations.csv \
        --no-background \
        --output-path results/confusion_no_bg
    ```

The header reads `Include background: False`, and the matrix has no `background` row or column.

### Binary Bird Detection (Just-Bird)

=== "Command"
    ```bash
    python src/evaluation/confusion_matrix_analysis.py \
        --detections results \
        --labels data/annotations.csv \
        --single-cls \
        --output-path results/confusion_binary
    ```
=== "Expected Output (excerpt)"
    ```text
    Found 1 unique species: bird
    ...
    Matched 22 detection-label pairs
    Unmatched detections: 50
    Unmatched labels: 64
    ...
               |       bird background
    ----------------------------------------------------------------------------------------------------
          bird |         22         50
    background |         64          0
    ```

## Typical Workflow

This script sits at **Step 4** of the standard evaluation pipeline:

```
Step 1  detect_birds.py --conf 0.001 --no-merge    ->  raw_detections.json
Step 2  f_beta_score_analysis.py                   ->  f_beta_score_analysis/optimal_thresholds.csv
Step 3  filter_and_merge_detections.py --conf 0.15 ->  simplified.csv
Step 4  confusion_matrix_analysis.py               ->  confusion_matrix_analysis/
```

!!! info "Input File Format"
    Both `--detections` and `--labels` must use the six-column schema described in [CSV schema](../data/annotations.md#csv-schema) on the Ground-Truth Labels page. Filenames are matched by stem only. For details, see [Filename matching](../data/annotations.md#filename-matching).

---

## References

<a id="ref-kuhn"></a>
**[15]** Kuhn, H. W. (1955). "The Hungarian method for the assignment problem." *Naval Research Logistics Quarterly*, 2(1–2):83–97.
