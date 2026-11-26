# Merge Hyperparameter Search

Optional grid search of the merge gap and, if you want, `--nms-iou` on a **frozen** checkpoint. Most users never need this. The defaults in [`run_pipeline.sh`](workflows.md#complete-workflow-script) are the usual evaluation path.

!!! note "Advanced / optional"
    Skip this page unless you are already evaluating against labeled audio and want to retune merge parameters. For detection and F-beta with stock settings, see [Typical Workflows](workflows.md).

Each setting is scored with the same F-beta protocol as [`f_beta_score_analysis.py`](f-beta-score-analysis.md). Raw detections are filtered by confidence and merged. Then they are matched to `annotations.csv` with the Hungarian algorithm on 2D IoU. Confidence is swept at every grid point. The reported number is the best **Overall_Micro** F-beta.

The gap that is searched depends on the merge method. Both models merge with window fusion, so by default the search tries `--phrase-gaps`. With `--merge reconstruct` it tries `--song-gaps` instead.

Learning-rate and augmentation search lives in [BirdBox-Train](https://github.com/birdnet-team/BirdBox-Train) and is also optional. Do not mix the two. Freeze the merge settings while training, then run this search on the winning weights.

---

## Workflow

[`run_hpo.sh`](https://github.com/birdnet-team/BirdBox/blob/main/run_hpo.sh) at the repository root is the intended way to run this search. Edit the variables at the top and leave exactly one option enabled. It works like `run_pipeline.sh` but is not part of the default pipeline.

**Option A** (default, cheap) reuses one `raw_detections.json` and only sweeps the phrase gap. **Option B** (expensive) re-runs `detect_birds.py --no-merge` once per `--nms-iou` value, then sweeps the phrase gap on each dump. Detection dominates wall time. Use B only when you need to search NMS.

=== "Linux / macOS"

    ```bash
    --8<-- "run_hpo.sh"
    ```

=== "Windows (PowerShell)"

    ```powershell
    # Option A (cheap): phrase gap only, reuse an existing --no-merge dump (no new detect)
    python src/evaluation/hyperopt_merge.py `
        --raw-detections results/Just-Bird `
        --labels datasets/Northeastern-US_testset/annotations.csv `
        --output-path results/Just-Bird/hyperopt_merge `
        --phrase-gaps 0.0 0.025 0.05 0.1 0.2 `
        --num-workers 9 `
        --no-plot `
        --single-cls
    ```

=== "Windows (CMD)"

    ```cmd
    rem Option A (cheap): phrase gap only, reuse an existing --no-merge dump (no new detect)
    python src/evaluation/hyperopt_merge.py ^
        --raw-detections results/Just-Bird ^
        --labels datasets/Northeastern-US_testset/annotations.csv ^
        --output-path results/Just-Bird/hyperopt_merge ^
        --phrase-gaps 0.0 0.025 0.05 0.1 0.2 ^
        --num-workers 9 ^
        --no-plot ^
        --single-cls
    ```

---

## CLI Synopsis

Direct call to `hyperopt_merge.py` without the shell driver:

=== "Linux / macOS"
    ```bash
    python src/evaluation/hyperopt_merge.py \
        --raw-detections results/Just-Bird \
        --labels datasets/Northeastern-US_testset/annotations.csv \
        --phrase-gaps 0.0 0.025 0.05 0.1 0.2 \
        --single-cls
    ```
=== "Windows (PowerShell)"
    ```powershell
    python src/evaluation/hyperopt_merge.py `
        --raw-detections results/Just-Bird `
        --labels datasets/Northeastern-US_testset/annotations.csv `
        --phrase-gaps 0.0 0.025 0.05 0.1 0.2 `
        --single-cls
    ```
=== "Windows (CMD)"
    ```cmd
    python src/evaluation/hyperopt_merge.py ^
        --raw-detections results/Just-Bird ^
        --labels datasets/Northeastern-US_testset/annotations.csv ^
        --phrase-gaps 0.0 0.025 0.05 0.1 0.2 ^
        --single-cls
    ```

!!! warning "Checkpoint Must Be Frozen"
    This search does not train. It only changes how existing boxes are suppressed (`--nms-iou`) and merged (the gap). Pass the same `.pt` you would use in [`detect_birds.py`](detect-birds.md).

---

## What Is Searched

| Parameter | When it is applied | Needs a new detect pass? |
| :--- | :--- | :--- |
| `--phrase-gaps` | In F-beta, after filtering by confidence. Window fusion only (the default). | No. One `raw_detections.json` is enough. |
| `--song-gaps` | In F-beta, after filtering by confidence. Only with `--merge reconstruct`. | No. One `raw_detections.json` is enough. |
| `--nms-ious` | Inside YOLO during [`detect_birds.py --no-merge`](detect-birds.md) | **Yes.** One raw dump per NMS value. |
| Confidence | Swept automatically at every grid point | No |

`--iou-threshold` (matching detections to labels) is **not** part of this grid. Keep it at the evaluation default (`0.25`) so scores stay comparable to `run_pipeline.sh`.

!!! warning "Detection is the expensive part"
    The gap is applied in F-beta on an existing dump, so many gap values are cheap. `--nms-iou` is applied inside YOLO, so each NMS value is a full inference pass over the test set. Option A in `run_hpo.sh` never re-detects. Option B runs detection once per `--nms-ious` value.

---

## Parameters

| Parameter | Type / Default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `--raw-detections` | `PATH` / none | Yes, unless `--nms-ious` | Existing raw detections **file** or results directory from [`detect_birds.py --no-merge`](detect-birds.md). Used for a gap-only search. |
| `--labels` | `PATH` / none | **Yes** | Ground truth `annotations.csv`. Filenames are matched without extensions. |
| `--output-path` | `PATH` / `results/hyperopt_merge` | No | Root directory for per-setting F-beta runs and `summary.csv`. |
| `--merge` | `CHOICE` / `auto` | No | Merge method to score: `auto`, `fusion` or `reconstruct`. `auto` uses the method in the detections JSON, which is window fusion for both models. |
| `--phrase-gaps` | `FLOAT [...]` / `0.0 0.025 0.05 0.1 0.2` | No | Phrase gaps in seconds to try with window fusion. |
| `--song-gaps` | `FLOAT [...]` / `0.0 0.1 0.2 0.5 1.0 2.0` | No | Song gaps in seconds to try with `--merge reconstruct`. Ignored with fusion. |
| `--nms-ious` | `FLOAT [...]` / unset | No | If set, re-run detect `--no-merge` **once per NMS IoU** (the expensive path), then sweep the gap on each dump. Requires `--model`, `--audio`, and `--species-mapping`. Omit this to reuse one dump (Option A). |
| `--model` | `PATH` / none | With `--nms-ious` | Frozen YOLO checkpoint. |
| `--audio` | `PATH` / none | With `--nms-ious` | Audio file or directory (typically the test set `soundscape_data/` folder). |
| `--species-mapping` | `CHOICE` / none | With `--nms-ious` | `Just-Bird` or `Birds-With-Labels`. Must match the model. See [species mapping](../data/audio-and-model.md#species-mapping). |
| `--detect-conf` | `FLOAT` / `0.001` | No | Confidence used when re-running detect for `--nms-ious`. Keep it low so F-beta can sweep later. |
| `--detect-workers` | `INT` / `8` | No | CPU workers for those detect passes. |
| `--iou-threshold` | `FLOAT` / `0.25` | No | IoU for matching detections to labels. Use the same value as [`f_beta_score_analysis.py`](f-beta-score-analysis.md). |
| `--beta` | `FLOAT` / `1.0` | No | F-beta weighting. Same meaning as in F-beta analysis. |
| `--conf-range` | `MIN MAX STEP` / `0.00 1.0 0.01` | No | Confidence grid at each (`nms_iou`, gap) pair. |
| `--num-workers` | `INT` / `8` | No | Worker processes for each F-beta confidence sweep. |
| `--no-plot` | flag / off | No | Skip F-beta plots (faster; `run_hpo.sh` sets this). |
| `--single-cls` | flag / off | No | Collapse all species into one class. Use it for Just-Bird. |
| `--single-cls-name` | `STR` / `bird` | No | Class name when `--single-cls` is set. |

---

## Parameter Deep-Dives

### Phrase Gaps

`--phrase-gaps` lists the phrase gaps to try. Window fusion first joins the boxes of overlapping windows. Then it joins boxes in the same frequency band that are closer in time than the phrase gap. A gap of 0 keeps one box per call. Larger gaps give longer phrase boxes. See [phrase gap](detect-birds.md#phrase-gap) for details.

The gap is applied **inside** F-beta, after filtering by confidence. That is the same filter-then-merge policy as [`f_beta_score_analysis.py --phrase-gap`](f-beta-score-analysis.md). Because merging happens after detection, one `--no-merge` dump is enough to try every gap.

### Song Gaps

`--song-gaps` is used only with `--merge reconstruct`. The song merge joins boxes of the same species when the gap between them is at most this value. See [song gap](detect-birds.md#song-gap-threshold).

### NMS IoU Values

`--nms-ious` lists the NMS IoU values to try. NMS removes duplicate boxes at detection time, inside YOLO on each 3-second clip. Changing it changes which boxes are written to `raw_detections.json`, so the script calls `detect_birds.py --no-merge` **once per value**. That is the main compute cost of Option B. Four NMS values mean four full test set inference runs.

!!! info "NMS and the merge are different stages"
    NMS removes overlapping boxes *within* a clip. The merge then joins the surviving boxes *across* clips and time. Searching both is a product grid (NMS values × gaps), and each cell still sweeps confidence.

### Training Search and This Search

Training hyperparameter search is a separate, optional tool in [BirdBox-Train](https://github.com/birdnet-team/BirdBox-Train). It retrains YOLO (learning rate, mixup and more) and scores every trial with one fixed set of merge settings. So trials differ only by training recipe.

If you later copy a winning `best.pt` into `models/`, this page is where you can retune the merge settings on labeled audio. Neither search is required for ordinary detection or evaluation.

---

## Output Files

All files are written under `--output-path`. Each grid cell gets its own folder, named after the NMS value and the gap:

```text
results/Just-Bird/hyperopt_merge/
  summary.csv
  nms_0.70/phrase_gap_0.050/
    f1.0_score_analysis.csv
    optimal_thresholds.csv
    score.json
  nms_0.70/phrase_gap_0.100/
    ...
```

With `--merge reconstruct` the cell folders are named `song_gap_<value>`. When `--nms-ious` is set, each NMS value also stores its detection dump at `nms_<value>/detect/raw_detections.json`.

| File | Description |
| :--- | :--- |
| `summary.csv` | One row per (`nms_iou`, gap) with the merge method, the gap, the best micro F-beta, its confidence, precision, and recall. |
| `score.json` | Same numbers for that cell. |
| `f{beta}_score_analysis.csv` | Full F-beta table for that cell (same schema as the standalone F-beta script). |
| `optimal_thresholds.csv` | Best confidence per species at that merge setting. |

Open `summary.csv` and take the row with the highest `micro_f1`. Then set `--nms-iou` and `MERGE_FLAGS=(--phrase-gap ...)` in [`run_pipeline.sh`](workflows.md#complete-workflow-script) to those values for the final test evaluation.

---

## Examples

### Phrase Gap Only (Option A, No New Detect)

Run [`detect_birds.py --no-merge`](detect-birds.md) once first (Step 1 in `run_pipeline.sh`). Then reuse that dump:

=== "Command"
    ```bash
    python src/evaluation/hyperopt_merge.py \
        --raw-detections results/Just-Bird \
        --labels datasets/Northeastern-US_testset/annotations.csv \
        --output-path results/Just-Bird/hyperopt_merge \
        --phrase-gaps 0.0 0.025 0.05 0.1 0.2 \
        --num-workers 9 \
        --no-plot \
        --single-cls
    ```
=== "Expected Output (excerpt)"
    ```text
    === nms_iou=0.70 fusion phrase_gap=0.0 ===
    ...
    micro F1.0=0.xxxx  conf=0.xx  P=0.xxxx  R=0.xxxx
    === nms_iou=0.70 fusion phrase_gap=0.025 ===
    ...
    Best: nms_iou=0.7 fusion phrase_gap=0.05 micro F1.0=0.xxxx conf=0.xx
    Wrote results/Just-Bird/hyperopt_merge/summary.csv
    ```

This is the default option in `run_hpo.sh`. Wall time is F-beta only (one dump, many gaps).

### NMS IoU and Phrase Gap Together (Option B, Re-Detect per NMS)

=== "Command"
    ```bash
    python src/evaluation/hyperopt_merge.py \
        --model models/Birds-With-Labels_fp32.pt \
        --audio datasets/Northeastern-US_testset/soundscape_data \
        --species-mapping Birds-With-Labels \
        --labels datasets/Northeastern-US_testset/annotations.csv \
        --output-path results/Birds-With-Labels/hyperopt_merge \
        --nms-ious 0.5 0.6 0.7 0.8 \
        --phrase-gaps 0.05 0.1 \
        --detect-workers 18 \
        --num-workers 9 \
        --no-plot
    ```

Uncomment Option B in `run_hpo.sh` for the same grid, and comment out Option A so only one search runs. Expect one full `--no-merge` detection pass per `--nms-ious` value.

---

## After the Search

The usual BirdBox path stays [`run_pipeline.sh`](workflows.md#complete-workflow-script). Use this page only if you want different merge values than the pipeline defaults.

```
Usual (most users)     run_pipeline.sh     detect + F-beta + confusion matrix
Optional (this page)   run_hpo.sh          pick phrase_gap (and nms_iou) on a frozen model
                       then run_pipeline.sh with those values
Optional (Train repo)  run_hpo.sh          search training hyperparameters
```

If you did run the search:

1. Read `summary.csv` and pick the best `nms_iou` and gap.
2. Set those values in `run_pipeline.sh`: `--nms-iou` for detection, and `MERGE_FLAGS=(--phrase-gap ...)` for steps 2 and 3.
3. Run the usual evaluation steps on the test set.

---

## See Also

- [Typical Workflows](workflows.md). The usual detect and evaluate path.
- [Detecting Birds](detect-birds.md): [phrase gap](detect-birds.md#phrase-gap), [song gap](detect-birds.md#song-gap-threshold) and [NMS IoU](detect-birds.md#nms-iou-threshold)
- [F-Beta Score Analysis](f-beta-score-analysis.md)
