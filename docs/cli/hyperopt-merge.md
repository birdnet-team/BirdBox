
Optional grid-search of inference merge hyperparameters (`--song-gap`, and optionally `--nms-iou`) on a **frozen** checkpoint. Most users never need this: the defaults in [`run_pipeline.sh`](workflows.md#complete-workflow-script) are the usual evaluation path.

!!! note "Advanced / optional"
    Skip this page unless you are already evaluating against labeled audio and want to retune merge parameters. For detection and F-beta with stock settings, see [Typical Workflows](workflows.md).

Each setting is scored with the same 1-minute event F-beta protocol as [`f_beta_score_analysis.py`](f-beta-score-analysis.md): filter raw detections by confidence, merge, then Hungarian 2D IoU against `annotations.csv`. Confidence is swept at every grid point; the reported number is the best **Overall_Micro** F-beta.

Learning-rate and augmentation search lives in [BirdBox-Train](https://github.com/birdnet-team/BirdBox-Train) and is also optional. Do not mix the two: freeze `--nms-iou` / `--song-gap` while training, then run this search on the winning weights.

---

## Workflow

[`run_hpo.sh`](https://github.com/birdnet-team/BirdBox/blob/main/run_hpo.sh) at the repository root is the intended way to run this search. Edit variables at the top and leave exactly one option enabled — the same interaction style as `run_pipeline.sh`, but this script is not part of the default pipeline.

**Option A** (default, cheap) reuses one `raw_detections.json` and only sweeps `--song-gap`. **Option B** (expensive) re-runs `detect_birds.py --no-merge` once per `--nms-iou`, then sweeps `--song-gap` on each dump. Detection dominates wall time; use B only when you need to search NMS.

=== "Linux / macOS"

    ```bash
    --8<-- "run_hpo.sh"
    ```

=== "Windows (PowerShell)"

    ```powershell
    # Option A (cheap): song_gap only, reuse an existing --no-merge dump (no new detect)
    python src/evaluation/hyperopt_merge.py `
        --raw-detections results/Northeastern-US `
        --labels datasets/Northeastern-US_testset/annotations.csv `
        --output-path results/Northeastern-US/hyperopt_merge `
        --song-gaps 0.0 0.1 0.2 0.5 1.0 2.0 `
        --num-workers 9 `
        --no-plot
    ```

=== "Windows (CMD)"

    ```cmd
    rem Option A (cheap): song_gap only, reuse an existing --no-merge dump (no new detect)
    python src/evaluation/hyperopt_merge.py ^
        --raw-detections results/Northeastern-US ^
        --labels datasets/Northeastern-US_testset/annotations.csv ^
        --output-path results/Northeastern-US/hyperopt_merge ^
        --song-gaps 0.0 0.1 0.2 0.5 1.0 2.0 ^
        --num-workers 9 ^
        --no-plot
    ```

---

## CLI synopsis

Direct call to `hyperopt_merge.py` without the shell driver:

=== "Linux / macOS"
    ```bash
    python src/evaluation/hyperopt_merge.py \
        --raw-detections results/Northeastern-US \
        --labels datasets/Northeastern-US_testset/annotations.csv \
        --song-gaps 0.0 0.1 0.2 0.5 1.0 2.0
    ```
=== "Windows (PowerShell)"
    ```powershell
    python src/evaluation/hyperopt_merge.py `
        --raw-detections results/Northeastern-US `
        --labels datasets/Northeastern-US_testset/annotations.csv `
        --song-gaps 0.0 0.1 0.2 0.5 1.0 2.0
    ```
=== "Windows (CMD)"
    ```cmd
    python src/evaluation/hyperopt_merge.py ^
        --raw-detections results/Northeastern-US ^
        --labels datasets/Northeastern-US_testset/annotations.csv ^
        --song-gaps 0.0 0.1 0.2 0.5 1.0 2.0
    ```

!!! warning "Checkpoint Must Be Frozen"
    This search does not train. It only changes how existing boxes are suppressed (`--nms-iou`) and stitched into songs (`--song-gap`). Pass the same `.pt` you would use in [`detect_birds.py`](detect-birds.md).

---

## What Is Searched

| Parameter | When it is applied | Needs a new detect pass? |
| :--- | :--- | :--- |
| `--song-gap` | In F-beta, after filtering by confidence (filter-then-merge) | No — reuse one `raw_detections.json` |
| `--nms-iou` | Inside YOLO during [`detect_birds.py --no-merge`](detect-birds.md) | **Yes** — one raw dump per NMS value |
| `--conf` | Swept automatically at every grid point | No |

`--iou-threshold` (matching detections to labels) is **not** part of this grid. Keep it at the evaluation default (`0.25`) so scores stay comparable to `run_pipeline.sh`.

!!! warning "Detection is the expensive part"
    `--song-gap` is applied in F-beta on an existing dump, so many gap values are cheap. `--nms-iou` is applied inside YOLO, so each NMS value is a full inference pass over the testset. Option A in `run_hpo.sh` never re-detects. Option B runs detect `len(--nms-ious)` times.

---

## Parameters

| Parameter | Type / Default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `--raw-detections` | `PATH` / — | Yes, unless `--nms-ious` | Existing raw detections **file** or results directory from [`detect_birds.py --no-merge`](detect-birds.md). Used for a `song_gap`-only search. |
| `--labels` | `PATH` / — | **Yes** | Ground truth `annotations.csv`. Filenames are matched without extensions. |
| `--output-path` | `PATH` / `results/hyperopt_merge` | No | Root directory for per-setting F-beta runs and `summary.csv`. |
| `--song-gaps` | `FLOAT [...]` / `0.0 0.1 0.2 0.5 1.0 2.0` | No | `song_gap` values in seconds to evaluate. |
| `--nms-ious` | `FLOAT [...]` / unset | No | If set, re-run detect `--no-merge` **once per NMS IoU** (the expensive path), then sweep `--song-gaps` on each dump. Requires `--model`, `--audio`, and `--species-mapping`. Omit this to reuse one dump (Option A). |
| `--model` | `PATH` / — | With `--nms-ious` | Frozen YOLO checkpoint. |
| `--audio` | `PATH` / — | With `--nms-ious` | Audio file or directory (typically the testset `soundscape_data/` folder). |
| `--species-mapping` | `CHOICE` / — | With `--nms-ious` | Must match the mapping the model was trained with. See [species mapping](../data/audio-and-model.md#species-mapping-species-mapping). |
| `--detect-conf` | `FLOAT` / `0.001` | No | Confidence used when re-running detect for `--nms-ious`. Keep low so F-beta can sweep later. |
| `--detect-workers` | `INT` / `8` | No | CPU workers for those detect passes. |
| `--iou-threshold` | `FLOAT` / `0.25` | No | IoU for matching events to labels. Use the same value as [`f_beta_score_analysis.py`](f-beta-score-analysis.md). |
| `--beta` | `FLOAT` / `1.0` | No | F-beta weighting. Same meaning as in F-beta analysis. |
| `--conf-range` | `MIN MAX STEP` / `0.00 1.0 0.01` | No | Confidence grid at each (`nms_iou`, `song_gap`) pair. |
| `--num-workers` | `INT` / `8` | No | Worker processes for each F-beta confidence sweep. |
| `--no-plot` | flag / off | No | Skip F-beta plots (faster; `run_hpo.sh` sets this). |
| `--single-cls` | flag / off | No | Collapse all species into one class. |
| `--single-cls-name` | `STR` / `bird` | No | Class name when `--single-cls` is set. |

---

## Parameter Deep-Dives

### `--song-gaps` — merge after filtering

`--song-gap` is applied **inside** F-beta: at each confidence, surviving raw boxes of the same species are merged when the gap between them is ≤ this value. That is the same filter-then-merge policy as the app and [`f_beta_score_analysis.py --song-gap`](f-beta-score-analysis.md).

Because merging happens after detect, one `--no-merge` dump is enough to try every gap.

### `--nms-ious` — suppress duplicates at detect time

NMS runs inside YOLO on each 3-second clip. Changing it changes which boxes are written to `raw_detections.json`, so the script calls `detect_birds.py --no-merge` **once per value**. That is the main compute cost of Option B: four NMS values means four full testset inference runs, before any `song_gap` sweep.

!!! info "Relationship to `--song-gap`"
    `--nms-iou` removes overlapping boxes *within* a clip. `--song-gap` then stitches surviving boxes *across* time into songs. They are different stages; searching both is a product grid (`nms × song_gap`), each cell still sweeping confidence.

### Training search vs this search

Training hyperparameter search is a separate, optional tool in [BirdBox-Train](https://github.com/birdnet-team/BirdBox-Train). It retrains YOLO (learning rate, mixup, …) and scores each trial with **one** frozen pair such as `nms_iou=0.7`, `song_gap=0.2`, so trials differ by training recipe rather than merge settings.

If you later copy a winning `best.pt` into `models/`, this page is where you can retune merge values on labeled audio. Neither search is required for ordinary detection or evaluation.

---

## Output Files

All files are written under `--output-path`. Each grid cell gets its own folder:

```text
results/<dataset>/hyperopt_merge/
  summary.csv
  nms_0.70/song_gap_0.20/
    f1.0_score_analysis.csv
    optimal_thresholds.csv
    score.json
  nms_0.70/song_gap_0.50/
    ...
```

When `--nms-ious` is set, each NMS value also stores its detect dump at `nms_<value>/detect/raw_detections.json`.

| File | Description |
| :--- | :--- |
| `summary.csv` | One row per (`nms_iou`, `song_gap`) with best micro F-beta, confidence, precision, and recall. |
| `score.json` | Same numbers for that cell. |
| `f{beta}_score_analysis.csv` | Full F-beta table for that cell (same schema as the standalone F-beta script). |
| `optimal_thresholds.csv` | Best confidence per species at that merge setting. |

Open `summary.csv` and take the row with the highest `micro_f1`. Then set `--nms-iou` / `--song-gap` in [`run_pipeline.sh`](workflows.md#complete-workflow-script) to those values for the final test evaluation.

---

## Examples

### `song_gap` only (Option A — no new detect)

Run [`detect_birds.py --no-merge`](detect-birds.md) once first (or uncomment Step 1 in `run_pipeline.sh`). Then reuse that dump:

=== "Command"
    ```bash
    python src/evaluation/hyperopt_merge.py \
        --raw-detections results/Northeastern-US \
        --labels datasets/Northeastern-US_testset/annotations.csv \
        --output-path results/Northeastern-US/hyperopt_merge \
        --song-gaps 0.0 0.1 0.2 0.5 1.0 2.0 \
        --num-workers 9 \
        --no-plot
    ```
=== "Expected Output"
    ```text
    === nms_iou=0.80 song_gap=0.0 ===
    micro F1=0.54..  conf=0.33  P=...  R=...
    ...
    Best: nms_iou=0.8 song_gap=0.2 micro F1=0.55.. conf=0.33
    Wrote results/Northeastern-US/hyperopt_merge/summary.csv
    ```

This is the default option in `run_hpo.sh`. Wall time is F-beta only (one dump, many gaps).

### `nms_iou` and `song_gap` together (Option B — re-detect per NMS)

=== "Command"
    ```bash
    python src/evaluation/hyperopt_merge.py \
        --model models/Northeastern-US.pt \
        --audio datasets/Northeastern-US_testset/soundscape_data \
        --species-mapping Northeastern-US \
        --labels datasets/Northeastern-US_testset/annotations.csv \
        --output-path results/Northeastern-US/hyperopt_merge \
        --nms-ious 0.5 0.6 0.7 0.8 \
        --song-gaps 0.1 0.2 \
        --detect-workers 18 \
        --num-workers 9 \
        --no-plot
    ```

Uncomment Option B in `run_hpo.sh` for the same grid (and comment Option A so only one search runs). Expect one full `--no-merge` detect pass per `--nms-ious` value.

---

## After the search

The usual BirdBox path stays [`run_pipeline.sh`](workflows.md#complete-workflow-script). Use this page only if you want different merge values than the pipeline defaults.

```
Usual (most users)     run_pipeline.sh     detect + F-beta + confusion matrix
Optional (this page)   run_hpo.sh          pick song_gap (and nms_iou) on a frozen model
                       then run_pipeline.sh with those values
Optional (Train repo)  run_hpo.sh          search training hyperparameters
```

If you did run the search:

1. Read `summary.csv` and pick the best `nms_iou` / `song_gap`.
2. Set those values in `run_pipeline.sh` (detect `--nms-iou`, F-beta `--song-gap`).
3. Run the usual evaluation steps on the testset.

---

## See Also

- [Typical Workflows](workflows.md) — the usual detect and evaluate path
- [Detecting Birds — `--song-gap`](detect-birds.md#-song-gap-song-gap-threshold) and [`--nms-iou`](detect-birds.md#-nms-iou-nms-iou-threshold)
- [F-Beta Score Analysis](f-beta-score-analysis.md)
