# Data and Formats

BirdBox moves data through a small set of well-defined file types: audio and models go in, detections and evaluation artifacts come out. This section is the schema contract for pipeline scripts, external tools, and anyone wiring BirdBox into a larger workflow.

## What goes where

| Stage | You provide | BirdBox produces |
| :--- | :--- | :--- |
| Inference | Audio files, a YOLO model, a species mapping | Detection files (`--output-format` on [`detect_birds.py`](../cli/detect-birds.md)) |
| Threshold tuning | Raw detection JSON + ground-truth CSV | F-beta tables and plots ([`f_beta_score_analysis.py`](../cli/f-beta-score-analysis.md)) |
| Post-processing | Raw detection JSON + chosen confidence | Filtered/merged detections ([`filter_and_merge_detections.py`](../cli/filter-and-merge-detections.md)) |
| Error analysis | Merged detection CSV + ground-truth CSV | Confusion matrices ([`confusion_matrix_analysis.py`](../cli/confusion-matrix-analysis.md)) |

## Typical file flows

### Detection only

For field use or exporting annotations—no ground-truth CSV and no evaluation scripts. You run inference once at a chosen `--conf`; song merging happens inside `detect_birds.py` unless you pass `--no-merge`.

```mermaid
flowchart TB
  subgraph inputs ["Inputs"]
    audio[Audio WAV/FLAC/OGG/MP3]
    model[YOLO model .pt / .onnx / .engine]
    mapping[species mapping]
  end

  subgraph detection ["Detection"]
    detect[detect_birds.py]
  end

  subgraph outputs ["Outputs (--output-format)"]
    outJSON[with_algorithm_metadata.json]
    outCSV[simplified.csv]
    outXC[xeno-canto-annota.json]
    outRaven[raven_selection_table.txt]
  end

  audio --> detect
  model --> detect
  mapping --> detect
  detect --> outJSON
  detect --> outCSV
  detect --> outXC
  detect --> outRaven
```

Use `--output-format all` to write every format in one run, or pick a single format. Details are on **Detection outputs** in this section.

### Detection and evaluation (ground truth)

For benchmarking on a labeled test set: run detection once at very low confidence with `--no-merge`, tune threshold against `annotations.csv`, then filter/merge and score errors. This matches `run_pipeline.sh` / `run_pipeline.bat` at the repository root.

```mermaid
flowchart TB
  subgraph inputs ["Inputs"]
    audio[Audio WAV/FLAC/OGG/MP3]
    model[YOLO model .pt / .onnx / .engine]
    labels[annotations.csv]
  end

  subgraph detection ["Detection"]
    detect[detect_birds.py --no-merge]
    rawJSON[raw_detections.json]
  end

  subgraph tuning ["Threshold tuning"]
    fbeta[f_beta_score_analysis.py]
    threshold[optimal --conf]
  end

  subgraph merge ["Filter and merge"]
    filter[filter_and_merge_detections.py]
    outCSV[simplified.csv]
  end

  subgraph metrics ["Error analysis"]
    cm[confusion_matrix_analysis.py]
  end

  audio --> detect
  model --> detect
  detect --> rawJSON
  rawJSON --> fbeta
  labels --> fbeta
  fbeta --> threshold
  rawJSON --> filter
  threshold --> filter
  filter --> outCSV
  outCSV --> cm
  labels --> cm
```

Step-by-step CLI usage for each script is under **CLI Reference** in the site navigation.
