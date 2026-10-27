---
title: Demo
template: demo.html
hide:
  - navigation
  - toc
  - footer
---

## About this demo

The demo runs the self-contained Just-Bird graph `docs/models/Just-Bird_fp16.onnx` with [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/){ target="_blank" rel="noopener noreferrer" } in your browser. The graph computes the PCEN spectrogram, runs the YOLO detector on 3 s windows every 1.5 s and applies NMS, just like [`detect_birds.py`](cli/detect-birds.md) with the Just-Bird model. There are no species labels: every box is a bird vocalization (`bird`).

**Instant sliders.** The model runs once per recording, at the lowest confidence of the slider (0.05). The demo asks the graph for its per-window boxes (`raw_all`) and repeats the graph's window fusion in JavaScript: window ownership and weighted box fusion, joining calls into phrases, removing nested boxes. NMS only lets higher-scoring boxes suppress lower ones, so the boxes for any confidence and phrase gap are the same as a graph run with those values.

**Spectrogram.** The viewer shows an STFT spectrogram by default. In its display settings, *PCEN* switches to the PCEN mel spectrogram BirdBox detects on.

**Review.** Boxes can be verified, moved, resized, removed and drawn. The exports use the BirdBox [output formats](data/outputs.md): the JSON of `--output-format json-with-algorithm-metadata`, the simplified CSV and the Raven selection table, plus a CSV with every column of the table.

!!! warning "Browser limits"
    - The model is downloaded once (about 20 MB) and then comes from the browser cache.
    - Audio is decoded locally, resampled to 32 kHz mono and truncated to **10 minutes**.
    - The browser's resampler is not librosa's, so files that are not recorded at 32 kHz can give slightly different boxes than the CLI.
    - ONNX Runtime runs single-threaded here: about 10 s for 2 minutes of audio on a laptop. Safari can be slower than Chromium or Firefox.
    - Use the [CLI](cli/detect-birds.md) for long files and batch jobs.

The sample soundscape is the example recording of [BirdNET-Analyzer](https://github.com/birdnet-team/BirdNET-Analyzer){ target="_blank" rel="noopener noreferrer" } (`birdnet_analyzer/example/soundscape.wav`, MIT License), resampled to 32 kHz. The [BirdNET website demo](https://birdnet.cornell.edu/demo/){ target="_blank" rel="noopener noreferrer" } uses the same file. Recordings loaded by number come from [Xeno-canto](https://xeno-canto.org){ target="_blank" rel="noopener noreferrer" } under the license of each recording.

## Adding models

The demo runs the first model in `docs/models/manifest.json` with `"merge_method": "fusion"`. Only window-fusion exports work: the instant sliders need the graph's per-window boxes (`raw_all`) and the JavaScript port of its window fusion (`docs/javascripts/demo/fusion.js`). The export uses window fusion for Just-Bird, Hawaii and Southern-Sierra-Nevada (`--merge auto`); `--merge fusion` forces it for the other models.

1. Export the model.

    === "Linux / macOS"
        ```bash
        python src/inference/onnx_export.py \
            --output-path docs/models/Just-Bird_fp16.onnx \
            --precision fp16
        ```

    === "Windows (PowerShell)"
        ```powershell
        python src/inference/onnx_export.py `
            --output-path docs/models/Just-Bird_fp16.onnx `
            --precision fp16
        ```

    === "Windows (CMD)"
        ```cmd
        python src/inference/onnx_export.py ^
            --output-path docs/models/Just-Bird_fp16.onnx ^
            --precision fp16
        ```

    `--precision fp16` stores the network weights in fp16, which halves the file (20 MB instead of 38 MB for Just-Bird). Feature extraction and merging stay in fp32. On 13 test recordings (`tests/onnx_graph_parity.py`), the fp16 graph reproduced 99.8% of the boxes of `detect_birds.py` with `Just-Bird.pt` (mean IoU 0.995). The export prints the merge it baked in (`Merge (baked in)`) and the suggested defaults (`Default conf`, `Default song_gap`), which it also stores in the ONNX metadata.

2. Place the `.onnx` file under `docs/models/` if the export path was elsewhere.

3. Add an entry to `docs/models/manifest.json`, first in the list if the demo should run it.

    ```json
    {
      "models": [
        {
          "file": "Just-Bird_fp16.onnx",
          "label": "Just-Bird_fp16.onnx",
          "merge_method": "fusion",
          "conf": 0.15,
          "phrase_gap": 0.05,
          "species_mapping": "Just-Bird",
          "names": { "0": "bird" }
        }
      ]
    }
    ```

    | Field | Meaning |
    | :--- | :--- |
    | `file` | The ONNX file, relative to `docs/models/` |
    | `label` | Name shown at the top of the demo |
    | `merge_method` | `fusion`, as printed by the export. Other entries are skipped. |
    | `conf` | Where the confidence slider starts: the printed `Default conf` (0.15 for Just-Bird, 0.2 for the other models) |
    | `phrase_gap` | Where the phrase gap slider starts: the printed `Default song_gap` (0.05 for Just-Bird, 0 for Hawaii and Southern-Sierra-Nevada) |
    | `species_mapping` | Written to `model_config` of the exported JSON |
    | `names` | Class labels for the exports, by class index (ONNX Runtime Web does not expose ONNX metadata) |

4. Check that the demo's fusion still matches the graph:

    ```bash
    cd tests/js && npm install
    node demo_fusion_parity.mjs ../../docs/models/Just-Bird_fp16.onnx
    ```

    The script compares `fusion.js` with the graph's own detections for 16 slider settings on `tests/test.wav` and fails on any difference. Run it after exporting a new model or changing the fusion in `src/inference/onnx_export.py`.

