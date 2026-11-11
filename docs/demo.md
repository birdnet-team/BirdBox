---
title: Demo
template: demo.html
hide:
  - navigation
  - toc
  - footer
---

## About this demo

The demo runs a self-contained BirdBox graph with [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/){ target="_blank" rel="noopener noreferrer" } in your browser. The graph computes the log-floor spectrogram, runs the YOLO detector on 3 s windows every 1.5 s and applies NMS, just like [`detect_birds.py`](cli/detect-birds.md) with the same model. The model menu at the top picks the graph:

- **Just-Bird** (`docs/models/Just-Bird_fp16.onnx`, the default): every box is a bird vocalization (`bird`), without species labels. For species, combine the boxes with [BirdNET](https://github.com/birdnet-team/BirdNET-Analyzer){ target="_blank" rel="noopener noreferrer" }.
- **Birds-With-Labels** (`docs/models/Birds-With-Labels_fp16.onnx`): the [Birds-With-Labels](models-and-metrics/birds-with-labels.md) model with its 282 classes. The table names the species of each box.
- **Just-Bird, rolling floor** (`docs/models/Just-Bird_rolling60_fp16.onnx`, experimental): the same model with each window's noise floor taken over the 60 s before the window's end. Boxes found while recording are final at once (see *Recording*). The CLI gets the same features with `BIRDBOX_FLOOR_WINDOW=60`.

**Boxes while it runs.** New boxes appear after every 3 s of audio, and the part not analyzed yet is dimmed. A segment's spectrogram is computed as a whole, because its noise floor depends on the whole 60 s, and its clips then go through the network two at a time. With the rolling-floor model the spectrogram is computed 6 s at a time instead and kept, so there is no pause at the start of each minute. The rows are the same as in one run of the whole graph (`tests/js/demo_chunk_parity.mjs`). Rows are final once they are there, but the next windows can still change the boxes near the end of the analyzed part: fuse them with theirs, hide them as nested, or join them into a phrase. Such boxes are dashed: those that end less than 1.5 s plus the gap before it, and the boxes linked to them, of the same class, at most the gap apart and, for window fusion, with overlapping bands. That covers a growing phrase, whose band widens as it takes in new calls, so that it can take in a box beside it that ended long before. Solid boxes do not change any more (`docs/javascripts/demo/pending.js`, `tests/js/demo_dashed_boxes.mjs`). Exports unlock when the whole recording is done. If you stop detection, *Continue Detecting* picks it up where it stopped, and the result is the same as an uninterrupted run. While it is stopped, you can export the boxes found so far, and a note under the export buttons says that the export does not cover the whole file. The file names of such exports end in `_partial`, and the JSON also states how far detection got (`demo.partial`, `demo.analyzed_until_s`, `demo.audio_duration_s`).

**Recording.** When you record, the spectrogram flies in from the right, and boxes appear 1 to 3 s after a sound, as soon as the windows around it are complete. Dashed boxes are provisional. The final boxes use the noise floor of the whole minute, which does not exist yet, so these use the last 20 s of audio instead. As soon as a minute and the 2 s after it are complete, its final boxes are found on the side and turn solid, on phones only after you stop. When you stop, the rest is detected, usually just the last minute. The result is the same as loading the saved file. With the rolling-floor model, a window's noise floor comes from the audio before its end, so each window is detected once, as soon as it is complete, and its rows are final. Only new audio becomes spectrogram frames, so a window takes a fraction of a second. Only the boxes the next windows may still change are dashed, as in a file, and when you stop, only the last few windows are left (`tests/js/rolling_floor_live.mjs`). Recordings are saved as 32 kHz, 16-bit WAV, what BirdBox reads, without the compression of a typical browser recording. Nothing leaves your device.

**Instant sliders.** The model runs once per recording, at the lowest confidence of the slider (0.05). The demo asks the graph for its per-window boxes (`raw_all`) and repeats the graph's merge in JavaScript. For Just-Bird that is window fusion: window ownership and weighted box fusion, removing nested boxes, joining calls into phrases. For Birds-With-Labels it is song reconstruction: boxes of one species closer than the song gap become one box. NMS only lets higher-scoring boxes suppress lower ones, so the boxes for any confidence and gap are the same as a graph run with those values.

**Spectrogram.** The viewer shows an STFT spectrogram by default. In its display settings, *Log-floor* switches to the log-floor mel spectrogram BirdBox detects on (without the 2 s of context BirdBox adds to each minute's noise floor, so it can differ slightly at minute boundaries).

**Review.** Boxes can be verified, moved, resized, removed and drawn. The exports use the BirdBox [output formats](data/outputs.md): the JSON of `--output-format json-with-algorithm-metadata`, the simplified CSV and the Raven selection table, plus a CSV with every column of the table.

!!! warning "Browser limits"
    - The model is downloaded once (about 20 MB for Just-Bird, 51 MB for Birds-With-Labels) and then comes from the browser cache.
    - Audio is decoded locally, resampled to 32 kHz mono and truncated to **10 minutes**.
    - The browser's resampler is not librosa's, so files that are not recorded at 32 kHz can give slightly different boxes than the CLI.
    - ONNX Runtime runs single-threaded here: about 10 s for 2 minutes of audio on a laptop. Safari can be slower than Chromium or Firefox.
    - Use the [CLI](cli/detect-birds.md) for long files and batch jobs.

The sample soundscape is the example recording of [BirdNET-Analyzer](https://github.com/birdnet-team/BirdNET-Analyzer){ target="_blank" rel="noopener noreferrer" } (`birdnet_analyzer/example/soundscape.wav`, MIT License), resampled to 32 kHz. The [BirdNET website demo](https://birdnet.cornell.edu/demo/){ target="_blank" rel="noopener noreferrer" } uses the same file. Recordings loaded by number come from [Xeno-canto](https://xeno-canto.org){ target="_blank" rel="noopener noreferrer" } under the license of each recording.

## Adding models

The model menu lists the models in `docs/models/manifest.json`; the first one is the default. The instant sliders need the graph's per-window boxes (`raw_all`) and a JavaScript port of its merge: `docs/javascripts/demo/fusion.js` for window fusion (`fusion`, the export's default for Just-Bird) and `docs/javascripts/demo/songs.js` for song reconstruction (`reconstruct`, the default for Birds-With-Labels). The Just-Bird frequency gate (`--merge reconstruct` with Just-Bird) has no port; such entries are skipped.

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

    For another model, add `--pt-model`, for example `--pt-model models/Birds-With-Labels_fp32.pt --output-path docs/models/Birds-With-Labels_fp16.onnx`.

    `--precision fp16` stores the network weights in fp16, which halves the file (20 MB instead of 38 MB for Just-Bird, 51 MB instead of 99 MB for Birds-With-Labels). Feature extraction and merging stay in fp32. On 13 test recordings (`tests/onnx_graph_parity.py`), the fp16 graphs reproduced the boxes of `detect_birds.py` with the `.pt` model: 99.8% for Just-Bird (mean IoU 0.995) and 99.7% for Birds-With-Labels, with the same species (mean IoU 0.997). The export prints the merge it baked in (`Merge (baked in)`) and the suggested defaults (`Default conf`, `Default song_gap`), which it also stores in the ONNX metadata.

2. Place the `.onnx` file under `docs/models/` if the export path was elsewhere.

3. Add an entry to `docs/models/manifest.json`, first in the list if the demo should start with it.

    ```json
    {
      "models": [
        {
          "file": "Just-Bird_fp16.onnx",
          "label": "Just-Bird_fp16.onnx (recommended)",
          "merge_method": "fusion",
          "conf": 0.15,
          "phrase_gap": 0.05,
          "species_mapping": "Just-Bird",
          "names": { "0": "bird" }
        },
        {
          "file": "Birds-With-Labels_fp16.onnx",
          "label": "Birds-With-Labels_fp16.onnx",
          "merge_method": "reconstruct",
          "conf": 0.2,
          "song_gap": 0.1,
          "species_mapping": "Birds-With-Labels",
          "names": { "0": "????", "1": "amabaw1", "…": "…" },
          "common_names": { "0": "Unknown", "1": "Amazonian Barred-Woodcreeper", "…": "…" }
        }
      ]
    }
    ```

    | Field | Meaning |
    | :--- | :--- |
    | `file` | The ONNX file, relative to `docs/models/` |
    | `label` | Name in the model menu |
    | `merge_method` | `fusion` or `reconstruct`, as printed by the export. Other entries are skipped. |
    | `conf` | Where the confidence slider starts: the printed `Default conf` (0.15 for Just-Bird, 0.2 for the other models) |
    | `phrase_gap` | `fusion` only: where the phrase gap slider starts, the printed `Default song_gap` (0.05 for Just-Bird, 0 for Birds-With-Labels) |
    | `song_gap` | `reconstruct` only: where the song gap slider starts, the printed `Default song_gap` (0.1) |
    | `species_mapping` | Written to `model_config` of the exported JSON |
    | `names` | eBird codes by class index, for the table and the exports (ONNX Runtime Web does not expose ONNX metadata) |
    | `common_names` | Optional: common names by class index, shown in the table instead of the codes |

    For a species model, `names` and `common_names` come from `id_to_ebird` and `ebird_to_name` of its mapping in `src/config.py`. With more than one class, the table gets a species column.

4. Check that the demo's merge still matches the graph:

    ```bash
    cd tests/js && npm install
    node demo_merge_parity.mjs ../../docs/models/Just-Bird_fp16.onnx
    ```

    The script compares `fusion.js` or `songs.js`, whichever the manifest entry's `merge_method` names, with the graph's own detections for 24 slider settings (16 for song reconstruction) on `tests/test.wav` and fails on any difference. Run it after exporting a new model or changing a merge in `src/inference/onnx_export.py`.

    `npm test` runs all demo tests for every model in the manifest: the merge, the chunked and streamed runs, the boxes while recording, which boxes are dashed. They run in parallel processes, each on one ONNX Runtime thread like the demo, so the rows are the browser's. That takes about 10 minutes with 14 processes on a 20-core machine (`--jobs=N` sets how many run at once), against about 45 minutes one after another (`npm run test:serial`).

