# Detection in the Demo

The [Demo](../demo.md) runs a self-contained BirdBox graph with [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/){ target="_blank" rel="noopener noreferrer" } in your browser. The graph computes the log-floor spectrogram, runs the YOLO detector on 3 s windows every 1.5 s and applies NMS, just like [`detect_birds.py`](../cli/detect-birds.md) with the same model. The model menu at the top picks the graph:

- **Just-Bird** (`docs/models/Just-Bird_fp16.onnx`, the default): every box is a bird vocalization (`bird`), without species labels. For species, combine the boxes with [BirdNET](https://github.com/birdnet-team/BirdNET-Analyzer){ target="_blank" rel="noopener noreferrer" }.
- **Birds-With-Labels** (`docs/models/Birds-With-Labels_fp16.onnx`): the [Birds-With-Labels](../models-and-metrics/birds-with-labels.md) model. It finds calls as Just-Bird does and names each box after its species, or `bird` when it is not sure. The table shows the other likely species on hover.

This page explains how the demo detects, while a file runs and while you record. [Spectrogram](demo-spectrogram.md) explains what the viewer shows, [Adding Models](demo-adding-models.md) how to add a model to the menu.

## Boxes While It Runs

New boxes appear after every 1.5 s of audio, and the part not analyzed yet is dimmed. Each window's noise floor comes from the 60 s before the window's end ([rolling floor](spectrogram.md#stage-5-log-floor-normalization)), so nothing depends on where a minute starts: the spectrogram is computed one window at a time and kept, and each window goes through the network as soon as its frames are there. The floor itself is computed in the page: each band's frames of the last 60 s are kept sorted, and from one window to the next only the 128 frames that leave and the 128 that arrive are merged in, so the network gets only the window's own frames (`tests/js/demo_floor.mjs`). The rows are the same as in one run of the whole graph (`tests/js/demo_chunk_parity.mjs`).

### Pace

The windows show evenly spaced: a few finished windows, about 0.6 s of work, are held back and shown at the mean pace, which covers windows that take longer (`docs/javascripts/demo/pacer.js`, `tests/js/demo_pacer.mjs`). The progress line and, at the end, the summary show how many seconds of audio a second of detection covers ("9.5× real time").

### Dashed Boxes

Rows are final once they are there, but the next windows can still change the boxes near the end of the analyzed part: fuse them with theirs, hide them as nested, or join them into a phrase. Such boxes are dashed: those that end less than 1.5 s plus the gap before it, and the boxes linked to them, of the same class, at most the gap apart and, for window fusion, with overlapping bands. That covers a growing phrase, whose band widens as it takes in new calls, so that it can take in a box beside it that ended long before. Solid boxes do not change any more (`docs/javascripts/demo/pending.js`, `tests/js/demo_dashed_boxes.mjs`). With window fusion, the boxes no later window can reach are fused once and kept, so an update only fuses the newest rows, with the same boxes as fusing all rows anew (`docs/javascripts/demo/fusion.js`, `tests/js/demo_fusion_parts.mjs`).

### Stop and Continue

Exports unlock when the whole recording is done. If you stop detection, *Continue Detecting* picks it up where it stopped, and the result is the same as an uninterrupted run. While it is stopped, you can export the boxes found so far, and a note under the export buttons says that the export does not cover the whole file. The file names of such exports end in `_partial`, and the JSON also states how far detection got (`demo.partial`, `demo.analyzed_until_s`, `demo.audio_duration_s`).

## Instant Sliders

The model runs once per recording, at the lowest confidence of the slider (0.05). The demo asks the graph for its per-window boxes (`raw_all`) and repeats the graph's merge in JavaScript. For both models that is window fusion: window ownership and weighted box fusion, removing nested boxes, joining calls into phrases. Birds-With-Labels merges its boxes without their species and then names each merged box from the boxes in it ([Species Names](overview.md#species-names)). NMS only lets higher-scoring boxes suppress lower ones, so the boxes for any confidence and gap are the same as a graph run with those values.

## Recording

When you record, the spectrogram flies in from the right, and boxes appear a few seconds after a sound. A window's noise floor comes from the audio before its end, so each window is detected once, as soon as it is complete, and its rows are final. Only new audio becomes spectrogram frames, so a window takes a fraction of a second. Only the boxes the next windows may still change are dashed, as in a file. When you stop, only the last few windows are left, and the result is the same as loading the saved file (`tests/js/rolling_floor_live.mjs`). Recordings are saved as 32 kHz, 16-bit WAV, what BirdBox reads, without the compression of a typical browser recording. Nothing leaves your device.

## Review and Export

Boxes can be verified, moved, resized, removed and drawn. The exports use the BirdBox [output formats](../data/outputs.md): the JSON of `--output-format json-with-algorithm-metadata`, the simplified CSV and the Raven selection table, plus a CSV with every column of the table.

!!! warning "Browser limits"
    - The model is downloaded once (about 20 MB for Just-Bird, 52 MB for Birds-With-Labels) and then comes from the browser cache.
    - Audio is decoded locally, resampled to 32 kHz mono and truncated to **10 minutes**. The detection worker gets it 10 s at a time, so it holds a few seconds of audio rather than a copy of the file.
    - The browser's resampler is not librosa's, so files that are not recorded at 32 kHz can give slightly different boxes than the CLI.
    - ONNX Runtime runs single-threaded here: about 10 s for 2 minutes of audio on a laptop. Safari can be slower than Chromium or Firefox.
    - Use the [CLI](../cli/detect-birds.md) for long files and batch jobs.

The sample soundscape is the example recording of [BirdNET-Analyzer](https://github.com/birdnet-team/BirdNET-Analyzer){ target="_blank" rel="noopener noreferrer" } (`birdnet_analyzer/example/soundscape.wav`, MIT License), resampled to 32 kHz. The [BirdNET website demo](https://birdnet.cornell.edu/demo/){ target="_blank" rel="noopener noreferrer" } uses the same file. Recordings loaded by number come from [Xeno-canto](https://xeno-canto.org){ target="_blank" rel="noopener noreferrer" } under the license of each recording.
