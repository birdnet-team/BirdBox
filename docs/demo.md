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

**Spectrogram.** The viewer shows an STFT spectrogram by default. In its display settings, *PCEN* switches to the PCEN mel spectrogram BirdBox detects on, as the [Streamlit demo](getting-started/demo_streamlit.md) shows it.

**Review.** Boxes can be verified, moved, resized, removed and drawn. The exports use the BirdBox [output formats](data/outputs.md): the JSON of `--output-format json-with-algorithm-metadata`, the simplified CSV and the Raven selection table, plus a CSV with every column of the table.

!!! warning "Browser limits"
    - The model is downloaded once (about 35 MB) and then comes from the browser cache.
    - Audio is decoded locally, resampled to 32 kHz mono and truncated to **10 minutes** (the same limit as the [Streamlit demo](getting-started/demo_streamlit.md)).
    - The browser's resampler is not librosa's, so files that are not recorded at 32 kHz can give slightly different boxes than the CLI.
    - ONNX Runtime runs single-threaded here: about 10 s for 2 minutes of audio on a laptop. Safari can be slower than Chromium or Firefox.
    - Use the [CLI](cli/detect-birds.md) for long files and batch jobs.

The 15 s sample is the first 15 seconds of recording `UHH_001_S01_20161121_150000` from Navine, Kahl, Tanimoto-Johnson, Klinck & Hart (2022), [*A collection of fully-annotated soundscape recordings from the Island of Hawai'i*](https://doi.org/10.5281/zenodo.7078499){ target="_blank" rel="noopener noreferrer" }, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/){ target="_blank" rel="noopener noreferrer" }. Recordings loaded by number come from [Xeno-canto](https://xeno-canto.org){ target="_blank" rel="noopener noreferrer" } under the license of each recording.
