<div align="center">
  <h1>BirdBox</h1>
  <img src="img/logo_birdbox.png" width="250" alt="BirdBox-Logo" />
  
  
  <p><strong>Deep Learning Bird Call Detection & Evaluation System</strong></p>
  
  <a href="#license"><img src="https://img.shields.io/badge/license-MIT%20code%20%7C%20AGPL--3.0%20project-brightgreen.svg" alt="License: MIT code, AGPL-3.0 project"></a>
  <a href="https://www.python.org/downloads/release/python-31213/" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/python-3.12-blue.svg" alt="Python 3.12"></a>

</div>

BirdBox is a comprehensive system for detecting and evaluating bird calls in audio recordings using deep learning. It leverages YOLO (You Only Look Once) object detection on spectrogram images to identify and localize bird vocalizations in time and frequency.

⚠️ **Note**: This project is still under active development. Performance may vary.

---

## Scope

This code repository focuses on model inference and post-processing for bioacoustic event detection.
Additionally, one can evaluate the performance of the model inference with ground truth annotations.
Metrics like precision, recall, F$\beta$ and confusion matrices are already implemented and ready to go.

What's not covered here is the model training.
For this see [BirdBox-Train](https://github.com/birdnet-team/BirdBox-Train){ target="_blank" rel="noopener noreferrer" } (currently only available from within the BirdNET-Team).

---

## Example Detection

The following image shows a visualization of a detection in an audio file.
The generated PCEN spectrogram reaches from 50 to 15,000 Hertz across a span of roughly 15 seconds.
The inference software found 4 different species vocalizations and was able to localize them in time and frequency.

<img src="img/example_viz.png" width="1000" alt="Detection Visualization">

!!! info "Spectrogram Not Stored"
    The inference software takes in audio files and outputs detections in various [formats](data/outputs.md). The spectrogram is computed internally but never written to disk unless explicitly visualized.

---

## Key Features

- **Browser Demo** - Runs Just-Bird in the browser, with no installation
- **Multiple Audio Formats** - Supports WAV, FLAC, OGG, MP3 (WAV/FLAC recommended for best results) 
- **Arbitrary-Length Audio Processing** - Handle audio from seconds to hours  
- **Song Reconstruction** - Automatically merge temporally adjacent detections into continuous bird songs
- **Batch Processing** - Process entire directories of audio files
**PCEN Normalization** - Per-Channel Energy Normalization for robust spectral features
- **Comprehensive Evaluation** - F-beta analysis, confusion matrices, optimal threshold finding  
- **Multiple Output Formats** - JSON with algorithm metadata, simplified CSV, Xeno-Canto Annota-JSON and Raven Selection Table  
- **Multiple Model Formats** - Works with `.pt`, `.onnx`, `.tflite` and `.engine` model formats

---

## Quick Links

- [Installation](installation.md) - set up the environment
- [Demo](demo.md) - BirdBox in your browser
- [How it works](how-it-works/overview.md) - pipeline description
- [Data In/Out](data/fileflows.md) - datasets and output-format descriptions
- [Models and Metrics](models-and-metrics/overview.md) - list of models with corresponding metrics
- [CLI Reference](cli/workflows.md) - command line interface
- [API Reference](api/api-type.md) - application programming interface
- [GitHub Repository](https://github.com/birdnet-team/BirdBox)

---

## License

BirdBox's own source code is licensed under the [MIT License](https://github.com/birdnet-team/BirdBox/blob/main/LICENSE).

BirdBox is built on [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics), which is licensed under the [AGPL-3.0](https://github.com/birdnet-team/BirdBox/blob/main/LICENSE-AGPL-3.0). Ultralytics also treats models trained or fine-tuned with it as AGPL-3.0, so the BirdBox model weights (`.pt` and every exported format) are AGPL-3.0 as well. As a result, the project as a whole is subject to the AGPL-3.0:

- **Using BirdBox with Ultralytics or the BirdBox models** (running, redistributing, or offering it over a network) has to comply with the AGPL-3.0. If you modify it and let users interact with it over a network, you must offer them its source code.
- **Reusing BirdBox's own code** on its own, in a project that uses neither Ultralytics nor the BirdBox models, is possible under the MIT terms.

A few files are under other licenses (a function adapted from librosa, xeno-canto audio samples, AviList-derived taxonomy data). See [NOTICE.md](https://github.com/birdnet-team/BirdBox/blob/main/NOTICE.md) for the full breakdown.
