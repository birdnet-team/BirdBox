# Data and Formats

This section defines the files BirdBox reads and writes across inference and evaluation.

## Input Side

- audio files (`.wav`, `.flac`, `.ogg`, `.mp3`)
- labels CSV for evaluation
- trained model files (`.pt`, `.onnx`, `.engine`)
- species mapping definitions (`src/config.py`)

## Output Side

- detections JSON with algorithm metadata
- simplified detections CSV
- Xeno-Canto Annota-JSON
- Raven selection tables
- F-beta analysis tables and plots
- confusion matrices and metadata

Use these pages as the schema contract for pipeline automation and external integrations.
