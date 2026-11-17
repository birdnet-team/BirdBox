This chapter explains how BirdBox turns a recording into boxes, in Python and in the browser.

## Inference

BirdBox detects with YOLO on a sliding window: it computes a log-floor mel spectrogram, cuts it into 3 s windows every 1.5 s, runs the model on each window and merges the boxes of overlapping windows into one set of detections.

- [Algorithm Overview](overview.md): the steps from the audio file to the detections, and how the boxes of overlapping windows are merged (window fusion and the song merge).
- [Spectrogram Generation](spectrogram.md): how the spectrogram and the model's 256 × 256 images are made, stage by stage.
- [Troubleshooting](trouble.md): what to try with no detections, too many false positives, boxes too long or too short, a silent recording in the demo, memory or runtime issues and mismatched file names.

## Demo

The [Demo](../demo.md) runs the same steps in the browser, as one ONNX graph that also computes the spectrogram.

- [Detection](demo-detection.md): the models in the menu, boxes while a file runs, the instant sliders, recording, review and export, and the browser's limits.
- [Spectrogram](demo-spectrogram.md): the viewer's STFT view next to the log-floor view the model is given.
- [Adding Models](demo-adding-models.md): exporting a model to ONNX and adding it to the model menu.
