# Just-Bird Model Types and Format Parity

BirdBox ships seven pretrained models. Two of them, Just-Bird and All-In-One, are released in multiple runtime formats to cover different deployment targets. This page covers the **Just-Bird** model. It documents every available format and shows how each one compares against the PyTorch baseline on an identical audio file, so you can confirm that conversion or quantization does not degrade detection quality.

---

## Supported model formats

Each format targets a different deployment scenario. Install the matching runtime with `python install.py --model-format <FORMAT>`. See [Installation](../getting-started/installation.md) for the full setup guide.

| Format | Typical use case | Runtime |
| :--- | :--- | :--- |
| `.pt` | Default. PyTorch checkpoint, easiest to get started. | PyTorch (CUDA or CPU) |
| `.onnx` | Cross-platform deployment, quantized variants available. | ONNX Runtime (GPU or CPU) |
| `.tflite` | Edge devices and mobile targets. | LiteRT / ai-edge-litert (CPU) |
| `.engine` | Maximum throughput on NVIDIA GPUs. | TensorRT (NVIDIA GPU required) |

!!! warning "Platform restrictions"
    `.engine` files are compiled for a specific GPU architecture. A model built on one card may not run on a different GPU generation.

!!! danger "Each format requires its own Python environment"
    Do not load a `.tflite`, `.onnx`, or `.engine` model from a `.pt` environment. The wrong environment will either raise an import error immediately or silently degrade results. Run `python install.py --model-format <FORMAT>` to install the correct runtime before switching formats. See [Install Parameters](../getting-started/installation.md#install-parameters) for the full table.

---

## Format parity test

The table below is produced automatically by running `python tests/model_format_parity.py` from the repository root. The PyTorch model is the baseline. Every other format runs inference on the same audio clip and its merged song segments are matched against the baseline box by box.

!!! note "Last run"
    Generated on 2026-07-08 16:42. Audio: `test.wav`, species mapping: `Just-Bird`, confidence threshold: `0.2`, baseline: `Just-Bird_fp32.pt` (94 detections).

### At a glance

| Model | Format | Size | Detections | Verdict |
| :--- | :--- | ---: | ---: | :---: |
| `Just-Bird_fp32.pt` | `.pt` | 18.2 MiB | 94 | _baseline_ |
| `Just-Bird_fp16.engine` | `.engine` | 20.3 MiB | 93 | :material-check-circle: PASS |
| `Just-Bird_fp16.onnx` | `.onnx` | 18.1 MiB | 93 | :material-check-circle: PASS |
| `Just-Bird_fp16.tflite` | `.tflite` | 18.2 MiB | 94 | :material-check-circle: PASS |
| `Just-Bird_int8.engine` | `.engine` | 10.8 MiB | 33 | :material-alert-circle: WARN |
| `Just-Bird_int8.onnx` | `.onnx` | 9.4 MiB | 1 | :material-alert-circle: WARN |
| `Just-Bird_int8.tflite` | `.tflite` | 9.6 MiB | 24 | :material-alert-circle: WARN |

### Detection matching

| Model | Matched | Missed | Extra | Match Rate | Mean IoU |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `Just-Bird_fp32.pt` | — | — | — | _baseline_ | — |
| `Just-Bird_fp16.engine` | 93 | 1 | 0 | 98.9% | 0.990 |
| `Just-Bird_fp16.onnx` | 93 | 1 | 0 | 98.9% | 0.993 |
| `Just-Bird_fp16.tflite` | 94 | 0 | 0 | 100.0% | 0.999 |
| `Just-Bird_int8.engine` | 31 | 63 | 2 | 33.0% | 0.840 |
| `Just-Bird_int8.onnx` | 0 | 94 | 1 | 0.0% | 0.000 |
| `Just-Bird_int8.tflite` | 19 | 75 | 5 | 20.2% | 0.744 |

### Confidence and timing

| Model | Mean Conf Δ | Max Conf Δ | Load (s) | Detect (s) |
| :--- | ---: | ---: | ---: | ---: |
| `Just-Bird_fp32.pt` | — | — | 0.1 | 11.6 |
| `Just-Bird_fp16.engine` | 0.0026 | 0.0502 | 0.1 | 11.2 |
| `Just-Bird_fp16.onnx` | 0.0018 | 0.0107 | 0.2 | 13.9 |
| `Just-Bird_fp16.tflite` | 0.0002 | 0.0013 | 0.1 | 28.1 |
| `Just-Bird_int8.engine` | 0.0939 | 0.2530 | 0.1 | 11.2 |
| `Just-Bird_int8.onnx` | 0.0000 | 0.0000 | 0.2 | 16.9 |
| `Just-Bird_int8.tflite` | 0.1480 | 0.4722 | 0.1 | 12.8 |

### Verdict criteria

!!! success "PASS"
    The model matches at least 90% of baseline detections, adds no more than 10% extra detections, and keeps the mean confidence difference at or below 0.05. The export is safe to use in place of the baseline.

!!! warning "WARN"
    One or more thresholds were exceeded. The model runs but performance may have degraded after conversion or quantization. Inspect the per-model detail below and compare detections on your own audio before deploying.

!!! failure "FAIL"
    The worker subprocess exited with an error. The model did not produce any detections. Check the per-model detail below for the full traceback.

### Metric glossary

Matching uses a greedy algorithm: for each baseline detection the candidate detection with the highest 2D IoU (time and frequency) above `0.3` and the same species is selected.

| Metric | What it tells you |
| :--- | :--- |
| **Mean IoU** | Average time-and-frequency box overlap between matched pairs. `1.000` is a perfect overlap. Lower values mean boxes drifted in position or size. |
| **Mean Conf Δ** | Average absolute confidence difference on matched detections. Near `0.000` means the converted model is as confident as the baseline. |
| **Max Conf Δ** | Largest single confidence difference among matched detections. Surfaces worst-case outliers that the mean hides. |
| **Load (s)** | Wall-clock seconds to load the model file and build the detector. |
| **Detect (s)** | Wall-clock seconds spent running inference on the audio clip. |

---

## Per-model detail

---

### Just-Bird_fp32.pt

!!! info "Baseline"
    All other formats are compared against this model. Its detections define what a correct result looks like.

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `bird` | 94 | 94 |

File size: 18.2 MiB. Load time: 0.1 s. Detection time: 11.6 s.

---

### Just-Bird_fp16.engine

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `bird` | 94 | 93 |

!!! success "PASS"
    Matched 93 of 94 baseline detections (98.9%), with 1 missed and 0 extra. Mean IoU: 0.990. Mean Conf Δ: 0.0026. Mean start-time shift: 0.000 s.

File size: 20.3 MiB. Load time: 0.1 s. Detection time: 11.2 s.

---

### Just-Bird_fp16.onnx

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `bird` | 94 | 93 |

!!! success "PASS"
    Matched 93 of 94 baseline detections (98.9%), with 1 missed and 0 extra. Mean IoU: 0.993. Mean Conf Δ: 0.0018. Mean start-time shift: 0.000 s.

File size: 18.1 MiB. Load time: 0.2 s. Detection time: 13.9 s.

---

### Just-Bird_fp16.tflite

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `bird` | 94 | 94 |

!!! success "PASS"
    Matched 94 of 94 baseline detections (100.0%), with 0 missed and 0 extra. Mean IoU: 0.999. Mean Conf Δ: 0.0002. Mean start-time shift: 0.000 s.

File size: 18.2 MiB. Load time: 0.1 s. Detection time: 28.1 s.

---

### Just-Bird_int8.engine

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `bird` | 94 | 33 |

!!! warning "WARN"
    Matched 31 of 94 baseline detections (33.0%), with 63 missed and 2 extra. Mean IoU: 0.840. Mean Conf Δ: 0.0939. Mean start-time shift: 0.014 s.

File size: 10.8 MiB. Load time: 0.1 s. Detection time: 11.2 s.

---

### Just-Bird_int8.onnx

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `bird` | 94 | 1 |

!!! warning "WARN"
    Matched 0 of 94 baseline detections (0.0%), with 94 missed and 1 extra. Mean IoU: 0.000. Mean Conf Δ: 0.0000. Mean start-time shift: 0.000 s.

File size: 9.4 MiB. Load time: 0.2 s. Detection time: 16.9 s.

---

### Just-Bird_int8.tflite

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `bird` | 94 | 24 |

!!! warning "WARN"
    Matched 19 of 94 baseline detections (20.2%), with 75 missed and 5 extra. Mean IoU: 0.744. Mean Conf Δ: 0.1480. Mean start-time shift: 0.057 s.

File size: 9.6 MiB. Load time: 0.1 s. Detection time: 12.8 s.

---
