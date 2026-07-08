# All-In-One Model Types and Format Parity

BirdBox ships seven pretrained models. Two of them, Just-Bird and All-In-One, are released in multiple runtime formats to cover different deployment targets. This page covers the **All-In-One** model. It documents every available format and shows how each one compares against the PyTorch baseline on an identical audio file, so you can confirm that conversion or quantization does not degrade detection quality.

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

---

## Format parity test

The table below is produced automatically by running `python tests/model_format_parity.py` from the repository root. The PyTorch model is the baseline. Every other format runs inference on the same audio clip and its merged song segments are matched against the baseline box by box.

!!! note "Last run"
    Generated on 2026-07-08 16:21. Audio: `test.wav`, species mapping: `All-In-One`, confidence threshold: `0.2`, baseline: `All-In-One_fp32.pt` (100 detections).

### At a glance

| Model | Format | Size | Detections | Verdict |
| :--- | :--- | ---: | ---: | :---: |
| `All-In-One_fp32.pt` | `.pt` | 49.2 MiB | 100 | _baseline_ |
| `All-In-One_fp16.engine` | `.engine` | 51.8 MiB | 99 | :material-check-circle: PASS |
| `All-In-One_fp16.onnx` | `.onnx` | 48.8 MiB | 99 | :material-check-circle: PASS |
| `All-In-One_fp16.tflite` | `.tflite` | 49.1 MiB | 100 | :material-check-circle: PASS |
| `All-In-One_int8.engine` | `.engine` | 27.2 MiB | 116 | :material-alert-circle: WARN |
| `All-In-One_int8.onnx` | `.onnx` | 25.2 MiB | 27 | :material-alert-circle: WARN |
| `All-In-One_int8.tflite` | `.tflite` | 25.6 MiB | 105 | :material-alert-circle: WARN |

### Detection matching

| Model | Matched | Missed | Extra | Match Rate | Mean IoU |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `All-In-One_fp32.pt` | — | — | — | _baseline_ | — |
| `All-In-One_fp16.engine` | 99 | 1 | 0 | 99.0% | 0.989 |
| `All-In-One_fp16.onnx` | 99 | 1 | 0 | 99.0% | 0.995 |
| `All-In-One_fp16.tflite` | 100 | 0 | 0 | 100.0% | 1.000 |
| `All-In-One_int8.engine` | 89 | 11 | 27 | 89.0% | 0.885 |
| `All-In-One_int8.onnx` | 27 | 73 | 0 | 27.0% | 0.883 |
| `All-In-One_int8.tflite` | 97 | 3 | 8 | 97.0% | 0.860 |

### Confidence and timing

| Model | Mean Conf Δ | Max Conf Δ | Load (s) | Detect (s) |
| :--- | ---: | ---: | ---: | ---: |
| `All-In-One_fp32.pt` | — | — | 0.2 | 12.2 |
| `All-In-One_fp16.engine` | 0.0016 | 0.0505 | 0.1 | 11.4 |
| `All-In-One_fp16.onnx` | 0.0012 | 0.0504 | 0.2 | 18.2 |
| `All-In-One_fp16.tflite` | 0.0002 | 0.0025 | 0.1 | 78.9 |
| `All-In-One_int8.engine` | 0.0488 | 0.1828 | 0.1 | 11.4 |
| `All-In-One_int8.onnx` | 0.5211 | 0.8497 | 0.2 | 24.5 |
| `All-In-One_int8.tflite` | 0.0638 | 0.3254 | 0.1 | 17.9 |

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

### All-In-One_fp32.pt

!!! info "Baseline"
    All other formats are compared against this model. Its detections define what a correct result looks like.

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `ercfra` | 4 | 4 |
| `hawama` | 53 | 53 |
| `houfin` | 13 | 13 |
| `jabwar` | 30 | 30 |

File size: 49.2 MiB. Load time: 0.2 s. Detection time: 12.2 s.

---

### All-In-One_fp16.engine

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `ercfra` | 4 | 4 |
| `hawama` | 53 | 53 |
| `houfin` | 13 | 12 |
| `jabwar` | 30 | 30 |

!!! success "PASS"
    Matched 99 of 100 baseline detections (99.0%), with 1 missed and 0 extra. Mean IoU: 0.989. Mean Conf Δ: 0.0016. Mean start-time shift: 0.001 s.

File size: 51.8 MiB. Load time: 0.1 s. Detection time: 11.4 s.

---

### All-In-One_fp16.onnx

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `ercfra` | 4 | 4 |
| `hawama` | 53 | 53 |
| `houfin` | 13 | 12 |
| `jabwar` | 30 | 30 |

!!! success "PASS"
    Matched 99 of 100 baseline detections (99.0%), with 1 missed and 0 extra. Mean IoU: 0.995. Mean Conf Δ: 0.0012. Mean start-time shift: 0.000 s.

File size: 48.8 MiB. Load time: 0.2 s. Detection time: 18.2 s.

---

### All-In-One_fp16.tflite

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `ercfra` | 4 | 4 |
| `hawama` | 53 | 53 |
| `houfin` | 13 | 13 |
| `jabwar` | 30 | 30 |

!!! success "PASS"
    Matched 100 of 100 baseline detections (100.0%), with 0 missed and 0 extra. Mean IoU: 1.000. Mean Conf Δ: 0.0002. Mean start-time shift: 0.000 s.

File size: 49.1 MiB. Load time: 0.1 s. Detection time: 78.9 s.

---

### All-In-One_int8.engine

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `amepip` | 0 | 2 |
| `ercfra` | 4 | 4 |
| `hawama` | 53 | 59 |
| `houfin` | 13 | 18 |
| `jabwar` | 30 | 31 |
| `skylar` | 0 | 1 |
| `warwhe1` | 0 | 1 |

!!! warning "WARN"
    Matched 89 of 100 baseline detections (89.0%), with 11 missed and 27 extra. Mean IoU: 0.885. Mean Conf Δ: 0.0488. Mean start-time shift: 0.007 s.

File size: 27.2 MiB. Load time: 0.1 s. Detection time: 11.4 s.

---

### All-In-One_int8.onnx

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `ercfra` | 4 | 4 |
| `hawama` | 53 | 20 |
| `houfin` | 13 | 3 |
| `jabwar` | 30 | 0 |

!!! warning "WARN"
    Matched 27 of 100 baseline detections (27.0%), with 73 missed and 0 extra. Mean IoU: 0.883. Mean Conf Δ: 0.5211. Mean start-time shift: 0.019 s.

File size: 25.2 MiB. Load time: 0.2 s. Detection time: 24.5 s.

---

### All-In-One_int8.tflite

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `amepip` | 0 | 1 |
| `ercfra` | 4 | 4 |
| `hawama` | 53 | 52 |
| `houfin` | 13 | 14 |
| `jabwar` | 30 | 34 |

!!! warning "WARN"
    Matched 97 of 100 baseline detections (97.0%), with 3 missed and 8 extra. Mean IoU: 0.860. Mean Conf Δ: 0.0638. Mean start-time shift: 0.017 s.

File size: 25.6 MiB. Load time: 0.1 s. Detection time: 17.9 s.

---
