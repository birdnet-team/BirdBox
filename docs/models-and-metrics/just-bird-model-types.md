# Just-Bird Model Types and Format Parity

BirdBox ships seven pretrained models. Two of them, Just-Bird and All-In-One, are released in multiple runtime formats to cover different deployment targets. This page covers the **Just-Bird** model. It documents every available format and shows how each one compares against the PyTorch baseline on identical recordings, so you can confirm that conversion or quantization does not degrade detection quality.

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

The table below is produced automatically by running `python tests/model_format_parity.py` from the repository root. The PyTorch model is the baseline. Every other format runs inference on the same recordings and its merged song segments are matched against the baseline box by box.

!!! note "Last run"
    Generated on 2026-10-20 13:52. Audio: 13 recordings, 36 minutes (see [Test recordings](#test-recordings)), species mapping: `Just-Bird`, confidence threshold: `0.2`, baseline: `Just-Bird_fp32.pt` (1632 detections).

### At a glance

| Model | Format | Size | Detections | Verdict |
| :--- | :--- | ---: | ---: | :---: |
| `Just-Bird_fp32.pt` | `.pt` | 18.2 MiB | 1632 | _baseline_ |
| `Just-Bird_fp16.engine` | `.engine` | 20.1 MiB | 1626 | :material-check-circle: PASS |
| `Just-Bird_fp16.onnx` | `.onnx` | 18.1 MiB | 1631 | :material-check-circle: PASS |
| `Just-Bird_fp16.tflite` | `.tflite` | 18.2 MiB | 1635 | :material-check-circle: PASS |
| `Just-Bird_int8.engine` | `.engine` | 10.8 MiB | 1613 | :material-alert-circle: WARN |
| `Just-Bird_int8.onnx` | `.onnx` | 9.4 MiB | 344 | :material-alert-circle: WARN |
| `Just-Bird_int8.tflite` | `.tflite` | 9.6 MiB | 1717 | :material-alert-circle: WARN |

### Detection matching

| Model | Matched | Missed | Extra | Match Rate | Mean IoU |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `Just-Bird_fp32.pt` | — | — | — | _baseline_ | — |
| `Just-Bird_fp16.engine` | 1619 | 13 | 7 | 99.2% | 0.992 |
| `Just-Bird_fp16.onnx` | 1628 | 4 | 3 | 99.8% | 0.996 |
| `Just-Bird_fp16.tflite` | 1632 | 0 | 3 | 100.0% | 0.999 |
| `Just-Bird_int8.engine` | 1378 | 254 | 235 | 84.4% | 0.889 |
| `Just-Bird_int8.onnx` | 245 | 1387 | 99 | 15.0% | 0.831 |
| `Just-Bird_int8.tflite` | 1475 | 157 | 242 | 90.4% | 0.883 |

### Match rate per recording

Share of the baseline detections in each recording that the other formats reproduce. A format that only fails on some recordings shows up here.

| Recording | Length | Baseline | `fp16.engine` | `fp16.onnx` | `fp16.tflite` | `int8.engine` | `int8.onnx` | `int8.tflite` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `test.wav` | 5:00 | 86 | 100% | 100% | 100% | 85% | 8% | 97% |
| `XC1010642_Rwanda.mp3` | 4:56 | 157 | 99% | 100% | 100% | 86% | 5% | 90% |
| `XC1073923_Ethiopia.wav` | 2:16 | 153 | 97% | 99% | 100% | 86% | 10% | 95% |
| `XC1170114_South-Africa.wav` | 4:19 | 310 | 99% | 99% | 100% | 87% | 17% | 88% |
| `XC1174845_France.mp3` | 2:00 | 73 | 100% | 100% | 100% | 92% | 29% | 90% |
| `XC1180150_Estonia.wav` | 3:21 | 167 | 100% | 100% | 100% | 79% | 8% | 96% |
| `XC1181792_Spain.mp3` | 4:39 | 197 | 100% | 100% | 100% | 78% | 14% | 84% |
| `XC1184153_Spain.wav` | 1:03 | 17 | 100% | 100% | 100% | 100% | 0% | 100% |
| `XC1184684_Austria.wav` | 3:20 | 189 | 99% | 100% | 100% | 84% | 22% | 92% |
| `XC1185879_Sweden.wav` | 1:29 | 79 | 99% | 100% | 100% | 92% | 42% | 94% |
| `XC692092_Kenya.mp3` | 1:02 | 73 | 100% | 100% | 100% | 85% | 18% | 92% |
| `XC782384_Sierra-Leone.wav` | 1:07 | 58 | 100% | 100% | 100% | 86% | 12% | 84% |
| `XC889825_Zimbabwe.wav` | 1:46 | 73 | 99% | 100% | 100% | 75% | 8% | 84% |

### Confidence and timing

| Model | Mean Conf Δ | Max Conf Δ | Load (s) | Detect (s) |
| :--- | ---: | ---: | ---: | ---: |
| `Just-Bird_fp32.pt` | — | — | 0.2 | 83.8 |
| `Just-Bird_fp16.engine` | 0.0024 | 0.2237 | 0.2 | 80.7 |
| `Just-Bird_fp16.onnx` | 0.0013 | 0.1827 | 0.2 | 109.9 |
| `Just-Bird_fp16.tflite` | 0.0005 | 0.1039 | 0.1 | 206.4 |
| `Just-Bird_int8.engine` | 0.0509 | 0.3111 | 0.2 | 81.2 |
| `Just-Bird_int8.onnx` | 0.4410 | 0.6949 | 0.2 | 132.6 |
| `Just-Bird_int8.tflite` | 0.0659 | 0.4874 | 0.1 | 94.4 |

### Verdict criteria

!!! success "PASS"
    The model matches at least 90% of baseline detections, adds no more than 10% extra detections, and keeps the mean confidence difference at or below 0.05. The export is safe to use in place of the baseline.

!!! warning "WARN"
    One or more thresholds were exceeded. The model runs but performance may have degraded after conversion or quantization. Inspect the per-model detail below and compare detections on your own audio before deploying.

!!! failure "FAIL"
    The worker subprocess exited with an error. The model did not produce any detections. Check the per-model detail below for the full traceback.

### Metric glossary

Matching is one-to-one and recording by recording: a candidate detection can match a baseline detection of the same species when their 2D IoU (time and frequency) is at least `0.5`. Pairs are taken in order of decreasing IoU, and each detection is used at most once.

| Metric | What it tells you |
| :--- | :--- |
| **Mean IoU** | Average time-and-frequency box overlap between matched pairs. `1.000` is a perfect overlap. Lower values mean boxes drifted in position or size. |
| **Mean Conf Δ** | Average absolute confidence difference on matched detections. Near `0.000` means the converted model is as confident as the baseline. |
| **Max Conf Δ** | Largest single confidence difference among matched detections. Surfaces worst-case outliers that the mean hides. |
| **Load (s)** | Wall-clock seconds to load the model file and build the detector. |
| **Detect (s)** | Wall-clock seconds spent running inference on all recordings. |

### Test recordings { #test-recordings }

| Recording | Length | Source |
| :--- | ---: | :--- |
| `test.wav` | 5:00 | [UHH_001_S01_20161121_150000](https://doi.org/10.5281/zenodo.7078499), Hawaii, Navine et al. (2022), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `XC1010642_Rwanda.mp3` | 4:56 | [XC1010642](https://xeno-canto.org/1010642), Rwanda, Alain Malengreau, [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| `XC1073923_Ethiopia.wav` | 2:16 | [XC1073923](https://xeno-canto.org/1073923), Ethiopia, He Wenjin/文进, [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| `XC1170114_South-Africa.wav` | 4:19 | [XC1170114](https://xeno-canto.org/1170114), South Africa, Walter Wallner, [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| `XC1174845_France.mp3` | 2:00 | [XC1174845](https://xeno-canto.org/1174845), France, Lucien Hueber, [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| `XC1180150_Estonia.wav` | 3:21 | [XC1180150](https://xeno-canto.org/1180150), Estonia, Fernand Deroussen, [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |
| `XC1181792_Spain.mp3` | 4:39 | [XC1181792](https://xeno-canto.org/1181792), Spain, Esperanza Poveda, [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| `XC1184153_Spain.wav` | 1:03 | [XC1184153](https://xeno-canto.org/1184153), Spain, Esperanza Poveda, [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| `XC1184684_Austria.wav` | 3:20 | [XC1184684](https://xeno-canto.org/1184684), Austria, Walter Wallner, [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| `XC1185879_Sweden.wav` | 1:29 | [XC1185879](https://xeno-canto.org/1185879), Sweden, Ulf Elman, [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) |
| `XC692092_Kenya.mp3` | 1:02 | [XC692092](https://xeno-canto.org/692092), Kenya, isaac kilusu, [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| `XC782384_Sierra-Leone.wav` | 1:07 | [XC782384](https://xeno-canto.org/782384), Sierra Leone, Joost van Bruggen, [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) |
| `XC889825_Zimbabwe.wav` | 1:46 | [XC889825](https://xeno-canto.org/889825), Zimbabwe, Peter Ward, [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |

The xeno-canto soundscapes are not part of the repository. Fetch them with `python tests/parity_audio/download.py`.

---

## Per-model detail

---

### Just-Bird_fp32.pt

!!! info "Baseline"
    All other formats are compared against this model. Its detections define what a correct result looks like.

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `bird` | 1632 | 1632 |

File size: 18.2 MiB. Load time: 0.2 s. Detection time: 83.8 s.

---

### Just-Bird_fp16.engine

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `bird` | 1632 | 1626 |

!!! success "PASS"
    Matched 1619 of 1632 baseline detections (99.2%), with 13 missed and 7 extra. Mean IoU: 0.992. Mean Conf Δ: 0.0024. Mean start-time shift: 0.004 s.

File size: 20.1 MiB. Load time: 0.2 s. Detection time: 80.7 s.

---

### Just-Bird_fp16.onnx

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `bird` | 1632 | 1631 |

!!! success "PASS"
    Matched 1628 of 1632 baseline detections (99.8%), with 4 missed and 3 extra. Mean IoU: 0.996. Mean Conf Δ: 0.0013. Mean start-time shift: 0.003 s.

File size: 18.1 MiB. Load time: 0.2 s. Detection time: 109.9 s.

---

### Just-Bird_fp16.tflite

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `bird` | 1632 | 1635 |

!!! success "PASS"
    Matched 1632 of 1632 baseline detections (100.0%), with 0 missed and 3 extra. Mean IoU: 0.999. Mean Conf Δ: 0.0005. Mean start-time shift: 0.001 s.

File size: 18.2 MiB. Load time: 0.1 s. Detection time: 206.4 s.

---

### Just-Bird_int8.engine

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `bird` | 1632 | 1613 |

!!! warning "WARN"
    Matched 1378 of 1632 baseline detections (84.4%), with 254 missed and 235 extra. Mean IoU: 0.889. Mean Conf Δ: 0.0509. Mean start-time shift: 0.063 s.

File size: 10.8 MiB. Load time: 0.2 s. Detection time: 81.2 s.

---

### Just-Bird_int8.onnx

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `bird` | 1632 | 344 |

!!! warning "WARN"
    Matched 245 of 1632 baseline detections (15.0%), with 1387 missed and 99 extra. Mean IoU: 0.831. Mean Conf Δ: 0.4410. Mean start-time shift: 0.123 s.

File size: 9.4 MiB. Load time: 0.2 s. Detection time: 132.6 s.

---

### Just-Bird_int8.tflite

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `bird` | 1632 | 1717 |

!!! warning "WARN"
    Matched 1475 of 1632 baseline detections (90.4%), with 157 missed and 242 extra. Mean IoU: 0.883. Mean Conf Δ: 0.0659. Mean start-time shift: 0.057 s.

File size: 9.6 MiB. Load time: 0.1 s. Detection time: 94.4 s.

---
