# Just-Bird Model Types and Format Parity

BirdBox ships two pretrained models, Just-Bird and Birds-With-Labels. Both are released in multiple runtime formats to cover different deployment targets. This page covers the **Just-Bird** model. It documents every available format and shows how each one compares against the PyTorch baseline on identical recordings, so you can confirm that conversion or quantization does not degrade detection quality.

---

## Supported model formats

Each format targets a different deployment scenario. Install the matching runtime with `python install.py --model-format <FORMAT>`. See [Installation](../installation.md) for the full setup guide.

| Format | Typical use case | Runtime |
| :--- | :--- | :--- |
| `.pt` | Default. PyTorch checkpoint, easiest to get started. | PyTorch (CUDA or CPU) |
| `.onnx` | Cross-platform deployment, quantized variants available. | ONNX Runtime (GPU or CPU) |
| `.tflite` | Edge devices and mobile targets. | LiteRT / ai-edge-litert (CPU) |
| `.engine` | Maximum throughput on NVIDIA GPUs. | TensorRT (NVIDIA GPU required) |

!!! warning "Platform restrictions"
    `.engine` files are compiled for a specific GPU architecture. A model built on one card may not run on a different GPU generation.

!!! danger "Each format requires its own Python environment"
    Do not load a `.tflite`, `.onnx`, or `.engine` model from a `.pt` environment. The wrong environment will either raise an import error immediately or silently degrade results. Run `python install.py --model-format <FORMAT>` to install the correct runtime before switching formats. See [Install Parameters](../installation.md#install-parameters) for the full table.

---

## Format parity test

The table below is produced automatically by running `python tests/model_format_parity.py` from the repository root. The PyTorch model is the baseline. Every other format runs inference on the same recordings and its merged song segments are matched against the baseline box by box.

!!! note "Last run"
    Generated on 2026-11-11 14:07. Audio: 13 recordings, 36 minutes (see [Test recordings](#test-recordings)), species mapping: `Just-Bird`, confidence threshold: `0.2`, baseline: `Just-Bird_fp32.pt` (1312 detections).

### At a glance

| Model | Format | Size | Detections | Verdict |
| :--- | :--- | ---: | ---: | :---: |
| `Just-Bird_fp32.pt` | `.pt` | 18.2 MiB | 1312 | _baseline_ |
| `Just-Bird_fp16.onnx` | `.onnx` | 18.1 MiB | 1310 | :material-check-circle: PASS |
| `Just-Bird_fp16.tflite` | `.tflite` | 18.2 MiB | 1312 | :material-check-circle: PASS |
| `Just-Bird_int8.onnx` | `.onnx` | 13.1 MiB | 1294 | :material-check-circle: PASS |
| `Just-Bird_int8.tflite` | `.tflite` | 13.2 MiB | 1276 | :material-check-circle: PASS |

### Detection matching

| Model | Matched | Missed | Extra | Match Rate | Mean IoU |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `Just-Bird_fp32.pt` | — | — | — | _baseline_ | — |
| `Just-Bird_fp16.onnx` | 1306 | 6 | 4 | 99.5% | 0.997 |
| `Just-Bird_fp16.tflite` | 1311 | 1 | 1 | 99.9% | 0.999 |
| `Just-Bird_int8.onnx` | 1213 | 99 | 81 | 92.5% | 0.960 |
| `Just-Bird_int8.tflite` | 1214 | 98 | 62 | 92.5% | 0.958 |

### Match rate per recording

Share of the baseline detections in each recording that the other formats reproduce. A format that only fails on some recordings shows up here.

**FP16**

| Recording | Length | Baseline | `fp16.onnx` | `fp16.tflite` |
| :--- | ---: | ---: | ---: | ---: |
| `test.wav` | 5:00 | 89 | 100% | 100% |
| `XC1010642_Rwanda.mp3` | 4:56 | 149 | 100% | 100% |
| `XC1073923_Ethiopia.wav` | 2:16 | 84 | 100% | 100% |
| `XC1170114_South-Africa.wav` | 4:19 | 237 | 100% | 100% |
| `XC1174845_France.mp3` | 2:00 | 62 | 98% | 98% |
| `XC1180150_Estonia.wav` | 3:21 | 185 | 100% | 100% |
| `XC1181792_Spain.mp3` | 4:39 | 189 | 98% | 100% |
| `XC1184153_Spain.wav` | 1:03 | 11 | 100% | 100% |
| `XC1184684_Austria.wav` | 3:20 | 117 | 99% | 100% |
| `XC1185879_Sweden.wav` | 1:29 | 76 | 100% | 100% |
| `XC692092_Kenya.mp3` | 1:02 | 51 | 100% | 100% |
| `XC782384_Sierra-Leone.wav` | 1:07 | 31 | 100% | 100% |
| `XC889825_Zimbabwe.wav` | 1:46 | 31 | 100% | 100% |

**INT8**

| Recording | Length | Baseline | `int8.onnx` | `int8.tflite` |
| :--- | ---: | ---: | ---: | ---: |
| `test.wav` | 5:00 | 89 | 94% | 92% |
| `XC1010642_Rwanda.mp3` | 4:56 | 149 | 91% | 91% |
| `XC1073923_Ethiopia.wav` | 2:16 | 84 | 89% | 92% |
| `XC1170114_South-Africa.wav` | 4:19 | 237 | 93% | 94% |
| `XC1174845_France.mp3` | 2:00 | 62 | 90% | 94% |
| `XC1180150_Estonia.wav` | 3:21 | 185 | 97% | 94% |
| `XC1181792_Spain.mp3` | 4:39 | 189 | 87% | 93% |
| `XC1184153_Spain.wav` | 1:03 | 11 | 91% | 91% |
| `XC1184684_Austria.wav` | 3:20 | 117 | 94% | 94% |
| `XC1185879_Sweden.wav` | 1:29 | 76 | 95% | 93% |
| `XC692092_Kenya.mp3` | 1:02 | 51 | 96% | 90% |
| `XC782384_Sierra-Leone.wav` | 1:07 | 31 | 94% | 97% |
| `XC889825_Zimbabwe.wav` | 1:46 | 31 | 87% | 81% |

### Confidence and timing

| Model | Mean Conf Δ | Max Conf Δ | Load (s) | Detect (s) |
| :--- | ---: | ---: | ---: | ---: |
| `Just-Bird_fp32.pt` | — | — | 1.0 | 345.0 |
| `Just-Bird_fp16.onnx` | 0.0007 | 0.1274 | 0.8 | 498.0 |
| `Just-Bird_fp16.tflite` | 0.0003 | 0.0576 | 0.9 | 396.3 |
| `Just-Bird_int8.onnx` | 0.0220 | 0.3480 | 0.4 | 179.1 |
| `Just-Bird_int8.tflite` | 0.0210 | 0.3136 | 0.3 | 131.0 |

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
| `bird` | 1312 | 1312 |

File size: 18.2 MiB. Load time: 1.0 s. Detection time: 345.0 s.

---

### Just-Bird_fp16.onnx

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `bird` | 1312 | 1310 |

!!! success "PASS"
    Matched 1306 of 1312 baseline detections (99.5%), with 6 missed and 4 extra. Mean IoU: 0.997. Mean Conf Δ: 0.0007. Mean start-time shift: 0.000 s.

File size: 18.1 MiB. Load time: 0.8 s. Detection time: 498.0 s.

---

### Just-Bird_fp16.tflite

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `bird` | 1312 | 1312 |

!!! success "PASS"
    Matched 1311 of 1312 baseline detections (99.9%), with 1 missed and 1 extra. Mean IoU: 0.999. Mean Conf Δ: 0.0003. Mean start-time shift: 0.000 s.

File size: 18.2 MiB. Load time: 0.9 s. Detection time: 396.3 s.

---

### Just-Bird_int8.onnx

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `bird` | 1312 | 1294 |

!!! success "PASS"
    Matched 1213 of 1312 baseline detections (92.5%), with 99 missed and 81 extra. Mean IoU: 0.960. Mean Conf Δ: 0.0220. Mean start-time shift: 0.034 s.

File size: 13.1 MiB. Load time: 0.4 s. Detection time: 179.1 s.

---

### Just-Bird_int8.tflite

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `bird` | 1312 | 1276 |

!!! success "PASS"
    Matched 1214 of 1312 baseline detections (92.5%), with 98 missed and 62 extra. Mean IoU: 0.958. Mean Conf Δ: 0.0210. Mean start-time shift: 0.025 s.

File size: 13.2 MiB. Load time: 0.3 s. Detection time: 131.0 s.

---
