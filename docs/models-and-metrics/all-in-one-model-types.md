# All-In-One Model Types and Format Parity

BirdBox ships seven pretrained models. Two of them, Just-Bird and All-In-One, are released in multiple runtime formats to cover different deployment targets. This page covers the **All-In-One** model. It documents every available format and shows how each one compares against the PyTorch baseline on identical recordings, so you can confirm that conversion or quantization does not degrade detection quality.

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
    Generated on 2026-10-20 21:21. Audio: 13 recordings, 36 minutes (see [Test recordings](#test-recordings)), species mapping: `All-In-One`, confidence threshold: `0.2`, baseline: `All-In-One_fp32.pt` (1070 detections).

### At a glance

| Model | Format | Size | Detections | Verdict |
| :--- | :--- | ---: | ---: | :---: |
| `All-In-One_fp32.pt` | `.pt` | 49.2 MiB | 1070 | _baseline_ |
| `All-In-One_fp16.engine` | `.engine` | 51.8 MiB | 1070 | :material-check-circle: PASS |
| `All-In-One_fp16.onnx` | `.onnx` | 48.8 MiB | 1069 | :material-check-circle: PASS |
| `All-In-One_fp16.tflite` | `.tflite` | 49.1 MiB | 1069 | :material-check-circle: PASS |
| `All-In-One_int8.engine` | `.engine` | 35.9 MiB | 1064 | :material-alert-circle: WARN |
| `All-In-One_int8.onnx` | `.onnx` | 38.8 MiB | 1063 | :material-alert-circle: WARN |
| `All-In-One_int8.tflite` | `.tflite` | 38.8 MiB | 1025 | :material-alert-circle: WARN |

### Detection matching

| Model | Matched | Missed | Extra | Match Rate | Mean IoU |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `All-In-One_fp32.pt` | — | — | — | _baseline_ | — |
| `All-In-One_fp16.engine` | 1064 | 6 | 6 | 99.4% | 0.994 |
| `All-In-One_fp16.onnx` | 1067 | 3 | 2 | 99.7% | 0.997 |
| `All-In-One_fp16.tflite` | 1069 | 1 | 0 | 99.9% | 0.999 |
| `All-In-One_int8.engine` | 960 | 110 | 104 | 89.7% | 0.952 |
| `All-In-One_int8.onnx` | 934 | 136 | 129 | 87.3% | 0.946 |
| `All-In-One_int8.tflite` | 892 | 178 | 133 | 83.4% | 0.940 |

### Match rate per recording

Share of the baseline detections in each recording that the other formats reproduce. A format that only fails on some recordings shows up here.

| Recording | Length | Baseline | `fp16.engine` | `fp16.onnx` | `fp16.tflite` | `int8.engine` | `int8.onnx` | `int8.tflite` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `test.wav` | 5:00 | 101 | 99% | 99% | 100% | 98% | 92% | 97% |
| `XC1010642_Rwanda.mp3` | 4:56 | 132 | 100% | 100% | 100% | 83% | 86% | 82% |
| `XC1073923_Ethiopia.wav` | 2:16 | 53 | 100% | 100% | 100% | 83% | 87% | 79% |
| `XC1170114_South-Africa.wav` | 4:19 | 155 | 99% | 100% | 100% | 92% | 87% | 84% |
| `XC1174845_France.mp3` | 2:00 | 48 | 100% | 100% | 100% | 88% | 92% | 83% |
| `XC1180150_Estonia.wav` | 3:21 | 185 | 99% | 99% | 100% | 91% | 89% | 84% |
| `XC1181792_Spain.mp3` | 4:39 | 107 | 100% | 100% | 100% | 90% | 84% | 77% |
| `XC1184153_Spain.wav` | 1:03 | 3 | 100% | 100% | 100% | 67% | 100% | 67% |
| `XC1184684_Austria.wav` | 3:20 | 114 | 97% | 99% | 99% | 89% | 85% | 86% |
| `XC1185879_Sweden.wav` | 1:29 | 37 | 100% | 100% | 100% | 95% | 84% | 73% |
| `XC692092_Kenya.mp3` | 1:02 | 53 | 100% | 100% | 100% | 89% | 94% | 87% |
| `XC782384_Sierra-Leone.wav` | 1:07 | 49 | 100% | 100% | 100% | 88% | 82% | 76% |
| `XC889825_Zimbabwe.wav` | 1:46 | 33 | 100% | 100% | 100% | 88% | 82% | 82% |

### Confidence and timing

| Model | Mean Conf Δ | Max Conf Δ | Load (s) | Detect (s) |
| :--- | ---: | ---: | ---: | ---: |
| `All-In-One_fp32.pt` | — | — | 0.2 | 85.5 |
| `All-In-One_fp16.engine` | 0.0028 | 0.0968 | 0.2 | 81.0 |
| `All-In-One_fp16.onnx` | 0.0016 | 0.1958 | 0.2 | 132.1 |
| `All-In-One_fp16.tflite` | 0.0007 | 0.1964 | 0.1 | 578.2 |
| `All-In-One_int8.engine` | 0.0257 | 0.1723 | 0.2 | 81.5 |
| `All-In-One_int8.onnx` | 0.0290 | 0.2123 | 0.2 | 128.9 |
| `All-In-One_int8.tflite` | 0.0338 | 0.2373 | 0.1 | 159.8 |

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

### All-In-One_fp32.pt

!!! info "Baseline"
    All other formats are compared against this model. Its detections define what a correct result looks like.

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `????` | 1 | 1 |
| `akepa1` | 21 | 21 |
| `amecro` | 1 | 1 |
| `amegfi` | 2 | 2 |
| `amerob` | 28 | 28 |
| `apapan` | 104 | 104 |
| `brdowl` | 1 | 1 |
| `bubgro2` | 1 | 1 |
| `calqua` | 2 | 2 |
| `clanut` | 1 | 1 |
| `daejun` | 3 | 3 |
| `eawpew` | 1 | 1 |
| `elepai` | 37 | 37 |
| `ercfra` | 39 | 39 |
| `gcrfin` | 3 | 3 |
| `grycat` | 13 | 13 |
| `haiwoo` | 2 | 2 |
| `hawama` | 149 | 149 |
| `hawcre` | 41 | 41 |
| `hawpet1` | 7 | 7 |
| `houfin` | 29 | 29 |
| `iiwi` | 75 | 75 |
| `jabwar` | 32 | 32 |
| `melthr` | 10 | 10 |
| `norcar` | 8 | 8 |
| `norwat` | 1 | 1 |
| `omao` | 80 | 80 |
| `ovenbi1` | 2 | 2 |
| `palila` | 3 | 3 |
| `reblei` | 176 | 176 |
| `rufant3` | 1 | 1 |
| `scatan` | 1 | 1 |
| `skylar` | 39 | 39 |
| `stejay` | 18 | 18 |
| `veery` | 6 | 6 |
| `warwhe1` | 119 | 119 |
| `whcspa` | 5 | 5 |
| `wiltur` | 1 | 1 |
| `woothr` | 5 | 5 |
| `yebsap` | 1 | 1 |
| `yetvir` | 1 | 1 |

File size: 49.2 MiB. Load time: 0.2 s. Detection time: 85.5 s.

---

### All-In-One_fp16.engine

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `????` | 1 | 1 |
| `akepa1` | 21 | 21 |
| `amecro` | 1 | 1 |
| `amegfi` | 2 | 2 |
| `amerob` | 28 | 28 |
| `apapan` | 104 | 103 |
| `brdowl` | 1 | 1 |
| `bubgro2` | 1 | 1 |
| `calqua` | 2 | 2 |
| `clanut` | 1 | 1 |
| `daejun` | 3 | 3 |
| `eawpew` | 1 | 1 |
| `elepai` | 37 | 37 |
| `ercfra` | 39 | 39 |
| `gcrfin` | 3 | 3 |
| `grycat` | 13 | 13 |
| `haiwoo` | 2 | 2 |
| `hawama` | 149 | 149 |
| `hawcre` | 41 | 41 |
| `hawpet1` | 7 | 7 |
| `houfin` | 29 | 29 |
| `iiwi` | 75 | 74 |
| `jabwar` | 32 | 32 |
| `melthr` | 10 | 10 |
| `norcar` | 8 | 8 |
| `norwat` | 1 | 1 |
| `omao` | 80 | 80 |
| `ovenbi1` | 2 | 2 |
| `palila` | 3 | 3 |
| `reblei` | 176 | 176 |
| `rufant3` | 1 | 1 |
| `scatan` | 1 | 1 |
| `skylar` | 39 | 39 |
| `stejay` | 18 | 18 |
| `veery` | 6 | 6 |
| `warwhe1` | 119 | 120 |
| `whcspa` | 5 | 6 |
| `wiltur` | 1 | 1 |
| `woothr` | 5 | 5 |
| `yebsap` | 1 | 1 |
| `yetvir` | 1 | 1 |

!!! success "PASS"
    Matched 1064 of 1070 baseline detections (99.4%), with 6 missed and 6 extra. Mean IoU: 0.994. Mean Conf Δ: 0.0028. Mean start-time shift: 0.001 s.

File size: 51.8 MiB. Load time: 0.2 s. Detection time: 81.0 s.

---

### All-In-One_fp16.onnx

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `????` | 1 | 1 |
| `akepa1` | 21 | 21 |
| `amecro` | 1 | 1 |
| `amegfi` | 2 | 2 |
| `amerob` | 28 | 28 |
| `apapan` | 104 | 103 |
| `brdowl` | 1 | 1 |
| `bubgro2` | 1 | 1 |
| `calqua` | 2 | 2 |
| `clanut` | 1 | 1 |
| `daejun` | 3 | 3 |
| `eawpew` | 1 | 1 |
| `elepai` | 37 | 37 |
| `ercfra` | 39 | 39 |
| `gcrfin` | 3 | 3 |
| `grycat` | 13 | 13 |
| `haiwoo` | 2 | 2 |
| `hawama` | 149 | 149 |
| `hawcre` | 41 | 41 |
| `hawpet1` | 7 | 7 |
| `houfin` | 29 | 28 |
| `iiwi` | 75 | 74 |
| `jabwar` | 32 | 32 |
| `melthr` | 10 | 10 |
| `norcar` | 8 | 8 |
| `norwat` | 1 | 1 |
| `omao` | 80 | 80 |
| `ovenbi1` | 2 | 2 |
| `palila` | 3 | 3 |
| `reblei` | 176 | 177 |
| `rufant3` | 1 | 1 |
| `scatan` | 1 | 1 |
| `skylar` | 39 | 39 |
| `stejay` | 18 | 18 |
| `veery` | 6 | 6 |
| `warwhe1` | 119 | 120 |
| `whcspa` | 5 | 5 |
| `wiltur` | 1 | 1 |
| `woothr` | 5 | 5 |
| `yebsap` | 1 | 1 |
| `yetvir` | 1 | 1 |

!!! success "PASS"
    Matched 1067 of 1070 baseline detections (99.7%), with 3 missed and 2 extra. Mean IoU: 0.997. Mean Conf Δ: 0.0016. Mean start-time shift: 0.001 s.

File size: 48.8 MiB. Load time: 0.2 s. Detection time: 132.1 s.

---

### All-In-One_fp16.tflite

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `????` | 1 | 1 |
| `akepa1` | 21 | 21 |
| `amecro` | 1 | 1 |
| `amegfi` | 2 | 2 |
| `amerob` | 28 | 28 |
| `apapan` | 104 | 104 |
| `brdowl` | 1 | 1 |
| `bubgro2` | 1 | 1 |
| `calqua` | 2 | 2 |
| `clanut` | 1 | 1 |
| `daejun` | 3 | 3 |
| `eawpew` | 1 | 1 |
| `elepai` | 37 | 37 |
| `ercfra` | 39 | 39 |
| `gcrfin` | 3 | 3 |
| `grycat` | 13 | 13 |
| `haiwoo` | 2 | 2 |
| `hawama` | 149 | 149 |
| `hawcre` | 41 | 41 |
| `hawpet1` | 7 | 7 |
| `houfin` | 29 | 29 |
| `iiwi` | 75 | 74 |
| `jabwar` | 32 | 32 |
| `melthr` | 10 | 10 |
| `norcar` | 8 | 8 |
| `norwat` | 1 | 1 |
| `omao` | 80 | 80 |
| `ovenbi1` | 2 | 2 |
| `palila` | 3 | 3 |
| `reblei` | 176 | 176 |
| `rufant3` | 1 | 1 |
| `scatan` | 1 | 1 |
| `skylar` | 39 | 39 |
| `stejay` | 18 | 18 |
| `veery` | 6 | 6 |
| `warwhe1` | 119 | 119 |
| `whcspa` | 5 | 5 |
| `wiltur` | 1 | 1 |
| `woothr` | 5 | 5 |
| `yebsap` | 1 | 1 |
| `yetvir` | 1 | 1 |

!!! success "PASS"
    Matched 1069 of 1070 baseline detections (99.9%), with 1 missed and 0 extra. Mean IoU: 0.999. Mean Conf Δ: 0.0007. Mean start-time shift: 0.000 s.

File size: 49.1 MiB. Load time: 0.1 s. Detection time: 578.2 s.

---

### All-In-One_int8.engine

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `????` | 1 | 2 |
| `akepa1` | 21 | 23 |
| `amecro` | 1 | 1 |
| `amegfi` | 2 | 2 |
| `amepip` | 0 | 1 |
| `amerob` | 28 | 28 |
| `apapan` | 104 | 104 |
| `brdowl` | 1 | 1 |
| `bubgro2` | 1 | 1 |
| `butwoo1` | 0 | 1 |
| `calqua` | 2 | 2 |
| `clanut` | 1 | 0 |
| `daejun` | 3 | 4 |
| `eawpew` | 1 | 1 |
| `elepai` | 37 | 29 |
| `ercfra` | 39 | 40 |
| `gcrfin` | 3 | 4 |
| `grycat` | 13 | 15 |
| `haiwoo` | 2 | 2 |
| `hawama` | 149 | 141 |
| `hawcre` | 41 | 43 |
| `hawpet1` | 7 | 6 |
| `houfin` | 29 | 32 |
| `iiwi` | 75 | 59 |
| `jabwar` | 32 | 32 |
| `melthr` | 10 | 12 |
| `norcar` | 8 | 6 |
| `norwat` | 1 | 1 |
| `omao` | 80 | 79 |
| `ovenbi1` | 2 | 1 |
| `palila` | 3 | 3 |
| `reblei` | 176 | 178 |
| `rufant3` | 1 | 1 |
| `scatan` | 1 | 1 |
| `skylar` | 39 | 40 |
| `stejay` | 18 | 17 |
| `veery` | 6 | 7 |
| `warwhe1` | 119 | 129 |
| `whcspa` | 5 | 6 |
| `wiltur` | 1 | 1 |
| `woothr` | 5 | 6 |
| `yebsap` | 1 | 1 |
| `yetvir` | 1 | 1 |

!!! warning "WARN"
    Matched 960 of 1070 baseline detections (89.7%), with 110 missed and 104 extra. Mean IoU: 0.952. Mean Conf Δ: 0.0257. Mean start-time shift: 0.025 s.

File size: 35.9 MiB. Load time: 0.2 s. Detection time: 81.5 s.

---

### All-In-One_int8.onnx

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `????` | 1 | 3 |
| `akepa1` | 21 | 21 |
| `amecro` | 1 | 1 |
| `amegfi` | 2 | 2 |
| `amepip` | 0 | 1 |
| `amerob` | 28 | 31 |
| `apapan` | 104 | 96 |
| `brdowl` | 1 | 1 |
| `bubgro2` | 1 | 1 |
| `calqua` | 2 | 1 |
| `clanut` | 1 | 1 |
| `daejun` | 3 | 4 |
| `eawpew` | 1 | 1 |
| `elepai` | 37 | 29 |
| `ercfra` | 39 | 41 |
| `gcrfin` | 3 | 3 |
| `grycat` | 13 | 17 |
| `haiwoo` | 2 | 2 |
| `hawama` | 149 | 142 |
| `hawcre` | 41 | 51 |
| `hawpet1` | 7 | 8 |
| `houfin` | 29 | 33 |
| `iiwi` | 75 | 58 |
| `jabwar` | 32 | 30 |
| `melthr` | 10 | 11 |
| `norcar` | 8 | 9 |
| `norwat` | 1 | 1 |
| `omao` | 80 | 80 |
| `ovenbi1` | 2 | 1 |
| `palila` | 3 | 3 |
| `reblei` | 176 | 173 |
| `rewbla` | 0 | 1 |
| `rufant3` | 1 | 1 |
| `scatan` | 1 | 1 |
| `skylar` | 39 | 40 |
| `sonspa` | 0 | 1 |
| `stejay` | 18 | 20 |
| `veery` | 6 | 7 |
| `warwhe1` | 119 | 119 |
| `whcspa` | 5 | 7 |
| `whtspa` | 0 | 1 |
| `wiltur` | 1 | 1 |
| `woothr` | 5 | 6 |
| `yebsap` | 1 | 1 |
| `yetvir` | 1 | 1 |

!!! warning "WARN"
    Matched 934 of 1070 baseline detections (87.3%), with 136 missed and 129 extra. Mean IoU: 0.946. Mean Conf Δ: 0.0290. Mean start-time shift: 0.022 s.

File size: 38.8 MiB. Load time: 0.2 s. Detection time: 128.9 s.

---

### All-In-One_int8.tflite

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `????` | 1 | 2 |
| `akepa1` | 21 | 25 |
| `amecro` | 1 | 1 |
| `amegfi` | 2 | 2 |
| `amepip` | 0 | 1 |
| `amerob` | 28 | 26 |
| `apapan` | 104 | 96 |
| `brdowl` | 1 | 1 |
| `bubgro2` | 1 | 1 |
| `butwoo1` | 0 | 1 |
| `calqua` | 2 | 2 |
| `clanut` | 1 | 1 |
| `daejun` | 3 | 3 |
| `eawpew` | 1 | 1 |
| `elepai` | 37 | 28 |
| `ercfra` | 39 | 40 |
| `gcrfin` | 3 | 4 |
| `grycat` | 13 | 19 |
| `haiwoo` | 2 | 1 |
| `hawama` | 149 | 136 |
| `hawcre` | 41 | 42 |
| `hawpet1` | 7 | 7 |
| `houfin` | 29 | 26 |
| `iiwi` | 75 | 51 |
| `jabwar` | 32 | 35 |
| `meapar` | 0 | 1 |
| `melthr` | 10 | 13 |
| `moudov` | 0 | 1 |
| `norcar` | 8 | 9 |
| `norwat` | 1 | 1 |
| `omao` | 80 | 73 |
| `ovenbi1` | 2 | 1 |
| `palila` | 3 | 3 |
| `reblei` | 176 | 167 |
| `rewbla` | 0 | 1 |
| `rufant3` | 1 | 1 |
| `scatan` | 1 | 1 |
| `skylar` | 39 | 40 |
| `stejay` | 18 | 23 |
| `veery` | 6 | 7 |
| `warwhe1` | 119 | 113 |
| `whcspa` | 5 | 8 |
| `whtspa` | 0 | 1 |
| `wiltur` | 1 | 1 |
| `woothr` | 5 | 6 |
| `yebsap` | 1 | 1 |
| `yetvir` | 1 | 1 |

!!! warning "WARN"
    Matched 892 of 1070 baseline detections (83.4%), with 178 missed and 133 extra. Mean IoU: 0.940. Mean Conf Δ: 0.0338. Mean start-time shift: 0.026 s.

File size: 38.8 MiB. Load time: 0.1 s. Detection time: 159.8 s.

---
