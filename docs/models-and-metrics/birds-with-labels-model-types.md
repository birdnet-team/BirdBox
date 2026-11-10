# Birds-With-Labels Model Types and Format Parity

BirdBox ships two pretrained models, Just-Bird and Birds-With-Labels. Both are released in multiple runtime formats to cover different deployment targets. This page covers the **Birds-With-Labels** model. It documents every available format and shows how each one compares against the PyTorch baseline on identical recordings, so you can confirm that conversion or quantization does not degrade detection quality.

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
    Generated on 2026-11-05 10:14. Audio: 13 recordings, 36 minutes (see [Test recordings](#test-recordings)), species mapping: `Birds-With-Labels`, confidence threshold: `0.2`, baseline: `Birds-With-Labels_fp32.pt` (1149 detections).

### At a glance

| Model | Format | Size | Detections | Verdict |
| :--- | :--- | ---: | ---: | :---: |
| `Birds-With-Labels_fp32.pt` | `.pt` | 49.2 MiB | 1149 | _baseline_ |
| `Birds-With-Labels_fp16.onnx` | `.onnx` | 48.8 MiB | 1147 | :material-check-circle: PASS |
| `Birds-With-Labels_fp16.tflite` | `.tflite` | 49.1 MiB | 1149 | :material-check-circle: PASS |
| `Birds-With-Labels_int8.onnx` | `.onnx` | 38.8 MiB | 1055 | :material-alert-circle: WARN |
| `Birds-With-Labels_int8.tflite` | `.tflite` | 38.8 MiB | 1094 | :material-alert-circle: WARN |

### Detection matching

| Model | Matched | Missed | Extra | Match Rate | Mean IoU |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `Birds-With-Labels_fp32.pt` | — | — | — | _baseline_ | — |
| `Birds-With-Labels_fp16.onnx` | 1146 | 3 | 1 | 99.7% | 0.997 |
| `Birds-With-Labels_fp16.tflite` | 1148 | 1 | 1 | 99.9% | 0.999 |
| `Birds-With-Labels_int8.onnx` | 956 | 193 | 99 | 83.2% | 0.936 |
| `Birds-With-Labels_int8.tflite` | 824 | 325 | 270 | 71.7% | 0.903 |

### Match rate per recording

Share of the baseline detections in each recording that the other formats reproduce. A format that only fails on some recordings shows up here.

**FP16**

| Recording | Length | Baseline | `fp16.onnx` | `fp16.tflite` |
| :--- | ---: | ---: | ---: | ---: |
| `test.wav` | 5:00 | 109 | 100% | 100% |
| `XC1010642_Rwanda.mp3` | 4:56 | 124 | 100% | 100% |
| `XC1073923_Ethiopia.wav` | 2:16 | 69 | 100% | 100% |
| `XC1170114_South-Africa.wav` | 4:19 | 159 | 99% | 100% |
| `XC1174845_France.mp3` | 2:00 | 97 | 100% | 100% |
| `XC1180150_Estonia.wav` | 3:21 | 104 | 99% | 100% |
| `XC1181792_Spain.mp3` | 4:39 | 187 | 100% | 100% |
| `XC1184153_Spain.wav` | 1:03 | 1 | 100% | 100% |
| `XC1184684_Austria.wav` | 3:20 | 106 | 100% | 100% |
| `XC1185879_Sweden.wav` | 1:29 | 54 | 100% | 100% |
| `XC692092_Kenya.mp3` | 1:02 | 46 | 100% | 100% |
| `XC782384_Sierra-Leone.wav` | 1:07 | 41 | 100% | 98% |
| `XC889825_Zimbabwe.wav` | 1:46 | 52 | 98% | 100% |

**INT8**

| Recording | Length | Baseline | `int8.onnx` | `int8.tflite` |
| :--- | ---: | ---: | ---: | ---: |
| `test.wav` | 5:00 | 109 | 99% | 100% |
| `XC1010642_Rwanda.mp3` | 4:56 | 124 | 82% | 79% |
| `XC1073923_Ethiopia.wav` | 2:16 | 69 | 80% | 74% |
| `XC1170114_South-Africa.wav` | 4:19 | 159 | 78% | 61% |
| `XC1174845_France.mp3` | 2:00 | 97 | 79% | 84% |
| `XC1180150_Estonia.wav` | 3:21 | 104 | 80% | 77% |
| `XC1181792_Spain.mp3` | 4:39 | 187 | 82% | 52% |
| `XC1184153_Spain.wav` | 1:03 | 1 | 100% | 100% |
| `XC1184684_Austria.wav` | 3:20 | 106 | 76% | 69% |
| `XC1185879_Sweden.wav` | 1:29 | 54 | 93% | 78% |
| `XC692092_Kenya.mp3` | 1:02 | 46 | 85% | 65% |
| `XC782384_Sierra-Leone.wav` | 1:07 | 41 | 85% | 56% |
| `XC889825_Zimbabwe.wav` | 1:46 | 52 | 90% | 81% |

### Confidence and timing

| Model | Mean Conf Δ | Max Conf Δ | Load (s) | Detect (s) |
| :--- | ---: | ---: | ---: | ---: |
| `Birds-With-Labels_fp32.pt` | — | — | 0.3 | 79.9 |
| `Birds-With-Labels_fp16.onnx` | 0.0016 | 0.1619 | 0.4 | 127.6 |
| `Birds-With-Labels_fp16.tflite` | 0.0007 | 0.0823 | 0.3 | 566.8 |
| `Birds-With-Labels_int8.onnx` | 0.0389 | 0.2685 | 0.2 | 121.7 |
| `Birds-With-Labels_int8.tflite` | 0.0634 | 0.4869 | 0.2 | 151.9 |

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

### Birds-With-Labels_fp32.pt

!!! info "Baseline"
    All other formats are compared against this model. Its detections define what a correct result looks like.

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `????` | 7 | 7 |
| `akepa1` | 9 | 9 |
| `aldfly` | 1 | 1 |
| `amecro` | 1 | 1 |
| `amegfi` | 6 | 6 |
| `amepip` | 1 | 1 |
| `amerob` | 5 | 5 |
| `apapan` | 61 | 61 |
| `balori` | 1 | 1 |
| `bkcchi` | 1 | 1 |
| `blujay` | 3 | 3 |
| `bubgro2` | 1 | 1 |
| `butwoo1` | 1 | 1 |
| `calqua` | 4 | 4 |
| `cangoo` | 6 | 6 |
| `chukar` | 1 | 1 |
| `comgra` | 1 | 1 |
| `comyel` | 4 | 4 |
| `elepai` | 16 | 16 |
| `ercfra` | 46 | 46 |
| `eursta` | 2 | 2 |
| `foxspa` | 1 | 1 |
| `gcrfin` | 1 | 1 |
| `grfdov1` | 4 | 4 |
| `grycat` | 56 | 56 |
| `hawama` | 188 | 188 |
| `hawcre` | 8 | 8 |
| `hawpet1` | 10 | 10 |
| `houfin` | 82 | 82 |
| `houwre` | 2 | 2 |
| `iiwi` | 60 | 60 |
| `jabwar` | 42 | 42 |
| `meapar` | 4 | 4 |
| `melthr` | 23 | 23 |
| `norcar` | 10 | 10 |
| `norfli` | 2 | 2 |
| `omao` | 55 | 55 |
| `ovenbi1` | 2 | 2 |
| `palila` | 2 | 2 |
| `reblei` | 190 | 190 |
| `rebmac2` | 2 | 2 |
| `reevir1` | 1 | 1 |
| `rewbla` | 10 | 10 |
| `ruboro1` | 2 | 2 |
| `scatan` | 1 | 1 |
| `scrpih1` | 2 | 2 |
| `skylar` | 56 | 56 |
| `sonspa` | 3 | 3 |
| `stejay` | 8 | 8 |
| `thlwre1` | 1 | 1 |
| `tuftit` | 1 | 1 |
| `veery` | 3 | 3 |
| `warwhe1` | 126 | 126 |
| `whtspa` | 5 | 5 |
| `wiltur` | 1 | 1 |
| `woothr` | 4 | 4 |
| `yefcan` | 1 | 1 |
| `yelwar` | 1 | 1 |
| `yetvir` | 1 | 1 |

File size: 49.2 MiB. Load time: 0.3 s. Detection time: 79.9 s.

---

### Birds-With-Labels_fp16.onnx

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `????` | 7 | 7 |
| `akepa1` | 9 | 9 |
| `aldfly` | 1 | 1 |
| `amecro` | 1 | 1 |
| `amegfi` | 6 | 6 |
| `amepip` | 1 | 1 |
| `amerob` | 5 | 5 |
| `apapan` | 61 | 61 |
| `balori` | 1 | 1 |
| `bkcchi` | 1 | 1 |
| `blujay` | 3 | 3 |
| `bubgro2` | 1 | 1 |
| `butwoo1` | 1 | 1 |
| `calqua` | 4 | 4 |
| `cangoo` | 6 | 6 |
| `chukar` | 1 | 1 |
| `comgra` | 1 | 1 |
| `comyel` | 4 | 4 |
| `elepai` | 16 | 16 |
| `ercfra` | 46 | 45 |
| `eursta` | 2 | 2 |
| `foxspa` | 1 | 1 |
| `gcrfin` | 1 | 1 |
| `grfdov1` | 4 | 4 |
| `grycat` | 56 | 56 |
| `hawama` | 188 | 188 |
| `hawcre` | 8 | 8 |
| `hawpet1` | 10 | 10 |
| `houfin` | 82 | 82 |
| `houwre` | 2 | 2 |
| `iiwi` | 60 | 59 |
| `jabwar` | 42 | 42 |
| `meapar` | 4 | 4 |
| `melthr` | 23 | 23 |
| `norcar` | 10 | 10 |
| `norfli` | 2 | 2 |
| `omao` | 55 | 55 |
| `ovenbi1` | 2 | 2 |
| `palila` | 2 | 2 |
| `reblei` | 190 | 190 |
| `rebmac2` | 2 | 2 |
| `reevir1` | 1 | 1 |
| `rewbla` | 10 | 10 |
| `ruboro1` | 2 | 2 |
| `scatan` | 1 | 1 |
| `scrpih1` | 2 | 2 |
| `skylar` | 56 | 56 |
| `sonspa` | 3 | 3 |
| `stejay` | 8 | 8 |
| `thlwre1` | 1 | 1 |
| `tuftit` | 1 | 1 |
| `veery` | 3 | 3 |
| `warwhe1` | 126 | 126 |
| `whtspa` | 5 | 5 |
| `wiltur` | 1 | 1 |
| `woothr` | 4 | 4 |
| `yefcan` | 1 | 1 |
| `yelwar` | 1 | 1 |
| `yetvir` | 1 | 1 |

!!! success "PASS"
    Matched 1146 of 1149 baseline detections (99.7%), with 3 missed and 1 extra. Mean IoU: 0.997. Mean Conf Δ: 0.0016. Mean start-time shift: 0.002 s.

File size: 48.8 MiB. Load time: 0.4 s. Detection time: 127.6 s.

---

### Birds-With-Labels_fp16.tflite

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `????` | 7 | 7 |
| `akepa1` | 9 | 9 |
| `aldfly` | 1 | 1 |
| `amecro` | 1 | 1 |
| `amegfi` | 6 | 6 |
| `amepip` | 1 | 1 |
| `amerob` | 5 | 5 |
| `apapan` | 61 | 61 |
| `balori` | 1 | 1 |
| `bkcchi` | 1 | 1 |
| `blujay` | 3 | 3 |
| `bubgro2` | 1 | 1 |
| `butwoo1` | 1 | 1 |
| `calqua` | 4 | 4 |
| `cangoo` | 6 | 6 |
| `chukar` | 1 | 1 |
| `comgra` | 1 | 1 |
| `comyel` | 4 | 4 |
| `elepai` | 16 | 16 |
| `ercfra` | 46 | 46 |
| `eursta` | 2 | 2 |
| `foxspa` | 1 | 1 |
| `gcrfin` | 1 | 1 |
| `grfdov1` | 4 | 4 |
| `grycat` | 56 | 56 |
| `hawama` | 188 | 188 |
| `hawcre` | 8 | 8 |
| `hawpet1` | 10 | 10 |
| `houfin` | 82 | 83 |
| `houwre` | 2 | 2 |
| `iiwi` | 60 | 60 |
| `jabwar` | 42 | 42 |
| `meapar` | 4 | 4 |
| `melthr` | 23 | 23 |
| `norcar` | 10 | 10 |
| `norfli` | 2 | 2 |
| `omao` | 55 | 55 |
| `ovenbi1` | 2 | 2 |
| `palila` | 2 | 2 |
| `reblei` | 190 | 190 |
| `rebmac2` | 2 | 2 |
| `reevir1` | 1 | 1 |
| `rewbla` | 10 | 10 |
| `ruboro1` | 2 | 1 |
| `scatan` | 1 | 1 |
| `scrpih1` | 2 | 2 |
| `skylar` | 56 | 56 |
| `sonspa` | 3 | 3 |
| `stejay` | 8 | 8 |
| `thlwre1` | 1 | 1 |
| `tuftit` | 1 | 1 |
| `veery` | 3 | 3 |
| `warwhe1` | 126 | 126 |
| `whtspa` | 5 | 5 |
| `wiltur` | 1 | 1 |
| `woothr` | 4 | 4 |
| `yefcan` | 1 | 1 |
| `yelwar` | 1 | 1 |
| `yetvir` | 1 | 1 |

!!! success "PASS"
    Matched 1148 of 1149 baseline detections (99.9%), with 1 missed and 1 extra. Mean IoU: 0.999. Mean Conf Δ: 0.0007. Mean start-time shift: 0.001 s.

File size: 49.1 MiB. Load time: 0.3 s. Detection time: 566.8 s.

---

### Birds-With-Labels_int8.onnx

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `????` | 7 | 7 |
| `akepa1` | 9 | 8 |
| `aldfly` | 1 | 0 |
| `amecro` | 1 | 1 |
| `amegfi` | 6 | 3 |
| `amepip` | 1 | 1 |
| `amerob` | 5 | 3 |
| `apapan` | 61 | 59 |
| `balori` | 1 | 0 |
| `bkcchi` | 1 | 1 |
| `blujay` | 3 | 3 |
| `bubgro2` | 1 | 1 |
| `butwoo1` | 1 | 1 |
| `calqua` | 4 | 3 |
| `cangoo` | 6 | 5 |
| `chukar` | 1 | 0 |
| `comgra` | 1 | 1 |
| `comyel` | 4 | 3 |
| `elepai` | 16 | 17 |
| `ercfra` | 46 | 43 |
| `eursta` | 2 | 1 |
| `foxspa` | 1 | 1 |
| `gcrfin` | 1 | 1 |
| `grfdov1` | 4 | 3 |
| `grycat` | 56 | 50 |
| `hawama` | 188 | 162 |
| `hawcre` | 8 | 6 |
| `hawpet1` | 10 | 6 |
| `houfin` | 82 | 71 |
| `houwre` | 2 | 2 |
| `iiwi` | 60 | 50 |
| `jabwar` | 42 | 43 |
| `meapar` | 4 | 4 |
| `melthr` | 23 | 16 |
| `norcar` | 10 | 9 |
| `norfli` | 2 | 2 |
| `omao` | 55 | 56 |
| `ovenbi1` | 2 | 5 |
| `palila` | 2 | 1 |
| `reblei` | 190 | 196 |
| `rebmac2` | 2 | 3 |
| `reevir1` | 1 | 1 |
| `rewbla` | 10 | 11 |
| `ruboro1` | 2 | 2 |
| `scatan` | 1 | 2 |
| `scrpih1` | 2 | 2 |
| `skylar` | 56 | 51 |
| `sonspa` | 3 | 2 |
| `stejay` | 8 | 4 |
| `thlwre1` | 1 | 1 |
| `tuftit` | 1 | 1 |
| `veery` | 3 | 3 |
| `warwhe1` | 126 | 113 |
| `whcspa` | 0 | 1 |
| `whtspa` | 5 | 5 |
| `wiltur` | 1 | 1 |
| `woothr` | 4 | 4 |
| `yefcan` | 1 | 1 |
| `yelwar` | 1 | 1 |
| `yetvir` | 1 | 1 |

!!! warning "WARN"
    Matched 956 of 1149 baseline detections (83.2%), with 193 missed and 99 extra. Mean IoU: 0.936. Mean Conf Δ: 0.0389. Mean start-time shift: 0.032 s.

File size: 38.8 MiB. Load time: 0.2 s. Detection time: 121.7 s.

---

### Birds-With-Labels_int8.tflite

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `????` | 7 | 5 |
| `akepa1` | 9 | 10 |
| `aldfly` | 1 | 1 |
| `amecro` | 1 | 2 |
| `amegfi` | 6 | 9 |
| `amepip` | 1 | 1 |
| `amerob` | 5 | 2 |
| `apapan` | 61 | 77 |
| `balori` | 1 | 0 |
| `bkcchi` | 1 | 2 |
| `blujay` | 3 | 4 |
| `bubgro2` | 1 | 1 |
| `bucmot4` | 0 | 2 |
| `butwoo1` | 1 | 1 |
| `calqua` | 4 | 2 |
| `cangoo` | 6 | 5 |
| `chukar` | 1 | 0 |
| `comgra` | 1 | 1 |
| `comwax` | 0 | 1 |
| `comyel` | 4 | 6 |
| `eastow` | 0 | 1 |
| `eawpew` | 0 | 1 |
| `elepai` | 16 | 17 |
| `ercfra` | 46 | 37 |
| `eursta` | 2 | 1 |
| `foxspa` | 1 | 1 |
| `gcrfin` | 1 | 1 |
| `grfdov1` | 4 | 2 |
| `grycat` | 56 | 52 |
| `hauthr1` | 0 | 2 |
| `hawama` | 188 | 122 |
| `hawcre` | 8 | 10 |
| `hawpet1` | 10 | 7 |
| `houfin` | 82 | 58 |
| `houwre` | 2 | 2 |
| `iiwi` | 60 | 43 |
| `jabwar` | 42 | 48 |
| `meapar` | 4 | 4 |
| `melthr` | 23 | 19 |
| `norcar` | 10 | 10 |
| `norfli` | 2 | 2 |
| `omao` | 55 | 82 |
| `ovenbi1` | 2 | 11 |
| `palila` | 2 | 1 |
| `plupig2` | 0 | 1 |
| `reblei` | 190 | 218 |
| `rebmac2` | 2 | 3 |
| `rebwoo` | 0 | 1 |
| `reevir1` | 1 | 2 |
| `rewbla` | 10 | 13 |
| `ruboro1` | 2 | 0 |
| `scatan` | 1 | 0 |
| `scrpih1` | 2 | 2 |
| `skylar` | 56 | 44 |
| `sonspa` | 3 | 3 |
| `stejay` | 8 | 1 |
| `swaspa` | 0 | 1 |
| `thlwre1` | 1 | 3 |
| `tuftit` | 1 | 1 |
| `veery` | 3 | 4 |
| `warwhe1` | 126 | 117 |
| `whcspa` | 0 | 1 |
| `whtspa` | 5 | 5 |
| `wiltur` | 1 | 1 |
| `woothr` | 4 | 4 |
| `yefcan` | 1 | 1 |
| `yelwar` | 1 | 1 |
| `yetvir` | 1 | 1 |

!!! warning "WARN"
    Matched 824 of 1149 baseline detections (71.7%), with 325 missed and 270 extra. Mean IoU: 0.903. Mean Conf Δ: 0.0634. Mean start-time shift: 0.035 s.

File size: 38.8 MiB. Load time: 0.2 s. Detection time: 151.9 s.

---
