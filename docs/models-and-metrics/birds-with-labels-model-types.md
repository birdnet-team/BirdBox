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
    Generated on 2026-11-12 01:42. Audio: 13 recordings, 36 minutes (see [Test recordings](#test-recordings)), species mapping: `Birds-With-Labels`, confidence threshold: `0.2`, baseline: `Birds-With-Labels_fp32.pt` (692 detections).

### At a glance

| Model | Format | Size | Detections | Verdict |
| :--- | :--- | ---: | ---: | :---: |
| `Birds-With-Labels_fp32.pt` | `.pt` | 49.7 MiB | 692 | _baseline_ |
| `Birds-With-Labels_fp16.onnx` | `.onnx` | 49.4 MiB | 695 | :material-check-circle: PASS |
| `Birds-With-Labels_fp16.tflite` | `.tflite` | 49.7 MiB | 692 | :material-check-circle: PASS |
| `Birds-With-Labels_int8.onnx` | `.onnx` | 39.8 MiB | 684 | :material-alert-circle: WARN |
| `Birds-With-Labels_int8.tflite` | `.tflite` | 39.9 MiB | 634 | :material-alert-circle: WARN |

### Detection matching

| Model | Matched | Missed | Extra | Match Rate | Mean IoU |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `Birds-With-Labels_fp32.pt` | — | — | — | _baseline_ | — |
| `Birds-With-Labels_fp16.onnx` | 690 | 2 | 5 | 99.7% | 0.997 |
| `Birds-With-Labels_fp16.tflite` | 691 | 1 | 1 | 99.9% | 0.999 |
| `Birds-With-Labels_int8.onnx` | 576 | 116 | 108 | 83.2% | 0.950 |
| `Birds-With-Labels_int8.tflite` | 570 | 122 | 64 | 82.4% | 0.952 |

### Match rate per recording

Share of the baseline detections in each recording that the other formats reproduce. A format that only fails on some recordings shows up here.

**FP16**

| Recording | Length | Baseline | `fp16.onnx` | `fp16.tflite` |
| :--- | ---: | ---: | ---: | ---: |
| `test.wav` | 5:00 | 95 | 100% | 100% |
| `XC1010642_Rwanda.mp3` | 4:56 | 75 | 100% | 100% |
| `XC1073923_Ethiopia.wav` | 2:16 | 40 | 100% | 100% |
| `XC1170114_South-Africa.wav` | 4:19 | 99 | 100% | 100% |
| `XC1174845_France.mp3` | 2:00 | 76 | 100% | 100% |
| `XC1180150_Estonia.wav` | 3:21 | 42 | 100% | 100% |
| `XC1181792_Spain.mp3` | 4:39 | 89 | 99% | 100% |
| `XC1184153_Spain.wav` | 1:03 | 2 | 100% | 100% |
| `XC1184684_Austria.wav` | 3:20 | 71 | 100% | 100% |
| `XC1185879_Sweden.wav` | 1:29 | 52 | 100% | 100% |
| `XC692092_Kenya.mp3` | 1:02 | 11 | 100% | 100% |
| `XC782384_Sierra-Leone.wav` | 1:07 | 13 | 92% | 92% |
| `XC889825_Zimbabwe.wav` | 1:46 | 27 | 100% | 100% |

**INT8**

| Recording | Length | Baseline | `int8.onnx` | `int8.tflite` |
| :--- | ---: | ---: | ---: | ---: |
| `test.wav` | 5:00 | 95 | 96% | 98% |
| `XC1010642_Rwanda.mp3` | 4:56 | 75 | 80% | 73% |
| `XC1073923_Ethiopia.wav` | 2:16 | 40 | 80% | 78% |
| `XC1170114_South-Africa.wav` | 4:19 | 99 | 87% | 84% |
| `XC1174845_France.mp3` | 2:00 | 76 | 87% | 92% |
| `XC1180150_Estonia.wav` | 3:21 | 42 | 79% | 79% |
| `XC1181792_Spain.mp3` | 4:39 | 89 | 78% | 79% |
| `XC1184153_Spain.wav` | 1:03 | 2 | 100% | 50% |
| `XC1184684_Austria.wav` | 3:20 | 71 | 75% | 72% |
| `XC1185879_Sweden.wav` | 1:29 | 52 | 85% | 83% |
| `XC692092_Kenya.mp3` | 1:02 | 11 | 91% | 91% |
| `XC782384_Sierra-Leone.wav` | 1:07 | 13 | 62% | 69% |
| `XC889825_Zimbabwe.wav` | 1:46 | 27 | 81% | 78% |

### Confidence and timing

| Model | Mean Conf Δ | Max Conf Δ | Load (s) | Detect (s) |
| :--- | ---: | ---: | ---: | ---: |
| `Birds-With-Labels_fp32.pt` | — | — | 0.5 | 166.4 |
| `Birds-With-Labels_fp16.onnx` | 0.0024 | 0.0763 | 0.5 | 379.0 |
| `Birds-With-Labels_fp16.tflite` | 0.0012 | 0.0774 | 0.6 | 1011.1 |
| `Birds-With-Labels_int8.onnx` | 0.0421 | 0.2539 | 0.4 | 337.6 |
| `Birds-With-Labels_int8.tflite` | 0.0409 | 0.2681 | 0.4 | 371.8 |

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
| `????` | 57 | 57 |
| `akepa1` | 4 | 4 |
| `amepip` | 2 | 2 |
| `amerob` | 1 | 1 |
| `apapan` | 6 | 6 |
| `apptet1` | 2 | 2 |
| `arcnew1` | 3 | 3 |
| `ashcus2` | 3 | 3 |
| `astgna1` | 1 | 1 |
| `ausmag2` | 1 | 1 |
| `bawcow4` | 5 | 5 |
| `bbfgle1` | 1 | 1 |
| `bknsti` | 28 | 28 |
| `blackc1` | 2 | 2 |
| `blagro1` | 11 | 11 |
| `blapar1` | 1 | 1 |
| `blbman1` | 1 | 1 |
| `bllser1` | 1 | 1 |
| `blutit` | 1 | 1 |
| `brncac1` | 1 | 1 |
| `bufpar` | 3 | 3 |
| `cangoo` | 1 | 1 |
| `chacha1` | 4 | 4 |
| `chswar` | 1 | 1 |
| `cintyr1` | 8 | 8 |
| `coatit2` | 1 | 1 |
| `comcha` | 22 | 22 |
| `comchi1` | 14 | 14 |
| `comcuc` | 1 | 1 |
| `comnig1` | 1 | 1 |
| `comros` | 1 | 1 |
| `comyel` | 1 | 1 |
| `cowpig1` | 9 | 9 |
| `crbthr1` | 3 | 3 |
| `cregal1` | 1 | 1 |
| `crehor1` | 21 | 21 |
| `eardov1` | 1 | 1 |
| `easros1` | 3 | 3 |
| `elepai` | 2 | 2 |
| `ercfra` | 11 | 11 |
| `eupfly1` | 1 | 1 |
| `eurbla` | 5 | 5 |
| `eurjay1` | 1 | 1 |
| `eurrob1` | 3 | 3 |
| `firecr1` | 11 | 11 |
| `gcrwar` | 1 | 1 |
| `gobsal1` | 9 | 9 |
| `gogwoo1` | 3 | 3 |
| `grekis` | 2 | 2 |
| `gretit1` | 1 | 1 |
| `grwtyr1` | 31 | 31 |
| `grycat` | 19 | 19 |
| `gycwor1` | 1 | 1 |
| `gyhlov1` | 3 | 3 |
| `hawama` | 56 | 56 |
| `hawcre` | 5 | 5 |
| `hawfin` | 1 | 1 |
| `hawhaw` | 1 | 1 |
| `hawpet1` | 1 | 1 |
| `hoocro1` | 4 | 4 |
| `houfin` | 8 | 8 |
| `houspa` | 2 | 2 |
| `houwre` | 3 | 3 |
| `iiwi` | 11 | 11 |
| `jabwar` | 26 | 26 |
| `leswhi4` | 1 | 1 |
| `lotjae` | 1 | 1 |
| `madbul1` | 20 | 20 |
| `madcou1` | 7 | 7 |
| `madhoo1` | 8 | 8 |
| `mapfly1` | 11 | 11 |
| `masgna1` | 2 | 2 |
| `mitpar` | 2 | 2 |
| `monpar` | 33 | 33 |
| `nelwea1` | 1 | 1 |
| `norcar` | 2 | 2 |
| `omao` | 11 | 11 |
| `palila` | 1 | 1 |
| `pigdov1` | 3 | 3 |
| `pluibi1` | 1 | 1 |
| `polpar1` | 1 | 1 |
| `reblei` | 27 | 27 |
| `redfod1` | 3 | 3 |
| `redjun` | 4 | 4 |
| `rewbla` | 1 | 1 |
| `rubpep1` | 5 | 5 |
| `rucspa1` | 3 | 3 |
| `rufhor2` | 4 | 4 |
| `saffin` | 1 | 1 |
| `sander` | 11 | 11 |
| `saytan1` | 1 | 1 |
| `sinhon1` | 1 | 1 |
| `slathr2` | 3 | 3 |
| `slbblc1` | 1 | 1 |
| `smbela1` | 3 | 3 |
| `sobtyr1` | 1 | 1 |
| `sonspa` | 1 | 1 |
| `sonthr1` | 1 | 1 |
| `soulap1` | 2 | 2 |
| `souscr1` | 4 | 4 |
| `spbpuf1` | 1 | 1 |
| `spwpig3` | 2 | 2 |
| `stcspi2` | 3 | 3 |
| `suifly1` | 18 | 18 |
| `tui1` | 11 | 11 |
| `twbwar1` | 3 | 3 |
| `vartit1` | 1 | 1 |
| `warwhe1` | 7 | 7 |
| `whhvan1` | 2 | 2 |
| `whtspa` | 1 | 1 |
| `winwre4` | 8 | 8 |
| `wlwwar` | 2 | 2 |
| `woothr` | 1 | 1 |

File size: 49.7 MiB. Load time: 0.5 s. Detection time: 166.4 s.

---

### Birds-With-Labels_fp16.onnx

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `????` | 57 | 57 |
| `akepa1` | 4 | 4 |
| `amepip` | 2 | 2 |
| `amerob` | 1 | 1 |
| `apapan` | 6 | 6 |
| `apptet1` | 2 | 2 |
| `arcnew1` | 3 | 3 |
| `ashcus2` | 3 | 3 |
| `astgna1` | 1 | 1 |
| `ausmag2` | 1 | 1 |
| `bawcow4` | 5 | 5 |
| `bbfgle1` | 1 | 1 |
| `bknsti` | 28 | 28 |
| `blackc1` | 2 | 2 |
| `blagro1` | 11 | 11 |
| `blapar1` | 1 | 1 |
| `blbman1` | 1 | 1 |
| `bllser1` | 1 | 1 |
| `blutit` | 1 | 1 |
| `brncac1` | 1 | 1 |
| `bufpar` | 3 | 3 |
| `cangoo` | 1 | 1 |
| `chacha1` | 4 | 4 |
| `chswar` | 1 | 1 |
| `cintyr1` | 8 | 8 |
| `coatit2` | 1 | 1 |
| `comcha` | 22 | 22 |
| `comchi1` | 14 | 14 |
| `comcuc` | 1 | 1 |
| `comnig1` | 1 | 1 |
| `comros` | 1 | 1 |
| `comyel` | 1 | 1 |
| `cowpig1` | 9 | 9 |
| `crbthr1` | 3 | 3 |
| `cregal1` | 1 | 1 |
| `crehor1` | 21 | 21 |
| `eardov1` | 1 | 1 |
| `easros1` | 3 | 2 |
| `elepai` | 2 | 2 |
| `ercfra` | 11 | 11 |
| `eupfly1` | 1 | 1 |
| `eurbla` | 5 | 5 |
| `eurjay1` | 1 | 1 |
| `eurrob1` | 3 | 3 |
| `firecr1` | 11 | 11 |
| `gcrwar` | 1 | 1 |
| `gobsal1` | 9 | 9 |
| `gogwoo1` | 3 | 3 |
| `grekis` | 2 | 2 |
| `gretit1` | 1 | 1 |
| `grwtyr1` | 31 | 33 |
| `grycat` | 19 | 19 |
| `gycwor1` | 1 | 1 |
| `gyhlov1` | 3 | 3 |
| `hawama` | 56 | 56 |
| `hawcre` | 5 | 5 |
| `hawfin` | 1 | 1 |
| `hawhaw` | 1 | 1 |
| `hawpet1` | 1 | 1 |
| `hoocro1` | 4 | 4 |
| `houfin` | 8 | 8 |
| `houspa` | 2 | 2 |
| `houwre` | 3 | 3 |
| `iiwi` | 11 | 12 |
| `jabwar` | 26 | 26 |
| `leswhi4` | 1 | 1 |
| `lotjae` | 1 | 1 |
| `madbul1` | 20 | 20 |
| `madcou1` | 7 | 7 |
| `madhoo1` | 8 | 8 |
| `mapfly1` | 11 | 11 |
| `masgna1` | 2 | 2 |
| `mitpar` | 2 | 2 |
| `monpar` | 33 | 34 |
| `nelwea1` | 1 | 1 |
| `norcar` | 2 | 2 |
| `omao` | 11 | 11 |
| `palila` | 1 | 1 |
| `pigdov1` | 3 | 3 |
| `pluibi1` | 1 | 1 |
| `polpar1` | 1 | 1 |
| `reblei` | 27 | 27 |
| `redfod1` | 3 | 3 |
| `redjun` | 4 | 4 |
| `rewbla` | 1 | 1 |
| `rubpep1` | 5 | 5 |
| `rucspa1` | 3 | 3 |
| `rufhor2` | 4 | 4 |
| `saffin` | 1 | 1 |
| `sander` | 11 | 11 |
| `saytan1` | 1 | 1 |
| `sinhon1` | 1 | 1 |
| `slathr2` | 3 | 3 |
| `slbblc1` | 1 | 1 |
| `smbela1` | 3 | 3 |
| `sobtyr1` | 1 | 1 |
| `sonspa` | 1 | 1 |
| `sonthr1` | 1 | 1 |
| `soulap1` | 2 | 2 |
| `souscr1` | 4 | 4 |
| `spbpuf1` | 1 | 1 |
| `spwpig3` | 2 | 2 |
| `stcspi2` | 3 | 3 |
| `suifly1` | 18 | 18 |
| `tui1` | 11 | 11 |
| `twbwar1` | 3 | 3 |
| `vartit1` | 1 | 1 |
| `warwhe1` | 7 | 7 |
| `whhvan1` | 2 | 2 |
| `whtspa` | 1 | 1 |
| `winwre4` | 8 | 8 |
| `wlwwar` | 2 | 2 |
| `woothr` | 1 | 1 |

!!! success "PASS"
    Matched 690 of 692 baseline detections (99.7%), with 2 missed and 5 extra. Mean IoU: 0.997. Mean Conf Δ: 0.0024. Mean start-time shift: 0.001 s.

File size: 49.4 MiB. Load time: 0.5 s. Detection time: 379.0 s.

---

### Birds-With-Labels_fp16.tflite

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `????` | 57 | 57 |
| `akepa1` | 4 | 4 |
| `amepip` | 2 | 2 |
| `amerob` | 1 | 1 |
| `apapan` | 6 | 6 |
| `apptet1` | 2 | 2 |
| `arcnew1` | 3 | 3 |
| `ashcus2` | 3 | 3 |
| `astgna1` | 1 | 1 |
| `ausmag2` | 1 | 1 |
| `bawcow4` | 5 | 5 |
| `bbfgle1` | 1 | 1 |
| `bknsti` | 28 | 28 |
| `blackc1` | 2 | 2 |
| `blagro1` | 11 | 11 |
| `blapar1` | 1 | 1 |
| `blbman1` | 1 | 1 |
| `bllser1` | 1 | 1 |
| `blutit` | 1 | 1 |
| `brncac1` | 1 | 1 |
| `bufpar` | 3 | 3 |
| `cangoo` | 1 | 1 |
| `chacha1` | 4 | 4 |
| `chswar` | 1 | 1 |
| `cintyr1` | 8 | 8 |
| `coatit2` | 1 | 1 |
| `comcha` | 22 | 22 |
| `comchi1` | 14 | 14 |
| `comcuc` | 1 | 1 |
| `comnig1` | 1 | 1 |
| `comros` | 1 | 1 |
| `comyel` | 1 | 1 |
| `cowpig1` | 9 | 9 |
| `crbthr1` | 3 | 3 |
| `cregal1` | 1 | 1 |
| `crehor1` | 21 | 21 |
| `eardov1` | 1 | 1 |
| `easros1` | 3 | 2 |
| `elepai` | 2 | 2 |
| `ercfra` | 11 | 11 |
| `eupfly1` | 1 | 1 |
| `eurbla` | 5 | 5 |
| `eurjay1` | 1 | 1 |
| `eurrob1` | 3 | 3 |
| `firecr1` | 11 | 11 |
| `gcrwar` | 1 | 1 |
| `gobsal1` | 9 | 9 |
| `gogwoo1` | 3 | 3 |
| `grekis` | 2 | 2 |
| `gretit1` | 1 | 1 |
| `grwtyr1` | 31 | 31 |
| `grycat` | 19 | 19 |
| `gycwor1` | 1 | 1 |
| `gyhlov1` | 3 | 3 |
| `hawama` | 56 | 56 |
| `hawcre` | 5 | 5 |
| `hawfin` | 1 | 1 |
| `hawhaw` | 1 | 1 |
| `hawpet1` | 1 | 1 |
| `hoocro1` | 4 | 4 |
| `houfin` | 8 | 8 |
| `houspa` | 2 | 2 |
| `houwre` | 3 | 3 |
| `iiwi` | 11 | 11 |
| `jabwar` | 26 | 26 |
| `leswhi4` | 1 | 1 |
| `lotjae` | 1 | 1 |
| `madbul1` | 20 | 20 |
| `madcou1` | 7 | 7 |
| `madhoo1` | 8 | 8 |
| `mapfly1` | 11 | 11 |
| `masgna1` | 2 | 2 |
| `mitpar` | 2 | 2 |
| `monpar` | 33 | 33 |
| `nelwea1` | 1 | 1 |
| `norcar` | 2 | 2 |
| `omao` | 11 | 11 |
| `palila` | 1 | 1 |
| `pigdov1` | 3 | 3 |
| `pluibi1` | 1 | 1 |
| `polpar1` | 1 | 1 |
| `reblei` | 27 | 27 |
| `redfod1` | 3 | 3 |
| `redjun` | 4 | 4 |
| `rewbla` | 1 | 1 |
| `rubpep1` | 5 | 5 |
| `rucspa1` | 3 | 3 |
| `rufhor2` | 4 | 5 |
| `saffin` | 1 | 1 |
| `sander` | 11 | 11 |
| `saytan1` | 1 | 1 |
| `sinhon1` | 1 | 1 |
| `slathr2` | 3 | 3 |
| `slbblc1` | 1 | 1 |
| `smbela1` | 3 | 3 |
| `sobtyr1` | 1 | 1 |
| `sonspa` | 1 | 1 |
| `sonthr1` | 1 | 1 |
| `soulap1` | 2 | 2 |
| `souscr1` | 4 | 4 |
| `spbpuf1` | 1 | 1 |
| `spwpig3` | 2 | 2 |
| `stcspi2` | 3 | 3 |
| `suifly1` | 18 | 18 |
| `tui1` | 11 | 11 |
| `twbwar1` | 3 | 3 |
| `vartit1` | 1 | 1 |
| `warwhe1` | 7 | 7 |
| `whhvan1` | 2 | 2 |
| `whtspa` | 1 | 1 |
| `winwre4` | 8 | 8 |
| `wlwwar` | 2 | 2 |
| `woothr` | 1 | 1 |

!!! success "PASS"
    Matched 691 of 692 baseline detections (99.9%), with 1 missed and 1 extra. Mean IoU: 0.999. Mean Conf Δ: 0.0012. Mean start-time shift: 0.000 s.

File size: 49.7 MiB. Load time: 0.6 s. Detection time: 1011.1 s.

---

### Birds-With-Labels_int8.onnx

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `????` | 57 | 61 |
| `akepa1` | 4 | 3 |
| `amepip` | 2 | 3 |
| `amerob` | 1 | 0 |
| `apapan` | 6 | 2 |
| `apptet1` | 2 | 2 |
| `arcnew1` | 3 | 1 |
| `ashcus2` | 3 | 3 |
| `astgna1` | 1 | 1 |
| `ausmag2` | 1 | 1 |
| `barant1` | 0 | 1 |
| `bawcow4` | 5 | 5 |
| `bbfgle1` | 1 | 1 |
| `bknsti` | 28 | 24 |
| `blackc1` | 2 | 2 |
| `blagro1` | 11 | 9 |
| `blapar1` | 1 | 1 |
| `blbgra1` | 0 | 1 |
| `blbman1` | 1 | 1 |
| `bllser1` | 1 | 3 |
| `blutit` | 1 | 1 |
| `brncac1` | 1 | 1 |
| `bufpar` | 3 | 2 |
| `cangoo` | 1 | 1 |
| `chacha1` | 4 | 3 |
| `chswar` | 1 | 1 |
| `cintyr1` | 8 | 8 |
| `citspa1` | 0 | 1 |
| `coatit2` | 1 | 3 |
| `comcha` | 22 | 22 |
| `comchi1` | 14 | 15 |
| `comcuc` | 1 | 1 |
| `comjer1` | 0 | 1 |
| `comnig1` | 1 | 0 |
| `comros` | 1 | 0 |
| `comwax` | 0 | 1 |
| `comyel` | 1 | 1 |
| `cowpig1` | 9 | 9 |
| `crbthr1` | 3 | 2 |
| `cregal1` | 1 | 1 |
| `crehor1` | 21 | 16 |
| `eardov1` | 1 | 0 |
| `easros1` | 3 | 3 |
| `elepai` | 2 | 1 |
| `ercfra` | 11 | 8 |
| `eupfly1` | 1 | 1 |
| `eurbla` | 5 | 5 |
| `eurcur` | 0 | 1 |
| `eurjay1` | 1 | 1 |
| `eurrob1` | 3 | 5 |
| `firecr1` | 11 | 10 |
| `gcrwar` | 1 | 2 |
| `gobsal1` | 9 | 7 |
| `gogwoo1` | 3 | 2 |
| `grekis` | 2 | 2 |
| `gretit1` | 1 | 1 |
| `grwtyr1` | 31 | 37 |
| `grycat` | 19 | 25 |
| `gycwor1` | 1 | 2 |
| `gyhlov1` | 3 | 5 |
| `hawama` | 56 | 56 |
| `hawcre` | 5 | 1 |
| `hawfin` | 1 | 1 |
| `hawhaw` | 1 | 1 |
| `hawpet1` | 1 | 1 |
| `hoocro1` | 4 | 4 |
| `houfin` | 8 | 7 |
| `houspa` | 2 | 4 |
| `houwre` | 3 | 3 |
| `iiwi` | 11 | 9 |
| `jabwar` | 26 | 24 |
| `larela1` | 0 | 1 |
| `leswhi4` | 1 | 0 |
| `lotjae` | 1 | 1 |
| `madbul1` | 20 | 20 |
| `madcou1` | 7 | 7 |
| `madhoo1` | 8 | 11 |
| `mamrob1` | 0 | 1 |
| `mapfly1` | 11 | 12 |
| `masgna1` | 2 | 2 |
| `matdov1` | 0 | 1 |
| `mitpar` | 2 | 1 |
| `monpar` | 33 | 26 |
| `nelwea1` | 1 | 1 |
| `norcar` | 2 | 2 |
| `omao` | 11 | 9 |
| `palila` | 1 | 0 |
| `pigdov1` | 3 | 3 |
| `pluibi1` | 1 | 1 |
| `polpar1` | 1 | 1 |
| `puteup1` | 0 | 1 |
| `reblei` | 27 | 31 |
| `redfod1` | 3 | 3 |
| `redjun` | 4 | 3 |
| `rewbla` | 1 | 1 |
| `rubpep1` | 5 | 4 |
| `rucspa1` | 3 | 2 |
| `rufhor2` | 4 | 6 |
| `rufwhi1` | 0 | 2 |
| `saffin` | 1 | 1 |
| `sander` | 11 | 11 |
| `saytan1` | 1 | 1 |
| `schpar1` | 0 | 1 |
| `sinhon1` | 1 | 0 |
| `skylar` | 0 | 1 |
| `slathr2` | 3 | 4 |
| `slbblc1` | 1 | 1 |
| `smbela1` | 3 | 2 |
| `sobtyr1` | 1 | 1 |
| `sonspa` | 1 | 2 |
| `sonthr1` | 1 | 0 |
| `soulap1` | 2 | 0 |
| `souscr1` | 4 | 3 |
| `spbpuf1` | 1 | 1 |
| `spwpig3` | 2 | 2 |
| `stcspi2` | 3 | 3 |
| `suifly1` | 18 | 19 |
| `tui1` | 11 | 10 |
| `twbwar1` | 3 | 3 |
| `vartit1` | 1 | 1 |
| `warwhe1` | 7 | 5 |
| `whhvan1` | 2 | 1 |
| `whtspa` | 1 | 0 |
| `winwre4` | 8 | 10 |
| `wlwwar` | 2 | 2 |
| `woothr` | 1 | 1 |
| `yetvir` | 0 | 1 |

!!! warning "WARN"
    Matched 576 of 692 baseline detections (83.2%), with 116 missed and 108 extra. Mean IoU: 0.950. Mean Conf Δ: 0.0421. Mean start-time shift: 0.029 s.

File size: 39.8 MiB. Load time: 0.4 s. Detection time: 337.6 s.

---

### Birds-With-Labels_int8.tflite

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `????` | 57 | 49 |
| `akepa1` | 4 | 3 |
| `amepip` | 2 | 3 |
| `amerob` | 1 | 1 |
| `apapan` | 6 | 2 |
| `apptet1` | 2 | 2 |
| `arcnew1` | 3 | 1 |
| `ashcus2` | 3 | 3 |
| `astgna1` | 1 | 1 |
| `ausmag2` | 1 | 1 |
| `bawcow4` | 5 | 5 |
| `bbfgle1` | 1 | 1 |
| `bknsti` | 28 | 16 |
| `blackc1` | 2 | 2 |
| `blagro1` | 11 | 9 |
| `blapar1` | 1 | 2 |
| `blbman1` | 1 | 1 |
| `bllser1` | 1 | 1 |
| `blutit` | 1 | 2 |
| `brncac1` | 1 | 1 |
| `bufpar` | 3 | 2 |
| `cangoo` | 1 | 1 |
| `chacha1` | 4 | 3 |
| `chswar` | 1 | 1 |
| `cintyr1` | 8 | 8 |
| `citspa1` | 0 | 1 |
| `coatit2` | 1 | 2 |
| `comcha` | 22 | 24 |
| `comchi1` | 14 | 14 |
| `comcuc` | 1 | 1 |
| `comjer1` | 0 | 1 |
| `comnig1` | 1 | 0 |
| `comros` | 1 | 1 |
| `comyel` | 1 | 1 |
| `cowpig1` | 9 | 9 |
| `crbthr1` | 3 | 1 |
| `cregal1` | 1 | 1 |
| `crehor1` | 21 | 17 |
| `eardov1` | 1 | 0 |
| `easros1` | 3 | 2 |
| `elepai` | 2 | 1 |
| `ercfra` | 11 | 8 |
| `eupfly1` | 1 | 0 |
| `eurbla` | 5 | 5 |
| `eurjay1` | 1 | 0 |
| `eurrob1` | 3 | 5 |
| `firecr1` | 11 | 11 |
| `garwar1` | 0 | 1 |
| `gcrwar` | 1 | 1 |
| `gobsal1` | 9 | 9 |
| `gogwoo1` | 3 | 2 |
| `grekis` | 2 | 2 |
| `gretit1` | 1 | 1 |
| `grwtyr1` | 31 | 32 |
| `grycat` | 19 | 22 |
| `gycwor1` | 1 | 1 |
| `gyhlov1` | 3 | 1 |
| `hawama` | 56 | 57 |
| `hawcre` | 5 | 2 |
| `hawfin` | 1 | 1 |
| `hawhaw` | 1 | 1 |
| `hawpet1` | 1 | 1 |
| `hoocro1` | 4 | 4 |
| `houfin` | 8 | 9 |
| `houspa` | 2 | 3 |
| `houwre` | 3 | 4 |
| `iiwi` | 11 | 10 |
| `jabwar` | 26 | 27 |
| `larela1` | 0 | 1 |
| `leswhi4` | 1 | 0 |
| `lotjae` | 1 | 0 |
| `madbul1` | 20 | 18 |
| `madcou1` | 7 | 6 |
| `madhoo1` | 8 | 6 |
| `mapfly1` | 11 | 5 |
| `masgna1` | 2 | 2 |
| `mitpar` | 2 | 1 |
| `monpar` | 33 | 28 |
| `nelwea1` | 1 | 1 |
| `norcar` | 2 | 1 |
| `omao` | 11 | 10 |
| `ovenbi1` | 0 | 1 |
| `palila` | 1 | 1 |
| `pigdov1` | 3 | 3 |
| `pluibi1` | 1 | 2 |
| `polpar1` | 1 | 1 |
| `puteup1` | 0 | 1 |
| `reblei` | 27 | 28 |
| `redfod1` | 3 | 3 |
| `redjun` | 4 | 3 |
| `rewbla` | 1 | 1 |
| `rubpep1` | 5 | 3 |
| `rucspa1` | 3 | 3 |
| `rufhor2` | 4 | 2 |
| `rufwhi1` | 0 | 1 |
| `saffin` | 1 | 2 |
| `sander` | 11 | 10 |
| `saytan1` | 1 | 1 |
| `schpar1` | 0 | 1 |
| `sinhon1` | 1 | 0 |
| `skylar` | 0 | 1 |
| `slathr2` | 3 | 3 |
| `slbblc1` | 1 | 0 |
| `smbela1` | 3 | 2 |
| `sobtyr1` | 1 | 1 |
| `sonspa` | 1 | 1 |
| `sonthr1` | 1 | 1 |
| `soulap1` | 2 | 0 |
| `souscr1` | 4 | 4 |
| `spbpuf1` | 1 | 1 |
| `spwpig3` | 2 | 1 |
| `stcspi2` | 3 | 3 |
| `suifly1` | 18 | 18 |
| `tui1` | 11 | 10 |
| `twbwar1` | 3 | 3 |
| `vartit1` | 1 | 1 |
| `vaspar1` | 0 | 1 |
| `warwhe1` | 7 | 4 |
| `whhvan1` | 2 | 2 |
| `whtspa` | 1 | 0 |
| `winwre4` | 8 | 10 |
| `wlwwar` | 2 | 2 |
| `woothr` | 1 | 1 |
| `yetvir` | 0 | 1 |

!!! warning "WARN"
    Matched 570 of 692 baseline detections (82.4%), with 122 missed and 64 extra. Mean IoU: 0.952. Mean Conf Δ: 0.0409. Mean start-time shift: 0.033 s.

File size: 39.9 MiB. Load time: 0.4 s. Detection time: 371.8 s.

---
