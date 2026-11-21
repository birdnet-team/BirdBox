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
    Generated on 2026-11-20 15:04. Audio: 13 recordings, 36 minutes (see [Test recordings](#test-recordings)), species mapping: `Birds-With-Labels`, confidence threshold: `0.21`, baseline: `Birds-With-Labels_fp32.pt` (1495 detections).

### At a glance

| Model | Format | Size | Detections | Verdict |
| :--- | :--- | ---: | ---: | :---: |
| `Birds-With-Labels_fp32.pt` | `.pt` | 49.7 MiB | 1495 | _baseline_ |
| `Birds-With-Labels_fp16.engine` | `.engine` | 52.4 MiB | 1502 | :material-check-circle: PASS |
| `Birds-With-Labels_fp16.onnx` | `.onnx` | 49.4 MiB | 1497 | :material-check-circle: PASS |
| `Birds-With-Labels_fp16.tflite` | `.tflite` | 49.7 MiB | 1496 | :material-check-circle: PASS |
| `Birds-With-Labels_int8.engine` | `.engine` | 36.5 MiB | 1494 | :material-alert-circle: WARN |
| `Birds-With-Labels_int8.onnx` | `.onnx` | 39.8 MiB | 1446 | :material-alert-circle: WARN |
| `Birds-With-Labels_int8.tflite` | `.tflite` | 39.9 MiB | 1478 | :material-alert-circle: WARN |

### Detection matching

| Model | Matched | Missed | Extra | Match Rate | Mean IoU |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `Birds-With-Labels_fp32.pt` | — | — | — | _baseline_ | — |
| `Birds-With-Labels_fp16.engine` | 1479 | 16 | 23 | 98.9% | 0.993 |
| `Birds-With-Labels_fp16.onnx` | 1482 | 13 | 15 | 99.1% | 0.997 |
| `Birds-With-Labels_fp16.tflite` | 1493 | 2 | 3 | 99.9% | 0.999 |
| `Birds-With-Labels_int8.engine` | 1301 | 194 | 193 | 87.0% | 0.952 |
| `Birds-With-Labels_int8.onnx` | 1259 | 236 | 187 | 84.2% | 0.938 |
| `Birds-With-Labels_int8.tflite` | 1221 | 274 | 257 | 81.7% | 0.931 |

### Match rate per recording

Share of the baseline detections in each recording that the other formats reproduce. A format that only fails on some recordings shows up here.

**FP16**

| Recording | Length | Baseline | `fp16.engine` | `fp16.onnx` | `fp16.tflite` |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `test.wav` | 5:00 | 130 | 100% | 99% | 100% |
| `XC1010642_Rwanda.mp3` | 4:56 | 147 | 99% | 99% | 100% |
| `XC1073923_Ethiopia.wav` | 2:16 | 97 | 99% | 100% | 100% |
| `XC1170114_South-Africa.wav` | 4:19 | 253 | 99% | 99% | 100% |
| `XC1174845_France.mp3` | 2:00 | 86 | 99% | 100% | 100% |
| `XC1180150_Estonia.wav` | 3:21 | 140 | 100% | 99% | 100% |
| `XC1181792_Spain.mp3` | 4:39 | 247 | 99% | 100% | 100% |
| `XC1184153_Spain.wav` | 1:03 | 14 | 100% | 100% | 100% |
| `XC1184684_Austria.wav` | 3:20 | 148 | 98% | 99% | 100% |
| `XC1185879_Sweden.wav` | 1:29 | 71 | 100% | 100% | 100% |
| `XC692092_Kenya.mp3` | 1:02 | 67 | 97% | 99% | 100% |
| `XC782384_Sierra-Leone.wav` | 1:07 | 35 | 97% | 97% | 100% |
| `XC889825_Zimbabwe.wav` | 1:46 | 60 | 97% | 95% | 100% |

**INT8**

| Recording | Length | Baseline | `int8.engine` | `int8.onnx` | `int8.tflite` |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `test.wav` | 5:00 | 130 | 88% | 88% | 83% |
| `XC1010642_Rwanda.mp3` | 4:56 | 147 | 86% | 82% | 81% |
| `XC1073923_Ethiopia.wav` | 2:16 | 97 | 86% | 90% | 82% |
| `XC1170114_South-Africa.wav` | 4:19 | 253 | 87% | 87% | 84% |
| `XC1174845_France.mp3` | 2:00 | 86 | 90% | 94% | 91% |
| `XC1180150_Estonia.wav` | 3:21 | 140 | 91% | 86% | 86% |
| `XC1181792_Spain.mp3` | 4:39 | 247 | 87% | 84% | 83% |
| `XC1184153_Spain.wav` | 1:03 | 14 | 93% | 93% | 79% |
| `XC1184684_Austria.wav` | 3:20 | 148 | 83% | 77% | 78% |
| `XC1185879_Sweden.wav` | 1:29 | 71 | 87% | 85% | 80% |
| `XC692092_Kenya.mp3` | 1:02 | 67 | 87% | 78% | 66% |
| `XC782384_Sierra-Leone.wav` | 1:07 | 35 | 91% | 80% | 86% |
| `XC889825_Zimbabwe.wav` | 1:46 | 60 | 82% | 70% | 70% |

### Confidence and timing

| Model | Mean Conf Δ | Max Conf Δ | Load (s) | Detect (s) |
| :--- | ---: | ---: | ---: | ---: |
| `Birds-With-Labels_fp32.pt` | — | — | 1.6 | 115.3 |
| `Birds-With-Labels_fp16.engine` | 0.0032 | 0.4302 | 2.2 | 108.8 |
| `Birds-With-Labels_fp16.onnx` | 0.0018 | 0.1737 | 2.4 | 306.9 |
| `Birds-With-Labels_fp16.tflite` | 0.0007 | 0.1738 | 1.4 | 741.2 |
| `Birds-With-Labels_int8.engine` | 0.0258 | 0.2632 | 1.9 | 132.4 |
| `Birds-With-Labels_int8.onnx` | 0.0336 | 0.2897 | 2.0 | 206.1 |
| `Birds-With-Labels_int8.tflite` | 0.0369 | 0.2534 | 1.8 | 270.5 |

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
| `????` | 59 | 59 |
| `akepa1` | 3 | 3 |
| `amerob` | 2 | 2 |
| `apapan` | 1 | 1 |
| `ashcus2` | 1 | 1 |
| `ausmag2` | 1 | 1 |
| `bawcow4` | 5 | 5 |
| `bird` | 935 | 935 |
| `bknsti` | 24 | 24 |
| `blackc1` | 4 | 4 |
| `blagro1` | 16 | 16 |
| `blapar1` | 1 | 1 |
| `blbman1` | 1 | 1 |
| `bllser1` | 1 | 1 |
| `brncac1` | 2 | 2 |
| `brncre` | 1 | 1 |
| `bufpar` | 3 | 3 |
| `cangoo` | 1 | 1 |
| `chacha1` | 1 | 1 |
| `chswar` | 1 | 1 |
| `coatit2` | 3 | 3 |
| `comcha` | 29 | 29 |
| `comchi1` | 19 | 19 |
| `comcuc` | 1 | 1 |
| `comwax` | 1 | 1 |
| `cowpig1` | 12 | 12 |
| `crbthr1` | 1 | 1 |
| `credro1` | 2 | 2 |
| `crehor1` | 12 | 12 |
| `cretit2` | 1 | 1 |
| `croslf1` | 1 | 1 |
| `elepai` | 2 | 2 |
| `ercfra` | 5 | 5 |
| `eulfly1` | 2 | 2 |
| `eupfly1` | 1 | 1 |
| `eurbla` | 6 | 6 |
| `eurjay1` | 1 | 1 |
| `eurrob1` | 1 | 1 |
| `firecr1` | 4 | 4 |
| `gcrwar` | 1 | 1 |
| `gobsal1` | 11 | 11 |
| `gogwoo1` | 10 | 10 |
| `gowcac1` | 1 | 1 |
| `grekis` | 1 | 1 |
| `gretit1` | 2 | 2 |
| `grwtyr1` | 18 | 18 |
| `grycat` | 5 | 5 |
| `gyctyr2` | 1 | 1 |
| `gycwor1` | 1 | 1 |
| `gyhlov1` | 3 | 3 |
| `hawama` | 68 | 68 |
| `hawfin` | 1 | 1 |
| `houfin` | 7 | 7 |
| `houspa` | 1 | 1 |
| `houwre` | 2 | 2 |
| `iiwi` | 6 | 6 |
| `jabwar` | 27 | 27 |
| `madbul1` | 26 | 26 |
| `madcou1` | 5 | 5 |
| `malbrw1` | 1 | 1 |
| `mamrob1` | 1 | 1 |
| `mapfly1` | 6 | 6 |
| `masgna1` | 1 | 1 |
| `monpar` | 33 | 33 |
| `nabwoo1` | 1 | 1 |
| `omao` | 4 | 4 |
| `pigdov1` | 4 | 4 |
| `puteup1` | 1 | 1 |
| `reblei` | 28 | 28 |
| `redjun` | 5 | 5 |
| `retvan1` | 1 | 1 |
| `rubpep1` | 2 | 2 |
| `rucspa1` | 3 | 3 |
| `rufhor2` | 4 | 4 |
| `schpar1` | 4 | 4 |
| `skylar` | 1 | 1 |
| `slathr2` | 2 | 2 |
| `smbela1` | 3 | 3 |
| `sonthr1` | 1 | 1 |
| `souscr1` | 2 | 2 |
| `spwpig3` | 1 | 1 |
| `stcspi2` | 1 | 1 |
| `suifly1` | 8 | 8 |
| `swathr` | 1 | 1 |
| `warwhe1` | 4 | 4 |
| `whhvan1` | 2 | 2 |
| `winwre4` | 6 | 6 |

File size: 49.7 MiB. Load time: 1.6 s. Detection time: 115.3 s.

---

### Birds-With-Labels_fp16.engine

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `????` | 59 | 61 |
| `akepa1` | 3 | 3 |
| `amerob` | 2 | 2 |
| `apapan` | 1 | 1 |
| `ashcus2` | 1 | 1 |
| `ausmag2` | 1 | 1 |
| `bawcow4` | 5 | 5 |
| `bird` | 935 | 942 |
| `bknsti` | 24 | 23 |
| `blackc1` | 4 | 3 |
| `blagro1` | 16 | 16 |
| `blapar1` | 1 | 1 |
| `blbman1` | 1 | 1 |
| `bllser1` | 1 | 1 |
| `bobfly1` | 0 | 1 |
| `brncac1` | 2 | 2 |
| `brncre` | 1 | 1 |
| `bufpar` | 3 | 3 |
| `cangoo` | 1 | 1 |
| `chacha1` | 1 | 1 |
| `chswar` | 1 | 1 |
| `coatit2` | 3 | 3 |
| `comcha` | 29 | 28 |
| `comchi1` | 19 | 19 |
| `comcuc` | 1 | 1 |
| `comwax` | 1 | 1 |
| `cowpig1` | 12 | 12 |
| `crbthr1` | 1 | 1 |
| `credro1` | 2 | 2 |
| `crehor1` | 12 | 12 |
| `cretit2` | 1 | 1 |
| `croslf1` | 1 | 1 |
| `elepai` | 2 | 2 |
| `ercfra` | 5 | 5 |
| `eulfly1` | 2 | 1 |
| `eupfly1` | 1 | 1 |
| `eurbla` | 6 | 6 |
| `eurjay1` | 1 | 1 |
| `eurrob1` | 1 | 1 |
| `firecr1` | 4 | 4 |
| `gcrwar` | 1 | 1 |
| `gobsal1` | 11 | 11 |
| `gogwoo1` | 10 | 10 |
| `gowcac1` | 1 | 1 |
| `grekis` | 1 | 1 |
| `gretit1` | 2 | 2 |
| `grwtyr1` | 18 | 17 |
| `grycat` | 5 | 5 |
| `gyctyr2` | 1 | 1 |
| `gycwor1` | 1 | 1 |
| `gyhlov1` | 3 | 3 |
| `hawama` | 68 | 68 |
| `hawfin` | 1 | 1 |
| `houfin` | 7 | 7 |
| `houspa` | 1 | 1 |
| `houwre` | 2 | 2 |
| `iiwi` | 6 | 6 |
| `jabwar` | 27 | 27 |
| `madbul1` | 26 | 26 |
| `madcou1` | 5 | 5 |
| `malbrw1` | 1 | 1 |
| `mamrob1` | 1 | 1 |
| `mapfly1` | 6 | 6 |
| `masgna1` | 1 | 1 |
| `monpar` | 33 | 33 |
| `nabwoo1` | 1 | 2 |
| `omao` | 4 | 4 |
| `pigdov1` | 4 | 4 |
| `puteup1` | 1 | 1 |
| `reblei` | 28 | 28 |
| `redjun` | 5 | 5 |
| `retvan1` | 1 | 1 |
| `rubpep1` | 2 | 2 |
| `rucspa1` | 3 | 4 |
| `rufhor2` | 4 | 5 |
| `schpar1` | 4 | 4 |
| `skylar` | 1 | 1 |
| `slathr2` | 2 | 2 |
| `smbela1` | 3 | 3 |
| `sonthr1` | 1 | 0 |
| `souscr1` | 2 | 2 |
| `spwpig3` | 1 | 1 |
| `stcspi2` | 1 | 1 |
| `suifly1` | 8 | 8 |
| `swathr` | 1 | 1 |
| `warwhe1` | 4 | 4 |
| `whhvan1` | 2 | 2 |
| `winwre4` | 6 | 6 |

!!! success "PASS"
    Matched 1479 of 1495 baseline detections (98.9%), with 16 missed and 23 extra. Mean IoU: 0.993. Mean Conf Δ: 0.0032. Mean start-time shift: 0.004 s.

File size: 52.4 MiB. Load time: 2.2 s. Detection time: 108.8 s.

---

### Birds-With-Labels_fp16.onnx

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `????` | 59 | 59 |
| `akepa1` | 3 | 3 |
| `amerob` | 2 | 2 |
| `apapan` | 1 | 1 |
| `ashcus2` | 1 | 1 |
| `ausmag2` | 1 | 1 |
| `bawcow4` | 5 | 5 |
| `bird` | 935 | 938 |
| `bknsti` | 24 | 23 |
| `blackc1` | 4 | 4 |
| `blagro1` | 16 | 16 |
| `blapar1` | 1 | 1 |
| `blbman1` | 1 | 1 |
| `bllser1` | 1 | 1 |
| `brncac1` | 2 | 2 |
| `brncre` | 1 | 1 |
| `bufpar` | 3 | 1 |
| `cangoo` | 1 | 1 |
| `chacha1` | 1 | 1 |
| `chswar` | 1 | 1 |
| `coatit2` | 3 | 3 |
| `comcha` | 29 | 29 |
| `comchi1` | 19 | 19 |
| `comcuc` | 1 | 1 |
| `comwax` | 1 | 1 |
| `cowpig1` | 12 | 12 |
| `crbthr1` | 1 | 1 |
| `credro1` | 2 | 2 |
| `crehor1` | 12 | 12 |
| `cretit2` | 1 | 1 |
| `croslf1` | 1 | 1 |
| `elepai` | 2 | 2 |
| `ercfra` | 5 | 5 |
| `eulfly1` | 2 | 1 |
| `eupfly1` | 1 | 1 |
| `eurbla` | 6 | 6 |
| `eurjay1` | 1 | 1 |
| `eurrob1` | 1 | 1 |
| `firecr1` | 4 | 4 |
| `gcrwar` | 1 | 1 |
| `gobsal1` | 11 | 11 |
| `gogwoo1` | 10 | 10 |
| `gowcac1` | 1 | 1 |
| `grekis` | 1 | 1 |
| `gretit1` | 2 | 2 |
| `grwtyr1` | 18 | 19 |
| `grycat` | 5 | 5 |
| `gyctyr2` | 1 | 1 |
| `gycwor1` | 1 | 1 |
| `gyhlov1` | 3 | 3 |
| `hawama` | 68 | 67 |
| `hawfin` | 1 | 1 |
| `houfin` | 7 | 8 |
| `houspa` | 1 | 1 |
| `houwre` | 2 | 2 |
| `iiwi` | 6 | 6 |
| `jabwar` | 27 | 27 |
| `madbul1` | 26 | 25 |
| `madcou1` | 5 | 5 |
| `malbrw1` | 1 | 1 |
| `mamrob1` | 1 | 1 |
| `mapfly1` | 6 | 6 |
| `masgna1` | 1 | 1 |
| `monpar` | 33 | 33 |
| `nabwoo1` | 1 | 2 |
| `omao` | 4 | 4 |
| `pigdov1` | 4 | 4 |
| `puteup1` | 1 | 1 |
| `reblei` | 28 | 28 |
| `redjun` | 5 | 5 |
| `retvan1` | 1 | 1 |
| `rubpep1` | 2 | 2 |
| `rucspa1` | 3 | 3 |
| `rufhor2` | 4 | 5 |
| `schpar1` | 4 | 4 |
| `skylar` | 1 | 1 |
| `slathr2` | 2 | 2 |
| `smbela1` | 3 | 4 |
| `sonthr1` | 1 | 1 |
| `souscr1` | 2 | 2 |
| `spwpig3` | 1 | 1 |
| `stcspi2` | 1 | 1 |
| `suifly1` | 8 | 8 |
| `swathr` | 1 | 1 |
| `warwhe1` | 4 | 4 |
| `whhvan1` | 2 | 2 |
| `winwre4` | 6 | 6 |

!!! success "PASS"
    Matched 1482 of 1495 baseline detections (99.1%), with 13 missed and 15 extra. Mean IoU: 0.997. Mean Conf Δ: 0.0018. Mean start-time shift: 0.001 s.

File size: 49.4 MiB. Load time: 2.4 s. Detection time: 306.9 s.

---

### Birds-With-Labels_fp16.tflite

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `????` | 59 | 59 |
| `akepa1` | 3 | 3 |
| `amerob` | 2 | 2 |
| `apapan` | 1 | 1 |
| `ashcus2` | 1 | 1 |
| `ausmag2` | 1 | 1 |
| `bawcow4` | 5 | 5 |
| `bird` | 935 | 938 |
| `bknsti` | 24 | 24 |
| `blackc1` | 4 | 4 |
| `blagro1` | 16 | 16 |
| `blapar1` | 1 | 1 |
| `blbman1` | 1 | 1 |
| `bllser1` | 1 | 1 |
| `brncac1` | 2 | 2 |
| `brncre` | 1 | 1 |
| `bufpar` | 3 | 3 |
| `cangoo` | 1 | 1 |
| `chacha1` | 1 | 1 |
| `chswar` | 1 | 1 |
| `coatit2` | 3 | 3 |
| `comcha` | 29 | 29 |
| `comchi1` | 19 | 19 |
| `comcuc` | 1 | 1 |
| `comwax` | 1 | 1 |
| `cowpig1` | 12 | 12 |
| `crbthr1` | 1 | 1 |
| `credro1` | 2 | 2 |
| `crehor1` | 12 | 12 |
| `cretit2` | 1 | 1 |
| `croslf1` | 1 | 1 |
| `elepai` | 2 | 2 |
| `ercfra` | 5 | 5 |
| `eulfly1` | 2 | 2 |
| `eupfly1` | 1 | 1 |
| `eurbla` | 6 | 6 |
| `eurjay1` | 1 | 1 |
| `eurrob1` | 1 | 1 |
| `firecr1` | 4 | 4 |
| `gcrwar` | 1 | 1 |
| `gobsal1` | 11 | 11 |
| `gogwoo1` | 10 | 10 |
| `gowcac1` | 1 | 1 |
| `grekis` | 1 | 1 |
| `gretit1` | 2 | 2 |
| `grwtyr1` | 18 | 18 |
| `grycat` | 5 | 5 |
| `gyctyr2` | 1 | 1 |
| `gycwor1` | 1 | 1 |
| `gyhlov1` | 3 | 3 |
| `hawama` | 68 | 68 |
| `hawfin` | 1 | 1 |
| `houfin` | 7 | 7 |
| `houspa` | 1 | 1 |
| `houwre` | 2 | 2 |
| `iiwi` | 6 | 6 |
| `jabwar` | 27 | 27 |
| `madbul1` | 26 | 25 |
| `madcou1` | 5 | 5 |
| `malbrw1` | 1 | 1 |
| `mamrob1` | 1 | 1 |
| `mapfly1` | 6 | 6 |
| `masgna1` | 1 | 1 |
| `monpar` | 33 | 33 |
| `nabwoo1` | 1 | 1 |
| `omao` | 4 | 4 |
| `pigdov1` | 4 | 4 |
| `puteup1` | 1 | 1 |
| `reblei` | 28 | 28 |
| `redjun` | 5 | 5 |
| `retvan1` | 1 | 1 |
| `rubpep1` | 2 | 2 |
| `rucspa1` | 3 | 3 |
| `rufhor2` | 4 | 4 |
| `schpar1` | 4 | 4 |
| `skylar` | 1 | 1 |
| `slathr2` | 2 | 2 |
| `smbela1` | 3 | 3 |
| `sonthr1` | 1 | 1 |
| `souscr1` | 2 | 2 |
| `spwpig3` | 1 | 1 |
| `stcspi2` | 1 | 1 |
| `suifly1` | 8 | 8 |
| `swathr` | 1 | 0 |
| `warwhe1` | 4 | 4 |
| `whhvan1` | 2 | 2 |
| `winwre4` | 6 | 6 |

!!! success "PASS"
    Matched 1493 of 1495 baseline detections (99.9%), with 2 missed and 3 extra. Mean IoU: 0.999. Mean Conf Δ: 0.0007. Mean start-time shift: 0.000 s.

File size: 49.7 MiB. Load time: 1.4 s. Detection time: 741.2 s.

---

### Birds-With-Labels_int8.engine

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `????` | 59 | 50 |
| `akepa1` | 3 | 3 |
| `amerob` | 2 | 1 |
| `apapan` | 1 | 1 |
| `ashcus2` | 1 | 1 |
| `ausmag2` | 1 | 1 |
| `bawcow4` | 5 | 5 |
| `bird` | 935 | 960 |
| `bknsti` | 24 | 24 |
| `blackc1` | 4 | 4 |
| `blagro1` | 16 | 15 |
| `blapar1` | 1 | 1 |
| `blbman1` | 1 | 1 |
| `bllser1` | 1 | 1 |
| `brncac1` | 2 | 2 |
| `brncre` | 1 | 0 |
| `bufpar` | 3 | 1 |
| `cangoo` | 1 | 1 |
| `chacha1` | 1 | 1 |
| `chswar` | 1 | 1 |
| `cintyr1` | 0 | 1 |
| `coatit2` | 3 | 2 |
| `comcha` | 29 | 27 |
| `comchi1` | 19 | 18 |
| `comcuc` | 1 | 1 |
| `comwax` | 1 | 1 |
| `cowpig1` | 12 | 10 |
| `crbthr1` | 1 | 1 |
| `credro1` | 2 | 2 |
| `crehor1` | 12 | 12 |
| `cretit2` | 1 | 1 |
| `croslf1` | 1 | 0 |
| `elepai` | 2 | 2 |
| `ercfra` | 5 | 5 |
| `eulfly1` | 2 | 0 |
| `eupfly1` | 1 | 1 |
| `eurbla` | 6 | 6 |
| `eurjay1` | 1 | 1 |
| `eurrob1` | 1 | 1 |
| `firecr1` | 4 | 3 |
| `gcrwar` | 1 | 0 |
| `gobsal1` | 11 | 11 |
| `gogwoo1` | 10 | 7 |
| `gowcac1` | 1 | 0 |
| `grekis` | 1 | 1 |
| `gretit1` | 2 | 2 |
| `grwtyr1` | 18 | 19 |
| `grycat` | 5 | 5 |
| `gyctyr2` | 1 | 1 |
| `gycwor1` | 1 | 1 |
| `gyhlov1` | 3 | 3 |
| `hawama` | 68 | 63 |
| `hawfin` | 1 | 1 |
| `houfin` | 7 | 6 |
| `houspa` | 1 | 1 |
| `houwre` | 2 | 2 |
| `iiwi` | 6 | 7 |
| `jabwar` | 27 | 22 |
| `madbul1` | 26 | 21 |
| `madcou1` | 5 | 6 |
| `malbrw1` | 1 | 1 |
| `mamrob1` | 1 | 1 |
| `mapfly1` | 6 | 6 |
| `masgna1` | 1 | 1 |
| `mitpar` | 0 | 1 |
| `monpar` | 33 | 42 |
| `nabwoo1` | 1 | 2 |
| `omao` | 4 | 5 |
| `pigdov1` | 4 | 6 |
| `pluibi1` | 0 | 1 |
| `puteup1` | 1 | 2 |
| `reblei` | 28 | 30 |
| `redjun` | 5 | 6 |
| `retvan1` | 1 | 1 |
| `rubpep1` | 2 | 1 |
| `rucspa1` | 3 | 0 |
| `rufhor2` | 4 | 3 |
| `schpar1` | 4 | 4 |
| `skylar` | 1 | 0 |
| `slathr2` | 2 | 2 |
| `smbela1` | 3 | 4 |
| `sonthr1` | 1 | 0 |
| `souscr1` | 2 | 1 |
| `spwpig3` | 1 | 1 |
| `stcspi2` | 1 | 1 |
| `strfly1` | 0 | 1 |
| `suifly1` | 8 | 13 |
| `swathr` | 1 | 1 |
| `warwhe1` | 4 | 2 |
| `whhvan1` | 2 | 2 |
| `winwre4` | 6 | 5 |

!!! warning "WARN"
    Matched 1301 of 1495 baseline detections (87.0%), with 194 missed and 193 extra. Mean IoU: 0.952. Mean Conf Δ: 0.0258. Mean start-time shift: 0.024 s.

File size: 36.5 MiB. Load time: 1.9 s. Detection time: 132.4 s.

---

### Birds-With-Labels_int8.onnx

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `????` | 59 | 54 |
| `akepa1` | 3 | 2 |
| `amerob` | 2 | 2 |
| `apapan` | 1 | 0 |
| `ashcus2` | 1 | 1 |
| `ausmag2` | 1 | 1 |
| `bawcow4` | 5 | 5 |
| `bird` | 935 | 903 |
| `bknsti` | 24 | 30 |
| `blackc1` | 4 | 3 |
| `blagro1` | 16 | 16 |
| `blapar1` | 1 | 0 |
| `blbman1` | 1 | 0 |
| `bllser1` | 1 | 0 |
| `bobfly1` | 0 | 1 |
| `brncac1` | 2 | 1 |
| `brncre` | 1 | 0 |
| `bufpar` | 3 | 1 |
| `cangoo` | 1 | 1 |
| `carcro1` | 0 | 1 |
| `chacha1` | 1 | 1 |
| `chswar` | 1 | 1 |
| `coatit2` | 3 | 3 |
| `comcha` | 29 | 26 |
| `comchi1` | 19 | 19 |
| `comcuc` | 1 | 1 |
| `comwax` | 1 | 1 |
| `cowpig1` | 12 | 11 |
| `crbthr1` | 1 | 1 |
| `credro1` | 2 | 2 |
| `crehor1` | 12 | 11 |
| `cretit2` | 1 | 1 |
| `croslf1` | 1 | 0 |
| `elepai` | 2 | 0 |
| `ercfra` | 5 | 9 |
| `eulfly1` | 2 | 1 |
| `eupfly1` | 1 | 2 |
| `eurbla` | 6 | 6 |
| `eurjay1` | 1 | 1 |
| `eurrob1` | 1 | 1 |
| `firecr1` | 4 | 4 |
| `gcrwar` | 1 | 2 |
| `gobsal1` | 11 | 10 |
| `gogwoo1` | 10 | 10 |
| `gowcac1` | 1 | 1 |
| `grekis` | 1 | 1 |
| `gretit1` | 2 | 2 |
| `grtgra` | 0 | 1 |
| `grwtyr1` | 18 | 23 |
| `grycat` | 5 | 4 |
| `gyctyr2` | 1 | 1 |
| `gycwor1` | 1 | 1 |
| `gyhlov1` | 3 | 2 |
| `hawama` | 68 | 66 |
| `hawfin` | 1 | 1 |
| `houfin` | 7 | 6 |
| `houspa` | 1 | 0 |
| `houwre` | 2 | 2 |
| `iiwi` | 6 | 4 |
| `jabwar` | 27 | 22 |
| `madbul1` | 26 | 23 |
| `madcou1` | 5 | 6 |
| `malbrw1` | 1 | 1 |
| `mamrob1` | 1 | 1 |
| `mapfly1` | 6 | 5 |
| `masgna1` | 1 | 1 |
| `monpar` | 33 | 34 |
| `nabwoo1` | 1 | 0 |
| `omao` | 4 | 4 |
| `picpig2` | 0 | 1 |
| `pigdov1` | 4 | 4 |
| `puteup1` | 1 | 2 |
| `reblei` | 28 | 29 |
| `redjun` | 5 | 5 |
| `retvan1` | 1 | 1 |
| `rubpep1` | 2 | 1 |
| `rucspa1` | 3 | 2 |
| `rufhor2` | 4 | 4 |
| `schpar1` | 4 | 4 |
| `skylar` | 1 | 1 |
| `slathr2` | 2 | 3 |
| `smbela1` | 3 | 5 |
| `sonthr1` | 1 | 0 |
| `souscr1` | 2 | 2 |
| `spwpig3` | 1 | 0 |
| `stbori` | 0 | 1 |
| `stcspi2` | 1 | 1 |
| `strfly1` | 0 | 1 |
| `suifly1` | 8 | 11 |
| `swathr` | 1 | 0 |
| `vartit1` | 0 | 1 |
| `warwhe1` | 4 | 3 |
| `whhvan1` | 2 | 1 |
| `winwre4` | 6 | 5 |

!!! warning "WARN"
    Matched 1259 of 1495 baseline detections (84.2%), with 236 missed and 187 extra. Mean IoU: 0.938. Mean Conf Δ: 0.0336. Mean start-time shift: 0.031 s.

File size: 39.8 MiB. Load time: 2.0 s. Detection time: 206.1 s.

---

### Birds-With-Labels_int8.tflite

| Species | Baseline | This model |
| :--- | ---: | ---: |
| `????` | 59 | 52 |
| `akepa1` | 3 | 2 |
| `amerob` | 2 | 3 |
| `apapan` | 1 | 1 |
| `ashcus2` | 1 | 1 |
| `ausmag2` | 1 | 1 |
| `bawcow4` | 5 | 7 |
| `bird` | 935 | 952 |
| `bknsti` | 24 | 27 |
| `blackc1` | 4 | 4 |
| `blagro1` | 16 | 15 |
| `blapar1` | 1 | 1 |
| `blbman1` | 1 | 1 |
| `bllser1` | 1 | 0 |
| `bobfly1` | 0 | 1 |
| `brncac1` | 2 | 3 |
| `brncre` | 1 | 0 |
| `bufpar` | 3 | 1 |
| `cangoo` | 1 | 1 |
| `carcro1` | 0 | 1 |
| `chacha1` | 1 | 1 |
| `chswar` | 1 | 1 |
| `cintyr1` | 0 | 2 |
| `coatit2` | 3 | 3 |
| `comcha` | 29 | 27 |
| `comchi1` | 19 | 19 |
| `comcuc` | 1 | 1 |
| `comnig1` | 0 | 1 |
| `comwax` | 1 | 1 |
| `cowpig1` | 12 | 6 |
| `crbthr1` | 1 | 0 |
| `credro1` | 2 | 1 |
| `crehor1` | 12 | 7 |
| `cretit2` | 1 | 1 |
| `croslf1` | 1 | 0 |
| `elepai` | 2 | 1 |
| `ercfra` | 5 | 5 |
| `eulfly1` | 2 | 1 |
| `eupfly1` | 1 | 1 |
| `eurbla` | 6 | 5 |
| `eurjay1` | 1 | 1 |
| `eurrob1` | 1 | 1 |
| `firecr1` | 4 | 3 |
| `gcrwar` | 1 | 1 |
| `gobsal1` | 11 | 10 |
| `gogwoo1` | 10 | 8 |
| `gowcac1` | 1 | 0 |
| `grekis` | 1 | 0 |
| `gretit1` | 2 | 0 |
| `grwtyr1` | 18 | 25 |
| `grycat` | 5 | 6 |
| `gyctyr2` | 1 | 1 |
| `gycwor1` | 1 | 0 |
| `gyhlov1` | 3 | 2 |
| `hawama` | 68 | 60 |
| `hawfin` | 1 | 1 |
| `houfin` | 7 | 6 |
| `houspa` | 1 | 1 |
| `houwre` | 2 | 1 |
| `iiwi` | 6 | 7 |
| `jabwar` | 27 | 23 |
| `madbul1` | 26 | 25 |
| `madcou1` | 5 | 6 |
| `malbrw1` | 1 | 0 |
| `mamrob1` | 1 | 1 |
| `mapfly1` | 6 | 2 |
| `masgna1` | 1 | 1 |
| `monpar` | 33 | 42 |
| `nabwoo1` | 1 | 0 |
| `norcar` | 0 | 2 |
| `omao` | 4 | 5 |
| `pigdov1` | 4 | 1 |
| `pluibi1` | 0 | 1 |
| `puteup1` | 1 | 2 |
| `reblei` | 28 | 33 |
| `redjun` | 5 | 4 |
| `retvan1` | 1 | 2 |
| `rubpep1` | 2 | 1 |
| `rucspa1` | 3 | 0 |
| `rufhor2` | 4 | 1 |
| `schpar1` | 4 | 4 |
| `skylar` | 1 | 1 |
| `slathr2` | 2 | 1 |
| `smbela1` | 3 | 4 |
| `sonthr1` | 1 | 0 |
| `souscr1` | 2 | 2 |
| `spwpig3` | 1 | 0 |
| `stbori` | 0 | 1 |
| `stcspi2` | 1 | 1 |
| `suifly1` | 8 | 10 |
| `swathr` | 1 | 0 |
| `warwhe1` | 4 | 4 |
| `whhvan1` | 2 | 1 |
| `winwre4` | 6 | 5 |

!!! warning "WARN"
    Matched 1221 of 1495 baseline detections (81.7%), with 274 missed and 257 extra. Mean IoU: 0.931. Mean Conf Δ: 0.0369. Mean start-time shift: 0.030 s.

File size: 39.9 MiB. Load time: 1.8 s. Detection time: 270.5 s.

---
