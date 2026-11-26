# Audio and Model Inputs

BirdBox reads field recordings and a trained YOLO weights file, then maps neural-network class IDs to species using a named mapping in `src/config.py`. This page documents the accepted formats for the `--audio`, `--model`, and `--species-mapping` inputs. For CLI flags, defaults, and invocation examples see the [detect-birds](../cli/detect-birds.md) reference.

---

## Audio inputs

`--audio` may be a **single file** or a **directory**. Directories are searched **recursively** for supported files.

### Supported extensions

| Extension | Recommended? | Notes |
| :--- | :--- | :--- |
| `.wav` | Yes | Lossless. Fully supported. |
| `.flac` | Yes | Lossless. Fully supported. |
| `.ogg` | Use with care | Lossy. May reduce recall on faint or high-frequency calls. |
| `.mp3` | Use with care | Lossy. Same caveat as OGG. |

Case is ignored (`.WAV` and `.wav` are both found).

### Processing behavior

- **Mono conversion:** Stereo recordings are averaged to mono before the spectrogram is computed.
- **Sample rate:** Audio at any other rate is resampled to 32 kHz.
- **Arbitrary length:** Recordings of any length are processed in 3 s windows that overlap by half (see [Merging the Windows](../how-it-works/overview.md#merging-the-windows)).
- **Batch mode:** Point `--audio` at a folder to process every supported file under it in one run. Outputs and evaluation identify recordings by file name only, so give each recording a unique name across subfolders.

!!! warning "Lossy audio formats"
    MP3 and OGG are supported via `soundfile`, but compression can hide faint calls. In a study of avian recordings, MP3 compression reduced the detections of human listeners and automated recognisers [[22]](#ref-macphail). If detections look weak on a compressed file, re-run the same recording as WAV or FLAC before tuning thresholds.

---

## Model files

`--model` must point to a **YOLO-compatible** weights file loaded by Ultralytics. BirdBox supports three formats:

| Format | Typical use | Runtime |
| :--- | :--- | :--- |
| `.pt` | PyTorch checkpoint. Default for BirdBox releases. | PyTorch (CUDA or CPU) |
| `.onnx` | Cross-platform deployment. Quantized variants available. | ONNX Runtime (GPU or CPU) |
| `.tflite` | Edge devices and mobile targets. | LiteRT / ai-edge-litert (CPU) |

The pretrained models on [TUC-Cloud](https://tuc.cloud/index.php/s/ET4KE4LdSaysSSL){ target="_blank" rel="noopener noreferrer" } come as `.pt` in fp32 and as `.onnx` and `.tflite` in fp16 and int8 (see [Model Download](../installation.md#model-download)). Custom models can be trained with [BirdBox-Train](https://github.com/birdnet-team/BirdBox-Train){ target="_blank" rel="noopener noreferrer" }.

Each format requires its own Python environment. Install the correct runtime before switching formats.

!!! danger "Environment must match the model format"
    Loading a `.tflite` or `.onnx` model from a `.pt` environment fails with an import error. Run `python install.py --model-format <FORMAT>` to set up the right environment. See [Install Parameters](../installation.md#install-parameters) for the full format-to-runtime table.

!!! info "Format parity"
    Detection quality of the `.onnx` and `.tflite` formats is validated by running inference on the same audio and comparing results against the `.pt` baseline. See [Just-Bird format parity](../models-and-metrics/just-bird-model-types.md) and [Birds-With-Labels format parity](../models-and-metrics/birds-with-labels-model-types.md) for the current scores.

---

## Species Mapping

The `--species-mapping` name must match the label space the model was trained on. It is **not** inferred from the weights filename. You pass it explicitly.
For specific mappings see `SPECIES_MAPPING` in [src/config.py](https://github.com/birdnet-team/BirdBox/blob/main/src/config.py){ target="_blank" rel="noopener noreferrer" }.

Allowed values for the species-mapping parameter:

| Value | Classes |
| :--- | :--- |
| `Just-Bird` | One `bird` class for calls of any species |
| `Birds-With-Labels` | 632 species, a `????` class for calls of unknown species, and a `bird` class (see [Birds-With-Labels](../models-and-metrics/birds-with-labels.md)) |

Each key resolves to `id_to_ebird`, `ebird_to_name`, and display colors in `src/config.py`. The mapping name is stored in detection JSON as `model_config.species_mapping`.

!!! danger "Mapping must match the model"
    If `--species-mapping` does not match the class list the model was trained with, outputs carry **wrong eBird codes** with no error. Always pair model file and mapping from the same release (e.g. `Birds-With-Labels_fp32.pt` with `Birds-With-Labels`).

---

## References

<a id="ref-macphail"></a>
**[22]** MacPhail, A. G., Yip, D. A., Knight, E. C., Hedley, R., Knaggs, M., Shonfield, J., Upham-Mills, E., & Bayne, E. M. (2024). "Audio data compression affects acoustic indices and reduces detections of birds by human listening and automated recognisers." *Bioacoustics*, 33(1):74–90.
