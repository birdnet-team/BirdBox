# Audio and Model Inputs

BirdBox reads field recordings and a trained YOLO weights file, then maps neural-network class IDs to species using a named mapping in `src/config.py`. This page covers the `--audio`, `--model`, and `--species-mapping` arguments to `src/inference/detect_birds.py`.

---

## Usage synopsis

```bash
python src/inference/detect_birds.py \
    --audio path/to/recording.wav \
    --model models/Western-US.pt \
    --species-mapping Western-US
```

## Audio inputs

`--audio` may be a **single file** or a **directory**. Directories are searched **recursively** for supported files.

### Supported extensions

| Extension | Recommended? | Notes |
| :--- | :--- | :--- |
| `.wav` | Yes | Lossless; matches training data best. |
| `.flac` | Yes | Lossless; fully supported. |
| `.ogg` | Use with care | Lossy; may reduce recall on faint or high-frequency calls. |
| `.mp3` | Use with care | Lossy; same caveat as OGG. |

Case is ignored (`.WAV` and `.wav` are both found).

### Processing behavior

- **Mono conversion:** Stereo recordings are collapsed to mono before spectrogram generation.
- **Arbitrary length:** Clips are processed in overlapping windows.
- **Batch mode:** Point `--audio` at a folder to process every supported file under it in one run.

!!! warning "Lossy audio formats"
    MP3 and OGG are supported via `soundfile`, but the models were trained on lossless WAV. If detections look weak on a compressed file, re-run the same recording as WAV or FLAC before tuning thresholds.

## Model files

`--model` must point to a **YOLO-compatible** weights file loaded by Ultralytics. Common formats:

| Format | Typical use |
| :--- | :--- |
| `.pt` | PyTorch checkpoint (default for BirdBox releases) |
| `.onnx` | Cross-runtime / deployment export |
| `.engine` | TensorRT engine (NVIDIA GPU) |

Other formats supported by your Ultralytics install may work; pretrained releases on [TUC-Cloud](https://tuc.cloud/index.php/s/ET4KE4LdSaysSSL) ship as `.pt`. Custom models can be trained with [BirdBox-Train](https://github.com/birdnet-team/BirdBox-Train).

## Species mapping (`--species-mapping`)

The mapping name must match the label space the model was trained on. It is **not** inferred from the weights filename. You pass it explicitly.
For specific mappings see [Species Mapping](https://github.com/birdnet-team/BirdBox/blob/main/src/config.py#L19){ target="_blank" rel="noopener noreferrer" }.

Allowed values for the species-mapping parameter:

| Value | Region / role |
| :--- | :--- |
| `Just-Bird` | Binary bird vs. background |
| `All-In-One` | Multi-region combined model |
| `Hawaii` | Hawaii |
| `Northeastern-US` | Northeastern United States |
| `Southern-Sierra-Nevada` | Southern Sierra Nevada |
| `Western-US` | Western United States |
| `Amazon-Basin` | Southwestern Amazon Basin |

Each key resolves to `id_to_ebird`, `ebird_to_name`, and display colors in `src/config.py`. The mapping name is stored in detection JSON as `model_config.species_mapping`.

!!! danger "Mapping must match the model"
    If `--species-mapping` does not match the model's training `conf.yaml` / class list, outputs will carry **wrong eBird codes** with no error. Always pair model file and mapping from the same release (e.g. `Western-US.pt` with `Western-US`).

## Parameters (inference inputs)

| Parameter | Type / default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `--audio` | `PATH` / — | **Yes** | Single audio file or directory (recursive search). |
| `--model` | `PATH` / — | **Yes** | YOLO weights (`.pt`, `.onnx`, `.engine`, …). |
| `--species-mapping` | `CHOICE` / — | **Yes** | Dataset key in `src/config.py`; must match training. |

Other detection flags (`--conf`, `--no-merge`, `--output-format`, …) are documented on the **detect-birds** CLI page in this site.

## Examples

=== "Single WAV file"
    ```bash
    python src/inference/detect_birds.py \
        --audio datasets/Western-US_testset/audio/SNE_001_17.wav \
        --model models/Western-US.pt \
        --species-mapping Western-US
    ```
=== "Recursive folder (FLAC + WAV)"
    ```bash
    python src/inference/detect_birds.py \
        --audio path/to/recordings/ \
        --model models/Northeastern-US.pt \
        --species-mapping Northeastern-US \
        --output-path results/batch \
        --workers 2
    ```
