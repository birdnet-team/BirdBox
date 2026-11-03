# Installation

The installation process may take roughly ten minutes because BirdBox depends on large deep learning libraries such as PyTorch and Ultralytics.

The [install.py](https://github.com/birdnet-team/BirdBox/blob/main/install.py){ target="_blank" rel="noopener noreferrer" } script auto-detects your hardware and installs the correct wheels. Pass `--model-format` to target the runtime for your chosen model type. The default installs native PyTorch (`.pt`) support.

---

## Prerequisites

- [Python 3.12](https://www.python.org/downloads/release/python-31213/){ target="_blank" rel="noopener noreferrer" } must be installed before running the script. `install.py` exits on any other version. The plain `python` command is often a different version, so the scripts below call Python 3.12 explicitly (`python3.12` on Linux and macOS, `py -3.12` on Windows).
- macOS is supported on Apple Silicon (M1 and later) only. PyTorch 2.5.1 and ONNX Runtime 1.27.0 publish no wheels for Intel Macs.
- Disk space varies by format and by whether CUDA libraries are included. Measured sizes of the environment: about 2 GB for `tflite` and CPU-only installs, about 6.5 GB for `pt` and `onnx` with CUDA, and about 13 GB for `engine`.

---

## Recommended

- A CUDA-capable GPU significantly speeds up model inference.
- A CPU-only setup works, but inference takes considerably longer on large files.
- The install script detects your hardware automatically. NVIDIA GPUs receive CUDA wheels. Apple Silicon Macs receive CPU/MPS wheels. All other systems receive CPU wheels by default.

---

## Installation Scripts

Copy the script below that matches your operating system. Each script creates a virtual environment and runs `install.py` inside it. Alternatively, activate a conda environment and run `install.py` directly.

Pass `--model-format <FORMAT>` to install a runtime other than the default `.pt`. See [Install Parameters](#install-parameters) below for all options.

=== "Linux / macOS"

    ```bash
    # 1. Clone the repository
    git clone https://github.com/birdnet-team/BirdBox.git
    cd BirdBox

    # 2. Create a virtual environment with Python 3.12
    python3.12 -m venv .venv

    # 3. Activate the environment
    source .venv/bin/activate

    # 4. Install dependencies  (add --model-format <FORMAT> to change the runtime)
    python install.py
    ```

=== "Windows (PowerShell)"

    ```powershell
    # 1. Clone the repository
    git clone https://github.com/birdnet-team/BirdBox.git
    cd BirdBox

    # 2. Create a virtual environment with Python 3.12
    py -3.12 -m venv .venv

    # 3. Activate the environment
    .venv\Scripts\Activate.ps1

    # 4. Install dependencies  (add --model-format <FORMAT> to change the runtime)
    python install.py
    ```

=== "Windows (CMD)"

    ```cmd
    rem 1. Clone the repository
    git clone https://github.com/birdnet-team/BirdBox.git
    cd BirdBox

    rem 2. Create a virtual environment with Python 3.12
    py -3.12 -m venv .venv

    rem 3. Activate the environment
    .venv\Scripts\activate.bat

    rem 4. Install dependencies  (add --model-format <FORMAT> to change the runtime)
    python install.py
    ```

---

## Install Parameters

`install.py` accepts two optional arguments. Both default to sensible values so a plain `python install.py` always works.

| Parameter | Type / Default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `--model-format` | `CHOICE` / `pt` | No | Model format to install the inference runtime for. Each value installs a different set of packages. |
| `--mode` | `CHOICE` / `auto` | No | Compute mode: `auto` detects the GPU, `cpu` forces CPU-only wheels, `cuda` forces CUDA wheels. Affects which PyTorch and ONNX Runtime wheels are installed. |

### Model Formats

Each format installs exactly the packages required to run inference on that model type. Packages needed only for model conversion or export are excluded.

| `--model-format` | Model file | Extra runtime installed | Platform |
| :--- | :--- | :--- | :--- |
| `pt` (default) | `.pt` | GPU-aware PyTorch via CUDA | Any |
| `onnx` | `.onnx` | GPU-aware PyTorch (CUDA 12) + GPU-aware `onnxruntime` | Any |
| `tflite` | `.tflite` | CPU PyTorch + `ai-edge-litert` | Any |
| `engine` | `.engine` | CUDA 12 PyTorch + `tensorrt-cu12` 11.1 + `onnxruntime-gpu` | NVIDIA GPU required |

A conversion benchmark for some individual model types is given at [Birds-With-Labels-Model-Types](models-and-metrics/birds-with-labels-model-types.md) and [Just-Bird-Model-Types](models-and-metrics/just-bird-model-types.md).

!!! warning "Platform Restrictions"
    `--model-format engine` requires an NVIDIA GPU and will exit with an error on CPU-only or macOS machines.

    TensorRT is pinned to version 11.1.0.106. A `.engine` file only loads in the TensorRT version that built it, so build your engines in this environment.

!!! info "GPU compute for ONNX models"
    Ultralytics runs `.onnx` models through ONNX Runtime, but only picks the CUDA execution provider when PyTorch itself has CUDA. On NVIDIA systems, `--model-format onnx` therefore installs CUDA 12 PyTorch next to `onnxruntime-gpu`. Both share the same CUDA 12 libraries, which need an NVIDIA driver for CUDA 12 (version 525 or newer). If the GPU runtime does not work on your system, the script falls back to CPU `onnxruntime` and removes the unused CUDA libraries again. Pass `--mode cuda` to force the GPU install on systems where auto-detection does not pick up the GPU, or `--mode cpu` for a smaller CPU-only environment.

!!! note "CLI models and the demo graph"
    `--model-format onnx` is for network-only ONNX exports, as made by the Ultralytics exporter: they take a spectrogram image and return raw YOLO output, and `detect_birds.py` does the pre- and postprocessing in Python. The [Demo](demo.md) graph (`docs/models/*.onnx`, built by `src/inference/onnx_export.py`) is a different artefact: it takes raw audio and returns merged detections, with PCEN, spectrogram rendering, NMS and merging inside the graph. It only needs an ONNX runtime and cannot be loaded by `detect_birds.py`.

---

## Model Download

The YOLO models are not included in the BirdBox repository. Only the models you need have to be downloaded.

Recommended: Once downloaded, store the model files in your own local [models/](https://github.com/birdnet-team/BirdBox/tree/main/models){ target="_blank" rel="noopener noreferrer" } directory.

---

### TUC-Cloud

Trained YOLO models can be found on the [TUC-Cloud](https://tuc.cloud/index.php/s/ET4KE4LdSaysSSL){ target="_blank" rel="noopener noreferrer" }.
Each model has its own folder there. File names carry the weight data type: `.pt` comes as `fp32`, `.onnx` and `.tflite` as `fp16` and `int8`.
Start with `Just-Bird_fp32.pt` from the `Just-Bird` folder. [Just-Bird](models-and-metrics/just-bird.md) is the recommended model. Place the file in the `models/` directory, where the pipeline scripts and the examples in these docs expect `models/Just-Bird_fp32.pt`.
For details see [Models and Metrics](models-and-metrics/overview.md).

---

### GitHub Releases

Each [BirdBox release](https://github.com/birdnet-team/BirdBox/releases){ target="_blank" rel="noopener noreferrer" } has the model files it was released with attached as assets, with the same names as on the TUC-Cloud. The TUC-Cloud always has the current files and may be updated later; a release keeps its files unchanged. Download from the release to work with exactly the weights of a given BirdBox version, for example to reproduce published results.

`SHA256SUMS.txt` in each release lists the checksum of every model file. To check downloaded files, run this in the folder that holds them:

```bash
sha256sum -c SHA256SUMS.txt --ignore-missing
```

`.engine` files are not attached, because they only run on the GPU and TensorRT version they were built for.

---

### Custom Model Training

Alternatively, train your own model on a custom dataset using [BirdBox-Train](https://github.com/birdnet-team/BirdBox-Train){ target="_blank" rel="noopener noreferrer" }.

!!! info "Restricted Access"
    BirdBox-Train is currently only available to members of the BirdNET Team.
