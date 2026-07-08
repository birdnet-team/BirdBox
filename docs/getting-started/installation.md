# Installation

The installation process may take roughly ten minutes because BirdBox depends on large deep learning libraries such as PyTorch and Ultralytics.

The [install.py](https://github.com/birdnet-team/BirdBox/blob/main/install.py){ target="_blank" rel="noopener noreferrer" } script auto-detects your hardware and installs the correct wheels. Pass `--model-format` to target the runtime for your chosen model type. The default installs native PyTorch (`.pt`) support.

---

## Prerequisites

- [Python 3.12](https://www.python.org/downloads/release/python-31213/){ target="_blank" rel="noopener noreferrer" } must be installed before running the script.
- Disk space varies by format. Expect 2–6 GB depending on whether CUDA libraries are included.

---

## Recommended

- A CUDA-capable GPU significantly speeds up model inference.
- A CPU-only setup works, but inference takes considerably longer on large files.
- The install script detects your hardware automatically. NVIDIA GPUs receive CUDA wheels. macOS systems receive CPU/MPS wheels. All other systems receive CPU wheels by default.

---

## Installation Scripts

Copy the script below that matches your operating system. Each script creates a virtual environment and runs `install.py` inside it. Alternatively, activate a conda environment and run `install.py` directly.

Pass `--model-format <FORMAT>` to install a runtime other than the default `.pt`. See [Install Parameters](#install-parameters) below for all options.

=== "Linux / macOS"

    ```bash
    # 1. Clone the repository
    git clone https://github.com/birdnet-team/BirdBox.git
    cd BirdBox

    # 2. Create a virtual environment
    python -m venv .venv

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

    # 2. Create a virtual environment
    python -m venv .venv

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

    rem 2. Create a virtual environment
    python -m venv .venv

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
| `onnx` | `.onnx` | CPU PyTorch + GPU-aware `onnxruntime` | Any |
| `tflite` | `.tflite` | CPU PyTorch + `ai-edge-litert` | Any |
| `engine` | `.engine` | CUDA PyTorch + `tensorrt-cu12` + `onnxruntime-gpu` | NVIDIA GPU required |

A conversion benchmark for some individual model types is given at [All-in-One-Model-Types](../models-and-metrics/all-in-one-model-types.md) and [Just-Bird-Model-Types](../models-and-metrics/just-bird-model-types.md).

!!! warning "Platform Restrictions"
    `--model-format engine` requires an NVIDIA GPU and will exit with an error on CPU-only or macOS machines.

!!! info "GPU compute for ONNX models"
    When `--model-format onnx` is selected, PyTorch is installed as a CPU-only wheel. The GPU compute is handled by `onnxruntime-gpu` instead. This avoids shipping redundant CUDA libraries. Pass `--mode cuda` to force GPU ONNX Runtime on systems where auto-detection does not pick up the GPU.

---

## Model Download

The YOLO models are not included in the BirdBox repository. Only the models you need have to be downloaded.

Recommended: Once downloaded, store the model files in your own local [models/](https://github.com/birdnet-team/BirdBox/tree/main/models){ target="_blank" rel="noopener noreferrer" } directory.

---

### TUC-Cloud

Trained YOLO models can be found on the [TUC-Cloud](https://tuc.cloud/index.php/s/ET4KE4LdSaysSSL){ target="_blank" rel="noopener noreferrer" }.
For details see [Models and Metrics](../models-and-metrics/overview.md).

---

### Custom Model Training

Alternatively, train your own model on a custom dataset using [BirdBox-Train](https://github.com/birdnet-team/BirdBox-Train){ target="_blank" rel="noopener noreferrer" }.

!!! info "Restricted Access"
    BirdBox-Train is currently only available to members of the BirdNET Team.
