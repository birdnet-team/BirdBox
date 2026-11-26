# Installation

Installing BirdBox takes about ten minutes. Most of that time goes to large libraries such as PyTorch and Ultralytics.

The [install.py](https://github.com/birdnet-team/BirdBox/blob/main/install.py){ target="_blank" rel="noopener noreferrer" } script detects your hardware and installs the matching packages. Pass `--model-format` to install the runtime for a model format. Without it, the script installs support for `.pt` models.

---

## Prerequisites

- **Python 3.12.** `install.py` exits on any other version.
    - On Windows and macOS, use the installer of [Python 3.12.10](https://www.python.org/downloads/release/python-31210/){ target="_blank" rel="noopener noreferrer" }. It is the last 3.12 release with installers. On Windows, keep the `py` launcher option selected, because the commands below use it.
    - On Linux, install Python 3.12 with your package manager or with pyenv. On Debian and Ubuntu, also install the `python3.12-venv` package. Without it, creating the environment fails.
    - The plain `python` command is often a different version. So the commands below call Python 3.12 explicitly: `python3.12` on Linux and macOS, `py -3.12` on Windows.
- **Git**, to clone the repository. Without Git, download the repository from GitHub as a ZIP file (Code, then Download ZIP), unpack it and start at step 2.
- **macOS** works on Apple Silicon (M1 and later) only. PyTorch 2.5.1 and ONNX Runtime 1.27.0 publish no packages for Intel Macs.
- **Disk space** depends on the format and on whether CUDA libraries are included. Measured sizes of the environment: about 2 GB for `tflite` and CPU-only installs, about 6.5 GB for `pt` and `onnx` with CUDA, and about 13 GB for `engine`.

---

## Hardware

- An NVIDIA GPU with CUDA makes inference much faster. Without one, BirdBox runs on the CPU, which takes considerably longer on large files.
- `install.py` looks for an NVIDIA GPU with `nvidia-smi`. If it finds one, it installs CUDA packages. Otherwise it installs CPU packages.
- Macs always get CPU packages and run inference on the CPU. The Apple GPU (MPS) is not used.
- The NVIDIA driver must be new enough for the CUDA version of your format. The [Model Formats](#model-formats) table lists the minimum driver versions.

---

## Installation Commands

Copy the commands for your operating system. They create a virtual environment named `.venv` and run `install.py` inside it. The pipeline scripts (`run_pipeline.sh`, `run_pipeline.ps1`, `run_pipeline.bat`) activate `.venv` by themselves if it exists. Without `.venv`, they use the Python that is active.

Add `--model-format <FORMAT>` to the last command to install a runtime other than `.pt`. See [Install Parameters](#install-parameters) for all options.

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
    #    If PowerShell refuses to run the script, first run:
    #    Set-ExecutionPolicy -Scope Process RemoteSigned
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

=== "Conda"

    ```bash
    # 1. Clone the repository
    git clone https://github.com/birdnet-team/BirdBox.git
    cd BirdBox

    # 2. Create a conda environment with Python 3.12
    conda create -n birdbox python=3.12

    # 3. Activate the environment
    conda activate birdbox

    # 4. Install dependencies  (add --model-format <FORMAT> to change the runtime)
    python install.py
    ```

    The pipeline scripts do not activate conda environments. Activate the environment yourself before you run them.

### Check the Installation

Run this in the activated environment:

```bash
python -c "import torch, ultralytics; print(torch.__version__, torch.cuda.is_available())"
```

It prints the PyTorch version and whether PyTorch can use the GPU. On an NVIDIA system with a CUDA install, the output ends with `True`. On a Mac or with a CPU install, it ends with `False`.

Next, [download a model](#model-download) and run a first detection as in [Simple Workflow for Detection Only](cli/workflows.md#simple-workflow-for-detection-only).

---

## Install Parameters

`install.py` accepts two optional arguments. A plain `python install.py` sets up `.pt` support and detects the GPU.

| Parameter | Type / Default | Required? | Description |
| :--- | :--- | :--- | :--- |
| `--model-format` | `CHOICE` / `pt` | No | Model format to install the inference runtime for. Each value installs a different set of packages. |
| `--mode` | `CHOICE` / `auto` | No | Compute mode for PyTorch and ONNX Runtime. `auto` installs CUDA packages when `nvidia-smi` finds an NVIDIA GPU. `cpu` forces CPU packages. `cuda` forces CUDA packages, for example when `nvidia-smi` is not on the PATH. |

`.tflite` models always run on the CPU. So with `--model-format tflite`, `auto` installs CPU PyTorch even on NVIDIA systems. `--mode cuda` still installs CUDA PyTorch, which `.tflite` models do not use.

### Model Formats

Every environment gets the packages in `requirements.txt`. These include `onnx` and `onnxslim`, which BirdBox uses to read and export ONNX models. Each format adds its runtime on top.

| `--model-format` | Model file | Runtime installed | Platform |
| :--- | :--- | :--- | :--- |
| `pt` (default) | `.pt` | PyTorch. CUDA 11.8 build on NVIDIA systems, needs driver 450 or newer. | Any |
| `onnx` | `.onnx` | PyTorch and ONNX Runtime. CUDA 12.1 PyTorch on NVIDIA systems, needs driver 525 or newer. | Any |
| `tflite` | `.tflite` | CPU PyTorch and `ai-edge-litert` | Any |
| `engine` | `.engine` | CUDA 12.1 PyTorch, `tensorrt-cu12` 11.1 and ONNX Runtime | NVIDIA GPU required |

An environment runs `.pt` models and the format it was installed for. An `engine` environment also runs `.onnx` models. Loading any other format stops with an error that names the install command. To use a second format, create a second environment, for example `.venv-onnx`, and run `install.py` with that format in it.

Conversion benchmarks for the model types are given at [Birds-With-Labels-Model-Types](models-and-metrics/birds-with-labels-model-types.md) and [Just-Bird-Model-Types](models-and-metrics/just-bird-model-types.md).

!!! warning "Platform Restrictions"
    `--model-format engine` requires an NVIDIA GPU and will exit with an error on CPU-only or macOS machines.

    TensorRT is pinned to version 11.1.0.106. A `.engine` file only loads in the TensorRT version that built it, so build your engines in this environment.

!!! info "GPU compute for ONNX models"
    Ultralytics runs `.onnx` models through ONNX Runtime. It only uses the GPU when PyTorch has CUDA as well. So on NVIDIA systems, `--model-format onnx` installs CUDA PyTorch next to `onnxruntime-gpu`. `install.py` tries two ONNX Runtime builds and keeps the first one that runs on the GPU:

    1. `onnxruntime-gpu` 1.27.0 with its own CUDA 13 libraries. It needs driver 580 or newer.
    2. `onnxruntime-gpu` 1.26.0 with CUDA 12 libraries, which it shares with PyTorch. It needs driver 525 or newer.

    If neither works, the script removes the ONNX Runtime GPU packages and installs CPU `onnxruntime`. `.onnx` models then run on the CPU. PyTorch keeps CUDA, so `.pt` models in the same environment still use the GPU.

!!! note "CLI models and the demo graph"
    `--model-format onnx` is for network-only ONNX exports, as made by the Ultralytics exporter. They take a spectrogram image and return raw YOLO output, and `detect_birds.py` does the pre- and postprocessing in Python. The [Demo](demo.md) graph (`docs/models/*.onnx`, built by `src/inference/onnx_export.py`) is a different artefact. It takes raw audio and returns merged detections, with the log-floor spectrogram, NMS and merging inside the graph. It only needs an ONNX runtime and cannot be loaded by `detect_birds.py`.

---

## Model Download

The YOLO models are not included in the BirdBox repository. Download only the models you need. Store them in the [models/](https://github.com/birdnet-team/BirdBox/tree/main/models){ target="_blank" rel="noopener noreferrer" } directory of your clone, where the pipeline scripts and the examples in these docs expect them.

---

### TUC-Cloud

Trained YOLO models can be found on the [TUC-Cloud](https://tuc.cloud/index.php/s/ET4KE4LdSaysSSL){ target="_blank" rel="noopener noreferrer" }.
Each model has its own folder there. File names carry the weight data type: `.pt` comes as `fp32`, `.onnx` and `.tflite` as `fp16` and `int8`.
Start with `Just-Bird_fp32.pt` from the `Just-Bird` folder. [Just-Bird](models-and-metrics/just-bird.md) is the recommended model. The examples in these docs expect it at `models/Just-Bird_fp32.pt`.
For details see [Models and Metrics](models-and-metrics/overview.md).

---

### GitHub Releases

Each [BirdBox release](https://github.com/birdnet-team/BirdBox/releases){ target="_blank" rel="noopener noreferrer" } has the model files it was released with attached as assets, with the same names as on the TUC-Cloud. The TUC-Cloud always has the current files and may be updated later. A release keeps its files unchanged. Download from the release to work with exactly the weights of a given BirdBox version, for example to reproduce published results.

`SHA256SUMS.txt` in each release lists the checksum of every model file. To check downloaded files, run this in the folder that holds them:

=== "Linux"

    ```bash
    sha256sum -c SHA256SUMS.txt --ignore-missing
    ```

    Each checked file should report `OK`.

=== "macOS"

    ```bash
    shasum -a 256 -c SHA256SUMS.txt --ignore-missing
    ```

    Each checked file should report `OK`.

=== "Windows (PowerShell)"

    ```powershell
    Get-FileHash Just-Bird_fp32.pt -Algorithm SHA256
    ```

    Compare the printed hash with the line for that file in `SHA256SUMS.txt`.

---

### Custom Model Training

Alternatively, train your own model on a custom dataset using [BirdBox-Train](https://github.com/birdnet-team/BirdBox-Train){ target="_blank" rel="noopener noreferrer" }.

!!! info "Restricted Access"
    BirdBox-Train is currently only available to members of the BirdNET Team.
