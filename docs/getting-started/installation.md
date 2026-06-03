# Installation

The installation process may take roughly ten minutes because BirdBox relies on large deep learning libraries such as PyTorch and Ultralytics.

---

## Prerequisites

- [Python 3.12](https://www.python.org/downloads/release/python-31213/){ target="_blank" rel="noopener noreferrer" } has to be installed in advance
- ~4 GB disk space for dependencies (PyTorch, Ultralytics and CUDA binaries)

---

## Recommended

- CUDA-capable GPU for accelerated model inference 
- a non-CUDA setup is also possible, but model inference will take significantly longer
- the [installation script](https://github.com/birdnet-team/BirdBox/blob/main/install.py){ target="_blank" rel="noopener noreferrer" } will automatically detect GPU/CPU/Mac environments and install the appropriate dependencies and CUDA binaries

---

## Installation Scripts

Simply copy the script below that matches your operating system. 
All scripts will create a new virtual environment and install the dependencies into it via [install.py](https://github.com/birdnet-team/BirdBox/blob/main/install.py){ target="_blank" rel="noopener noreferrer" }. 
Alternatively, you can also run install.py inside a conda environment.

=== "Linux / macOS"

    ```bash
    # 1. Clone the repository
    git clone https://github.com/birdnet-team/BirdBox.git
    cd BirdBox

    # 2. Create a virtual environment
    python3 -m venv .venv

    # 3. Activate the environment
    source .venv/bin/activate

    # 4. Install dependencies
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

    # 4. Install dependencies
    python install.py
    ```

=== "Windows (CMD)"

    ```cmd
    :: 1. Clone the repository
    git clone https://github.com/birdnet-team/BirdBox.git
    cd BirdBox

    :: 2. Create a virtual environment
    python -m venv .venv

    :: 3. Activate the environment
    .venv\Scripts\activate.bat

    :: 4. Install dependencies
    python install.py
    ```

---

## Model Download

The YOLO-models are not included in the BirdBox GitHub-Repository.
This yields the advantage that only the required models have to be downloaded.

Recommended: Once downloaded, store the model files in your own local [models/](https://github.com/birdnet-team/BirdBox/tree/main/models){ target="_blank" rel="noopener noreferrer" } directory.

---

### TUC-Cloud

Trained YOLO-Models for this task can be found on the [TUC-Cloud](https://tuc.cloud/index.php/s/ET4KE4LdSaysSSL){ target="_blank" rel="noopener noreferrer" }.
For more details see [Models and Metrics](../models-and-metrics/index.md).

---

### Custom Model Training

Alternatively, you can train your own model on a custom dataset by using the code available in the [BirdBox-Train](https://github.com/birdnet-team/BirdBox-Train){ target="_blank" rel="noopener noreferrer" } repository (currently only available for the BirdNET Team).

