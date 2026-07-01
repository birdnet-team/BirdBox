"""
BirdBox dependency bootstrap script.

This script exists to resolve platform-specific ML runtime wheels automatically:
- PyTorch: CPU/MPS on macOS or non-NVIDIA systems, CUDA (cu118) on NVIDIA systems
- ONNX Runtime: CPU onnxruntime on CPU-only setups; on NVIDIA setups tries GPU
  wheels in order (CUDA 13 for 1.27.x, then CUDA 12 for 1.26.x), then CPU fallback

Without that resolution, setup would usually be just:
pip install -r requirements.txt

This script works for both venv and conda environments.
Just activate the environment before running the script.

Example:
```bash
conda activate my_env
python install.py
```
"""

import platform
import subprocess
import sys
import argparse

PYTHON_VERSION = (3, 12)
TORCH_VERSION = "2.5.1"
TORCHVISION_VERSION = "0.20.1"
PYTORCH_CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"
PYTORCH_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu118"
ONNXRUNTIME_VERSION = "1.27.0"
ONNXRUNTIME_GPU_CUDA12_VERSION = "1.26.0"
# Tried in order on NVIDIA setups only (--mode cuda, or --mode auto with a GPU).
ONNXRUNTIME_GPU_CANDIDATES = [
    (
        "CUDA 13",
        f"onnxruntime-gpu[cuda,cudnn]=={ONNXRUNTIME_VERSION}",
    ),
    (
        "CUDA 12",
        f"onnxruntime-gpu[cuda,cudnn]=={ONNXRUNTIME_GPU_CUDA12_VERSION}",
    ),
]


# -----------------------------
# Helpers
# -----------------------------

def run(cmd):
    print(f"\n>>> {' '.join(cmd)}\n")
    subprocess.check_call(cmd)


def pip_install(*args):
    run([sys.executable, "-m", "pip", "install", *args])


def pip_uninstall(*packages):
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", *packages],
        check=False,
    )


def pip_dry_run(*args):
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--dry-run", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
    return result.returncode == 0


def has_nvidia_gpu():
    """
    Detect NVIDIA GPU via nvidia-smi.
    Works on most Linux/Windows systems with drivers installed.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def is_macos():
    return platform.system() == "Darwin"


def should_use_cuda(mode):
    """
    Resolve whether CUDA-specific wheels should be installed.
    mode: cpu | cuda | auto
    """
    if mode == "cpu":
        return False
    if mode == "cuda":
        return True
    if is_macos():
        return False
    return has_nvidia_gpu()


def require_python_312():
    if sys.version_info[:2] != PYTHON_VERSION:
        expected = f"{PYTHON_VERSION[0]}.{PYTHON_VERSION[1]}"
        current = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        raise SystemExit(
            f"Unsupported Python version: {current}. "
            f"This project requires Python {expected}.x."
        )


# -----------------------------
# PyTorch installer logic
# -----------------------------

def install_torch(mode):
    """
    mode: cpu | cuda | auto
    """

    if mode == "cpu":
        print("Forcing CPU PyTorch install (matching environment-cpu.yml)")
        pip_install(
            f"torch=={TORCH_VERSION}",
            f"torchvision=={TORCHVISION_VERSION}",
            "--index-url",
            PYTORCH_CPU_INDEX_URL
        )
        return

    if mode == "cuda":
        print("Forcing CUDA PyTorch install (cu118, matching environment-gpu.yml)")
        pip_install(
            f"torch=={TORCH_VERSION}",
            f"torchvision=={TORCHVISION_VERSION}",
            "--index-url",
            PYTORCH_CUDA_INDEX_URL
        )
        return

    # AUTO mode
    if is_macos():
        print("Detected macOS -> installing CPU/MPS PyTorch")
        pip_install(
            f"torch=={TORCH_VERSION}",
            f"torchvision=={TORCHVISION_VERSION}"
        )
        return

    if has_nvidia_gpu():
        print("Detected NVIDIA GPU -> installing CUDA PyTorch (cu118)")
        pip_install(
            f"torch=={TORCH_VERSION}",
            f"torchvision=={TORCHVISION_VERSION}",
            "--index-url",
            PYTORCH_CUDA_INDEX_URL
        )
        return

    print("No GPU detected -> installing CPU PyTorch")
    pip_install(
        f"torch=={TORCH_VERSION}",
        f"torchvision=={TORCHVISION_VERSION}",
        "--index-url",
        PYTORCH_CPU_INDEX_URL
    )


# -----------------------------
# ONNX Runtime installer logic
# -----------------------------

def verify_onnxruntime_gpu():
    """
    Return True only when CUDAExecutionProvider actually works end-to-end.

    get_available_providers() lists CUDA even when the underlying CUDA library
    fails to load. A real InferenceSession test is the only reliable check:
    ORT silently falls back to CPUExecutionProvider on failure, so we inspect
    the active providers on the session instead of the global list.
    """
    try:
        import onnx
        import onnxruntime as ort
        from onnx import TensorProto, helper

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()

        if "CUDAExecutionProvider" not in ort.get_available_providers():
            return False

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1])
        node = helper.make_node("Identity", ["X"], ["Y"])
        graph = helper.make_graph([node], "cuda_probe", [X], [Y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
        model_bytes = model.SerializeToString()

        sess = ort.InferenceSession(
            model_bytes,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        return "CUDAExecutionProvider" in sess.get_providers()
    except Exception as exc:
        print(f"ONNX Runtime GPU verification failed: {exc}")
        return False


def install_onnxruntime_cpu():
    print(f"Installing ONNX Runtime CPU (onnxruntime=={ONNXRUNTIME_VERSION})")
    pip_uninstall("onnxruntime-gpu")
    pip_install(f"onnxruntime=={ONNXRUNTIME_VERSION}")


def try_install_onnxruntime_gpu():
    """
    Try each GPU wheel candidate until CUDAExecutionProvider is available.
    Returns True on success.
    """
    for label, package in ONNXRUNTIME_GPU_CANDIDATES:
        if not pip_dry_run(package):
            print(f"Skipping ONNX Runtime GPU ({label}): pip extras not available.")
            continue

        try:
            print(
                f"Installing ONNX Runtime GPU with bundled {label} libraries "
                f"({package})"
            )
            pip_install(package)
            if verify_onnxruntime_gpu():
                return True

            print(
                f"WARNING: ONNX Runtime GPU ({label}) installed but "
                "CUDAExecutionProvider is unavailable."
            )
        except subprocess.CalledProcessError:
            print(f"WARNING: ONNX Runtime GPU ({label}) install failed.")

        pip_uninstall("onnxruntime-gpu", "onnxruntime")

    return False


def install_onnxruntime(mode):
    """
    Install the ONNX Runtime wheel matching the selected compute mode.
    mode: cpu | cuda | auto

    CPU-only setups (--mode cpu, macOS, or no NVIDIA GPU) install CPU
    onnxruntime only. NVIDIA setups try GPU wheels first, then CPU fallback.
    """
    if not should_use_cuda(mode):
        install_onnxruntime_cpu()
        return

    if try_install_onnxruntime_gpu():
        return

    print(
        "WARNING: ONNX Runtime GPU is not usable on this system.\n"
        "         Falling back to CPU onnxruntime.\n"
        "         .pt models still use GPU via PyTorch. .onnx models run on CPU."
    )
    install_onnxruntime_cpu()


# -----------------------------
# Main dependency install
# -----------------------------

def install_requirements():
    print("Installing project dependencies from requirements.txt")
    pip_install("-r", "requirements.txt")


# -----------------------------
# Environment check
# -----------------------------

def print_env_info():
    print("\n==============================")
    print("BirdBox installation starting")
    print("==============================")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print("==============================\n")


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Installation mode (default: auto)"
    )

    args = parser.parse_args()

    print_env_info()
    require_python_312()

    # Upgrade pip first (important for PyTorch wheels)
    pip_install("--upgrade", "pip")

    # Install PyTorch first (critical dependency order)
    install_torch(args.mode)

    # Install ONNX Runtime before requirements.txt (avoids Ultralytics auto-install)
    install_onnxruntime(args.mode)

    # Install the rest of your stack
    install_requirements()

    print("\n===================================")
    print("BirdBox installation complete ✔")
    print("===================================\n")


if __name__ == "__main__":
    main()