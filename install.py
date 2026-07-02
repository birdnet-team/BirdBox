"""
BirdBox dependency bootstrap script.

Resolves platform-specific ML runtime wheels automatically, based on available
compute hardware and the chosen model format.

Without this script, setup would usually be:
    pip install -r requirements.txt

This script works for both venv and conda environments.
Activate the environment before running it.

Examples:
    python install.py                                    # .pt model, auto-detect GPU
    python install.py --model-format onnx                # .onnx, auto-detect GPU for ORT
    python install.py --model-format tflite              # .tflite, LiteRT CPU runtime
    python install.py --model-format engine              # .engine, NVIDIA only
    python install.py --model-format onnx --mode cuda    # force CUDA for ORT
    python install.py --mode cpu                         # force CPU for all runtimes
"""

import platform
import subprocess
import sys
import argparse
from pathlib import Path

PYTHON_VERSION = (3, 12)

TORCH_VERSION = "2.5.1"
TORCHVISION_VERSION = "0.20.1"
PYTORCH_CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"
PYTORCH_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu118"
# TensorRT cu12 requires a CUDA 12.x driver, so engine installs use cu121 wheels.
PYTORCH_CUDA12_INDEX_URL = "https://download.pytorch.org/whl/cu121"

# These two version constants drive the GPU candidate negotiation logic below.
# If you update onnx.txt, keep ONNXRUNTIME_VERSION in sync.
ONNXRUNTIME_VERSION = "1.27.0"
ONNXRUNTIME_GPU_CUDA12_VERSION = "1.26.0"
# Tried in order on NVIDIA setups (--mode cuda, or auto with a GPU detected).
ONNXRUNTIME_GPU_CANDIDATES = [
    ("CUDA 13", f"onnxruntime-gpu[cuda,cudnn]=={ONNXRUNTIME_VERSION}"),
    ("CUDA 12", f"onnxruntime-gpu[cuda,cudnn]=={ONNXRUNTIME_GPU_CUDA12_VERSION}"),
]

REQUIREMENTS_DIR = Path(__file__).parent / "model_format_requirements"


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


def load_format_packages(format_name):
    """
    Parse the inference requirement file for a model format.
    Returns a list of pip-installable package specs (comments and blank lines skipped).
    """
    path = REQUIREMENTS_DIR / f"{format_name}.txt"
    packages = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                packages.append(line)
    return packages


def pip_install_format(format_name):
    """Install all packages listed in the format's inference requirement file."""
    packages = load_format_packages(format_name)
    if packages:
        pip_install(*packages)


def has_nvidia_gpu():
    """Detect an NVIDIA GPU via nvidia-smi. Works on Linux/Windows with drivers."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
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
        current = (
            f"{sys.version_info.major}.{sys.version_info.minor}"
            f".{sys.version_info.micro}"
        )
        raise SystemExit(
            f"Unsupported Python version: {current}. "
            f"This project requires Python {expected}.x."
        )


# -----------------------------
# PyTorch installer
# -----------------------------

def install_torch(mode, prefer_gpu=True, cuda_index_url=None):
    """
    Install PyTorch and torchvision for the selected compute mode.

    prefer_gpu controls whether NVIDIA systems get CUDA wheels.
    Set prefer_gpu=False for formats whose GPU compute is handled by a
    separate runtime (e.g. onnxruntime-gpu, openvino), so PyTorch only
    runs the CPU-side pre/post-processing without needing CUDA itself.

    cuda_index_url overrides PYTORCH_CUDA_INDEX_URL for callers that need
    a specific CUDA toolkit version (e.g. cu121 for TensorRT cu12 installs).
    """
    index_url = cuda_index_url or PYTORCH_CUDA_INDEX_URL
    cuda_label = index_url.rstrip("/").split("/")[-1]

    if mode == "cpu":
        print("Forcing CPU PyTorch install.")
        pip_install(
            f"torch=={TORCH_VERSION}",
            f"torchvision=={TORCHVISION_VERSION}",
            "--index-url", PYTORCH_CPU_INDEX_URL,
        )
        return

    if mode == "cuda":
        print(f"Forcing CUDA PyTorch install ({cuda_label}).")
        pip_install(
            f"torch=={TORCH_VERSION}",
            f"torchvision=={TORCHVISION_VERSION}",
            "--index-url", index_url,
        )
        return

    # AUTO mode
    if is_macos():
        print("Detected macOS -> installing CPU/MPS PyTorch.")
        pip_install(
            f"torch=={TORCH_VERSION}",
            f"torchvision=={TORCHVISION_VERSION}",
        )
        return

    if prefer_gpu and has_nvidia_gpu():
        print(f"Detected NVIDIA GPU -> installing CUDA PyTorch ({cuda_label}).")
        pip_install(
            f"torch=={TORCH_VERSION}",
            f"torchvision=={TORCHVISION_VERSION}",
            "--index-url", index_url,
        )
        return

    if not prefer_gpu and has_nvidia_gpu():
        print(
            "NVIDIA GPU detected, but this model format uses a separate GPU runtime.\n"
            "Installing CPU PyTorch to avoid redundant CUDA libraries."
        )

    print("Installing CPU PyTorch.")
    pip_install(
        f"torch=={TORCH_VERSION}",
        f"torchvision=={TORCHVISION_VERSION}",
        "--index-url", PYTORCH_CPU_INDEX_URL,
    )


# -----------------------------
# ONNX Runtime installer
# -----------------------------

def verify_onnxruntime_gpu():
    """
    Return True only when CUDAExecutionProvider actually works end-to-end.

    get_available_providers() lists CUDA even when the underlying library
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
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 11)]
        )
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
    packages = load_format_packages("onnx")
    print(f"Installing ONNX Runtime CPU ({', '.join(packages)}).")
    pip_uninstall("onnxruntime-gpu")
    pip_install(*packages)


def try_install_onnxruntime_gpu():
    """Try each GPU wheel candidate until CUDAExecutionProvider is available."""
    for label, package in ONNXRUNTIME_GPU_CANDIDATES:
        if not pip_dry_run(package):
            print(f"Skipping ONNX Runtime GPU ({label}): pip extras not available.")
            continue

        try:
            print(
                f"Installing ONNX Runtime GPU with bundled {label} libraries "
                f"({package})."
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
    NVIDIA setups attempt GPU wheels first, then fall back to CPU.
    """
    if not should_use_cuda(mode):
        install_onnxruntime_cpu()
        return

    if try_install_onnxruntime_gpu():
        return

    print(
        "WARNING: ONNX Runtime GPU is not usable on this system.\n"
        "         Falling back to CPU onnxruntime.\n"
        "         .onnx models will run on CPU only."
    )
    install_onnxruntime_cpu()


# -----------------------------
# Base project requirements
# -----------------------------

def install_requirements():
    print("Installing project dependencies from requirements.txt.")
    pip_install("-r", "requirements.txt")


# -----------------------------
# Per-format install routines
# -----------------------------

def install_for_pt(mode):
    """
    Native PyTorch .pt inference.
    GPU compute is handled directly by PyTorch/CUDA.
    """
    print("Setting up runtime for .pt model format.")
    install_torch(mode, prefer_gpu=True)


def install_for_onnx(mode):
    """
    ONNX Runtime .onnx inference.
    GPU compute is handled by onnxruntime-gpu, not PyTorch, so CPU PyTorch
    is sufficient and avoids shipping redundant CUDA libraries.
    """
    print("Setting up runtime for .onnx model format.")
    install_torch(mode, prefer_gpu=False)
    install_onnxruntime(mode)


def install_for_tflite(mode):
    """
    LiteRT .tflite inference.
    TFLite models run on CPU; CUDA is not involved.
    """
    print("Setting up runtime for .tflite model format.")
    install_torch(mode, prefer_gpu=False)
    pip_install_format("tflite")


def install_for_engine(mode):
    """
    TensorRT .engine inference. Requires an NVIDIA GPU.
    .engine files are compiled for a specific GPU and cannot run without one.
    PyTorch is installed from the cu121 index to align with tensorrt-cu12,
    which requires a CUDA 12.x driver.
    """
    if not should_use_cuda(mode):
        raise SystemExit(
            "ERROR: --model-format engine requires an NVIDIA GPU and CUDA.\n"
            "       TensorRT .engine files are compiled for a specific GPU.\n"
            "       Run on an NVIDIA machine, or pass --mode cuda to override."
        )
    print("Setting up runtime for TensorRT .engine model format.")
    install_torch(mode, prefer_gpu=True, cuda_index_url=PYTORCH_CUDA12_INDEX_URL)
    pip_install_format("engine")
    install_onnxruntime(mode)


FORMAT_INSTALLERS = {
    "pt":     install_for_pt,
    "onnx":   install_for_onnx,
    "tflite": install_for_tflite,
    "engine": install_for_engine,
}


# -----------------------------
# Environment info
# -----------------------------

def print_env_info(model_format, mode):
    nvidia = has_nvidia_gpu()
    print("\n==============================")
    print("BirdBox installation starting")
    print("==============================")
    print(f"OS:           {platform.system()} {platform.release()}")
    print(f"Python:       {sys.version.split()[0]}")
    print(f"Model format: {model_format}")
    print(f"Mode:         {mode}")
    print(f"NVIDIA GPU:   {'yes' if nvidia else 'no'}")
    print("==============================\n")


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="BirdBox environment installer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
model formats and inference runtimes installed:
  pt      PyTorch (.pt)      GPU-aware PyTorch (CUDA when available)
  onnx    ONNX Runtime (.onnx)   CPU PyTorch + GPU-aware onnxruntime
  tflite  LiteRT (.tflite)   CPU PyTorch + ai-edge-litert
  engine  TensorRT (.engine) CUDA 12 PyTorch + tensorrt-cu12 + onnxruntime-gpu  [NVIDIA only]
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help=(
            "Compute mode: auto-detect GPU (default), force CPU, or force CUDA. "
            "Affects which PyTorch and ONNX Runtime wheels are installed."
        ),
    )
    parser.add_argument(
        "--model-format",
        choices=list(FORMAT_INSTALLERS.keys()),
        default="pt",
        metavar="FORMAT",
        help=(
            "Model format to install the inference runtime for. "
            f"Choices: {', '.join(FORMAT_INSTALLERS.keys())}. "
            "Default: pt."
        ),
    )

    args = parser.parse_args()

    print_env_info(args.model_format, args.mode)
    require_python_312()

    pip_install("--upgrade", "pip")

    FORMAT_INSTALLERS[args.model_format](args.mode)

    install_requirements()

    print("\n===================================")
    print(f"BirdBox installation complete  (format: {args.model_format})")
    print("===================================\n")


if __name__ == "__main__":
    main()
