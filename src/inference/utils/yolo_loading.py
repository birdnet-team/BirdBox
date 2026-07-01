"""
Helpers for loading YOLO models across .pt, .onnx, and .engine formats.

Ultralytics requests onnxruntime-gpu whenever PyTorch CUDA is available, even if
only CPU onnxruntime is installed. That triggers AutoUpdate and can replace a
working CPU package with a broken GPU wheel (missing libcudart.so.13).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

# Keep in sync with install.py ONNXRUNTIME_VERSION
ONNXRUNTIME_VERSION = "1.27.0"


def is_onnx_model(model_path: str) -> bool:
    return Path(model_path).suffix.lower() == ".onnx"


def disable_yolo_autoinstall() -> None:
    os.environ["YOLO_AUTOINSTALL"] = "false"


def onnxruntime_import_ok() -> bool:
    try:
        import onnxruntime as ort

        ort.get_available_providers()
        return True
    except Exception:
        return False


def _preload_ort_dlls() -> None:
    """
    Load bundled CUDA libraries from pip packages (nvidia-cublas-cu12, etc.).
    onnxruntime-gpu 1.19+ ships CUDA runtime libs as pip extras that land in
    site-packages/nvidia/*/lib/. Without preloading, libcublasLt.so.12 and
    friends are not on LD_LIBRARY_PATH and the CUDA provider fails to load.
    """
    try:
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
    except Exception:
        pass


def onnxruntime_cuda_available() -> bool:
    """
    Return True only when CUDAExecutionProvider actually works end-to-end.

    get_available_providers() lists CUDA even when the underlying library
    (libcublasLt.so.12, libcudart.so.13, …) fails to load, which causes
    Ultralytics to set up IO binding for CUDA and then crash on a CPU session.
    Creating a real InferenceSession with CUDAExecutionProvider exposes the
    true state: ORT silently falls back to CPU when the provider cannot load,
    so we verify the active providers on the session rather than the global list.
    """
    try:
        import onnx
        import onnxruntime as ort
        from onnx import TensorProto, helper

        _preload_ort_dlls()

        if "CUDAExecutionProvider" not in ort.get_available_providers():
            return False

        # Build a minimal single-node ONNX model to test the CUDA provider.
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
    except Exception:
        return False


def repair_onnxruntime_cpu() -> None:
    """Replace a broken onnxruntime-gpu install with the CPU package."""
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime-gpu", "onnxruntime"],
        check=False,
        capture_output=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", f"onnxruntime=={ONNXRUNTIME_VERSION}"],
        check=True,
    )


def prepare_yolo_for_model(model_path: str) -> Optional[str]:
    """
    Prepare the environment for a YOLO model and return an optional device override.

    Returns:
        None to use Ultralytics defaults (.pt / .engine, or ONNX with working GPU ORT).
        "cpu" to force CPU ONNX inference when GPU ORT is unavailable.
    """
    if not is_onnx_model(model_path):
        return None

    disable_yolo_autoinstall()

    if not onnxruntime_import_ok():
        print("Repairing ONNX Runtime (replacing broken GPU install with CPU package)...")
        repair_onnxruntime_cpu()

    _preload_ort_dlls()

    if onnxruntime_cuda_available():
        return None

    print(
        "ONNX model: GPU ONNX Runtime unavailable. Using CPU for .onnx inference "
        "(.pt models still use GPU via PyTorch)."
    )
    return "cpu"


def load_yolo(model_path: str) -> YOLO:
    """Load a YOLO model. ONNX exports get task=detect to avoid Ultralytics warnings."""
    kwargs = {"task": "detect"} if is_onnx_model(model_path) else {}
    return YOLO(model_path, **kwargs)


def yolo_predict_kwargs(device: Optional[str], **kwargs):
    """Build keyword arguments for YOLO predict, including an optional device override."""
    if device:
        kwargs["device"] = device
    return kwargs
