"""
Limit BLAS / OpenMP / OpenCV / PyTorch threads so process pools do not oversubscribe.
"""

from __future__ import annotations

import os
from typing import Optional


_WORKER_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def apply_worker_thread_limits(n: int = 1) -> None:
    """Pin numeric libraries to ``n`` threads. Safe to call more than once."""
    value = str(n)
    for key in _WORKER_ENV_VARS:
        os.environ[key] = value
    try:
        import threadpoolctl

        threadpoolctl.threadpool_limits(limits=n)
    except Exception:
        pass
    try:
        import cv2

        cv2.setNumThreads(n)
    except Exception:
        pass


def configure_parent_inference_threads(num_workers: int, yolo_device: Optional[str]) -> None:
    """
    Stop the parent process from using every core for intra-op threads.

    On GPU, pin the parent to 1 host thread and leave cores for workers.
    On CPU YOLO, leave leftover cores for the model.
    When ``num_workers > 1``, also pin the parent's BLAS so librosa in the
    parent does not compete with the pool.
    """
    try:
        import cv2

        cv2.setNumThreads(1)
    except Exception:
        pass

    try:
        import torch
    except Exception:
        torch = None

    cpu_count = os.cpu_count() or 1
    use_cpu_yolo = False
    if torch is not None:
        use_cpu_yolo = (
            yolo_device == "cpu"
            or (yolo_device is None and not torch.cuda.is_available())
        )
        if use_cpu_yolo:
            infer_threads = max(1, cpu_count - num_workers) if num_workers > 1 else cpu_count
            torch.set_num_threads(infer_threads)
        else:
            torch.set_num_threads(1)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass

    if num_workers > 1:
        apply_worker_thread_limits(1)
        if torch is not None and use_cpu_yolo:
            # Restore YOLO intra-op threads after BLAS pin.
            infer_threads = max(1, cpu_count - num_workers)
            torch.set_num_threads(infer_threads)
