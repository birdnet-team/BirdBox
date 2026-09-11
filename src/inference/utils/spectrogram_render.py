"""
Spectrogram image rendering for inference.

Kept in a torch-free module so ProcessPoolExecutor spawn workers do not
import Ultralytics/PyTorch or initialize CUDA.
"""

from __future__ import annotations

from typing import Dict

try:
    from inference.utils.thread_limits import apply_worker_thread_limits
except ImportError:
    from utils.thread_limits import apply_worker_thread_limits

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import librosa.display
import numpy as np


def render_spectrogram_image(
    pcen_data: np.ndarray,
    output_path: str,
    sr: int,
    hop_length: int,
    colormap: str,
    vmin: float,
    vmax: float,
) -> None:
    """Write a training-matched spectrogram PNG from PCEN features."""
    fig = Figure(figsize=(2.56, 2.56), dpi=100)
    FigureCanvas(fig)
    ax = fig.add_subplot(111)

    try:
        librosa.display.specshow(
            pcen_data,
            sr=sr,
            hop_length=hop_length,
            ax=ax,
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=100)
    finally:
        fig.clear()
        plt.close(fig)


def render_spectrogram_worker(job: Dict) -> str:
    """
    Process-pool entry point. ``job`` must be picklable (no detector/model).

    Returns the written image path.
    """
    apply_worker_thread_limits(1)
    render_spectrogram_image(
        pcen_data=job["pcen"],
        output_path=job["output_path"],
        sr=job["sr"],
        hop_length=job["hop_length"],
        colormap=job["colormap"],
        vmin=job["vmin"],
        vmax=job["vmax"],
    )
    return job["output_path"]
