#!/usr/bin/env python3
"""
Export a trained YOLO bird detection model to a self-contained ONNX graph.

The exported graph takes raw audio samples and returns merged song segments.
Everything that ``src/inference/detect_birds.py`` does around the network is
part of the graph: PCEN feature extraction, spectrogram image rendering, clip
windowing, non-maximum suppression, pixel to time/frequency mapping, and song
reconstruction. A consumer therefore only needs an ONNX runtime, no librosa,
matplotlib, or Ultralytics.

The Ultralytics exporter is deliberately not used here. It only converts the
network itself and leaves pre- and postprocessing to the caller.

Graph interface
---------------
Input  ``audio``       float32, shape ``(num_samples,)``
                       Mono waveform, 32000 Hz, range [-1, 1], at least 3 s.
Input  ``conf``        float32, shape ``(1,)``
                       Confidence threshold. Required. Suggested default 0.18
                       (also stored in ONNX metadata as ``default_conf``).
Input  ``song_gap``    float32, shape ``(1,)``
                       Max gap in seconds between detections to merge into the
                       same song. Required. Suggested default 0.1 (also stored
                       in ONNX metadata as ``default_song_gap``).
Output ``detections``  float32, shape ``(num_detections, 8)`` with columns
                       ``[time_start, time_end, freq_low_hz, freq_high_hz,
                       avg_confidence, max_confidence, detections_merged,
                       class_id]``.

Times are seconds from the start of the input audio. Frequencies are Hz.
Rows are reconstructed songs, matching the default
``detect_birds.py`` / Streamlit merge path.

Reimplemented dependencies
--------------------------
ONNX cannot express librosa, scipy, or matplotlib calls, so the pieces used by
the detection pipeline are reimplemented here with numpy (constants baked into
the graph) and torch (graph operations):

* flattop analysis window (``scipy.signal.get_window``)
* STFT, as a strided convolution with a DFT basis (``librosa.stft``)
* HTK mel filterbank (``librosa.filters.mel``)
* PCEN, as a parallel-prefix EMA instead of ``scipy.signal.lfilter``
* colormap lookup and figure rasterization (``librosa.display.specshow``)
* NMS and box decoding (``ultralytics`` postprocessing)
* song reconstruction (``reconstruct_songs`` in ``detect_birds.py``)

The rendered spectrogram images are bit-identical to the PNG files the reference
pipeline writes, and PCEN agrees to about 1e-6 of its value range.

Known deviations from detect_birds.py
-------------------------------------
* No resampling. The graph expects 32000 Hz audio, because a fixed graph cannot
  express an arbitrary sample rate conversion. Resample before feeding audio in.
* PCEN restarts every 60 s inside the graph, matching
  ``pcen_inference.compute_pcen_for_inference``. The exported ONNX file wraps
  that walk in an ONNX ``Loop`` so long files stay closer to ``detect_birds.py``
  and peak memory stays near one minute of features plus that segment's clips.
* PCEN warm-up uses 100 frames. The reference uses ``frames // 4`` when that is
  smaller, which only happens for inputs below roughly 4.3 s.
* ``--max-det`` limits detections per class and clip, not per clip. The
  Ultralytics pre-NMS cap of 30000 boxes is not applied. Both only matter at
  very low confidence thresholds.

Usage:
    python src/inference/onnx_export.py
    python src/inference/onnx_export.py --pt-model models/Hawaii.pt --precision fp16
    python src/inference/onnx_export.py --precision fp32 --output-path build/bird.onnx
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# Add parent directory to path to import config (same approach as detect_birds.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

DEFAULT_PT_MODEL = "models/Just-Bird.pt"
DEFAULT_OUTPUT_TEMPLATE = "private_models/{model_name}_{precision}.onnx"
DEFAULT_OPSET = 17  # operation set version

# --------------------------------------------------------------------------- #
# Pipeline constants
#
# These mirror src/inference/utils/pcen_inference.get_fft_and_pcen_settings()
# and src/config.py. Keep them in sync with the training settings.
# --------------------------------------------------------------------------- #

SAMPLE_RATE = 32000
N_FFT = 2048
WIN_LENGTH = 2048
HOP_LENGTH = 375
N_MELS = 256
FMIN = 50
FMAX = 15000

PCEN_GAIN = 0.75          # pcen_norm_exponent
PCEN_BIAS = 1.0           # pcen_delta
PCEN_POWER = 0.35
PCEN_TIME_CONSTANT = 1.0
PCEN_EPS = 1e-6           # librosa.pcen default
PCEN_LEFT_PAD_SECONDS = 0.5
PCEN_WARMUP_FRAMES = 100  # filter warm-up frames prepended before PCEN
PCEN_SEGMENT_SECONDS = float(config.PCEN_SEGMENT_LENGTH)  # 60
# Context padding around each PCEN segment (pcen_inference.compute_pcen_for_inference).
SEGMENT_CONTEXT_SECONDS = 2.0
SEGMENT_SAMPLES = int(PCEN_SEGMENT_SECONDS * SAMPLE_RATE)
CONTEXT_SAMPLES = int(SEGMENT_CONTEXT_SECONDS * SAMPLE_RATE)

# detect_birds.py maps audio to the range [-2**31, 2**31[ before the STFT.
AUDIO_SCALE = float(2 ** 31)

CLIP_LENGTH_SECONDS = float(config.CLIP_LENGTH)          # 3
CLIP_HOP_SECONDS = CLIP_LENGTH_SECONDS / 2               # 1.5
CLIP_FRAMES = 252                                        # frames per 3 s clip
CLIP_HOP_FRAMES = int(CLIP_HOP_SECONDS * SAMPLE_RATE) // HOP_LENGTH  # 128

IMAGE_SIZE = config.HEIGHT_AND_WIDTH_IN_PIXELS           # 256
COLORMAP = "inferno"
COLORMAP_LEVELS = 256
VMIN = 0.0
VMAX = 100.0

# Frequency range of the spectrogram images (detect_birds.BirdCallDetector).
MAX_FREQ = 15000
MIN_FREQ = 50

DEFAULT_CONF = 0.18
DEFAULT_NMS_IOU = 0.7
DEFAULT_SONG_GAP = 0.1
DEFAULT_MAX_DET = 300

PRECISION_CHOICES = ("fp32", "fp16", "native")
TORCH_DTYPE_FOR_PRECISION = {"fp32": torch.float32, "fp16": torch.float16}

MIN_AUDIO_SECONDS = CLIP_LENGTH_SECONDS
TRACE_SECONDS = 6.0  # waveform length used while tracing the graph

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VERIFY_AUDIO = REPO_ROOT / "tests" / "test.wav"

# Merged-song output layout (matches reconstruct_songs fields).
SONG_COLUMNS = 8
SONG_COL_TIME_START = 0
SONG_COL_TIME_END = 1
SONG_COL_FREQ_LOW = 2
SONG_COL_FREQ_HIGH = 3
SONG_COL_AVG_CONF = 4
SONG_COL_MAX_CONF = 5
SONG_COL_MERGED = 6
SONG_COL_CLASS = 7

# Raw per-clip detection layout before song reconstruction.
RAW_COL_CLIP = 0
RAW_COL_TIME_START = 1
RAW_COL_TIME_END = 2
RAW_COL_FREQ_LOW = 3
RAW_COL_FREQ_HIGH = 4
RAW_COL_CONF = 5
RAW_COL_CLASS = 6
RAW_COLUMNS = 7


# --------------------------------------------------------------------------- #
# Reimplemented librosa / scipy / matplotlib helpers (numpy, export time only)
# --------------------------------------------------------------------------- #

def hz_to_mel_htk(frequency: np.ndarray) -> np.ndarray:
    """Convert Hz to mel on the HTK scale (librosa.hz_to_mel with htk=True)."""
    return 2595.0 * np.log10(1.0 + np.asarray(frequency, dtype=np.float64) / 700.0)


def mel_to_hz_htk(mel: np.ndarray) -> np.ndarray:
    """Convert mel to Hz on the HTK scale (librosa.mel_to_hz with htk=True)."""
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def flattop_window(length: int) -> np.ndarray:
    """
    Build the periodic flattop window used by the STFT.

    Equivalent to ``scipy.signal.get_window("flattop", length, fftbins=True)``,
    which librosa calls internally. The periodic variant is a symmetric window
    of ``length + 1`` points with the last sample dropped.
    """
    coefficients = (0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368)
    angle = np.linspace(-np.pi, np.pi, length + 1)
    window = np.zeros(length + 1, dtype=np.float64)
    for order, coefficient in enumerate(coefficients):
        window += coefficient * np.cos(order * angle)
    return window[:-1]


def mel_filterbank(
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
    fmin: float = FMIN,
    fmax: float = FMAX,
) -> np.ndarray:
    """
    Build the HTK mel filterbank with Slaney normalization.

    Equivalent to ``librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
    fmin=fmin, fmax=fmax, htk=True)``.

    Returns:
        Filterbank of shape ``(n_mels, 1 + n_fft // 2)``
    """
    n_bins = 1 + n_fft // 2
    fft_frequencies = np.linspace(0, sr / 2.0, n_bins, dtype=np.float64)

    mel_edges = mel_to_hz_htk(
        np.linspace(hz_to_mel_htk(fmin), hz_to_mel_htk(fmax), n_mels + 2)
    )
    edge_widths = np.diff(mel_edges)
    ramps = np.subtract.outer(mel_edges, fft_frequencies)

    weights = np.zeros((n_mels, n_bins), dtype=np.float64)
    for band in range(n_mels):
        lower = -ramps[band] / edge_widths[band]
        upper = ramps[band + 2] / edge_widths[band + 1]
        weights[band] = np.maximum(0.0, np.minimum(lower, upper))

    # Slaney normalization: equal area per band.
    enorm = 2.0 / (mel_edges[2 : n_mels + 2] - mel_edges[:n_mels])
    weights *= enorm[:, np.newaxis]

    return weights


def dft_convolution_kernel(window: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Build a convolution kernel that computes a windowed DFT.

    Applying this kernel with ``stride=hop_length`` reproduces
    ``librosa.stft(..., center=False)``: output channel ``k`` holds the real
    part of bin ``k`` and channel ``n_bins + k`` the imaginary part.

    A convolution is used instead of an ONNX STFT node because Conv and MatMul
    are supported by every runtime and execution provider, while the signal
    operators of opset 17 are frequently missing (WebGL, WebGPU, trimmed WASM
    builds).

    Returns:
        Kernel of shape ``(2 * n_bins, 1, len(window))``
    """
    n_fft = len(window)
    sample_index = np.arange(n_fft, dtype=np.int64)
    bin_index = np.arange(n_bins, dtype=np.int64)[:, np.newaxis]

    # Reduce the phase modulo n_fft before the trigonometric call to keep the
    # argument small and the constants accurate.
    phase = 2.0 * np.pi * ((bin_index * sample_index) % n_fft) / n_fft
    real = np.cos(phase) * window
    imaginary = -np.sin(phase) * window

    kernel = np.concatenate([real, imaginary], axis=0)
    return kernel[:, np.newaxis, :]


def pcen_smoothing_coefficient(
    sr: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
    time_constant: float = PCEN_TIME_CONSTANT,
) -> float:
    """Return the EMA coefficient ``b`` that librosa.pcen derives internally."""
    t_frames = time_constant * sr / float(hop_length)
    return float((np.sqrt(1.0 + 4.0 * t_frames ** 2) - 1.0) / (2.0 * t_frames ** 2))


def pcen_prefix_steps(smoothing: float, tolerance: float = 1e-12) -> int:
    """
    Number of parallel-prefix doubling steps needed for the PCEN smoother.

    ``librosa.pcen`` runs an exponential moving average over time as an IIR
    filter, which ONNX cannot express without a Loop. ``k`` doubling steps turn
    the recursion into an FIR filter with ``2 ** k`` taps, so pick ``k`` such
    that the dropped tail stays below ``tolerance``.
    """
    decay = 1.0 - smoothing
    taps = math.log(tolerance) / math.log(decay)
    return max(1, int(math.ceil(math.log2(taps))))


def colormap_lookup_table(name: str = COLORMAP) -> np.ndarray:
    """
    Return the 8-bit RGB lookup table of a matplotlib colormap, scaled to [0, 1].

    ``librosa.display.specshow`` rasterizes PCEN values through this table, and
    Ultralytics then divides the resulting image by 255. Indexing the colormap
    with integers returns its entries directly, without going through a
    normalization step.

    The Agg renderer rounds colors to 8 bit, unlike ``bytes=True``, which
    truncates. Rounding here keeps the table bit-identical to the PNG files the
    reference pipeline writes.

    Returns:
        Table of shape ``(256, 3)``
    """
    import matplotlib

    colormap = matplotlib.colormaps[name]
    table = np.asarray(colormap(np.arange(COLORMAP_LEVELS)))[:, :3]
    return np.round(table * 255.0) / 255.0


def nearest_column_index(source_width: int, target_width: int) -> np.ndarray:
    """
    Map output pixel columns to source columns for a nearest-neighbour resize.

    ``pcolormesh`` draws one hard-edged quad per PCEN frame with antialiasing
    disabled, so rasterizing 252 frames into a 256 pixel wide figure picks the
    frame that covers each pixel center.
    """
    centers = (np.arange(target_width, dtype=np.float64) + 0.5) * source_width / target_width
    return np.floor(centers).astype(np.int64)


# --------------------------------------------------------------------------- #
# Non-maximum suppression
# --------------------------------------------------------------------------- #

class _NonMaxSuppression(torch.autograd.Function):
    """
    ONNX ``NonMaxSuppression`` with an eager fallback used while tracing.

    The ONNX operator suppresses per batch entry and per class in one call,
    which is exactly what the Ultralytics postprocessing does with its class
    offset trick. Boxes stay in ``xywh``, so ``center_point_box=1``.
    """

    @staticmethod
    def forward(ctx, boxes, scores, max_output_per_class, iou_threshold, score_threshold):
        from torchvision.ops import nms

        limit = int(max_output_per_class.item())
        iou = float(iou_threshold.item())
        minimum_score = float(score_threshold.item())

        half_size = boxes[..., 2:] / 2.0
        corners = torch.cat([boxes[..., :2] - half_size, boxes[..., :2] + half_size], dim=-1)

        selected = []
        for batch in range(boxes.shape[0]):
            for class_id in range(scores.shape[1]):
                class_scores = scores[batch, class_id]
                candidates = (class_scores > minimum_score).nonzero().flatten()
                if candidates.numel() == 0:
                    continue
                kept = nms(
                    corners[batch].index_select(0, candidates),
                    class_scores.index_select(0, candidates),
                    iou,
                )[:limit]
                for box_index in candidates.index_select(0, kept).tolist():
                    selected.append([batch, class_id, box_index])

        return torch.tensor(selected, dtype=torch.int64, device=boxes.device).reshape(-1, 3)

    @staticmethod
    def symbolic(g, boxes, scores, max_output_per_class, iou_threshold, score_threshold):
        return g.op(
            "NonMaxSuppression",
            boxes,
            scores,
            max_output_per_class,
            iou_threshold,
            score_threshold,
            center_point_box_i=1,
        )


# --------------------------------------------------------------------------- #
# Song reconstruction
# --------------------------------------------------------------------------- #

@torch.jit.script
def reconstruct_songs_tensor(raw: torch.Tensor, song_gap: torch.Tensor) -> torch.Tensor:
    """
    Merge temporally adjacent detections into songs.

    Matches ``reconstruct_songs`` in ``detect_birds.py`` for a single file:
    group by class, sort by start time, extend a song while the gap to the
    current song end stays within ``song_gap``, then sort the result by start
    time.

    State updates use ``torch.where`` on a packed vector. Plain Python
    ``if``/``else`` assignments on scalar carries miscompile under the
    TorchScript ONNX Loop lowering and corrupt running confidence sums.

    Args:
        raw: Raw detections ``(N, 7)`` with columns
            ``[clip_index, time_start, time_end, freq_low, freq_high, score, class]``.
        song_gap: Shape ``(1,)`` max gap in seconds.

    Returns:
        Songs ``(M, 8)`` with columns
        ``[time_start, time_end, freq_low, freq_high, avg_conf, max_conf,
        detections_merged, class_id]``.
    """
    empty = torch.zeros((0, 8), dtype=raw.dtype, device=raw.device)
    if raw.size(0) == 0:
        return empty

    gap = song_gap.reshape(-1)[0].to(dtype=raw.dtype)
    one = torch.ones((), dtype=raw.dtype, device=raw.device)
    class_stride = torch.tensor(1.0e6, dtype=raw.dtype, device=raw.device)

    # Sort by class, then by start time. 1e6 seconds is far beyond any clip.
    sort_key = raw[:, 6] * class_stride + raw[:, 1]
    order = torch.argsort(sort_key)
    dets = raw.index_select(0, order)

    songs = torch.zeros((dets.size(0), 8), dtype=raw.dtype, device=raw.device)
    written = 0
    # Packed state: start, end, flo, fhi, sum, max, n, cls
    state = torch.stack(
        (
            dets[0, 1],
            dets[0, 2],
            dets[0, 3],
            dets[0, 4],
            dets[0, 5],
            dets[0, 5],
            one,
            dets[0, 6],
        )
    )

    for i in range(1, dets.size(0)):
        t0 = dets[i, 1]
        t1 = dets[i, 2]
        flo = dets[i, 3]
        fhi = dets[i, 4]
        score = dets[i, 5]
        cls = dets[i, 6]
        merge = (cls == state[7]) & ((t0 - state[1]) <= gap)
        if torch.logical_not(merge):
            songs[written, 0] = state[0]
            songs[written, 1] = state[1]
            songs[written, 2] = state[2]
            songs[written, 3] = state[3]
            songs[written, 4] = state[4]
            songs[written, 5] = state[5]
            songs[written, 6] = state[6]
            songs[written, 7] = state[7]
            written = written + 1
        merged_state = torch.stack(
            (
                state[0],
                torch.maximum(state[1], t1),
                torch.minimum(state[2], flo),
                torch.maximum(state[3], fhi),
                state[4] + score,
                torch.maximum(state[5], score),
                state[6] + one,
                state[7],
            )
        )
        fresh_state = torch.stack((t0, t1, flo, fhi, score, score, one, cls))
        state = torch.where(merge, merged_state, fresh_state)

    songs[written, 0] = state[0]
    songs[written, 1] = state[1]
    songs[written, 2] = state[2]
    songs[written, 3] = state[3]
    songs[written, 4] = state[4]
    songs[written, 5] = state[5]
    songs[written, 6] = state[6]
    songs[written, 7] = state[7]
    written = written + 1

    songs = songs[:written]
    avg = songs[:, 4] / songs[:, 6]
    songs = torch.stack(
        (
            songs[:, 0],
            songs[:, 1],
            songs[:, 2],
            songs[:, 3],
            avg,
            songs[:, 5],
            songs[:, 6],
            songs[:, 7],
        ),
        dim=1,
    )
    return songs.index_select(0, torch.argsort(songs[:, 0]))


# --------------------------------------------------------------------------- #
# The graph
# --------------------------------------------------------------------------- #

class BirdDetectionGraph(nn.Module):
    """
    Audio in, merged songs out.

    Wraps a fused Ultralytics detection model with the feature extraction of
    ``src/inference/utils/pcen_inference.py``, the box decoding of
    ``src/inference/detect_birds.py``, and song reconstruction.

    Feature extraction always runs in float32. The audio is scaled to
    ``[-2**31, 2**31[`` before the STFT, and squared magnitudes reach 1e25,
    far beyond the float16 range. Only the network and its input image use the
    export precision.

    ``conf`` and ``song_gap`` are required graph inputs so callers can tune
    them without re-exporting. NMS IoU stays baked in at export time.
    """

    def __init__(
        self,
        detection_model: nn.Module,
        num_classes: int,
        num_anchors: int,
        image_size: int,
        network_dtype: torch.dtype = torch.float32,
        nms_iou_threshold: float = DEFAULT_NMS_IOU,
        max_detections: int = DEFAULT_MAX_DET,
    ):
        super().__init__()

        self.network = detection_model
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.image_size = image_size
        self.network_dtype = network_dtype

        mel_basis = mel_filterbank()
        # Bands above FMAX carry zero weight, so the DFT can stop early.
        self.num_bins = int(np.max(np.nonzero(mel_basis.sum(axis=0))[0])) + 1
        mel_basis = mel_basis[:, : self.num_bins]
        kernel = dft_convolution_kernel(flattop_window(WIN_LENGTH), self.num_bins)

        self.pcen_smoothing = pcen_smoothing_coefficient()
        self.pcen_decay = 1.0 - self.pcen_smoothing
        self.pcen_steps = pcen_prefix_steps(self.pcen_smoothing)
        self.pcen_pad_samples = int(PCEN_LEFT_PAD_SECONDS * SAMPLE_RATE)
        self.pcen_trim_frames = PCEN_WARMUP_FRAMES + self.pcen_pad_samples // HOP_LENGTH

        # Frequency axis of the spectrogram images, see
        # BirdCallDetector.pixels_to_hz.
        self.min_mel = float(hz_to_mel_htk(MIN_FREQ))
        self.mel_range = float(hz_to_mel_htk(MAX_FREQ)) - self.min_mel

        self.register_buffer("dft_kernel", torch.from_numpy(kernel).float())
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float())
        self.register_buffer("color_table", torch.from_numpy(colormap_lookup_table()).float())
        self.register_buffer(
            "column_index",
            torch.from_numpy(nearest_column_index(CLIP_FRAMES, IMAGE_SIZE)),
        )
        self.register_buffer("frame_index", torch.arange(CLIP_FRAMES, dtype=torch.int64))
        self.register_buffer("max_detections", torch.tensor([max_detections], dtype=torch.int64))
        self.register_buffer("nms_iou_threshold", torch.tensor([nms_iou_threshold], dtype=torch.float32))

    # -- feature extraction ------------------------------------------------- #

    def power_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """Windowed power spectrogram of the scaled and left-padded waveform."""
        spectrum = F.conv1d(audio, self.dft_kernel, stride=HOP_LENGTH)
        real = spectrum[:, : self.num_bins]
        imaginary = spectrum[:, self.num_bins :]
        return real * real + imaginary * imaginary

    def pcen(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Per-channel energy normalization over the time axis.

        The exponential moving average of ``librosa.pcen`` is evaluated with
        parallel prefix sums: after ``k`` doubling steps every frame has
        accumulated ``2 ** k`` of its predecessors, which is exact up to the
        truncated tail of the decay.
        """
        warmup = mel_spectrogram[:, :, :PCEN_WARMUP_FRAMES]
        looped = torch.cat([warmup, mel_spectrogram], dim=2)

        smoothed = self.pcen_smoothing * looped
        lag = 1
        for _ in range(self.pcen_steps):
            shifted = F.pad(smoothed, (lag, 0))[:, :, :-lag]
            smoothed = smoothed + (self.pcen_decay ** lag) * shifted
            lag *= 2

        gain = torch.exp(
            -PCEN_GAIN * (math.log(PCEN_EPS) + torch.log1p(smoothed / PCEN_EPS))
        )
        normalized = (looped * gain + PCEN_BIAS) ** PCEN_POWER - PCEN_BIAS ** PCEN_POWER
        return normalized[:, :, self.pcen_trim_frames :]

    def extract_clips(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cut PCEN features into overlapping 3 s clips on the global grid.

        Returns clips ``(N, n_mels, CLIP_FRAMES)`` and their absolute start
        times in seconds ``(N,)``. Trailing frames that cannot fill a clip are
        dropped, matching ``pcen_inference.compute_pcen_for_inference``.
        """
        num_clips = (features.shape[2] - CLIP_FRAMES) // CLIP_HOP_FRAMES + 1
        starts = torch.arange(num_clips, device=features.device) * CLIP_HOP_FRAMES
        index = (starts.reshape(-1, 1) + self.frame_index).reshape(-1)

        gathered = torch.index_select(features[0], 1, index)
        clips = gathered.reshape(N_MELS, -1, CLIP_FRAMES).permute(1, 0, 2)
        clip_times = starts.to(torch.float32) * (float(HOP_LENGTH) / float(SAMPLE_RATE))
        return clips, clip_times

    def extract_segment_clips(
        self,
        features: torch.Tensor,
        padded_start_time: torch.Tensor,
        seg_t0: torch.Tensor,
        seg_t1: torch.Tensor,
        file_duration: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gather clips whose global start times fall in ``[seg_t0, seg_t1)``.

        Frame offsets use truncating division toward zero, matching the
        ``int(...)`` conversion in ``pcen_inference``. A fixed clip-slot grid
        is masked to the live duration so ONNX keeps the length dynamic.
        """
        # ~3.4 h at 1.5 s hop. Long enough for Raven hour files with headroom.
        max_clip_slots = 8192
        device = features.device

        max_start = file_duration - CLIP_LENGTH_SECONDS
        num_global = torch.floor(torch.clamp(max_start, min=0.0) / CLIP_HOP_SECONDS).to(
            torch.int64
        ) + 1
        num_global = torch.clamp(num_global, min=0, max=max_clip_slots)

        clip_times = (
            torch.arange(max_clip_slots, device=device, dtype=torch.float32)
            * CLIP_HOP_SECONDS
        )
        slot = torch.arange(max_clip_slots, device=device)
        in_segment = (
            (clip_times >= seg_t0)
            & (clip_times < seg_t1)
            & (clip_times <= max_start)
            & (slot < num_global)
        )
        clip_times = clip_times[in_segment]
        frames = torch.div(
            (clip_times - padded_start_time) * float(SAMPLE_RATE),
            float(HOP_LENGTH),
            rounding_mode="trunc",
        ).to(torch.int64)
        valid = (frames >= 0) & (frames + CLIP_FRAMES <= features.shape[2])
        frames = frames[valid]
        clip_times = clip_times[valid]
        # When nothing is valid, build an empty gather that still type-checks.
        index = (frames.reshape(-1, 1) + self.frame_index).reshape(-1)
        gathered = torch.index_select(features[0], 1, index)
        clips = gathered.reshape(N_MELS, -1, CLIP_FRAMES).permute(1, 0, 2)
        return clips, clip_times

    def features_from_segment_audio(self, segment: torch.Tensor) -> torch.Tensor:
        """STFT + mel + PCEN for one context-padded audio slice."""
        waveform = segment.reshape(1, 1, -1).to(torch.float32) * AUDIO_SCALE
        waveform = torch.cat([waveform[:, :, : self.pcen_pad_samples], waveform], dim=2)
        power = self.power_spectrogram(waveform)
        mel_spectrogram = torch.matmul(self.mel_basis, power)
        return self.pcen(mel_spectrogram)

    def detect_segment(
        self,
        samples: torch.Tensor,
        seg_start_sample: int,
        seg_end_sample: int,
        file_duration: float,
        conf: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run PCEN, YOLO, and NMS for one minute-style segment.

        Matches the per-segment path of
        ``pcen_inference.compute_pcen_for_inference``: optional 2 s of context
        on each side, independent PCEN warm-up, and clips assigned by global
        start time.
        """
        num_samples = int(samples.shape[0])
        padded_start = max(0, seg_start_sample - CONTEXT_SAMPLES)
        padded_end = min(num_samples, seg_end_sample + CONTEXT_SAMPLES)
        segment = samples[padded_start:padded_end]

        if segment.numel() < 2 * N_FFT:
            return samples.new_zeros((0, RAW_COLUMNS))

        features = self.features_from_segment_audio(segment)
        padded_start_time = samples.new_tensor(padded_start / float(SAMPLE_RATE))
        seg_t0 = samples.new_tensor(seg_start_sample / float(SAMPLE_RATE))
        seg_t1 = samples.new_tensor(seg_end_sample / float(SAMPLE_RATE))
        duration = samples.new_tensor(file_duration)

        clips, clip_times = self.extract_segment_clips(
            features, padded_start_time, seg_t0, seg_t1, duration
        )
        if clips.shape[0] == 0:
            return samples.new_zeros((0, RAW_COLUMNS))

        images = self.render(clips)
        predictions = self.network(images.to(self.network_dtype)).to(torch.float32)
        return self.postprocess(predictions, conf, clip_times)

    def detect_segment_tensor(
        self,
        samples: torch.Tensor,
        segment_index: torch.Tensor,
        conf: torch.Tensor,
    ) -> torch.Tensor:
        """
        Tensor-index variant used by the ONNX segment body.

        ``segment_index`` is a scalar int64. Empty segments still run one dummy
        clip through the network so the exported graph never hits a 0-batch
        Reshape inside YOLO, then discards those rows.
        """
        samples = samples.reshape(-1).to(torch.float32)
        num_samples = samples.shape[0]
        file_duration = samples.new_zeros(()).float() + (
            num_samples / float(SAMPLE_RATE)
        )

        seg_start = segment_index.to(torch.int64).reshape(()) * int(SEGMENT_SAMPLES)
        num_samples_t = samples.new_zeros(()).long() + num_samples
        seg_end = torch.minimum(seg_start + int(SEGMENT_SAMPLES), num_samples_t)
        padded_start = torch.clamp(seg_start - int(CONTEXT_SAMPLES), min=0)
        padded_end = torch.minimum(seg_end + int(CONTEXT_SAMPLES), num_samples_t)

        segment = samples[padded_start:padded_end]
        if segment.numel() < 2 * N_FFT:
            # Keep a tiny valid slice so export does not see a dead branch.
            segment = samples.new_zeros((2 * N_FFT,))

        features = self.features_from_segment_audio(segment)
        padded_start_time = padded_start.to(torch.float32) / float(SAMPLE_RATE)
        seg_t0 = seg_start.to(torch.float32) / float(SAMPLE_RATE)
        seg_t1 = seg_end.to(torch.float32) / float(SAMPLE_RATE)

        clips, clip_times = self.extract_segment_clips(
            features, padded_start_time, seg_t0, seg_t1, file_duration
        )
        n_real = clips.shape[0]
        dummy_clip = features.new_zeros((1, N_MELS, CLIP_FRAMES))
        dummy_time = features.new_zeros((1,))
        safe_clips = torch.cat([clips, dummy_clip], dim=0)
        safe_times = torch.cat([clip_times, dummy_time], dim=0)
        # max(n_real, 1) without a Python max that constant-folds at export.
        one = features.new_ones((), dtype=torch.int64)
        n_safe = torch.maximum(features.new_zeros(()).long() + n_real, one)
        take = torch.arange(n_safe, device=features.device)
        safe_clips = safe_clips.index_select(0, take)
        safe_times = safe_times.index_select(0, take)

        images = self.render(safe_clips)
        predictions = self.network(images.to(self.network_dtype)).to(torch.float32)
        raw = self.postprocess(predictions, conf, safe_times)

        # Keep every detection when the segment had real clips. Drop all rows
        # for empty segments (the one dummy clip is only there for YOLO shape).
        n_real_t = features.new_zeros(()).long() + n_real
        keep = torch.arange(raw.shape[0], device=raw.device) >= 0
        keep = keep & (n_real_t > 0)
        return raw[keep]

    # -- image rendering ---------------------------------------------------- #

    def render(self, clips: torch.Tensor) -> torch.Tensor:
        """
        Turn PCEN clips into the RGB images the network was trained on.

        Reproduces the matplotlib figure of
        ``BirdCallDetector.create_spectrogram_image``: values are normalized to
        [VMIN, VMAX], quantized into the colormap, drawn with the low mel band
        at the bottom, and rasterized into a square image.
        """
        normalized = (clips - VMIN) / (VMAX - VMIN)
        level = torch.clamp(
            torch.floor(normalized * COLORMAP_LEVELS), 0.0, COLORMAP_LEVELS - 1
        ).to(torch.int64)

        image = self.color_table[level]            # (clips, mels, frames, 3)
        image = image.permute(0, 3, 1, 2)          # (clips, 3, mels, frames)
        image = torch.flip(image, dims=[2])        # mel band 0 at the bottom
        image = torch.index_select(image, 3, self.column_index)

        if self.image_size != IMAGE_SIZE:
            image = F.interpolate(
                image,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return image

    # -- postprocessing ----------------------------------------------------- #

    def postprocess(
        self,
        predictions: torch.Tensor,
        conf: torch.Tensor,
        clip_times: torch.Tensor,
    ) -> torch.Tensor:
        """
        Suppress overlapping boxes and convert pixels to seconds and Hz.

        Mirrors ``ultralytics.utils.ops.non_max_suppression`` with
        ``multi_label=False``: only the strongest class of a box competes, and
        suppression runs per class. Absolute clip start times come from
        ``clip_times``. Frequency mapping matches
        ``BirdCallDetector.pixels_to_hz``.

        Args:
            predictions: Network output ``(clips, 4 + num_classes, anchors)``.
            conf: Confidence threshold, shape ``(1,)``.
            clip_times: Absolute clip start times in seconds, shape ``(clips,)``.
        """
        boxes = predictions[:, :4, :].permute(0, 2, 1)
        scores = predictions[:, 4:, :]

        strongest = scores.amax(dim=1, keepdim=True)
        scores = torch.where(scores >= strongest, scores, torch.zeros_like(scores))

        score_threshold = conf.reshape(-1)[:1].to(dtype=torch.float32)
        selected = _NonMaxSuppression.apply(
            boxes,
            scores,
            self.max_detections,
            self.nms_iou_threshold,
            score_threshold,
        )
        clip_index = selected[:, 0]
        class_id = selected[:, 1]
        box_index = selected[:, 2]

        flat_boxes = boxes.reshape(-1, 4)
        box = flat_boxes.index_select(0, clip_index * self.num_anchors + box_index)
        confidence = scores.reshape(-1).index_select(
            0,
            (clip_index * self.num_classes + class_id) * self.num_anchors + box_index,
        )

        # Undo the letterbox scale, then clip to the image, as scale_boxes does.
        gain = self.image_size / IMAGE_SIZE
        center_x = box[:, 0] / gain
        center_y = box[:, 1] / gain
        half_width = box[:, 2] / (2.0 * gain)
        half_height = box[:, 3] / (2.0 * gain)

        x1 = torch.clamp(center_x - half_width, 0.0, IMAGE_SIZE)
        x2 = torch.clamp(center_x + half_width, 0.0, IMAGE_SIZE)
        y1 = torch.clamp(center_y - half_height, 0.0, IMAGE_SIZE)
        y2 = torch.clamp(center_y + half_height, 0.0, IMAGE_SIZE)

        clip_start = clip_times.index_select(0, clip_index)
        time_start = clip_start + x1 / IMAGE_SIZE * CLIP_LENGTH_SECONDS
        time_end = clip_start + x2 / IMAGE_SIZE * CLIP_LENGTH_SECONDS

        # y1 is the top of the box and therefore the high frequency.
        freq_high = self.pixels_to_hz(y1)
        freq_low = self.pixels_to_hz(y2)

        return torch.stack(
            [
                clip_index.to(torch.float32),
                time_start,
                time_end,
                freq_low,
                freq_high,
                confidence,
                class_id.to(torch.float32),
            ],
            dim=1,
        )

    def pixels_to_hz(self, y_pixel: torch.Tensor) -> torch.Tensor:
        """Convert image rows to Hz, inverting the mel mapping used for labels."""
        mel = (1.0 - y_pixel / IMAGE_SIZE) * self.mel_range + self.min_mel
        hz = 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)
        return torch.round(torch.clamp(hz, MIN_FREQ, MAX_FREQ))

    def detect_all_segments(
        self,
        audio: torch.Tensor,
        conf: torch.Tensor,
    ) -> torch.Tensor:
        """Walk the file in 60 s PCEN segments and concatenate raw detections."""
        samples = audio.reshape(-1).to(torch.float32)
        num_samples = int(samples.shape[0])
        file_duration = num_samples / float(SAMPLE_RATE)
        empty = samples.new_zeros((0, RAW_COLUMNS))
        if num_samples <= 0:
            return empty

        parts = []
        seg_start = 0
        while seg_start < num_samples:
            seg_end = min(seg_start + SEGMENT_SAMPLES, num_samples)
            part = self.detect_segment(samples, seg_start, seg_end, file_duration, conf)
            if part.shape[0] > 0:
                parts.append(part)
            seg_start = seg_end

        if not parts:
            return empty
        return torch.cat(parts, dim=0)

    def detect_continuous(
        self,
        audio: torch.Tensor,
        conf: torch.Tensor,
    ) -> torch.Tensor:
        """Single-pass PCEN over the whole waveform (debug / comparison helper)."""
        samples = audio.reshape(-1).to(torch.float32)
        features = self.features_from_segment_audio(samples)
        clips, clip_times = self.extract_clips(features)
        if clips.shape[0] == 0:
            return samples.new_zeros((0, RAW_COLUMNS))
        images = self.render(clips)
        predictions = self.network(images.to(self.network_dtype)).to(torch.float32)
        return self.postprocess(predictions, conf, clip_times)

    # -- graph -------------------------------------------------------------- #

    def forward(
        self,
        audio: torch.Tensor,
        conf: torch.Tensor,
        song_gap: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full detection pipeline.

        Walks 60 s PCEN segments (matching ``detect_birds.py``), then merges
        songs. The ONNX exporter builds the same walk as an ONNX ``Loop``.
        """
        raw = self.detect_all_segments(audio, conf)
        return reconstruct_songs_tensor(raw, song_gap.to(dtype=torch.float32))


class SegmentBody(nn.Module):
    """
    One PCEN segment (features + YOLO + NMS).

    Kept as a standalone module for experiments with ONNX Loop wrapping and for
    hosts that prefer to drive the 60 s walk themselves.
    """

    def __init__(self, core: BirdDetectionGraph):
        super().__init__()
        self.core = core

    def forward(
        self,
        audio: torch.Tensor,
        segment_index: torch.Tensor,
        conf: torch.Tensor,
    ) -> torch.Tensor:
        return self.core.detect_segment_tensor(audio, segment_index, conf)


class SongReconstructor(nn.Module):
    """Thin wrapper so song merging can be exported on its own."""

    def forward(self, raw: torch.Tensor, song_gap: torch.Tensor) -> torch.Tensor:
        return reconstruct_songs_tensor(raw, song_gap.to(dtype=torch.float32))


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #

def load_detection_model(pt_model: str) -> Tuple[nn.Module, int, Dict[int, str], torch.dtype]:
    """
    Load a YOLO checkpoint and prepare its network for export.

    Fusing Conv and BatchNorm and switching the detection head into export mode
    is what Ultralytics does before any conversion, so the graph matches what
    ``.pt`` inference computes.

    Returns:
        network: Fused detection model in eval mode
        image_size: Training image size stored in the checkpoint
        names: Class index to name mapping of the checkpoint
        dtype: Precision the checkpoint weights are stored in
    """
    from ultralytics import YOLO
    from ultralytics.nn.modules.head import Detect

    yolo = YOLO(pt_model)
    image_size = int(yolo.overrides.get("imgsz", IMAGE_SIZE))
    names = dict(yolo.model.names)
    dtype = next(yolo.model.parameters()).dtype

    network = deepcopy(yolo.model).float().eval()
    network = network.fuse()
    for module in network.modules():
        if isinstance(module, Detect):
            module.export = True
            module.dynamic = False
            module.format = "onnx"
    for parameter in network.parameters():
        parameter.requires_grad = False

    return network, image_size, names, dtype


def resolve_precision(precision: str, checkpoint_dtype: torch.dtype) -> Tuple[str, torch.dtype]:
    """
    Resolve the requested precision to a label and a torch dtype.

    ``native`` keeps whatever the checkpoint holds, so exports stay lossless by
    default.
    """
    if precision != "native":
        return precision, TORCH_DTYPE_FOR_PRECISION[precision]

    label = "fp16" if checkpoint_dtype == torch.float16 else "fp32"
    return label, TORCH_DTYPE_FOR_PRECISION[label]


def probe_network(network: nn.Module, image_size: int) -> Tuple[int, int]:
    """Run one dummy image through the network to read its output layout."""
    with torch.no_grad():
        predictions = network(torch.zeros(1, 3, image_size, image_size))
    if isinstance(predictions, (list, tuple)):
        predictions = predictions[0]
    return int(predictions.shape[1]) - 4, int(predictions.shape[2])


def species_names(pt_model: str, species_mapping: Optional[str], fallback: Dict[int, str]) -> Tuple[Optional[str], Dict[int, str]]:
    """
    Resolve the species mapping that turns class indices into eBird codes.

    Checkpoints only store placeholder class names, so the graph metadata uses
    the mapping from ``src/config.py`` whenever it can be determined.
    """
    if species_mapping is None:
        try:
            species_mapping = config.get_species_mapping_for_model(pt_model)
        except ValueError:
            return None, fallback

    return species_mapping, config.get_species_mapping(species_mapping)["id_to_ebird"]


def graph_metadata(
    pt_model: str,
    precision: str,
    species_mapping: Optional[str],
    names: Dict[int, str],
    default_conf: float,
    nms_iou_threshold: float,
    default_song_gap: float,
    max_detections: int,
) -> Dict[str, str]:
    """Describe the graph interface for consumers that only have the .onnx file."""
    return {
        "source_model": Path(pt_model).name,
        "precision": precision,
        "species_mapping": species_mapping or "unknown",
        "names": json.dumps({int(index): name for index, name in names.items()}),
        "sample_rate": str(SAMPLE_RATE),
        "channels": "1",
        "min_audio_seconds": str(MIN_AUDIO_SECONDS),
        "pcen_segment_seconds": str(PCEN_SEGMENT_SECONDS),
        "clip_length_seconds": str(CLIP_LENGTH_SECONDS),
        "clip_hop_seconds": str(CLIP_HOP_SECONDS),
        "default_conf": str(default_conf),
        "default_song_gap": str(default_song_gap),
        "nms_iou_threshold": str(nms_iou_threshold),
        "max_detections_per_class": str(max_detections),
        "input_audio": "audio: float32 (num_samples,), mono, [-1, 1]",
        "input_conf": (
            f"conf: float32 (1,), required, suggested default {default_conf}"
        ),
        "input_song_gap": (
            f"song_gap: float32 (1,), required, suggested default "
            f"{default_song_gap} seconds"
        ),
        "output": (
            "detections: float32 (num_detections, 8) "
            "[time_start, time_end, freq_low_hz, freq_high_hz, "
            "avg_confidence, max_confidence, detections_merged, class_id]"
        ),
    }


def write_metadata(output_path: Path, metadata: Dict[str, str]) -> None:
    """Attach the metadata to the ONNX file as string properties."""
    import onnx

    model = onnx.load(str(output_path))
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(model, str(output_path))


def write_metadata(output_path: Path, metadata: Dict[str, str]) -> None:
    """Attach the metadata to the ONNX file as string properties."""
    import onnx

    model = onnx.load(str(output_path))
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(model, str(output_path))


def export_onnx(
    graph: BirdDetectionGraph,
    output_path: Path,
    opset: int,
    dummy_seconds: float,
    default_conf: float,
    default_song_gap: float,
    dummy_audio: Optional[torch.Tensor] = None,
) -> None:
    """
    Export audio -> songs with a 60 s PCEN segment Loop.

    The segment body (features + YOLO + NMS) and song reconstructor are exported
    with the TorchScript ONNX path, then composed into one ORT-loadable model.
    ``conf`` and ``song_gap`` stay required graph inputs so browser, Raven, and
    desktop hosts all tune them the same way. Suggested defaults are written to
    ONNX metadata only.
    """
    import tempfile

    if dummy_audio is None:
        dummy_audio = torch.zeros(int(dummy_seconds * SAMPLE_RATE), dtype=torch.float32)
    else:
        dummy_audio = dummy_audio.reshape(-1).to(dtype=torch.float32)

    dummy_conf = torch.tensor([default_conf], dtype=torch.float32)
    dummy_song_gap = torch.tensor([default_song_gap], dtype=torch.float32)
    dummy_index = torch.zeros((), dtype=torch.int64)

    with torch.no_grad():
        for _ in range(2):
            graph(dummy_audio, dummy_conf, dummy_song_gap)

    body = SegmentBody(graph).eval()
    recon = SongReconstructor().eval()

    export_kwargs = dict(opset_version=opset, do_constant_folding=True)
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        body_path = tmp_dir / "segment_body.onnx"
        recon_path = tmp_dir / "recon.onnx"

        with torch.no_grad():
            torch.onnx.export(
                body,
                (dummy_audio, dummy_index, dummy_conf),
                str(body_path),
                input_names=["audio", "segment_index", "conf"],
                output_names=["raw"],
                dynamic_axes={
                    "audio": {0: "num_samples"},
                    "raw": {0: "num_raw"},
                },
                **export_kwargs,
            )
            raw_example = body(dummy_audio, dummy_index, dummy_conf)
            if raw_example.shape[0] == 0:
                raw_example = torch.zeros((1, RAW_COLUMNS), dtype=torch.float32)
            torch.onnx.export(
                recon,
                (raw_example, dummy_song_gap),
                str(recon_path),
                input_names=["raw", "song_gap"],
                output_names=["detections"],
                dynamic_axes={
                    "raw": {0: "num_raw"},
                    "detections": {0: "num_detections"},
                },
                **export_kwargs,
            )

        compose_segment_loop_onnx(body_path, recon_path, output_path, opset)


def compose_segment_loop_onnx(
    body_path: Path,
    recon_path: Path,
    output_path: Path,
    opset: int,
) -> None:
    """
    Build ``audio, conf, song_gap -> detections`` with an ONNX Loop of segments.

    Loop-carried state is ``(acc, audio, conf)``. ``audio`` and ``conf`` pass
    through unchanged each iteration. That matches the ONNX Loop signature.
    Passing them as non-carried extras previously made ORT abort on load.
    """
    import copy

    import onnx
    from onnx import TensorProto, helper, numpy_helper

    body_model = onnx.load(str(body_path))
    recon_model = onnx.load(str(recon_path))

    def map_name(name: str, prefix: str, keep_map: Dict[str, str]) -> str:
        if not name:
            return name
        if name in keep_map:
            return keep_map[name]
        return prefix + name

    def rename_graph_values(graph, prefix: str, keep_map: Dict[str, str]) -> None:
        def rename_node(node) -> None:
            node.input[:] = [map_name(n, prefix, keep_map) for n in node.input]
            node.output[:] = [map_name(n, prefix, keep_map) for n in node.output]
            for attr in node.attribute:
                if attr.HasField("g"):
                    rename_graph_values(attr.g, prefix, keep_map)

        for node in graph.node:
            rename_node(node)
        for tensor in graph.initializer:
            tensor.name = map_name(tensor.name, prefix, keep_map)
        for value in list(graph.value_info) + list(graph.input) + list(graph.output):
            value.name = map_name(value.name, prefix, keep_map)

    def replace_name(graph, old: str, new: str) -> None:
        for node in graph.node:
            node.input[:] = [new if x == old else x for x in node.input]
            node.output[:] = [new if x == old else x for x in node.output]
            for attr in node.attribute:
                if attr.HasField("g"):
                    replace_name(attr.g, old, new)
        for tensor in graph.initializer:
            if tensor.name == old:
                tensor.name = new
        for value in list(graph.value_info) + list(graph.input) + list(graph.output):
            if value.name == old:
                value.name = new

    body_graph = copy.deepcopy(body_model.graph)
    rename_graph_values(
        body_graph,
        "b_",
        {
            "audio": "audio_in",
            "segment_index": "segment_index",
            "conf": "conf_in",
            "raw": "raw",
        },
    )

    loop_nodes = [
        helper.make_node("Identity", ["iter"], ["segment_index"]),
        *list(body_graph.node),
        helper.make_node("Concat", ["acc_in", "raw"], ["acc_out"], axis=0),
        helper.make_node("Identity", ["audio_in"], ["audio_out"]),
        helper.make_node("Identity", ["conf_in"], ["conf_out"]),
        helper.make_node("Identity", ["cond_in"], ["cond_out"]),
    ]
    loop_body = helper.make_graph(
        loop_nodes,
        "segment_loop_body",
        [
            helper.make_tensor_value_info("iter", TensorProto.INT64, []),
            helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
            helper.make_tensor_value_info(
                "acc_in", TensorProto.FLOAT, ["acc_rows", RAW_COLUMNS]
            ),
            helper.make_tensor_value_info(
                "audio_in", TensorProto.FLOAT, ["num_samples"]
            ),
            helper.make_tensor_value_info("conf_in", TensorProto.FLOAT, [1]),
        ],
        [
            helper.make_tensor_value_info("cond_out", TensorProto.BOOL, []),
            helper.make_tensor_value_info(
                "acc_out", TensorProto.FLOAT, ["acc_rows_out", RAW_COLUMNS]
            ),
            helper.make_tensor_value_info(
                "audio_out", TensorProto.FLOAT, ["num_samples"]
            ),
            helper.make_tensor_value_info("conf_out", TensorProto.FLOAT, [1]),
        ],
        list(body_graph.initializer),
    )

    recon_graph = copy.deepcopy(recon_model.graph)
    rename_graph_values(
        recon_graph, "r_", {"song_gap": "song_gap", "detections": "detections"}
    )
    replace_name(recon_graph, "r_raw", "raw_all")

    initializers = list(recon_graph.initializer) + [
        numpy_helper.from_array(np.array(True), "true_const"),
        numpy_helper.from_array(
            np.zeros((0, RAW_COLUMNS), dtype=np.float32), "acc_empty"
        ),
        numpy_helper.from_array(np.array(SEGMENT_SAMPLES, dtype=np.int64), "seg_samples"),
        numpy_helper.from_array(
            np.array(SEGMENT_SAMPLES - 1, dtype=np.int64), "seg_minus_1"
        ),
        numpy_helper.from_array(np.array(0, dtype=np.int64), "zero_i"),
    ]

    parent_nodes = [
        helper.make_node("Shape", ["audio"], ["audio_shape"]),
        helper.make_node("Gather", ["audio_shape", "zero_i"], ["num_samples"], axis=0),
        helper.make_node("Add", ["num_samples", "seg_minus_1"], ["num_plus"]),
        helper.make_node("Div", ["num_plus", "seg_samples"], ["trip_count"]),
        helper.make_node(
            "Loop",
            ["trip_count", "true_const", "acc_empty", "audio", "conf"],
            ["raw_all", "audio_final", "conf_final"],
            body=loop_body,
        ),
        *list(recon_graph.node),
    ]

    parent = helper.make_graph(
        parent_nodes,
        "birdbox_segmented",
        [
            helper.make_tensor_value_info(
                "audio", TensorProto.FLOAT, ["num_samples"]
            ),
            helper.make_tensor_value_info("conf", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("song_gap", TensorProto.FLOAT, [1]),
        ],
        [
            helper.make_tensor_value_info(
                "detections", TensorProto.FLOAT, ["num_detections", SONG_COLUMNS]
            )
        ],
        initializers,
    )

    model = helper.make_model(
        parent,
        opset_imports=[helper.make_opsetid("", opset)],
        producer_name="birdbox.onnx_export",
    )
    model.ir_version = min(body_model.ir_version, recon_model.ir_version, 9)
    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))


def load_verification_audio(
    audio_path: str,
    max_seconds: Optional[float] = None,
) -> torch.Tensor:
    """
    Load mono audio for tracing and the verification pass.

    The graph has no resampler, so the file has to be recorded at the training
    sample rate. Pass ``max_seconds`` to truncate. Omit it to keep the whole
    file.
    """
    try:
        import soundfile as sf
    except ImportError:
        raise SystemExit(
            "Error: verification audio needs soundfile. Install it with "
            "'pip install soundfile' or pass --no-verify."
        )

    path = Path(audio_path)
    if not path.exists():
        raise SystemExit(f"Error: verification audio not found: {path}")

    audio, sr = sf.read(str(path), dtype="float32")
    if sr != SAMPLE_RATE:
        raise SystemExit(
            f"Error: verification audio must be {SAMPLE_RATE} Hz, got {sr} Hz. "
            "Resample the file first."
        )
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if max_seconds is not None:
        audio = audio[: int(max_seconds * SAMPLE_RATE)]
    return torch.from_numpy(np.ascontiguousarray(audio))


def verify_onnx(
    graph: BirdDetectionGraph,
    output_path: Path,
    default_conf: float,
    default_song_gap: float,
    audio_path: str,
) -> None:
    """
    Compare the exported graph against PyTorch on the same audio.

    This checks that the conversion preserved the computation and that every
    operator is supported by ONNX Runtime. It does not check parity with
    ``detect_birds.py``.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("ONNX Runtime not installed, skipping verification.")
        return

    audio = load_verification_audio(audio_path)
    source = f"{audio.numel() / SAMPLE_RATE:.1f}s of {Path(audio_path).name}"

    conf = torch.tensor([default_conf], dtype=torch.float32)
    song_gap = torch.tensor([default_song_gap], dtype=torch.float32)

    with torch.no_grad():
        expected = graph(audio, conf, song_gap).numpy()

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    actual = session.run(
        None,
        {
            "audio": audio.numpy(),
            "conf": conf.numpy(),
            "song_gap": song_gap.numpy(),
        },
    )[0]

    print(f"Verification on {source}:")
    print(f"  PyTorch detections:      {expected.shape[0]}")
    print(f"  ONNX Runtime detections: {actual.shape[0]}")

    if expected.shape != actual.shape:
        print("  WARNING: detection counts differ between PyTorch and ONNX Runtime.")
        return

    if expected.shape[0] == 0:
        print("  No detections in this audio, only graph execution was checked.")
        return

    deviation = float(np.abs(expected - actual).max())
    print(f"  Max absolute difference: {deviation:.6g}")
    # Rounded Hz columns can differ by 1 at float boundaries.
    freq_diff = float(np.abs(expected[:, 2:4] - actual[:, 2:4]).max())
    non_freq_cols = [0, 1, 4, 5, 6, 7]
    non_freq = float(np.abs(expected[:, non_freq_cols] - actual[:, non_freq_cols]).max())
    if non_freq > 1e-2 or freq_diff > 1.0 + 1e-6:
        print("  WARNING: outputs differ more than expected.")
    elif freq_diff > 0:
        print(
            f"  (Frequency columns differ by at most {freq_diff:g} Hz after rounding.)"
        )


def build_output_path(output_path: Optional[str], pt_model: str, precision: str) -> Path:
    """Fill in the default output template when no path was given."""
    if output_path is not None:
        return Path(output_path)

    return Path(
        DEFAULT_OUTPUT_TEMPLATE.format(
            model_name=Path(pt_model).stem,
            precision=precision,
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export a YOLO bird detection model to ONNX with audio "
            "preprocessing, NMS, and song reconstruction included."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: models/Just-Bird.pt -> private_models/Just-Bird_fp32.onnx
  python src/inference/onnx_export.py

  # Half precision network, for browser and mobile runtimes
  python src/inference/onnx_export.py --pt-model models/Hawaii.pt --precision fp16

  # Explicit output path
  python src/inference/onnx_export.py --output-path build/just-bird.onnx

  # Suggested conf / song_gap defaults stored in ONNX metadata
  python src/inference/onnx_export.py --conf 0.25 --song-gap 0.2
        """,
    )

    parser.add_argument(
        "--pt-model",
        type=str,
        default=DEFAULT_PT_MODEL,
        help=f"Path to the trained .pt model to convert (default: {DEFAULT_PT_MODEL})",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help=(
            "Path of the .onnx file to write "
            f"(default: {DEFAULT_OUTPUT_TEMPLATE})"
        ),
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=list(PRECISION_CHOICES),
        default="native",
        help=(
            "Numeric precision of the network weights. native keeps the dtype "
            "of the .pt file (default: native). Feature extraction always runs "
            "in fp32."
        ),
    )
    parser.add_argument(
        "--species-mapping",
        type=str,
        default=None,
        choices=sorted(config.SPECIES_MAPPING.keys()),
        help=(
            "Species mapping to store in the graph metadata "
            "(default: derived from the model filename)"
        ),
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF,
        help=(
            "Suggested confidence threshold stored in ONNX metadata "
            f"(default: {DEFAULT_CONF}). Hosts must pass a conf tensor on "
            "every run. This value is the usual starting point."
        ),
    )
    parser.add_argument(
        "--song-gap",
        type=float,
        default=DEFAULT_SONG_GAP,
        help=(
            "Suggested song merge gap in seconds stored in ONNX metadata "
            f"(default: {DEFAULT_SONG_GAP}). Hosts must pass a song_gap "
            "tensor on every run. This value is the usual starting point."
        ),
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=DEFAULT_NMS_IOU,
        help=f"NMS IoU threshold baked into the graph (default: {DEFAULT_NMS_IOU})",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=DEFAULT_MAX_DET,
        help=(
            "Maximum detections per class and clip "
            f"(default: {DEFAULT_MAX_DET})"
        ),
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=DEFAULT_OPSET,
        help=f"ONNX opset version (default: {DEFAULT_OPSET})",
    )
    parser.add_argument(
        "--verify-audio",
        type=str,
        default=str(DEFAULT_VERIFY_AUDIO),
        help=(
            f"Audio file ({SAMPLE_RATE} Hz) for tracing and the verification "
            f"pass (default: {DEFAULT_VERIFY_AUDIO})"
        ),
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the ONNX Runtime check that compares the export against PyTorch",
    )

    args = parser.parse_args()

    if not Path(args.pt_model).exists():
        print(f"Error: Model file not found: {args.pt_model}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model: {args.pt_model}")
    network, image_size, checkpoint_names, source_dtype = load_detection_model(args.pt_model)
    num_classes, num_anchors = probe_network(network, image_size)
    mapping_name, names = species_names(args.pt_model, args.species_mapping, checkpoint_names)

    precision, network_dtype = resolve_precision(args.precision, source_dtype)
    output_path = build_output_path(args.output_path, args.pt_model, precision)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if network_dtype == torch.float16:
        network = network.half()

    graph = BirdDetectionGraph(
        detection_model=network,
        num_classes=num_classes,
        num_anchors=num_anchors,
        image_size=image_size,
        network_dtype=network_dtype,
        nms_iou_threshold=args.nms_iou,
        max_detections=args.max_det,
    ).eval()

    print(f"Species mapping: {mapping_name or 'unknown (using checkpoint names)'}")
    print(f"Classes: {num_classes}")
    print(f"Network input: {image_size}x{image_size}, precision {precision}")
    print(f"Default conf (tunable): {args.conf}")
    print(f"Default song_gap (tunable): {args.song_gap}s")
    print(f"NMS IoU threshold (baked in): {args.nms_iou}")
    print(f"Max detections per class and clip: {args.max_det}")

    print(f"\nExporting to: {output_path}")
    trace_audio = None
    try:
        trace_audio = load_verification_audio(args.verify_audio, TRACE_SECONDS)
    except SystemExit as exc:
        if not args.no_verify:
            print(exc, file=sys.stderr)
            sys.exit(1)
        print(exc)
        print("Tracing with silence instead.")

    export_onnx(
        graph,
        output_path,
        args.opset,
        dummy_seconds=TRACE_SECONDS,
        default_conf=args.conf,
        default_song_gap=args.song_gap,
        dummy_audio=trace_audio,
    )

    write_metadata(
        output_path,
        graph_metadata(
            args.pt_model,
            precision,
            mapping_name,
            names,
            args.conf,
            args.nms_iou,
            args.song_gap,
            args.max_det,
        ),
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {output_path} ({size_mb:.1f} MB)\n")

    if not args.no_verify:
        verify_onnx(
            graph,
            output_path,
            default_conf=args.conf,
            default_song_gap=args.song_gap,
            audio_path=args.verify_audio,
        )

    print(
        "\nGraph interface:\n"
        f"  audio      float32 ({SAMPLE_RATE} Hz mono, at least "
        f"{MIN_AUDIO_SECONDS:g}s, shape (num_samples,))\n"
        f"  conf       float32 (1,), required, suggested default {args.conf}\n"
        f"  song_gap   float32 (1,), required, suggested default {args.song_gap}s\n"
        "  detections float32 (num_detections, 8): time_start, time_end, "
        "freq_low_hz, freq_high_hz, avg_confidence, max_confidence, "
        "detections_merged, class_id"
    )


if __name__ == "__main__":
    main()
