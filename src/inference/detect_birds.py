#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-2026 BirdNET-Team
# SPDX-License-Identifier: MIT
"""
Detect bird calls in arbitrary-length audio files.

This script loads audio files (WAV, FLAC, OGG, MP3), processes them using the same PCEN 
pipeline as training, and detects bird calls using a trained YOLO model. It returns 
timestamped detections with species labels and confidence scores.

Usage:
    python src/inference/detect_birds.py --audio path/to/audio.wav --model path/to/model.pt
    python src/inference/detect_birds.py --audio audio.flac --model model.pt --output-path results
    python src/inference/detect_birds.py --audio audio.mp3 --model model.pt --conf 0.25 --nms-iou 0.5 --verbose
"""

import os
import sys
import argparse
import json
import tempfile
import shutil
import threading
import csv
from collections import deque
from pathlib import Path
from typing import List, Dict, Tuple, Callable, Optional
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, Future, wait, FIRST_COMPLETED

# Try to import file locking (Unix/Linux)
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False
    # Windows fallback
    try:
        import msvcrt
        HAS_MSVCRT = True
    except ImportError:
        HAS_MSVCRT = False

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Spawn workers re-import this file as __mp_main__ when the CLI is the
# program entry point. Skip torch/YOLO so render processes stay CPU-only.
_IS_SPAWN_BOOTSTRAP = (
    __name__ == "__mp_main__"
    or multiprocessing.current_process().name != "MainProcess"
)

if _IS_SPAWN_BOOTSTRAP:
    try:
        from inference.utils.thread_limits import apply_worker_thread_limits
    except ImportError:
        from utils.thread_limits import apply_worker_thread_limits
    apply_worker_thread_limits(1)

import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

try:
    from inference.utils.spectrogram_render import (
        render_spectrogram_image,
        render_spectrogram_worker,
    )
except ImportError:
    from utils.spectrogram_render import (
        render_spectrogram_image,
        render_spectrogram_worker,
    )

if not _IS_SPAWN_BOOTSTRAP:
    import config

    try:
        from inference.utils import pcen_inference
        from inference.utils.audio_load import load_audio_signal
        from inference.utils.file_preprocess import preprocess_audio_file_worker
        from inference.utils.thread_limits import (
            apply_worker_thread_limits,
            configure_parent_inference_threads,
        )
        from inference.utils.raven_export import save_raven_tables
        from inference.utils.xeno_canto_export import build_xeno_canto_json
        from inference.utils.yolo_loading import (
            load_yolo,
            prepare_yolo_for_model,
            yolo_predict_kwargs,
            is_engine_model,
            is_onnx_model,
            is_tflite_model,
            onnx_static_batch,
        )
        from inference.utils.output_paths import (
            algorithm_metadata_json_path,
            format_output_path,
            ensure_output_directory,
            DEFAULT_RESULTS_DIR,
            prepare_results_directory,
            is_default_results_path,
            RAW_DETECTIONS_JSON,
            save_args_yaml,
        )
    except ImportError:
        # If running as script, try relative import
        from utils import pcen_inference
        from utils.audio_load import load_audio_signal
        from utils.file_preprocess import preprocess_audio_file_worker
        from utils.thread_limits import (
            apply_worker_thread_limits,
            configure_parent_inference_threads,
        )
        from utils.raven_export import save_raven_tables
        from utils.xeno_canto_export import build_xeno_canto_json
        from utils.yolo_loading import (
            load_yolo,
            prepare_yolo_for_model,
            yolo_predict_kwargs,
            is_engine_model,
            is_onnx_model,
            is_tflite_model,
            onnx_static_batch,
        )
        from utils.output_paths import (
            algorithm_metadata_json_path,
            format_output_path,
            ensure_output_directory,
            DEFAULT_RESULTS_DIR,
            prepare_results_directory,
            is_default_results_path,
            RAW_DETECTIONS_JSON,
            save_args_yaml,
        )


# Just-Bird has a single class, so time-only song merging chains simultaneous
# calls that sit in different frequency bands. Other mappings keep the time-only
# merger. A negative requested value forces the time-only merger for Just-Bird
# (used when comparing against the historical behavior).
JUST_BIRD_SPECIES_MAPPING = "Just-Bird"
JUST_BIRD_FREQ_IOU = 0.6


def resolve_freq_iou_threshold(
    species_mapping: Optional[str],
    requested: Optional[float],
) -> Optional[float]:
    """
    Frequency-IoU gate for song merging.

    Returns None for the time-only merger. Just-Bird is the only mapping that
    receives a gate. Other mappings ignore ``requested``.
    """
    if species_mapping != JUST_BIRD_SPECIES_MAPPING:
        return None
    if requested is None:
        return JUST_BIRD_FREQ_IOU
    if requested < 0:
        return None
    return requested


def _frequency_iou(low_a: float, high_a: float, low_b: float, high_b: float) -> float:
    """1D IoU of two frequency intervals. Zero when they do not overlap."""
    intersection = min(high_a, high_b) - max(low_a, low_b)
    if intersection <= 0:
        return 0.0
    union = (high_a - low_a) + (high_b - low_b) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _blank_song(det: Dict, filename: str) -> Dict:
    song = {
        'species': det['species'],
        'species_id': det['species_id'],
        'time_start': det['time_start'],
        'time_end': det['time_end'],
        'avg_confidence': det['confidence'],
        'max_confidence': det['confidence'],
        'detections_merged': det.get('detections_merged', 1),
        'freq_low_hz': det['freq_low_hz'],
        'freq_high_hz': det['freq_high_hz'],
    }
    if filename != '__single_file__':
        song['filename'] = filename
    return song


def _extend_song(song: Dict, det: Dict) -> None:
    # A fused box already stands for several raw detections; count those.
    count = det.get('detections_merged', 1)
    song['time_end'] = max(song['time_end'], det['time_end'])
    song['avg_confidence'] = (
        song['avg_confidence'] * song['detections_merged'] + det['confidence'] * count
    ) / (song['detections_merged'] + count)
    song['max_confidence'] = max(song['max_confidence'], det['confidence'])
    song['detections_merged'] += count
    song['freq_low_hz'] = min(song['freq_low_hz'], det['freq_low_hz'])
    song['freq_high_hz'] = max(song['freq_high_hz'], det['freq_high_hz'])


def _group_detections(detections: List[Dict]) -> Dict:
    """Group by (filename, species_id). Each file has its own timeline at 0."""
    groups = {}
    for det in detections:
        filename = det.get('filename', '__single_file__')
        key = (filename, det['species_id'])
        if key not in groups:
            groups[key] = []
        groups[key].append(det)
    return groups


def _reconstruct_songs_by_time(detections: List[Dict], song_gap_threshold: float) -> List[Dict]:
    """Merge same-species detections whose time gap is within the song gap."""
    merged_songs = []
    for (filename, _species_id), group_detections in _group_detections(detections).items():
        group_detections = sorted(group_detections, key=lambda x: x['time_start'])
        current_song = None
        for det in group_detections:
            if current_song is None:
                current_song = _blank_song(det, filename)
                continue
            gap = det['time_start'] - current_song['time_end']
            if gap <= song_gap_threshold:
                _extend_song(current_song, det)
            else:
                merged_songs.append(current_song)
                current_song = _blank_song(det, filename)
        if current_song is not None:
            merged_songs.append(current_song)
    return sorted(merged_songs, key=lambda x: (x.get('filename', ''), x['time_start']))


def _reconstruct_songs_by_frequency(
    detections: List[Dict],
    song_gap_threshold: float,
    freq_iou_threshold: float,
) -> List[Dict]:
    """
    Merge same-species detections that are close in time and in frequency.

    Several songs of one species can stay open at once, so a low-band call and
    a high-band call that overlap in time remain separate boxes.
    """
    merged_songs = []
    for (filename, _species_id), group_detections in _group_detections(detections).items():
        group_detections = sorted(
            group_detections,
            key=lambda x: (x['time_start'], x['freq_low_hz']),
        )
        active = []
        for det in group_detections:
            still = []
            best = None
            best_score = -1.0
            for song in active:
                if det['time_start'] - song['time_end'] > song_gap_threshold:
                    merged_songs.append(song)
                    continue
                still.append(song)
                score = _frequency_iou(
                    song['freq_low_hz'],
                    song['freq_high_hz'],
                    det['freq_low_hz'],
                    det['freq_high_hz'],
                )
                if score >= freq_iou_threshold and score > best_score:
                    best = song
                    best_score = score
            active = still
            if best is None:
                active.append(_blank_song(det, filename))
            else:
                _extend_song(best, det)
        merged_songs.extend(active)
    return sorted(merged_songs, key=lambda x: (x.get('filename', ''), x['time_start']))


def reconstruct_songs(
    detections: List[Dict],
    song_gap_threshold: float,
    freq_iou_threshold: Optional[float] = None,
) -> List[Dict]:
    """
    Reconstruct continuous bird songs by merging temporally adjacent detections.
    
    Standalone version usable without a BirdCallDetector instance (e.g. in evaluation
    after filtering raw detections by confidence). Matches app/detect_birds workflow:
    filter by confidence first, then merge.
    
    Args:
        detections: List of raw detections (each with time_start, time_end, species_id,
            species, confidence, freq_low_hz, freq_high_hz; optional 'filename' for multi-file)
        song_gap_threshold: Max gap (seconds) between detections to merge into same song
        freq_iou_threshold: If set, also require this frequency IoU before merging.
            None keeps the time-only merger used by multi-species models.
        
    Returns:
        List of merged song segments (with avg_confidence, max_confidence, detections_merged)
    """
    if len(detections) == 0:
        return []
    if freq_iou_threshold is None:
        return _reconstruct_songs_by_time(detections, song_gap_threshold)
    return _reconstruct_songs_by_frequency(
        detections, song_gap_threshold, freq_iou_threshold
    )


# Window fusion (default for the call-level models). Each call is seen by two
# 50%-overlap windows; the song merge above unions both views, which widens
# boxes and chains fast calls. Fusion instead:
#   1. ownership: keep a box only from the window whose middle half holds its
#      centre (the first window also owns the file start, the last the end).
#      These middles tile the file, so each call is kept once, from its most
#      central view;
#   2. weighted box fusion: boxes of one window that overlap at 2D IoU >= 0.4
#      (YOLO's per-window NMS at 0.7 leaves such near-duplicates) are averaged,
#      weighted by confidence, with the strongest member's confidence;
#   3. join boxes that overlap in time by at least 0.05 s with frequency IoU
#      >= 0.8 (a long call split across windows);
#   4. drop nested boxes: a box with at least 80% of its area inside a larger
#      box of the same species is removed (a syllable inside its phrase box).
#      Neutral on Hawaii val, +0.012 F1 at IoU 0.5 on the held-out site, and
#      about 10-25% fewer boxes.
# Chosen on Hawaii val full files and checked on a held-out site; see
# BirdBox-Train training_and_validation/runs/Just-Bird/holdout_ssn/RESULTS.md.
# With phrase_gap set, step 3 instead joins same-band boxes (frequency IoU
# >= 0.3) closer than phrase_gap seconds: syllables into one phrase box, for
# annotation conventions that box whole phrases.
MERGE_METHODS = ('fusion', 'reconstruct')
# Models whose labels box single calls. On their test sets fusion beats the
# song merge (IoU 0.5 micro F1: Hawaii 0.582 vs 0.515, Southern-Sierra-Nevada
# 0.637 vs 0.615). Models trained on whole-song labels keep the song merge,
# whose wide boxes match those labels better; see BirdBox-Train
# training_and_validation/runs/merge_multispecies/.
FUSION_SPECIES_MAPPINGS = (JUST_BIRD_SPECIES_MAPPING, 'Hawaii', 'Southern-Sierra-Nevada')
FUSION_WBF_IOU = 0.4
FUSION_JOIN_GAP = -0.05
FUSION_JOIN_FREQ_IOU = 0.8
FUSION_NESTED_RATIO = 0.8
PHRASE_FREQ_IOU = 0.3


def resolve_merge_method(species_mapping: Optional[str], requested: Optional[str]) -> str:
    """'fusion' for the call-level models (FUSION_SPECIES_MAPPINGS) and 'reconstruct'
    for the others, unless requested."""
    if requested in (None, 'auto'):
        return 'fusion' if species_mapping in FUSION_SPECIES_MAPPINGS else 'reconstruct'
    if requested not in MERGE_METHODS:
        raise ValueError(f"Unknown merge method: {requested}")
    return requested


def _box_iou_2d(a: Dict, b: Dict) -> float:
    t = min(a['time_end'], b['time_end']) - max(a['time_start'], b['time_start'])
    f = min(a['freq_high_hz'], b['freq_high_hz']) - max(a['freq_low_hz'], b['freq_low_hz'])
    if t <= 0 or f <= 0:
        return 0.0
    inter = t * f
    union = (
        (a['time_end'] - a['time_start']) * (a['freq_high_hz'] - a['freq_low_hz'])
        + (b['time_end'] - b['time_start']) * (b['freq_high_hz'] - b['freq_low_hz'])
        - inter
    )
    return inter / union if union > 0 else 0.0


def _owned(det: Dict, file_end: float) -> bool:
    """True if the box centre lies in the part of the file its window owns."""
    if 'clip_start' not in det or 'clip_end' not in det:
        return True
    quarter = (det['clip_end'] - det['clip_start']) / 4
    centre = (det['time_start'] + det['time_end']) / 2
    lo = det['clip_start'] + quarter if det['clip_start'] > 0 else float('-inf')
    hi = det['clip_end'] - quarter if det['clip_end'] < file_end else float('inf')
    return lo <= centre < hi


def _weighted_box_fusion(detections: List[Dict], iou_threshold: float) -> List[Dict]:
    """Greedy clusters by confidence (2D IoU >= threshold), confidence-weighted average."""
    order = sorted(detections, key=lambda d: d['confidence'], reverse=True)
    by_start = sorted(range(len(order)), key=lambda i: order[i]['time_start'])
    starts = [order[i]['time_start'] for i in by_start]
    max_len = max(d['time_end'] - d['time_start'] for d in order)
    assigned = [False] * len(order)
    fused = []
    for i, head in enumerate(order):
        if assigned[i]:
            continue
        assigned[i] = True
        members = [head]
        # Boxes can only overlap if their times overlap.
        lo = int(np.searchsorted(starts, head['time_start'] - max_len, 'left'))
        hi = int(np.searchsorted(starts, head['time_end'], 'right'))
        for j in by_start[lo:hi]:
            if not assigned[j] and _box_iou_2d(head, order[j]) >= iou_threshold:
                assigned[j] = True
                members.append(order[j])
        weights = np.array([m['confidence'] for m in members], dtype=np.float64)
        if weights.sum() <= 0:
            weights = np.ones(len(members))
        box = dict(head)
        for key in ('time_start', 'time_end', 'freq_low_hz', 'freq_high_hz'):
            box[key] = float(np.dot(weights, [m[key] for m in members]) / weights.sum())
        box['confidence'] = head['confidence']
        box['detections_merged'] = len(members)
        fused.append(box)
    return fused


def _drop_nested(songs: List[Dict], ratio: float) -> List[Dict]:
    """
    Remove boxes that lie at least ``ratio`` of their area inside a larger box
    of the same species and file. Of two boxes of equal area, the one with the
    lower max_confidence (then the later one) counts as the inner box.
    """
    kept = []
    for _key, group in _group_detections(songs).items():
        t0 = np.array([s['time_start'] for s in group])
        t1 = np.array([s['time_end'] for s in group])
        f0 = np.array([s['freq_low_hz'] for s in group])
        f1 = np.array([s['freq_high_hz'] for s in group])
        conf = np.array([s['max_confidence'] for s in group])
        area = (t1 - t0) * (f1 - f0)
        index = np.arange(len(group))
        for i, song in enumerate(group):
            inter = (np.clip(np.minimum(t1[i], t1) - np.maximum(t0[i], t0), 0, None)
                     * np.clip(np.minimum(f1[i], f1) - np.maximum(f0[i], f0), 0, None))
            larger = (area > area[i]) | ((area == area[i]) & (
                (conf > conf[i]) | ((conf == conf[i]) & (index < i))))
            inside = (inter >= ratio * area[i]) & larger & (index != i)
            if not inside.any():
                kept.append(song)
    return sorted(kept, key=lambda x: (x.get('filename', ''), x['time_start']))


def file_ends_from_raw(detections: List[Dict]) -> Dict[str, float]:
    """Latest clip_end per filename of a raw (unfiltered) detection list.

    Evaluation filters raw detections by confidence before merging. Taking the
    file end from the unfiltered list keeps window ownership the same as in
    live detection, where the last window is known.
    """
    ends: Dict[str, float] = {}
    for det in detections:
        name = det.get('filename', '__single_file__')
        end = det.get('clip_end', det['time_end'])
        if end > ends.get(name, float('-inf')):
            ends[name] = end
    return ends


def fuse_window_detections(
    detections: List[Dict],
    phrase_gap: Optional[float] = None,
    file_end=None,
) -> List[Dict]:
    """
    Window fusion of raw detections (see the comment above MERGE_METHODS).

    Args:
        detections: Raw detections with clip_start / clip_end (optional
            'filename' for multi-file input). Filter by confidence first.
        phrase_gap: If above 0, join same-band boxes closer than this many
            seconds into one phrase box. None or 0 keeps call-level boxes.
        file_end: End of the last window: a float (one file), a dict of
            filename -> end (file_ends_from_raw), or None to take it per file
            as the latest clip_end among the detections.

    Returns:
        Merged segments in the reconstruct_songs format.
    """
    if len(detections) == 0:
        return []
    fused = []
    for (filename, _species_id), group in _group_detections(detections).items():
        end = file_end.get(filename) if isinstance(file_end, dict) else file_end
        if end is None:
            end = max(d.get('clip_end', d['time_end']) for d in group)
        owned = [d for d in group if _owned(d, end)]
        if owned:
            fused.extend(_weighted_box_fusion(owned, FUSION_WBF_IOU))
    if phrase_gap is None or phrase_gap <= 0:
        joined = _reconstruct_songs_by_frequency(fused, FUSION_JOIN_GAP, FUSION_JOIN_FREQ_IOU)
    else:
        joined = _reconstruct_songs_by_frequency(fused, phrase_gap, PHRASE_FREQ_IOU)
    return _drop_nested(joined, FUSION_NESTED_RATIO)


def merge_detections(
    detections: List[Dict],
    merge_method: str,
    song_gap_threshold: float,
    freq_iou_threshold: Optional[float] = None,
    phrase_gap: Optional[float] = None,
    file_end=None,
) -> List[Dict]:
    """Merge raw detections with 'fusion' or 'reconstruct' (filter by confidence first).

    file_end is passed to fuse_window_detections (float, filename dict or None).
    """
    if merge_method == 'fusion':
        return fuse_window_detections(detections, phrase_gap=phrase_gap, file_end=file_end)
    return reconstruct_songs(detections, song_gap_threshold, freq_iou_threshold=freq_iou_threshold)


def resolve_merge_settings(
    model_config: Dict,
    song_gap: Optional[float] = None,
    freq_iou: Optional[float] = None,
    merge_method: Optional[str] = None,
    phrase_gap: Optional[float] = None,
) -> Dict:
    """
    merge_detections keyword arguments for a raw detections JSON.

    Explicit arguments win, then the JSON model_config, then the defaults
    (fusion for Just-Bird, Hawaii and Southern-Sierra-Nevada, song merge with
    gap 0.1 s for other mappings).
    """
    species_mapping = model_config.get('species_mapping')
    if song_gap is None:
        song_gap = float(model_config.get('song_gap_threshold', 0.1))
    if freq_iou is None:
        freq_iou = model_config.get('freq_iou_threshold')
    if merge_method in (None, 'auto'):
        merge_method = model_config.get('merge_method')
    method = resolve_merge_method(species_mapping, merge_method)
    if phrase_gap is None:
        phrase_gap = model_config.get('phrase_gap')
    return {
        'merge_method': method,
        'song_gap_threshold': song_gap,
        'freq_iou_threshold': resolve_freq_iou_threshold(species_mapping, freq_iou),
        'phrase_gap': phrase_gap if method == 'fusion' else None,
    }


def describe_merge_settings(settings: Dict) -> str:
    """One-line description of resolve_merge_settings output for logs."""
    if settings['merge_method'] == 'fusion':
        gap = settings['phrase_gap']
        return "fusion" + (f", phrase_gap={gap}s" if gap is not None else "")
    freq = settings['freq_iou_threshold']
    return (f"reconstruct, song_gap={settings['song_gap_threshold']}s, "
            f"freq_iou={'time-only' if freq is None else freq}")


class BirdCallDetector:
    """
    Detector for bird calls in audio files.
    
    This class handles the complete pipeline from audio loading to detection,
    using the same processing approach as training.
    """
    
    # Class-level locks for thread/process-safe YOLO model inference
    # YOLO models are not thread-safe when multiple instances run inference simultaneously
    _inference_lock = threading.Lock()  # For thread-level locking (same process)
    _lock_file_path = None  # For process-level locking (file-based, works across processes)
    
    @classmethod
    def _get_lock_file(cls):
        """Get or create the lock file for process-level locking."""
        if cls._lock_file_path is None:
            # Create lock file in system temp directory
            lock_dir = Path(tempfile.gettempdir())
            cls._lock_file_path = lock_dir / "birdbox_yolo_inference.lock"
            # Create the lock file if it doesn't exist
            cls._lock_file_path.touch(exist_ok=True)
        return cls._lock_file_path
    
    # Frequency range constants (same as in dataset_conversion/get_labels.py)
    MAX_FREQ = 15000  # Hz
    MIN_FREQ = 50     # Hz
    
    def __init__(self, model_path: str, species_mapping: str, conf_threshold: float = 0.001, 
                 nms_iou_threshold: float = 0.7, song_gap_threshold: float = 0.1,
                 num_workers: int = 1, verbose: bool = True,
                 freq_iou_threshold: Optional[float] = None,
                 merge_method: Optional[str] = None,
                 phrase_gap: Optional[float] = None):
        """
        Initialize the bird call detector.
        
        Args:
            model_path: Path to the trained YOLO model (.pt, .onnx, .engine, etc.)
            species_mapping: Dataset name for species mappings (e.g., 'Hawaii', 'Western-US', 'All-In-One')
            conf_threshold: Confidence threshold for detections (0-1)
            nms_iou_threshold: IoU threshold for NMS (per-clip and across time windows) (0-1)
            song_gap_threshold: Max gap (seconds) between detections to merge into same song (default: 0.1)
            num_workers: Number of CPU processes for PCEN, spectrogram rendering,
                and (when detecting a folder) whole-file preprocess (default: 1).
                YOLO inference stays in this process and runs in batches.
            verbose: If True, print per-file details and clip-level progress bars.
                If False, show a single file-level progress bar.
            freq_iou_threshold: Just-Bird frequency IoU gate. None uses the Just-Bird
                default. Other species mappings always keep the time-only merger.
                Used by the 'reconstruct' merge only.
            merge_method: 'fusion' (window ownership + weighted box fusion) or
                'reconstruct' (song merge). None / 'auto' picks fusion for
                Just-Bird, Hawaii and Southern-Sierra-Nevada and reconstruct for
                every other mapping.
            phrase_gap: Fusion only. Join same-band boxes closer than this many
                seconds into one phrase box. None keeps call-level boxes.
        """
        self.verbose = verbose
        self._yolo_device = prepare_yolo_for_model(model_path)
        configure_parent_inference_threads(num_workers, self._yolo_device)
        self.model = load_yolo(model_path)
        self.model_path = str(model_path)
        self.conf_threshold = conf_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.song_gap_threshold = song_gap_threshold
        self.freq_iou_threshold = resolve_freq_iou_threshold(species_mapping, freq_iou_threshold)
        self.merge_method = resolve_merge_method(species_mapping, merge_method)
        self.phrase_gap = phrase_gap if self.merge_method == 'fusion' else None
        self.num_workers = num_workers
        self._cpu_pool: Optional[ProcessPoolExecutor] = None
        self._cpu_pool_workers: Optional[int] = None
        self.settings = pcen_inference.get_fft_and_pcen_settings()
        
        # Load species-specific mappings
        self.species_mapping = species_mapping
        self.species_mappings = config.get_species_mapping(species_mapping)
        self.id_to_ebird = self.species_mappings['id_to_ebird']
        self.bird_colors = self.species_mappings['bird_colors']
        
        # PCEN and spectrogram settings (same as training)
        self.colormap = 'inferno'
        self.vmin = 0.0
        self.vmax = 100.0
        self.clip_length = self.species_mappings['clip_length']  # 3 seconds
        self.clip_hop = self.clip_length / 2  # 1.5 seconds (50% overlap)
        self.height_width = self.species_mappings['height_width']  # 256
        self.pcen_segment_length = self.species_mappings['pcen_segment_length']  # 60
        
        # Precompute mel scale range for frequency conversion
        self.max_mel = librosa.hz_to_mel(self.MAX_FREQ, htk=True)
        self.min_mel = librosa.hz_to_mel(self.MIN_FREQ, htk=True)
        self.mel_range = self.max_mel - self.min_mel
        
        self._log(f"Loaded model: {model_path}")
        self._log(f"Species mapping: {self.species_mapping}")
        self._log(f"Species count: {len(self.id_to_ebird)}")
        self._log(f"Confidence threshold: {conf_threshold}")
        self._log(f"NMS IoU threshold: {nms_iou_threshold}")
        self._log(f"Merge method: {self.merge_method}")
        if self.merge_method == 'fusion':
            if self.phrase_gap is not None:
                self._log(f"Phrase gap: {self.phrase_gap}s")
        else:
            self._log(f"Song gap threshold: {song_gap_threshold}s")
            if self.freq_iou_threshold is not None:
                self._log(f"Frequency IoU gate (Just-Bird): {self.freq_iou_threshold}")
        if num_workers > 1:
            self._log(
                f"CPU process pool: {num_workers} workers "
                f"(PCEN, spectrograms, multi-file preprocess)"
            )

    def _log(self, *args, **kwargs):
        """Print only when verbose logging is enabled."""
        if self.verbose:
            print(*args, **kwargs)
    
    def pixels_to_hz(self, y_pixel: float) -> float:
        """
        Convert y-axis pixel coordinate to frequency in Hz.
        
        This reverses the conversion done in dataset_conversion/get_labels.py (BirdBox-Train repository):
        1. Hz → Mel (HTK) → Normalize [0,1] → Invert y-axis → Pixels [0,256]
        
        Args:
            y_pixel: Y-coordinate in pixels (0-256, where 0 is top/high freq)
            
        Returns:
            Frequency in Hz
        """
        image_height = self.height_width  # 256
        
        # Normalize pixel to [0, 1]
        y_normalized = y_pixel / image_height
        
        # Un-invert y-axis (in get_labels.py: y_center = 1 - y_center)
        # Lower pixel values (top of image) = higher frequencies
        y_normalized = 1.0 - y_normalized
        
        # Convert from normalized [0,1] back to mel scale
        mel_value = y_normalized * self.mel_range + self.min_mel
        
        # Convert mel to Hz using HTK scale (same as training)
        freq_hz = librosa.mel_to_hz(mel_value, htk=True)
        
        # Clip to valid range
        freq_hz = np.clip(freq_hz, self.MIN_FREQ, self.MAX_FREQ)
        
        # Round to integer (same as original annotations)
        return int(round(freq_hz))
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file with automatic fallback for compatibility.
        
        Supports multiple formats: WAV, FLAC, OGG, MP3
        Uses soundfile (fast) with librosa fallback for problematic files.
        
        Note: Model was trained on WAV files. Lossy formats (MP3, OGG) may affect 
        detection performance, especially for faint calls or high frequencies.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            audio: Audio signal as numpy array
            sr: Sample rate
            
        Raises:
            Exception: If file cannot be loaded by any method
        """
        self._log(f"\nLoading audio: {audio_path}")
        
        # Check for lossy formats and warn user
        audio_path_obj = Path(audio_path)
        lossy_formats = {'.mp3', '.ogg'}
        if audio_path_obj.suffix.lower() in lossy_formats:
            self._log("⚠️  Warning: Lossy audio format detected (.mp3 or .ogg)")
            self._log("   Model was trained on lossless WAV files. For best results:")
            self._log("   - Use WAV or FLAC formats")
            self._log("   - If using MP3/OGG, ensure high bitrate (≥256 kbps)")
            self._log("   - Be aware of potential performance degradation for faint/distant calls\n")
        
        try:
            audio, sr = sf.read(audio_path, dtype='float32')
            loading_method = "soundfile"
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
        except Exception as sf_error:
            self._log(f"⚠️  soundfile failed ({sf_error})")
            self._log("   Attempting to load with librosa fallback...")
            try:
                audio, sr = load_audio_signal(audio_path)
                loading_method = "librosa"
                self._log("✓ Successfully loaded using librosa fallback")
            except Exception as librosa_error:
                raise Exception(str(librosa_error)) from librosa_error
        
        duration = len(audio) / sr
        self._log(f"Duration: {duration:.2f} seconds")
        self._log(f"Sample rate: {sr} Hz")
        if loading_method == "librosa":
            self._log("(Loaded via librosa fallback)")
        
        return audio, sr
    
    def create_spectrogram_image(self, pcen_data: np.ndarray, output_path: str):
        """
        Create a spectrogram image from PCEN data (same as training).
        
        Args:
            pcen_data: PCEN features
            output_path: Where to save the image
        """
        render_spectrogram_image(
            pcen_data,
            output_path,
            sr=self.settings["sr"],
            hop_length=self.settings["hop_length"],
            colormap=self.colormap,
            vmin=self.vmin,
            vmax=self.vmax,
        )
    
    def process_audio_to_clips(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """
        Process audio into PCEN clips with sliding window.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            List of clips with PCEN data and timing information
        """
        self._log("\nProcessing audio with PCEN...")
        
        # Use inference-specific PCEN processing that handles continuous audio
        # (unlike training which must avoid cross-boundary clips between chunks)
        executor = None
        workers = 1
        if self.num_workers > 1:
            executor = self._get_cpu_pool(self.num_workers)
            workers = self.num_workers
        clips, _ = pcen_inference.compute_pcen_for_inference(
            audio,
            sr,
            segment_length_seconds=self.pcen_segment_length,
            verbose=self.verbose,
            executor=executor,
            num_workers=workers,
        )
        
        return clips
    
    def detect_in_clip(self, clip_data: Dict, temp_dir: Path) -> List[Dict]:
        """
        Run detection on a single clip.
        
        Args:
            clip_data: Dictionary with 'pcen', 'start_time', 'end_time'
            temp_dir: Temporary directory for spectrogram images
            
        Returns:
            List of detections with timing and species information
        """
        # Create spectrogram image
        temp_image = temp_dir / f"temp_{clip_data['start_time']:.1f}s.png"
        self.create_spectrogram_image(clip_data['pcen'], str(temp_image))
        
        # Run inference with locks to prevent interference between concurrent sessions
        # YOLO models are not thread-safe when multiple instances run inference simultaneously
        # Use both thread lock (for same process) and file lock (for different processes)
        with BirdCallDetector._inference_lock:
            # File-based lock works across processes (e.g., multiple Streamlit workers)
            lock_file_path = BirdCallDetector._get_lock_file()
            lock_file = None
            try:
                lock_file = open(lock_file_path, 'w')
                
                # Try to acquire file lock (works across processes)
                if HAS_FCNTL:
                    # Unix/Linux: use fcntl
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                elif HAS_MSVCRT:
                    # Windows: use msvcrt
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
                
                # Run inference
                results = self.model(
                    str(temp_image),
                    **yolo_predict_kwargs(
                        self._yolo_device,
                        conf=self.conf_threshold,
                        iou=self.nms_iou_threshold,
                        verbose=False,
                    ),
                )[0]
                
            except (IOError, OSError, AttributeError):
                # Fallback: if file locking fails, just use the thread lock
                # (works within same process, which is the common case)
                results = self.model(
                    str(temp_image),
                    **yolo_predict_kwargs(
                        self._yolo_device,
                        conf=self.conf_threshold,
                        iou=self.nms_iou_threshold,
                        verbose=False,
                    ),
                )[0]
            finally:
                # Always release the file lock and close file
                if lock_file is not None:
                    try:
                        if HAS_FCNTL:
                            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                        elif HAS_MSVCRT:
                            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                    except (IOError, OSError, AttributeError):
                        pass  # Ignore unlock errors
                    finally:
                        lock_file.close()
        
        detections = self._parse_box_detections(results, clip_data)
        
        # Clean up temp image
        temp_image.unlink(missing_ok=True)
        
        return detections
    
    def _parse_box_detections(self, results, clip_data: Dict) -> List[Dict]:
        """
        Parse YOLO box results into detection dictionaries.
        
        Thread-safe: only reads immutable instance attributes (clip_length, id_to_ebird, etc.).
        """
        detections = []
        
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                image_width = 256
                clip_duration = self.clip_length
                
                x1_pixels, y1_pixels, x2_pixels, y2_pixels = xyxy
                
                time_start_in_clip = (x1_pixels / image_width) * clip_duration
                time_end_in_clip = (x2_pixels / image_width) * clip_duration
                
                abs_time_start = clip_data['start_time'] + time_start_in_clip
                abs_time_end = clip_data['start_time'] + time_end_in_clip

                # Silence padding extends a short file to one full clip.
                # Keep only the part that overlaps the original audio.
                content_end = clip_data.get('content_end_time')
                if content_end is not None:
                    if abs_time_start >= content_end:
                        continue
                    if abs_time_end > content_end:
                        abs_time_end = content_end
                    if abs_time_end <= abs_time_start:
                        continue
                
                species = self.id_to_ebird.get(cls, f"unknown_{cls}")
                
                # y1 (top of box) = high frequency, y2 (bottom of box) = low frequency
                freq_high_hz = self.pixels_to_hz(y1_pixels)
                freq_low_hz = self.pixels_to_hz(y2_pixels)
                
                detections.append({
                    'species': species,
                    'species_id': cls,
                    'confidence': conf,
                    'time_start': abs_time_start,
                    'time_end': abs_time_end,
                    'freq_low_hz': freq_low_hz,
                    'freq_high_hz': freq_high_hz,
                    'clip_start': clip_data['start_time'],
                    'clip_end': clip_data['end_time'],
                })
        
        return detections
    
    def _get_cpu_pool(self, num_workers: int) -> ProcessPoolExecutor:
        """Reuse a spawn process pool across files so workers are not respawned."""
        if self._cpu_pool is not None and self._cpu_pool_workers != num_workers:
            self._close_cpu_pool()
        if self._cpu_pool is None:
            ctx = multiprocessing.get_context("spawn")
            self._cpu_pool = ProcessPoolExecutor(
                max_workers=num_workers,
                mp_context=ctx,
                initializer=apply_worker_thread_limits,
                initargs=(1,),
            )
            self._cpu_pool_workers = num_workers
        return self._cpu_pool

    def _close_cpu_pool(self) -> None:
        """Shut down CPU workers if they were started."""
        if self._cpu_pool is not None:
            self._cpu_pool.shutdown(wait=True)
            self._cpu_pool = None
            self._cpu_pool_workers = None

    def _close_render_pool(self) -> None:
        """Backward-compatible alias used by detect() cleanup."""
        self._close_cpu_pool()

    def _render_kwargs(self) -> Dict:
        return {
            "sr": self.settings["sr"],
            "hop_length": self.settings["hop_length"],
            "colormap": self.colormap,
            "vmin": self.vmin,
            "vmax": self.vmax,
        }

    def _infer_batch_size(self) -> int:
        if is_engine_model(self.model_path) or is_tflite_model(self.model_path):
            return 1
        if is_onnx_model(self.model_path):
            if not hasattr(self, '_onnx_batch'):
                self._onnx_batch = onnx_static_batch(self.model_path)
            if self._onnx_batch:
                return self._onnx_batch
        return max(64, self.num_workers * 4)

    def _yolo_batch_kwargs(self, batch_size: int) -> Dict:
        predict_kwargs = yolo_predict_kwargs(
            self._yolo_device,
            conf=self.conf_threshold,
            iou=self.nms_iou_threshold,
            verbose=False,
            stream=False,
        )
        # TensorRT engines and TFLite often have a fixed/export-time batch size.
        if not is_engine_model(self.model_path) and not is_tflite_model(self.model_path):
            predict_kwargs["batch"] = batch_size
        return predict_kwargs

    def _infer_rendered_clips(self, clips: List[Dict]) -> List[Dict]:
        """Run batched YOLO on clips that already have an ``image_path``."""
        all_detections = []
        batch_size = self._infer_batch_size()
        for start in range(0, len(clips), batch_size):
            chunk = clips[start:start + batch_size]
            image_paths = [clip_data["image_path"] for clip_data in chunk]
            try:
                results = self.model(
                    image_paths,
                    **self._yolo_batch_kwargs(len(image_paths)),
                )
            finally:
                for image_path in image_paths:
                    Path(image_path).unlink(missing_ok=True)
            if not isinstance(results, list):
                results = [results]
            for result, clip_data in zip(results, chunk):
                all_detections.extend(self._parse_box_detections(result, clip_data))
        return all_detections

    def _submit_render_chunk(
        self,
        executor: ProcessPoolExecutor,
        chunk: List[Dict],
        temp_dir: Path,
        chunk_index: int,
    ) -> List[Future]:
        render_kwargs = self._render_kwargs()
        futures = []
        for j, clip_data in enumerate(chunk):
            image_path = (
                temp_dir
                / f"clip_{chunk_index:04d}_{j:04d}_{clip_data['start_time']:.3f}s.png"
            )
            job = {
                "pcen": np.ascontiguousarray(clip_data["pcen"], dtype=np.float32),
                "output_path": str(image_path),
                **render_kwargs,
            }
            futures.append(executor.submit(render_spectrogram_worker, job))
        return futures

    def _detect_clips_parallel(self, clips: List[Dict], temp_dir: Path,
                                progress_callback=None) -> List[Dict]:
        """
        Hybrid clip pipeline: process-pool spectrogram rendering, batched YOLO.

        Keeps two render chunks in flight so workers stay busy while the parent
        runs a larger YOLO batch.
        """
        num_workers = min(self.num_workers, len(clips))
        chunk_size = self._infer_batch_size()
        prefetch_chunks = 2
        chunks = [clips[i:i + chunk_size] for i in range(0, len(clips), chunk_size)]

        self._log(
            f"Rendering spectrograms with {num_workers} processes "
            f"(batched inference, chunk size {chunk_size})..."
        )
        executor = self._get_cpu_pool(self.num_workers)

        all_detections = []
        completed = 0
        pending: deque = deque()
        next_index = 0

        def submit_more():
            nonlocal next_index
            while next_index < len(chunks) and len(pending) < prefetch_chunks:
                pending.append((
                    chunks[next_index],
                    self._submit_render_chunk(executor, chunks[next_index], temp_dir, next_index),
                ))
                next_index += 1

        submit_more()
        progress_bar = None
        if progress_callback is None:
            progress_bar = tqdm(
                total=len(clips),
                desc=f"Pipeline ({num_workers} processes)",
                disable=not self.verbose,
                dynamic_ncols=True,
            )

        try:
            while pending:
                chunk, futures = pending.popleft()
                image_paths = [future.result() for future in futures]
                submit_more()
                try:
                    results = self.model(
                        image_paths,
                        **self._yolo_batch_kwargs(len(image_paths)),
                    )
                finally:
                    for image_path in image_paths:
                        Path(image_path).unlink(missing_ok=True)

                if not isinstance(results, list):
                    results = [results]

                for result, clip_data in zip(results, chunk):
                    all_detections.extend(self._parse_box_detections(result, clip_data))

                completed += len(chunk)
                if progress_callback:
                    progress_callback(
                        completed,
                        len(clips),
                        f"Rendering + detecting ({num_workers} processes)...",
                    )
                elif progress_bar is not None:
                    progress_bar.update(len(chunk))
        finally:
            while pending:
                _, futures = pending.popleft()
                for future in futures:
                    try:
                        future.result()
                    except Exception:
                        pass
            if progress_bar is not None:
                progress_bar.close()

        return all_detections
    
    def merge_overlapping_detections(
        self,
        detections: List[Dict],
        merge_mode: str = 'reconstruct',
        file_end: Optional[float] = None,
    ) -> List[Dict]:
        """
        Merge detections using different strategies.
        
        Args:
            detections: List of all detections from all clips
            merge_mode: Strategy for merging
                - 'nms': Traditional NMS to remove duplicates (keeps highest confidence)
                - 'reconstruct': the detector's merge_method ('fusion' or the song merge)
            file_end: End of the file's last window (fusion ownership). None takes
                the latest clip_end among the detections.
            
        Returns:
            Filtered/merged list of detections
        """
        if len(detections) == 0:
            return []
        
        if merge_mode == 'nms':
            return self._merge_with_nms(detections)
        elif merge_mode == 'reconstruct':
            return self._reconstruct_songs(detections, file_end=file_end)
        else:
            raise ValueError(f"Unknown merge_mode: {merge_mode}")
    
    def _merge_with_nms(self, detections: List[Dict]) -> List[Dict]:
        """
        Traditional NMS: Remove duplicate detections from overlapping windows.
        Keeps the detection with highest confidence.
        """
        # Sort by confidence (highest first)
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        
        for detection in detections:
            should_keep = True
            
            for kept in keep:
                # Only compare detections of the same species
                if detection['species_id'] != kept['species_id']:
                    continue
                
                # Calculate temporal IoU
                time_overlap_start = max(detection['time_start'], kept['time_start'])
                time_overlap_end = min(detection['time_end'], kept['time_end'])
                
                if time_overlap_start < time_overlap_end:
                    overlap_duration = time_overlap_end - time_overlap_start
                    detection_duration = detection['time_end'] - detection['time_start']
                    kept_duration = kept['time_end'] - kept['time_start']
                    
                    intersection = overlap_duration
                    union = detection_duration + kept_duration - intersection
                    iou = intersection / union if union > 0 else 0
                    
                    if iou > self.nms_iou_threshold:
                        should_keep = False
                        break
            
            if should_keep:
                keep.append(detection)
        
        # Sort by time
        keep = sorted(keep, key=lambda x: x['time_start'])
        return keep
    
    def _reconstruct_songs(self, detections: List[Dict], file_end: Optional[float] = None) -> List[Dict]:
        """
        Merge the overlapping windows' detections with the detector's merge method.
        Delegates to module-level merge_detections for consistency with evaluation.
        """
        return merge_detections(
            detections,
            self.merge_method,
            self.song_gap_threshold,
            freq_iou_threshold=self.freq_iou_threshold,
            phrase_gap=self.phrase_gap,
            file_end=file_end,
        )
    
    def detect_multiple_files(self, audio_paths: List[str], output_path: str = None, output_formats: List[str] = None, no_merge: bool = False) -> List[Dict]:
        """
        Detect bird calls in multiple audio files.
        
        Args:
            audio_paths: List of paths to WAV files
            output_path: Optional output directory for result files
            output_formats: One or more output formats - 'json-with-algorithm-metadata',
                'simplified-csv', 'xeno-canto-annota-json', 'raven-selection-table', or 'all'
            no_merge: If True, return raw (unmerged) detections; add filename to each for later merge.
            
        Returns:
            List of all detections from all files with timing and species information
        """
        if output_formats is None:
            output_formats = ['json-with-algorithm-metadata']

        self._log(f"\nProcessing {len(audio_paths)} audio files...")

        if self.num_workers > 1 and len(audio_paths) > 1:
            all_detections = self._detect_files_parallel(audio_paths, no_merge=no_merge)
        else:
            all_detections = self._detect_files_sequential(audio_paths, no_merge=no_merge)
        
        self._log(f"\n{'='*60}")
        self._log(f"TOTAL DETECTIONS ACROSS ALL FILES: {len(all_detections)}")
        self._log(f"{'='*60}")
        
        # Save results if output path is specified
        if output_path and all_detections:
            self.save_results(
                all_detections,
                output_path,
                audio_paths[0] if audio_paths else None,
                output_formats,
                no_merge=no_merge,
            )
        
        return all_detections

    def _attach_file_metadata(self, detections: List[Dict], audio_path: str) -> List[Dict]:
        filename = Path(audio_path).name
        for detection in detections:
            detection['filename'] = filename
            detection['file_path'] = str(audio_path)
        return detections

    def _report_file_error(self, audio_path: str, error: Exception) -> None:
        err = f"Error processing {audio_path}: {error}"
        if self.verbose:
            print(err)
        else:
            tqdm.write(err)

    def _detect_files_sequential(self, audio_paths: List[str], no_merge: bool = False) -> List[Dict]:
        all_detections = []
        path_iter = audio_paths
        if not self.verbose:
            path_iter = tqdm(audio_paths, desc="Processing files", unit="file", dynamic_ncols=True)

        for i, audio_path in enumerate(path_iter, 1):
            self._log(f"\n{'='*60}")
            self._log(f"Processing file {i}/{len(audio_paths)}: {Path(audio_path).name}")
            self._log(f"{'='*60}")
            try:
                file_detections = self.detect_single_file(audio_path, no_merge=no_merge)
                self._attach_file_metadata(file_detections, audio_path)
                all_detections.extend(file_detections)
                self._log(f"Found {len(file_detections)} detections in this file")
            except Exception as e:
                self._report_file_error(audio_path, e)
                continue
        return all_detections

    def _detect_files_parallel(self, audio_paths: List[str], no_merge: bool = False) -> List[Dict]:
        """
        Preprocess many files in the CPU pool (load + PCEN + render).
        YOLO stays in the parent and runs as each file's images are ready.
        """
        num_workers = min(self.num_workers, len(audio_paths))
        self._log(
            f"Preprocessing files with {num_workers} processes "
            f"(batched inference in parent)..."
        )
        executor = self._get_cpu_pool(num_workers)
        render_kwargs = self._render_kwargs()
        root_temp = Path(tempfile.mkdtemp(prefix="birdbox_pre_"))
        per_file: List[Optional[List[Dict]]] = [None] * len(audio_paths)

        try:
            jobs = []
            for index, audio_path in enumerate(audio_paths):
                file_temp = root_temp / f"{index:05d}_{Path(audio_path).stem}"
                file_temp.mkdir(parents=True, exist_ok=True)
                jobs.append({
                    "audio_path": audio_path,
                    "temp_dir": str(file_temp),
                    "pcen_segment_length": self.pcen_segment_length,
                    **render_kwargs,
                })

            in_flight = {}
            job_iter = enumerate(jobs)

            def submit_more():
                while len(in_flight) < num_workers:
                    try:
                        index, job = next(job_iter)
                    except StopIteration:
                        break
                    future = executor.submit(preprocess_audio_file_worker, job)
                    in_flight[future] = index

            submit_more()
            progress_bar = None
            if not self.verbose:
                progress_bar = tqdm(
                    total=len(audio_paths),
                    desc="Processing files",
                    unit="file",
                    dynamic_ncols=True,
                )

            while in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    index = in_flight.pop(future)
                    audio_path = audio_paths[index]
                    result = future.result()
                    try:
                        if result["error"]:
                            raise Exception(result["error"])
                        file_detections = self._infer_rendered_clips(result["clips"])
                        if not no_merge:
                            file_end = max((c["end_time"] for c in result["clips"]), default=None)
                            file_detections = self.merge_overlapping_detections(
                                file_detections, merge_mode='reconstruct', file_end=file_end
                            )
                        self._attach_file_metadata(file_detections, audio_path)
                        per_file[index] = file_detections
                        self._log(
                            f"Found {len(file_detections)} detections in {Path(audio_path).name}"
                        )
                    except Exception as e:
                        self._report_file_error(audio_path, e)
                        per_file[index] = []
                    submit_more()
                    if progress_bar is not None:
                        progress_bar.update(1)

            if progress_bar is not None:
                progress_bar.close()
        finally:
            shutil.rmtree(root_temp, ignore_errors=True)

        all_detections = []
        for file_detections in per_file:
            if file_detections:
                all_detections.extend(file_detections)
        return all_detections

    def detect_single_file(
        self,
        audio_path: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        no_merge: bool = False
    ) -> List[Dict]:
        """
        Detect bird calls in a single audio file (renamed from detect method).
        
        Args:
            audio_path: Path to the WAV file
            progress_callback: Optional callback function(current, total, message) for progress updates
            no_merge: If True, return raw (unmerged) detections for later filter-then-merge (e.g. F-score sweep).
            
        Returns:
            List of detections with timing and species information (merged or raw per no_merge)
        """
        # Load audio
        audio, sr = self.load_audio(audio_path)
        
        # Process to clips
        clips = self.process_audio_to_clips(audio, sr)
        
        # Create temporary directory for spectrogram images
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Run detection on each clip
            self._log(f"\nRunning detection on {len(clips)} clips...")
            
            if self.num_workers > 1 and len(clips) > 1:
                all_detections = self._detect_clips_parallel(clips, temp_dir, progress_callback)
            else:
                all_detections = []
                if progress_callback:
                    for i, clip_data in enumerate(clips):
                        clip_detections = self.detect_in_clip(clip_data, temp_dir)
                        all_detections.extend(clip_detections)
                        progress_callback(i + 1, len(clips), f"Detecting bird calls in {self.clip_length} second clips...")
                else:
                    for clip_data in tqdm(clips, desc="Detecting", disable=not self.verbose, dynamic_ncols=True):
                        clip_detections = self.detect_in_clip(clip_data, temp_dir)
                        all_detections.extend(clip_detections)
            
            self._log(f"\nFound {len(all_detections)} raw detections")
            
            if no_merge:
                return all_detections
            
            # Merge detections (default: reconstruct songs)
            if progress_callback:
                progress_callback(len(clips), len(clips), "Reconstructing bird songs...")
            self._log("Reconstructing continuous bird songs from detections...")
            file_end = max((c['end_time'] for c in clips), default=None)
            final_detections = self.merge_overlapping_detections(
                all_detections, merge_mode='reconstruct', file_end=file_end
            )
            
            self._log(f"Final count: {len(final_detections)} song segments")
            
            return final_detections
            
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    def detect(self, audio_path: str, output_path: str = None, output_formats: List[str] = None, no_merge: bool = False) -> List[Dict]:
        """
        Detect bird calls in an audio file or directory.
        
        Supports WAV, FLAC, OGG, and MP3 formats.
        
        Args:
            audio_path: Path to audio file or directory containing audio files
            output_path: Optional output directory for result files
            output_formats: One or more output formats - 'json-with-algorithm-metadata',
                'simplified-csv', 'xeno-canto-annota-json', 'raven-selection-table', or 'all'
            no_merge: If True, save raw (unmerged) detections for filter-then-merge workflows (e.g. F-score sweep).
            
        Returns:
            List of detections with timing and species information
        """
        if output_formats is None:
            output_formats = ['json-with-algorithm-metadata']
        # Find all audio files
        audio_files = find_audio_files(audio_path, verbose=self.verbose)
        
        if not audio_files:
            print("No audio files found to process")
            return []

        try:
            if len(audio_files) == 1:
                # Single file - use original logic
                if self.verbose:
                    detections = self.detect_single_file(audio_files[0], no_merge=no_merge)
                else:
                    with tqdm(total=len(audio_files), desc="Processing files", unit="file", dynamic_ncols=True) as pbar:
                        detections = self.detect_single_file(audio_files[0], no_merge=no_merge)
                        pbar.update(1)
                if output_path:
                    self.save_results(
                        detections,
                        output_path,
                        audio_files[0],
                        output_formats,
                        no_merge=no_merge,
                    )
                return detections
            else:
                # Multiple files - use new batch processing
                return self.detect_multiple_files(audio_files, output_path, output_formats, no_merge=no_merge)
        finally:
            self._close_render_pool()
    
    def _convert_to_json_serializable(self, obj):
        """
        Convert numpy types and other non-JSON-serializable objects to standard Python types.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        else:
            return obj

    def save_detections(self, detections: List[Dict], output_path: str, audio_path: str = None):
        """
        Save detections to JSON file.
        
        Args:
            detections: List of detections
            output_path: Path to save JSON file
            audio_path: Original audio file path (for metadata, optional for multi-file)
        """
        # Determine if this is multi-file output
        is_multi_file = any('filename' in det for det in detections)
        
        if is_multi_file:
            # Multi-file output
            unique_files = list(set(det.get('file_path', 'unknown') for det in detections))
            output = {
                'audio_files': unique_files,
                'file_count': len(unique_files),
                'model_config': {
                    'model': self.model_path,
                    'confidence_threshold': self.conf_threshold,
                    'nms_iou_threshold': self.nms_iou_threshold,
                    'song_gap_threshold': self.song_gap_threshold,
                    'species_mapping': self.species_mapping,
                    'merge_method': self.merge_method,
                    **(
                        {'freq_iou_threshold': self.freq_iou_threshold}
                        if self.freq_iou_threshold is not None else {}
                    ),
                    **({'phrase_gap': self.phrase_gap} if self.phrase_gap is not None else {}),
                },
                'detection_count': len(detections),
                'detections': detections
            }
        else:
            # Single file output
            output = {
                'audio_file': str(audio_path) if audio_path else 'unknown',
                'model_config': {
                    'model': self.model_path,
                    'confidence_threshold': self.conf_threshold,
                    'nms_iou_threshold': self.nms_iou_threshold,
                    'song_gap_threshold': self.song_gap_threshold,
                    'species_mapping': self.species_mapping,
                    'merge_method': self.merge_method,
                    **(
                        {'freq_iou_threshold': self.freq_iou_threshold}
                        if self.freq_iou_threshold is not None else {}
                    ),
                    **({'phrase_gap': self.phrase_gap} if self.phrase_gap is not None else {}),
                },
                'detection_count': len(detections),
                'detections': detections
            }
        
        # Convert all numpy types to JSON-serializable types
        output = self._convert_to_json_serializable(output)
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nSaved detections to: {output_path}")
    
    def save_detections_csv(self, detections: List[Dict], output_path: str, audio_path: str = None):
        """
        Save detections to CSV file in the same format as annotations.csv.
        
        Args:
            detections: List of detections
            output_path: Path to save CSV file
            audio_path: Original audio file path (for metadata, optional for multi-file)
        """
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header (same as annotations.csv, plus Confidence)
            writer.writerow(['Filename', 'Start Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)', 'Species eBird Code', 'Confidence'])
            
            # Write detection data
            for det in detections:
                # Use filename from detection if available (for multi-file), otherwise use audio_path
                if 'filename' in det:
                    filename = det['filename']
                elif audio_path:
                    filename = Path(audio_path).name
                else:
                    filename = 'unknown'
                
                # Merged detections have avg_confidence; raw have confidence
                confidence = det['avg_confidence'] if 'detections_merged' in det else det['confidence']
                
                writer.writerow([
                    filename,
                    f"{det['time_start']:.1f}",
                    f"{det['time_end']:.1f}",
                    det['freq_low_hz'],
                    det['freq_high_hz'],
                    det['species'],
                    f"{confidence:.3f}"
                ])
        
        print(f"\nSaved detections to CSV: {output_path}")

    def save_detections_raven_txt(self, detections: List[Dict], output_path: str, audio_path: str = None):
        """
        Save detections as Raven Selection Tables (.txt, tab-separated).

        Single-file mode writes one .txt file.
        Multi-file mode writes one .txt file per source audio into a directory.

        Args:
            detections: List of detections
            output_path: Output directory (multi-file Raven tables go in a raven/ subdirectory)
            audio_path: Original audio file path (optional for multi-file)
        """
        output_dir = Path(output_path)
        written = save_raven_tables(
            detections,
            single_file_path=format_output_path(str(output_dir), 'raven-selection-table'),
            multi_file_dir=output_dir / 'raven',
        )
        if written.is_dir():
            print(f"\nSaved Raven Selection Tables to directory: {written}")
        else:
            print(f"\nSaved Raven Selection Table: {written}")

    def save_detections_xc_json(self, detections: List[Dict], output_path: str, audio_path: str = None):
        """
        Save detections to Xeno-Canto Annota-JSON.

        Args:
            detections: List of detections
            output_path: Path to save Xeno-Canto Annota-JSON file
            audio_path: Original audio file path (for metadata, optional for multi-file)
        """
        output = build_xeno_canto_json(
            detections,
            audio_path=audio_path,
            species_mappings=self.species_mappings,
        )

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nSaved detections to Xeno-Canto Annota-JSON: {output_path}")
    
    def save_results(
        self,
        detections: List[Dict],
        output_path: str,
        audio_path: str = None,
        output_formats: List[str] = None,
        no_merge: bool = False,
    ):
        """
        Save detections in the specified format(s).
        
        Args:
            detections: List of detections
            output_path: Output directory for result files
            audio_path: Original audio file path (for metadata, optional for multi-file)
            output_formats: One or more output formats - 'json-with-algorithm-metadata',
                'simplified-csv', 'xeno-canto-annota-json', 'raven-selection-table', or 'all'
            no_merge: If True, write only raw_detections.json (evaluation pipeline).
        """
        if output_formats is None:
            output_formats = ['json-with-algorithm-metadata']

        if no_merge:
            self.save_detections(
                detections,
                str(algorithm_metadata_json_path(output_path, raw=True)),
                audio_path,
            )
            return

        use_all = 'all' in output_formats
        if use_all or 'json-with-algorithm-metadata' in output_formats:
            self.save_detections(
                detections,
                str(algorithm_metadata_json_path(output_path, raw=False)),
                audio_path,
            )

        if use_all or 'simplified-csv' in output_formats:
            self.save_detections_csv(
                detections,
                str(format_output_path(output_path, 'simplified-csv')),
                audio_path,
            )

        if use_all or 'xeno-canto-annota-json' in output_formats:
            self.save_detections_xc_json(
                detections,
                str(format_output_path(output_path, 'xeno-canto-annota-json')),
                audio_path,
            )

        if use_all or 'raven-selection-table' in output_formats:
            self.save_detections_raven_txt(detections, output_path, audio_path)
    
    def print_summary(self, detections: List[Dict]):
        """Print a summary of detections."""
        if len(detections) == 0:
            print("\nNo bird calls detected.")
            return
        
        print(f"\n{'='*80}")
        print("DETECTION SUMMARY")
        print(f"{'='*80}")
        
        # Group by species
        species_counts = {}
        for det in detections:
            species = det['species']
            if species not in species_counts:
                species_counts[species] = []
            species_counts[species].append(det)
        
        print(f"\nTotal detections: {len(detections)}")
        print(f"Species detected: {len(species_counts)}")
        
        # Check if these are reconstructed songs (have 'detections_merged' field)
        is_reconstructed = 'detections_merged' in detections[0] if detections else False
        
        print()
        
        for species, dets in sorted(species_counts.items()):
            print(f"{species}: {len(dets)} {'song segments' if is_reconstructed else 'detections'}")
            
            for det in dets[:5]:  # Show first 5 for each species
                duration = det['time_end'] - det['time_start']
                
                if is_reconstructed:
                    print(f"  {det['time_start']:6.2f}s - {det['time_end']:6.2f}s "
                          f"({duration:5.2f}s duration, "
                          f"{det['detections_merged']:2d} clips merged, "
                          f"avg conf: {det['avg_confidence']:.3f}, "
                          f"max conf: {det['max_confidence']:.3f})")
                else:
                    print(f"  {det['time_start']:6.2f}s - {det['time_end']:6.2f}s "
                          f"(confidence: {det['confidence']:.3f})")
            
            if len(dets) > 5:
                print(f"  ... and {len(dets) - 5} more")
            
            # Print statistics for this species
            if is_reconstructed:
                durations = [d['time_end'] - d['time_start'] for d in dets]
                merged_counts = [d['detections_merged'] for d in dets]
                print(f"  Stats: avg duration {sum(durations)/len(durations):.2f}s, "
                      f"avg clips merged {sum(merged_counts)/len(merged_counts):.1f}")
            
            print()


def find_audio_files(audio_path: str, verbose: bool = True) -> List[str]:
    """
    Find all supported audio files in the given path (file or directory).
    
    Supported formats: WAV, FLAC, OGG, MP3
    
    Args:
        audio_path: Path to a single audio file or directory containing audio files
        verbose: If True, print how many files were found in a directory
        
    Returns:
        List of paths to audio files
    """
    # Supported audio extensions (soundfile supports these natively)
    SUPPORTED_EXTENSIONS = {'.wav', '.flac', '.ogg', '.mp3'}
    
    audio_path_obj = Path(audio_path)
    
    if audio_path_obj.is_file():
        # Single file
        if audio_path_obj.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [str(audio_path_obj)]
        else:
            print(f"Warning: {audio_path} is not a supported audio file format. Supported: WAV, FLAC, OGG, MP3")
            return []
    
    elif audio_path_obj.is_dir():
        # Directory - find all supported audio files recursively
        audio_files = []
        for ext in SUPPORTED_EXTENSIONS:
            # Search for lowercase
            for audio_file in audio_path_obj.rglob(f'*{ext}'):
                audio_files.append(str(audio_file))
            # Search for uppercase
            for audio_file in audio_path_obj.rglob(f'*{ext.upper()}'):
                audio_files.append(str(audio_file))
        
        audio_files.sort()  # Sort for consistent ordering
        if verbose:
            print(f"Found {len(audio_files)} audio files in directory: {audio_path}")
        return audio_files
    
    else:
        print(f"Error: {audio_path} is neither a file nor a directory")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Detect bird calls in audio files using trained YOLO model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic detection (single file) - supports WAV, FLAC, OGG, MP3
  python src/inference/detect_birds.py --audio recording.wav --model models/Just-Bird.pt --species-mapping Just-Bird
  
  # Process entire folder of audio files
  python src/inference/detect_birds.py --audio /path/to/audio/folder --model models/Just-Bird.pt --species-mapping Just-Bird --output-path results --output-format all
  
  # Process FLAC file (JSON with algorithm metadata)
  python src/inference/detect_birds.py --audio recording.flac --model models/Just-Bird.pt --species-mapping Just-Bird --output-path results --output-format json-with-algorithm-metadata
  
  # Save results to simplified CSV
  python src/inference/detect_birds.py --audio recording.mp3 --model models/Just-Bird.pt --species-mapping Just-Bird --output-path results --output-format simplified-csv

  # Save results to Xeno-Canto Annota-JSON
  python src/inference/detect_birds.py --audio recording.wav --model models/Just-Bird.pt --species-mapping Just-Bird --output-path results --output-format xeno-canto-annota-json
  
  # Save Raven Selection Table (.txt)
  python src/inference/detect_birds.py --audio recording.wav --model models/Just-Bird.pt --species-mapping Just-Bird --output-path results --output-format raven-selection-table
  
  # Save all formats
  python src/inference/detect_birds.py --audio recording.ogg --model models/Just-Bird.pt --species-mapping Just-Bird --output-path results --output-format all

  # Save selected formats (space-separated)
  python src/inference/detect_birds.py --audio recording.wav --model models/Just-Bird.pt --species-mapping Just-Bird --output-path results --output-format json-with-algorithm-metadata simplified-csv
  
  # Adjust thresholds
  python src/inference/detect_birds.py --audio audio.wav --model models/Just-Bird.pt --species-mapping Just-Bird --conf 0.5 --nms-iou 0.6

  # Species-level detection (All-In-One model)
  python src/inference/detect_birds.py --audio recording.wav --model models/All-In-One.pt --species-mapping All-In-One --output-path results --output-format all

  # Verbose per-file logging
  python src/inference/detect_birds.py --audio /path/to/audio/folder --model models/Just-Bird.pt --species-mapping Just-Bird --verbose
        """
    )
    
    parser.add_argument(
        '--audio',
        type=str,
        required=True,
        help='Path to audio file (WAV/FLAC/OGG/MP3) or directory containing audio files'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to the trained model (.pt, .onnx, .engine, etc.)'
    )
    
    parser.add_argument(
        '--species-mapping',
        type=str,
        required=True,
        choices=[
            'Just-Bird',
            'All-In-One',
            'Hawaii',
            'Northeastern-US',
            'Southern-Sierra-Nevada',
            'Western-US',
            'Amazon-Basin',
        ],
        help='Dataset/species mapping used to train the model (REQUIRED)'
    )
    
    parser.add_argument(
        '--output-path',
        type=str,
        default=None,
        help=(
            f'Output directory for result files (default: {DEFAULT_RESULTS_DIR}; '
            'auto-versions to results/run_N if results/ already has outputs).'
        ),
    )
    
    parser.add_argument(
        '--output-format',
        nargs='+',
        choices=[
            'json-with-algorithm-metadata',
            'simplified-csv',
            'xeno-canto-annota-json',
            'raven-selection-table',
            'all',
        ],
        default=['json-with-algorithm-metadata'],
        help=(
            'One or more output formats (space-separated): json-with-algorithm-metadata, '
            'simplified-csv, xeno-canto-annota-json, raven-selection-table, or all'
        )
    )
    
    # the default value should work perfectly
    parser.add_argument(
        '--conf',
        type=float,
        default=0.2,
        help='Confidence threshold for detections'
    )
    
    # this value can be further explored
    parser.add_argument(
        '--nms-iou',
        type=float,
        default=0.7,
        help='NMS IoU threshold for per-clip and cross-window NMS (default: 0.7)'
    )
    
    # this value can be further explored
    parser.add_argument(
        '--song-gap',
        type=float,
        default=0.1,
        help='Max gap (seconds) between detections to merge into same song (default: 0.1)'
    )

    parser.add_argument(
        '--freq-iou',
        type=float,
        default=None,
        help=(
            'Just-Bird only. Minimum frequency IoU required to merge two detections '
            f'of class bird into one song (default: {JUST_BIRD_FREQ_IOU}). '
            'Pass a negative value to use the time-only merger. '
            'Ignored for every other species mapping.'
        ),
    )

    parser.add_argument(
        '--merge',
        choices=['auto', *MERGE_METHODS],
        default='auto',
        help=(
            'How detections from the 50%%-overlap windows are merged. fusion: keep each '
            'box from the window that sees it most centrally, average near-duplicates, '
            'join only overlapping boxes (tighter boxes). reconstruct: song merge with '
            '--song-gap / --freq-iou. auto (default): fusion for the call-level models '
            '(Just-Bird, Hawaii, Southern-Sierra-Nevada), reconstruct for the others.'
        ),
    )

    parser.add_argument(
        '--phrase-gap',
        type=float,
        default=None,
        metavar='SECONDS',
        help=(
            'Fusion only. Join same-band boxes closer than SECONDS (e.g. 0.1) into one '
            'phrase box, for annotation conventions that box whole phrases rather than '
            'single calls or syllables. Default: off (call-level boxes).'
        ),
    )
    
    # select amount of workers based on available hardware
    parser.add_argument(
        '--num-workers',
        type=int,
        default=1,
        metavar='N',
        help=(
            'Number of CPU processes for PCEN, spectrogram rendering, and '
            'multi-file preprocess. YOLO inference stays in the parent and '
            'runs in batches. (default: 1)'
        )
    )
    
    parser.add_argument(
        '--no-merge',
        action='store_true',
        help=(
            'Evaluation mode: write only raw_detections.json (ignores --output-format). '
            'Use low --conf (e.g. 0.001) for F-beta / filter-and-merge workflows.'
        ),
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help=(
            'Print per-file processing details and clip-level progress bars. '
            'Default is a single file-level progress bar.'
        ),
    )
    
    args = parser.parse_args()

    if args.no_merge and args.output_format != ['json-with-algorithm-metadata']:
        print(
            f"Note: --no-merge writes only {RAW_DETECTIONS_JSON}; "
            f"ignoring --output-format {args.output_format!r}."
        )
    output_formats = ['json-with-algorithm-metadata'] if args.no_merge else args.output_format
    if args.output_path is None or is_default_results_path(args.output_path):
        output_path = prepare_results_directory(args.output_path)
    else:
        output_path = args.output_path
    
    # Validate inputs
    if not Path(args.audio).exists():
        print(f"Error: Audio file or directory not found: {args.audio}", file=sys.stderr)
        sys.exit(1)
    
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}", file=sys.stderr)
        sys.exit(1)
    
    # Ensure output directory exists (ask user if it needs to be created)
    if output_path and not ensure_output_directory(output_path):
        sys.exit(1)

    args.output_path = output_path
    save_args_yaml(args, output_path)
    
    # Create detector
    detector = BirdCallDetector(
        model_path=args.model,
        species_mapping=args.species_mapping,
        conf_threshold=args.conf,
        nms_iou_threshold=args.nms_iou,
        song_gap_threshold=args.song_gap,
        num_workers=args.num_workers,
        verbose=args.verbose,
        freq_iou_threshold=args.freq_iou,
        merge_method=args.merge,
        phrase_gap=args.phrase_gap,
    )
    
    # Run detection
    detections = detector.detect(args.audio, output_path, output_formats, no_merge=args.no_merge)
    
    # Print summary
    if args.verbose:
        detector.print_summary(detections)


if __name__ == '__main__':
    main()

