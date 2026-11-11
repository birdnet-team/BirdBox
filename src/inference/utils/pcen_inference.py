# SPDX-FileCopyrightText: 2025-2026 BirdNET-Team
# SPDX-License-Identifier: MIT
"""
Spectrogram features optimized for inference on long continuous audio files.

The features are log-floor mel spectrograms: log power in dB minus each mel
band's noise floor (its 20th percentile over the 60 s segment), -3..45 dB above
the floor mapped to 0..100. A recorder gain is a constant dB offset and cancels,
and a loud sound does not dim what follows it. The models were trained on the
same features (BirdBox-Train, holdout_ssn/RESULTS.md, "Spectrogram frontend").
BirdBox used PCEN before November 2026; the module keeps its name.

Unlike training (which processes independent 60s chunks and must avoid
cross-boundary clips), inference processes continuous files where segment
boundaries are artificial (memory management only).

This version generates clips at regular intervals regardless of segment
boundaries, ensuring complete coverage of the audio.
"""

import numpy as np
import librosa
import gc
import sys
import os
import warnings

# Add src directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config

try:
    from inference.utils.thread_limits import apply_worker_thread_limits
except ImportError:
    try:
        from utils.thread_limits import apply_worker_thread_limits
    except ImportError:
        apply_worker_thread_limits = None


def get_fft_and_pcen_settings():
    """Return default FFT and log-floor settings for audio processing. This has to match the training settings"""
    return {
        "fmin": 50,
        "fmax": 15000,
        "hop_length": 375,
        "n_fft": 2048,
        "n_mels": 256,
        "floor_percentile": 20,  # noise floor: this percentile of each band over the floor window
        "floor_lo_db": -3.0,     # dB above the floor drawn as the lowest colour (0)
        "floor_hi_db": 45.0,     # dB above the floor drawn as the highest colour (100)
        "sr": 32000,
        "win_length": 2048,  # Should match n_fft to avoid zero-padding artifacts
        "window": "flattop",
        "left_pad_length": 0.5,
        # Seconds: each clip's floor over that much audio up to the clip's end (rolling
        # floor, the default). None: one floor per 60 s segment (deprecated)
        "floor_window": config.FLOOR_WINDOW_SECONDS,
    }


def log_floor(melspec, settings):
    """
    Log-floor features of a mel power spectrogram, on the 0..100 image scale.

    Frames of digital silence (the padding after the file end) are left out of
    the floor, so they do not pull it down.
    """
    db = 10 * np.log10(np.maximum(melspec, 1e-10))
    sound = melspec.sum(axis=0) > 0
    floor = np.percentile(
        db[:, sound] if sound.any() else db,
        settings["floor_percentile"],
        axis=1,
        keepdims=True,
    )
    lo, hi = settings["floor_lo_db"], settings["floor_hi_db"]
    return ((db - floor - lo) * (100.0 / (hi - lo))).astype(np.float32)


def history_db(audio, settings):
    """dB mel frames of a stretch of audio (center=False), and which frames hold sound."""
    stft = librosa.stft(
        audio,
        n_fft=settings["n_fft"],
        win_length=settings["win_length"],
        hop_length=settings["hop_length"],
        window=settings["window"],
        center=False,
    )
    power = np.abs(stft) ** 2
    del stft
    mel = librosa.feature.melspectrogram(
        S=power,
        sr=settings["sr"],
        n_fft=settings["n_fft"],
        n_mels=settings["n_mels"],
        fmin=settings["fmin"],
        fmax=settings["fmax"],
        htk=True,
    )
    del power
    return 10 * np.log10(np.maximum(mel, 1e-10)), mel.sum(axis=0) > 0


# The rolling floor's frame grid: frame m of a file starts at sample m * hop + FRAME_ORIGIN.
# -250 is where the segment code's frames fall in the first segment (its 0.5 s left pad,
# 42 frames of 375 samples, minus 16000), which is also how the training images were made.
FRAME_ORIGIN = 42 * 375 - 16000


def rolling_floor(db, sound, first_frame, end_sample, settings):
    """
    The noise floor of the clip that ends at end_sample: the floor percentile of
    each band over the frames that lie in the last floor_window seconds before
    end_sample. db and sound start at frame first_frame of the file's frame grid.
    Frames of digital silence are left out, as in log_floor.
    """
    hop, n_fft = settings["hop_length"], settings["n_fft"]
    window = int(round(settings["floor_window"] * settings["sr"]))
    # frame m covers [m hop + FRAME_ORIGIN, m hop + FRAME_ORIGIN + n_fft)
    hi = (end_sample - n_fft - FRAME_ORIGIN) // hop + 1 - first_frame
    lo = max(0, -(-(end_sample - window - FRAME_ORIGIN) // hop) - first_frame)
    win, keep = db[:, lo:hi], sound[lo:hi]
    return np.percentile(
        win[:, keep] if keep.any() else win,
        settings["floor_percentile"],
        axis=1,
        keepdims=True,
    )


def _plan_clip_times(total_duration, clip_length):
    clip_hop_seconds = clip_length / 2
    clip_times = []
    current_time = 0.0
    while current_time + clip_length <= total_duration:
        clip_times.append(current_time)
        current_time += clip_hop_seconds
    return clip_times


def _analysis_end_samples(num_samples, sr, clip_length):
    """Sample count that lets the last window on the clip grid finish.

    Clip starts sit on a 1.5 s grid. The returned length is the end of the
    first grid window that still contains the last real sample. Files that
    already end on that grid are unchanged.
    """
    clip_samples = int(round(float(clip_length) * sr))
    hop_samples = int(round(float(clip_length) * sr / 2.0))
    if clip_samples <= 0 or hop_samples <= 0 or num_samples <= 0:
        return max(num_samples, clip_samples)
    extra = max(0, num_samples - clip_samples)
    steps = (extra + hop_samples - 1) // hop_samples
    return clip_samples + steps * hop_samples


def trailing_silence_seconds(duration, sr, clip_length):
    """Seconds of silence added so the final clip window is complete."""
    if duration <= 0 or sr <= 0:
        return 0.0
    num_samples = int(round(float(duration) * sr))
    target = _analysis_end_samples(num_samples, sr, clip_length)
    return max(0.0, (target / float(sr)) - float(duration))


def _pad_to_clip_length(audio, sr, clip_length):
    """Right-pad with silence until the last clip window is complete.

    Returns ``(audio, content_end_time)``. ``content_end_time`` is the original
    duration in seconds when padding was applied, otherwise ``None``.
    """
    target_samples = _analysis_end_samples(len(audio), sr, clip_length)
    if target_samples <= 0 or len(audio) >= target_samples:
        return audio, None
    content_end_time = len(audio) / float(sr)
    padded = np.pad(audio, (0, target_samples - len(audio)), mode="constant")
    return np.ascontiguousarray(padded, dtype=np.float32), content_end_time


def _iter_pcen_segment_jobs(
    audio, sr, settings, segment_length_seconds, clip_times, clip_length, content_end_time
):
    """Yield picklable jobs, one per 60 s memory segment."""
    segment_samples = int(segment_length_seconds * sr)
    segment_start_sample = 0

    while segment_start_sample < len(audio):
        segment_end_sample = min(segment_start_sample + segment_samples, len(audio))

        pad_before = int(2 * sr)
        pad_after = int(2 * sr)
        padded_start = max(0, segment_start_sample - pad_before)
        padded_end = min(len(audio), segment_end_sample + pad_after)
        segment_audio = audio[padded_start:padded_end]

        if len(segment_audio) < 2 * settings["n_fft"]:
            break

        segment_start_time = segment_start_sample / sr
        segment_end_time = segment_end_sample / sr
        segment_clip_times = [
            clip_time for clip_time in clip_times
            if segment_start_time <= clip_time < segment_end_time
        ]

        history = {}
        if settings.get("floor_window"):
            # Rolling floor: one spectrogram for the segment's clips and their floors, on the
            # file's frame grid (FRAME_ORIGIN). It starts with the first frame of the first
            # clip's floor window. Before the file start come the samples the segment code's
            # left pad puts there (audio[15750:16000]).
            hop = settings["hop_length"]
            clip_samples = int(round(clip_length * sr))
            window = int(round(settings["floor_window"] * sr))
            first = max(0, -(-(segment_start_sample + clip_samples - window - FRAME_ORIGIN) // hop))
            start, end = first * hop + FRAME_ORIGIN, min(segment_end_sample + pad_after, len(audio))
            if start < 0:
                pre = int(settings["left_pad_length"] * sr)
                piece = np.concatenate([audio[pre + start:pre], audio[:end]])
            else:
                piece = audio[start:end]
            history = {
                "history_audio": np.ascontiguousarray(piece, dtype=np.float32),
                "history_first_frame": first,
            }

        yield {
            **history,
            "segment_audio": np.ascontiguousarray(segment_audio, dtype=np.float32),
            "sr": sr,
            "settings": settings,
            "segment_start_sample": segment_start_sample,
            "padded_start": padded_start,
            "segment_start_time": segment_start_time,
            "segment_end_time": segment_end_time,
            "clip_times": segment_clip_times,
            "clip_length": clip_length,
            "content_end_time": content_end_time,
        }
        segment_start_sample = segment_end_sample


def pcen_segment_worker(job):
    """
    Compute the feature clips of one memory segment.

    Used both sequentially and from a process pool. Pin BLAS threads when
    called as a worker so segments do not oversubscribe the machine.
    """
    import multiprocessing
    if (
        apply_worker_thread_limits is not None
        and multiprocessing.current_process().name != "MainProcess"
    ):
        apply_worker_thread_limits(1)

    settings = job["settings"]
    sr = job["sr"]
    segment_audio = job["segment_audio"]
    clip_length = job["clip_length"]

    if job.get("history_audio") is not None:
        return _rolling_segment_clips(job)

    pcen_pad_len = int(settings["left_pad_length"] * sr)
    segment_with_pcen_pad = np.concatenate([segment_audio[:pcen_pad_len], segment_audio])

    stft = librosa.stft(
        segment_with_pcen_pad,
        n_fft=settings["n_fft"],
        win_length=settings["win_length"],
        hop_length=settings["hop_length"],
        window=settings["window"],
        center=False,
    )

    abs2_stft = np.abs(stft) ** 2
    del stft
    gc.collect()

    melspec = librosa.feature.melspectrogram(
        S=abs2_stft,
        sr=sr,
        n_fft=settings["n_fft"],
        n_mels=settings["n_mels"],
        fmin=settings["fmin"],
        fmax=settings["fmax"],
        htk=True,
    )

    del abs2_stft
    gc.collect()

    # The first frames are repeated in front: the warm-up of the former PCEN
    # smoother. Log-floor needs none, but the training images were rendered with
    # these frames in the floor percentile, so they stay.
    loop_length = min(100, melspec.shape[1] // 4)
    if loop_length > 0:
        melspec_looped = np.concatenate([melspec[:, :loop_length], melspec], axis=1)
        del melspec
        gc.collect()
    else:
        melspec_looped = melspec

    pcen_looped = log_floor(melspec_looped, settings)

    del melspec_looped
    gc.collect()

    pcen_segment = pcen_looped[:, loop_length:] if loop_length > 0 else pcen_looped
    del pcen_looped
    gc.collect()

    pcen_pad_frames = pcen_pad_len // settings["hop_length"]
    pcen_segment = pcen_segment[:, pcen_pad_frames:].astype("float32")

    clip_length_frames = 252
    clips = []
    for clip_time in job["clip_times"]:
        time_in_padded_segment = clip_time - (job["padded_start"] / sr)
        clip_start_frame = int(time_in_padded_segment * sr / settings["hop_length"])

        if clip_start_frame >= 0 and clip_start_frame + clip_length_frames <= pcen_segment.shape[1]:
            clip = pcen_segment[:, clip_start_frame:clip_start_frame + clip_length_frames]
            clip_record = {
                "pcen": clip,
                "start_time": clip_time,
                "end_time": clip_time + clip_length,
                "start_frame": clip_start_frame,
            }
            if job.get("content_end_time") is not None:
                clip_record["content_end_time"] = job["content_end_time"]
            clips.append(clip_record)

    return clips


def _rolling_segment_clips(job):
    """
    The clips of one segment with the rolling floor. Clip c of the file is frames
    128 c to 128 c + 251 of the file's frame grid, so its pixels do not depend on
    where the segment starts, and a recording in progress has them as soon as the
    audio reaches the clip's end.
    """
    settings, sr, clip_length = job["settings"], job["sr"], job["clip_length"]
    hop, first = settings["hop_length"], job["history_first_frame"]
    db, sound = history_db(job["history_audio"], settings)
    scale = 100.0 / (settings["floor_hi_db"] - settings["floor_lo_db"])
    clip_length_frames = 252
    clips = []
    for clip_time in job["clip_times"]:
        # Clip starts are multiples of the hop (1.5 s = 128 hops): clip c begins with frame 128 c
        frame = int(round(clip_time * sr)) // hop - first
        if frame < 0 or frame + clip_length_frames > db.shape[1]:
            continue
        floor = rolling_floor(db, sound, first, int(round((clip_time + clip_length) * sr)), settings)
        clip = ((db[:, frame:frame + clip_length_frames] - settings["floor_lo_db"]) * scale).astype(np.float32)
        record = {
            "pcen": (clip - floor * scale).astype(np.float32),
            "start_time": clip_time,
            "end_time": clip_time + clip_length,
            "start_frame": frame,
        }
        if job.get("content_end_time") is not None:
            record["content_end_time"] = job["content_end_time"]
        clips.append(record)
    return clips


def compute_pcen_for_inference(
    audio,
    sr,
    segment_length_seconds=None,
    verbose=True,
    executor=None,
    num_workers=1,
):
    """
    Compute log-floor features on long audio with complete clip coverage for inference.
    
    Unlike the training version, this generates clips at regular intervals
    across the entire audio, ignoring artificial segment boundaries.

    When ``executor`` is a process pool and there are multiple segments,
    segments are processed in parallel.
    
    Args:
        audio: Input audio signal
        sr: Original sample rate
        segment_length_seconds: Length of the compute segments (default from config). With the
            deprecated segment floor (BIRDBOX_FLOOR_WINDOW=segment) also the floor's window
        verbose: If True, print clip-planning and extraction progress
        executor: Optional ProcessPoolExecutor for segment-level parallelism
        num_workers: Max parallel segment jobs when executor is set
    
    Returns:
        clips: List of dictionaries with each clip's feature array (key 'pcen', kept for compatibility)
        sr: Sample rate used for processing
    """
    if segment_length_seconds is None:
        segment_length_seconds = config.PCEN_SEGMENT_LENGTH
    
    settings = get_fft_and_pcen_settings()
    target_sr = settings["sr"]
    if not settings["floor_window"]:
        warnings.warn(config.SEGMENT_FLOOR_DEPRECATION, DeprecationWarning, stacklevel=2)
        if verbose:
            print(f"Warning: {config.SEGMENT_FLOOR_DEPRECATION}")
    
    # Validate audio
    librosa.util.valid_audio(audio)
    
    # Map to the range [-2**31, 2**31[
    audio = (audio * (2 ** 31)).astype("float32")
    
    # Resample if needed
    if not sr == target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
        if verbose:
            print(f"Resampled audio to {sr} Hz")
    
    clip_length = config.CLIP_LENGTH
    audio, content_end_time = _pad_to_clip_length(audio, sr, clip_length)
    if content_end_time is not None and verbose:
        print(
            f"Audio is {content_end_time:.2f}s. "
            f"Padding {len(audio) / sr - content_end_time:.2f}s of silence "
            f"so the final {clip_length}s clip is complete."
        )

    total_duration = len(audio) / sr
    clip_times = _plan_clip_times(total_duration, clip_length)
    
    if verbose:
        print(f"Planning to extract {len(clip_times)} clips from {total_duration:.1f}s audio")

    jobs = list(_iter_pcen_segment_jobs(
        audio, sr, settings, segment_length_seconds, clip_times, clip_length, content_end_time
    ))

    clips = []
    use_pool = executor is not None and num_workers > 1 and len(jobs) > 1
    if use_pool:
        for segment_clips in executor.map(pcen_segment_worker, jobs, chunksize=1):
            clips.extend(segment_clips)
    else:
        for job in jobs:
            clips.extend(pcen_segment_worker(job))
    
    if verbose:
        print(f"Successfully extracted {len(clips)} clips")
    return clips, sr
