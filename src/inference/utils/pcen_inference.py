"""
PCEN processing optimized for inference on long continuous audio files.

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
    """Return default FFT and PCEN settings for audio processing. This has to match the training settings"""
    return {
        "fmin": 50,
        "fmax": 15000,
        "hop_length": 375,
        "n_fft": 2048,
        "n_mels": 256,
        "pcen_delta": 1.0,
        "pcen_time_constant": 1.0,
        "pcen_norm_exponent": 0.75,
        "pcen_power": 0.35,
        "sr": 32000,
        "win_length": 2048,  # Should match n_fft to avoid zero-padding artifacts
        "window": "flattop",
        "left_pad_length": 0.5,
    }


def _plan_clip_times(total_duration, clip_length):
    clip_hop_seconds = clip_length / 2
    clip_times = []
    current_time = 0.0
    while current_time + clip_length <= total_duration:
        clip_times.append(current_time)
        current_time += clip_hop_seconds
    return clip_times


def _iter_pcen_segment_jobs(audio, sr, settings, segment_length_seconds, clip_times, clip_length):
    """Yield picklable jobs, one per PCEN memory segment."""
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

        yield {
            "segment_audio": np.ascontiguousarray(segment_audio, dtype=np.float32),
            "sr": sr,
            "settings": settings,
            "segment_start_sample": segment_start_sample,
            "padded_start": padded_start,
            "segment_start_time": segment_start_time,
            "segment_end_time": segment_end_time,
            "clip_times": segment_clip_times,
            "clip_length": clip_length,
        }
        segment_start_sample = segment_end_sample


def pcen_segment_worker(job):
    """
    Compute PCEN clips for one memory segment.

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

    loop_length = min(100, melspec.shape[1] // 4)
    if loop_length > 0:
        melspec_looped = np.concatenate([melspec[:, :loop_length], melspec], axis=1)
        del melspec
        gc.collect()
    else:
        melspec_looped = melspec

    pcen_looped = librosa.pcen(
        melspec_looped,
        sr=sr,
        hop_length=settings["hop_length"],
        gain=settings["pcen_norm_exponent"],
        bias=settings["pcen_delta"],
        power=settings["pcen_power"],
        time_constant=settings["pcen_time_constant"],
    )

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
            clips.append({
                "pcen": clip,
                "start_time": clip_time,
                "end_time": clip_time + clip_length,
                "start_frame": clip_start_frame,
            })

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
    Compute PCEN on long audio with complete clip coverage for inference.
    
    Unlike the training version, this generates clips at regular intervals
    across the entire audio, ignoring artificial segment boundaries.

    When ``executor`` is a process pool and there are multiple segments,
    segments are processed in parallel.
    
    Args:
        audio: Input audio signal
        sr: Original sample rate
        segment_length_seconds: Length of segments for PCEN computation (default from config)
        verbose: If True, print clip-planning and extraction progress
        executor: Optional ProcessPoolExecutor for segment-level parallelism
        num_workers: Max parallel segment jobs when executor is set
    
    Returns:
        clips: List of dictionaries containing PCEN feature arrays for each clip
        sr: Sample rate used for processing
    """
    if segment_length_seconds is None:
        segment_length_seconds = config.PCEN_SEGMENT_LENGTH
    
    settings = get_fft_and_pcen_settings()
    target_sr = settings["sr"]
    
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
    
    total_duration = len(audio) / sr
    clip_length = config.CLIP_LENGTH
    clip_times = _plan_clip_times(total_duration, clip_length)
    
    if verbose:
        print(f"Planning to extract {len(clip_times)} clips from {total_duration:.1f}s audio")

    jobs = list(_iter_pcen_segment_jobs(
        audio, sr, settings, segment_length_seconds, clip_times, clip_length
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
