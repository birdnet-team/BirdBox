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

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from dataset_conversion.utils.pcen import get_fft_and_pcen_settings


def compute_pcen_for_inference(audio, sr, segment_length_seconds=None):
    """
    Compute PCEN on long audio with complete clip coverage for inference.
    
    Unlike the training version, this generates clips at regular intervals
    across the entire audio, ignoring artificial segment boundaries.
    
    Args:
        audio: Input audio signal
        sr: Original sample rate
        segment_length_seconds: Length of segments for PCEN computation (default from config)
    
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
        print(f"Resampled audio to {sr} Hz")
    
    # Calculate total duration
    total_duration = len(audio) / sr
    
    # Pre-calculate all clip times we need
    clip_hop_seconds = config.CLIP_LENGTH / 2  # 1.5 seconds
    clip_times = []
    current_time = 0.0
    
    while current_time + config.CLIP_LENGTH <= total_duration:
        clip_times.append(current_time)
        current_time += clip_hop_seconds
    
    print(f"Planning to extract {len(clip_times)} clips from {total_duration:.1f}s audio")
    
    # Calculate segment length in samples
    segment_samples = int(segment_length_seconds * sr)
    
    # Process audio in segments for memory efficiency
    clips = []
    segment_start_sample = 0
    
    while segment_start_sample < len(audio):
        segment_end_sample = min(segment_start_sample + segment_samples, len(audio))
        
        # Add padding for context (need some audio before/after for clips at edges)
        pad_before = int(2 * sr)  # 2 seconds before
        pad_after = int(2 * sr)   # 2 seconds after
        
        padded_start = max(0, segment_start_sample - pad_before)
        padded_end = min(len(audio), segment_end_sample + pad_after)
        
        segment_audio = audio[padded_start:padded_end]
        
        # Skip segments that are too short
        if len(segment_audio) < 2 * settings["n_fft"]:
            print(f"Warning: Skipping segment - too short")
            break
        
        # Calculate time range for this segment
        segment_start_time = segment_start_sample / sr
        segment_end_time = segment_end_sample / sr
        
        # Pre-pad for PCEN
        pcen_pad_len = int(settings["left_pad_length"] * sr)
        segment_with_pcen_pad = np.concatenate([segment_audio[:pcen_pad_len], segment_audio])
        
        # Compute STFT
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
        
        # Compute mel spectrogram
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
        
        # Loop for PCEN warmup
        loop_length = min(100, melspec.shape[1] // 4)
        if loop_length > 0:
            melspec_looped = np.concatenate([melspec[:, :loop_length], melspec], axis=1)
            del melspec
            gc.collect()
        else:
            melspec_looped = melspec
        
        # Compute PCEN
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
        
        # Extract original segment
        pcen_segment = pcen_looped[:, loop_length:] if loop_length > 0 else pcen_looped
        del pcen_looped
        gc.collect()
        
        # Drop PCEN pad frames
        pcen_pad_frames = pcen_pad_len // settings["hop_length"]
        pcen_segment = pcen_segment[:, pcen_pad_frames:].astype("float32")
        
        # Account for the extra padding we added
        pad_before_frames = (segment_start_sample - padded_start) // settings["hop_length"]
        
        # Extract clips that fall within this segment's time range
        clip_length_frames = 252  # Standard for 3s clips
        
        for clip_time in clip_times:
            # Check if this clip falls within this segment
            if segment_start_time <= clip_time < segment_end_time:
                # Calculate frame position relative to padded segment
                time_in_padded_segment = clip_time - (padded_start / sr)
                clip_start_frame = int(time_in_padded_segment * sr / settings["hop_length"])
                
                # Check if we have enough frames
                if clip_start_frame >= 0 and clip_start_frame + clip_length_frames <= pcen_segment.shape[1]:
                    clip = pcen_segment[:, clip_start_frame:clip_start_frame + clip_length_frames]
                    
                    clips.append({
                        'pcen': clip,
                        'start_time': clip_time,
                        'end_time': clip_time + config.CLIP_LENGTH,
                        'start_frame': clip_start_frame
                    })
        
        # Move to next segment
        segment_start_sample = segment_end_sample
        
        # Clean up
        del segment_audio
        del segment_with_pcen_pad
        del pcen_segment
        gc.collect()
    
    print(f"Successfully extracted {len(clips)} clips")
    return clips, sr

