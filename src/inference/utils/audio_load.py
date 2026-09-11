"""Load audio for inference without pulling in YOLO / PyTorch."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import soundfile as sf


def load_audio_signal(audio_path: str) -> Tuple[np.ndarray, int]:
    """
    Load an audio file as mono float32.

    Tries soundfile first, then librosa. Raises Exception if both fail.
    """
    try:
        audio, sr = sf.read(audio_path, dtype="float32")
    except Exception as sf_error:
        try:
            import librosa

            audio, sr = librosa.load(audio_path, sr=None, mono=False, dtype=np.float32)
        except Exception as librosa_error:
            raise Exception(
                f"Failed to load audio file with both methods:\n"
                f"  - soundfile: {sf_error}\n"
                f"  - librosa: {librosa_error}\n"
                f"File may be corrupted or in an unsupported format.\n"
                f"Try re-encoding: ffmpeg -i {audio_path} -c:a flac output.flac"
            )

    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    return audio, sr
