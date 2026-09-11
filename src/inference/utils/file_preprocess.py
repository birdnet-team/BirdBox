"""
Per-file CPU preprocess for inference: load audio, PCEN, render clip images.

Torch-free so spawn workers do not initialize CUDA.
"""

from __future__ import annotations

from typing import Dict, List

try:
    from inference.utils.audio_load import load_audio_signal
    from inference.utils.pcen_inference import compute_pcen_for_inference
    from inference.utils.spectrogram_render import render_spectrogram_image
    from inference.utils.thread_limits import apply_worker_thread_limits
except ImportError:
    from utils.audio_load import load_audio_signal
    from utils.pcen_inference import compute_pcen_for_inference
    from utils.spectrogram_render import render_spectrogram_image
    from utils.thread_limits import apply_worker_thread_limits


def preprocess_audio_file_worker(job: Dict) -> Dict:
    """
    Process one audio file into spectrogram PNGs.

    ``job`` keys: audio_path, temp_dir, pcen_segment_length, sr, hop_length,
    colormap, vmin, vmax.
    """
    apply_worker_thread_limits(1)
    audio_path = job["audio_path"]
    try:
        audio, sr = load_audio_signal(audio_path)
        clips, _ = compute_pcen_for_inference(
            audio,
            sr,
            segment_length_seconds=job["pcen_segment_length"],
            verbose=False,
        )
        rendered: List[Dict] = []
        for index, clip in enumerate(clips):
            image_path = (
                f"{job['temp_dir']}/clip_{index:05d}_{clip['start_time']:.3f}s.png"
            )
            render_spectrogram_image(
                clip["pcen"],
                image_path,
                sr=job["sr"],
                hop_length=job["hop_length"],
                colormap=job["colormap"],
                vmin=job["vmin"],
                vmax=job["vmax"],
            )
            rendered.append({
                "start_time": clip["start_time"],
                "end_time": clip["end_time"],
                "image_path": image_path,
            })
        return {"audio_path": audio_path, "clips": rendered, "error": None}
    except Exception as exc:
        return {"audio_path": audio_path, "clips": [], "error": str(exc)}
