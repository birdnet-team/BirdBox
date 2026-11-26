# SPDX-FileCopyrightText: 2026 BirdNET-Team
# SPDX-License-Identifier: MIT
"""
Make the example image of the documentation's home page (docs/img/example_viz.png).

It shows 15 s of tests/test.wav as Just-Bird sees it, the log-floor mel spectrogram
from 50 Hz to 15 kHz, with Just-Bird's boxes at the CLI's default settings and their
confidence. Frame j of the picture is drawn as the first window that contains it sees
it: window c = max(0, ceil((j - 251) / 128)), with that window's rolling noise floor,
as in the demo's log-floor view.

Run it from the repository root with the Just-Bird weights:

    python scripts/make_example_viz.py models/Just-Bird_fp32.pt

Options: --from=195 and --to=210 (seconds), --out=docs/img/example_viz.png.
The PNG is reduced to a 256-colour palette, which keeps it near 1.3 MB.
"""

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import librosa
import matplotlib
import numpy as np
import soundfile as sf
from PIL import Image

matplotlib.use("Agg")
from matplotlib import patheffects  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from inference.utils.log_floor import (  # noqa: E402
    compute_features_for_inference,
    get_spectrogram_settings,
)

AUDIO = ROOT / "tests" / "test.wav"
CLIP_FRAMES, CLIP_HOP_FRAMES = 252, 128   # a 3 s window, every 1.5 s
BOX = "#4fc3f7"
LABEL = "#0277bd"
AXIS, SHADE = "#fcffa4", "#000004"   # inferno's ends
AXIS_SIZE, TIME_SIZE = 9, 10.5


def options(argv):
    opts = {"from": 195.0, "to": 210.0, "out": str(ROOT / "docs" / "img" / "example_viz.png")}
    model = None
    for arg in argv:
        if arg.startswith("--"):
            key, _, value = arg[2:].partition("=")
            if key not in opts:
                sys.exit(f"Unknown option --{key}")
            opts[key] = value if key == "out" else float(value)
        else:
            model = arg
    if model is None:
        sys.exit(__doc__)
    return model, opts


def detections(model):
    """Just-Bird's boxes in tests/test.wav, from the CLI at its default settings."""
    with tempfile.TemporaryDirectory() as out:
        subprocess.run(
            [sys.executable, str(ROOT / "src" / "inference" / "detect_birds.py"),
             "--audio", str(AUDIO), "--model", model, "--species-mapping", "Just-Bird",
             "--output-path", out],
            check=True, stdout=subprocess.DEVNULL,
        )
        result = json.loads((Path(out) / "with_algorithm_metadata.json").read_text())
    return result["detections"], result["model_config"]


def picture(t0, t1):
    """The log-floor features (0..100) of the frames from t0 to t1, as the model sees them."""
    settings = get_spectrogram_settings()
    audio, sr = sf.read(str(AUDIO), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    clips, sr = compute_features_for_inference(audio, sr, verbose=False)
    hop = settings["hop_length"]
    by_start = {int(round(c["start_time"] * sr)) // hop: c["features"] for c in clips}
    last = max(by_start) + CLIP_FRAMES
    f0, f1 = int(round(t0 * sr / hop)), min(int(round(t1 * sr / hop)), last)
    img = np.empty((settings["n_mels"], f1 - f0), np.float32)
    for j in range(f0, f1):
        c = max(0, math.ceil((j - (CLIP_FRAMES - 1)) / CLIP_HOP_FRAMES))
        img[:, j - f0] = by_start[CLIP_HOP_FRAMES * c][:, j - CLIP_HOP_FRAMES * c]
    return img, (f1 - f0) * hop / sr, settings


def main():
    model, opts = options(sys.argv[1:])
    t0, t1 = opts["from"], opts["to"]
    dets, config = detections(model)
    img, dur, settings = picture(t0, t1)
    t1 = t0 + dur
    rows = img.shape[0]
    mel_lo = librosa.hz_to_mel(settings["fmin"], htk=True)
    mel_hi = librosa.hz_to_mel(settings["fmax"], htk=True)

    def row(hz):
        return (librosa.hz_to_mel(hz, htk=True) - mel_lo) / (mel_hi - mel_lo) * rows

    fig = plt.figure(figsize=(10, 3.7), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, origin="lower", aspect="auto", cmap="inferno", vmin=0, vmax=100,
              extent=(t0, t1, 0, rows))

    shown = [d for d in dets if d["time_end"] > t0 and d["time_start"] < t1]
    for d in shown:
        y0, y1 = row(d["freq_low_hz"]), row(d["freq_high_hz"])
        ax.add_patch(plt.Rectangle((d["time_start"], y0), d["time_end"] - d["time_start"], y1 - y0,
                                   fill=False, ec=BOX, lw=1.2))
        ax.text(d["time_start"], y1 + 2, f"{d['max_confidence']:.2f}", color="white", fontsize=6.5,
                va="bottom", ha="left", bbox=dict(boxstyle="square,pad=0.15", fc=LABEL, ec="none"))

    # Axes on the picture in the colormap's own colours: its lightest yellow, outlined in its
    # near-black so the labels also read on the bright low frequencies
    halo = [patheffects.withStroke(linewidth=2.4, foreground=SHADE, alpha=0.8)]
    text = dict(color=AXIS, fontsize=AXIS_SIZE, path_effects=halo)
    tick = dict(color=AXIS, lw=1, solid_capstyle="butt", path_effects=halo)
    for khz in (1, 2, 5, 10):
        y = row(khz * 1000)
        ax.plot([t0, t0 + 0.15], [y, y], **tick)
        ax.text(t0 + 0.22, y, f"{khz} kHz", va="center", ha="left", **text)
    # The time sits on the brightest part of the picture: a little larger, with a denser outline
    time_text = dict(text, fontsize=TIME_SIZE,
                     path_effects=[patheffects.withStroke(linewidth=3, foreground=SHADE, alpha=0.9)])
    for s in range(math.ceil(t0), math.floor(t1) + 1):
        ax.plot([s, s], [0, 6 if s % 5 == 0 else 3.5], **tick)
        if s % 5 == 0:
            x, ha = (s + 0.08, "left") if s == t0 else (s - 0.08, "right") if s == t1 else (s, "center")
            ax.text(x, 8, f"{s} s", ha=ha, va="bottom", **time_text)

    ax.set_xlim(t0, t1)
    ax.set_ylim(0, rows)
    ax.axis("off")
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        fig.savefig(tmp.name)
        rgb = Image.open(tmp.name).convert("RGB")
    rgb.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(
        opts["out"], optimize=True)
    print(f"{opts['out']}: {t0:g}-{t1:g} s, {len(shown)} boxes, model settings {json.dumps(config)}")


if __name__ == "__main__":
    main()
