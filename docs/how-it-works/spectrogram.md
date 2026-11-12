# Spectrogram Generation

BirdBox converts raw audio into log-floor mel spectrograms before running YOLO detection: log power in decibels minus each frequency band's noise floor.
This page documents the complete preprocessing pipeline.
Following these steps exactly lets you reproduce the same spectrogram representation used during both training and inference.

For the full implementation details, you can have a look at [`pcen_inference.py`](https://github.com/birdnet-team/BirdBox/blob/main/src/inference/utils/pcen_inference.py){ target="_blank" rel="noopener noreferrer" }.

## Pipeline at a Glance

The pipeline runs in six sequential stages:

1. Load and normalize audio to a common amplitude range
2. Resample to 32,000 Hz, pad the end and split the audio into one-minute segments
3. Compute the Short-Time Fourier Transform (STFT)
4. Apply a mel filterbank to compress frequency bins
5. Convert to decibels and subtract each band's noise floor (log-floor)
6. Extract fixed-length clips and render them as 256×256 pixel images

Stages 3 to 6 run once per segment, and segments only bound the memory: a clip's pixels do not depend on where a segment starts. Each step is described in detail below, including all parameter values.

---

## Stage 1: Audio Loading and Normalization

BirdBox reads audio using `soundfile`, with a `librosa` fallback for problematic files. Stereo recordings are mixed down to mono by averaging both channels.

The amplitude is then scaled to the integer range before any spectral processing:

```python
audio = (audio * (2 ** 31)).astype("float32")
```

This maps the normalized float input, typically in [-1.0, 1.0], into the range of a 32-bit signed integer. The scaling is kept from the training pipeline. Log-floor itself does not depend on it: any constant factor cancels when the noise floor is subtracted.

---

## Stage 2: Resampling, End Padding and Segments

All audio is resampled to a fixed sample rate using `librosa.resample`. Files already at the target rate are passed through without modification.

The clips start every 1.5 s. If the file does not end on that grid, or is shorter than 3 s, silence is added at the end until the last clip is complete. Boxes that the model finds in this silence are cut at the real end of the file, and boxes that start after it are dropped.

The spectrogram is not computed for the whole file at once. To bound the memory, the audio is split into one-minute segments, and each clip is taken from the segment in which it starts. A segment's spectrogram reaches back to the start of its first clip's [noise-floor window](#stage-5-log-floor-normalization) (up to 60 s before the segment) and 2 s past its end, so every clip of the segment and its floor window are complete. All segments use one frame grid for the whole file, so a clip's pixels are the same whichever segment computes them. Segments are independent, so for a single file `--num-workers` computes them in parallel (for several files it processes the files in parallel).

| Parameter | Value |
| :--- | :--- |
| Target sample rate | `32,000 Hz` |
| Segment length | `60 s` |
| Audio before a segment | its first clip's floor window, up to `60 s` |
| Audio after a segment | `2 s` |
| End padding | silence up to the end of the last 3 s clip |

---

## Stage 3: Short-Time Fourier Transform

The STFT is computed using `librosa.stft`[[20]](#ref-mcfee) with the following parameters. Note that the window length equals the FFT size to avoid zero-padding artifacts inside the analysis window.

| Parameter | Value | Notes |
| :--- | :--- | :--- |
| `n_fft` | `2048` | FFT size; sets frequency resolution |
| `win_length` | `2048` | Equals `n_fft`; no zero-padding inside window |
| `hop_length` | `375` | Frame step in samples (~11.72 ms per frame) |
| `window` | `"flattop"` | Flat-top window for accurate amplitude representation |
| `center` | `False` | No edge-padding; frames start at sample 0 |

The power spectrogram is derived from the complex STFT output:

```python
stft = librosa.stft(audio, n_fft=2048, win_length=2048,
                    hop_length=375, window="flattop", center=False)
abs2_stft = np.abs(stft) ** 2
```

### Why flat-top?

A flat-top window has a very flat passband in the frequency domain. This minimizes amplitude error for tonal signals, which is important for accurately representing bird vocalizations that may be near bin boundaries. This error between bins is known as scalloping loss [[5]](#ref-harris), and flat-top windows are designed to keep it small [[6]](#ref-heinzel).

---

## Stage 4: Mel Filterbank

The power spectrogram is mapped onto a mel-scale filterbank using `librosa.feature.melspectrogram`. BirdBox uses the HTK mel formula throughout, consistent with both training and inference.

| Parameter | Value | Notes |
| :--- | :--- | :--- |
| `n_mels` | `256` | Number of mel bands (output height) |
| `fmin` | `50 Hz` | Lowest frequency included |
| `fmax` | `15,000 Hz` | Highest frequency included |
| `htk` | `True` | Uses the HTK formula, not the Slaney formula |

```python
melspec = librosa.feature.melspectrogram(
    S=abs2_stft, sr=32000, n_fft=2048,
    n_mels=256, fmin=50, fmax=15000, htk=True
)
```

!!! info "Frequency Range"
    The 50–15,000 Hz range covers the vast majority of bird vocalizations while discarding low-frequency rumble and ultrasonic content above the range of interest. The HTK formula spaces mel bands differently than the Slaney formula. Using the wrong formula at inference will produce mismatched filterbanks and degrade detection performance.

---

## Stage 5: Log-Floor Normalization

The mel power is converted to decibels, and each mel band's noise floor is subtracted: the 20th percentile of that band's levels over the 60 s up to the end of the clip (the rolling floor; fewer seconds at the start of a file). The image therefore shows how far each sound rises above the background of its own band. Values from 3 dB below to 45 dB above the floor fill the colour range (0 to 100 on the image scale); louder sounds take the brightest colour.

```python
db = 10 * np.log10(np.maximum(melspec, 1e-10))     # frames of the file's frame grid
sound = melspec.sum(axis=0) > 0                    # leave out digital silence
# the frames that end by the clip's end and start at most 60 s before it
window = slice(first_frame_of_floor_window, last_frame_of_clip + 1)
floor = np.percentile(db[:, window][:, sound[window]], 20, axis=1, keepdims=True)
features = (db[:, clip_frames] - floor + 3.0) * (100.0 / 48.0)   # -3..45 dB above the floor -> 0..100
```

| Parameter | Value | Role |
| :--- | :--- | :--- |
| Floor percentile | `20` | The background level of each band over the floor window |
| Floor window | the `60 s` up to the clip's end | One floor per band and clip; no audio after the clip enters it |
| Lowest colour | `3 dB` below the floor | Quiet background stays dark but visible |
| Brightest colour | `45 dB` above the floor | Nearly no call reaches it (less than 0.15% of the pixels inside labelled calls) |
| Digital silence | left out of the floor | The padding at the end of a file does not pull the floor down |

**What it gives.** A recorder's gain adds the same number of decibels to every band and cancels when the floor is subtracted, so the same scene gives the same image whatever the recording level. A loud sound barely moves a 20th percentile, so the quiet tail after a loud call stays visible.

**Why a rolling floor.** Nothing after a clip changes its pixels, so a clip's boxes are final as soon as the audio reaches its end. A recording in progress gets the boxes of the saved file while it is still recording (see the [demo](../demo.md)), and the boxes of a clip do not depend on where a minute starts. Until November 2026 the floor was taken over each clip's whole 60 s segment. For Just-Bird, which was trained on that floor, the rolling floor scores level on the validation and hold-out recordings, and slightly better on long files (results in the [BirdBox-Train repository](https://github.com/birdnet-team/BirdBox-Train){ target="_blank" rel="noopener noreferrer" }, `holdout_ssn/RESULTS.md`). Birds-With-Labels was trained on rolling-floor images and scores the same with both floors (see its [Floor](../models-and-metrics/birds-with-labels.md#floor) section).

!!! warning "The segment floor is deprecated"
    `BIRDBOX_FLOOR_WINDOW=segment` still takes the floor over each clip's 60 s segment (with 2 s of context on each side, and the warm-up frames below), for comparisons with earlier results. It prints a deprecation warning and will be removed in the next release. `BIRDBOX_FLOOR_WINDOW=<seconds>` sets another window length; the models were trained with 60.

**Why not PCEN.** BirdBox used Per-Channel Energy Normalization (PCEN) until November 2026. PCEN was introduced for keyword spotting [[29]](#ref-wang); on recordings of natural environments it brings the distribution of magnitudes close to a Gaussian and decorrelates the frequency bands [[16]](#ref-lostanlen2019), and for bird flight calls it reduced overfitting to the time of day [[17]](#ref-lostanlenplos). With BirdBox's settings it kept about a quarter of the recording level in the image, so a model learned the level of its training recordings: with the Southern Sierra Nevada recordings played 12 dB louder, recall dropped from 0.53 to 0.38, and with the Hawai'i recordings 12 dB quieter it made many more faint false alarms. Its 1 s smoother also dimmed the quiet tail after a loud onset. Log-floor matched or beat PCEN on a site the models never saw, and its output does not change with the recording level (results in the [BirdBox-Train repository](https://github.com/birdnet-team/BirdBox-Train){ target="_blank" rel="noopener noreferrer" }, `holdout_ssn/RESULTS.md`).

### Frame Grid

Frame `m` of a file covers the samples from `375 m - 250` to `375 m - 250 + 2048`, and clip `c` (starting at `1.5 c` s) is frames `128 c` to `128 c + 251`. The offset of 250 samples comes from the PCEN pipeline the training images were first made with: it copied the first 0.5 s of the audio in front of it before the STFT and dropped the 42 frames that covered it, so the first kept frame started 250 samples before the file. The samples before the file start are therefore that copy (`audio[15750:16000]`), as in training.

The deprecated segment floor also repeats the first frames of each segment's mel spectrogram (up to 100) in front of it, and they enter its floor percentile. The rolling floor has no such frames.

| Parameter | Value |
| :--- | :--- |
| Repeated frames | `min(100, frames // 4)` |
| `left_pad_length` | `0.5 s` (16,000 samples at 32 kHz) |
| Pad frames dropped afterwards | `42` (16,000 // 375) |

---

## Stage 6: Clip Extraction and Image Rendering

### Clip Parameters

Fixed-length clips are extracted from the log-floor output using a sliding window with 50% overlap. Each clip covers exactly 3 seconds of audio.

| Parameter | Value |
| :--- | :--- |
| Clip length | `3 s` (252 frames) |
| Clip hop | `1.5 s` (50% overlap) |
| Output shape per clip | `256 × 252` (mel bands × time frames) |

The 252-frame clip length follows from the hop length: `252 × 375 / 32,000 ≈ 2.953 s`, which BirdBox rounds to 3 seconds.

### Image Rendering

Each clip is rendered as a 256×256 pixel PNG using matplotlib, with all axes and labels stripped. The image is produced at exactly 100 DPI from a 2.56×2.56-inch figure.

```python
fig = Figure(figsize=(2.56, 2.56), dpi=100)
ax = fig.add_subplot(111)
librosa.display.specshow(
    features, sr=32000, hop_length=375,
    ax=ax, cmap="inferno", vmin=0.0, vmax=100.0
)
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
```

| Parameter | Value |
| :--- | :--- |
| Figure size | `2.56 × 2.56 inches` |
| DPI | `100` (yields 256×256 pixels) |
| Colormap | `inferno` |
| `vmin` | `0.0` |
| `vmax` | `100.0` |

!!! warning "Exact Parameters Required"
    The `vmin` and `vmax` values are baked into the model's training distribution. Using a different colormap or rescaling will shift pixel intensities relative to what the YOLO model was trained on, causing unpredictable detection results.

---

## Complete Pipeline Reference

The table below summarizes every numerical parameter across all stages for quick reference.

| Stage | Parameter | Value |
| :--- | :--- | :--- |
| Normalization | Amplitude scale | `× 2³¹` |
| Resampling | Target sample rate | `32,000 Hz` |
| Segments | Length | `60 s`, with the floor window before and `2 s` after (memory only) |
| Segments | End padding | silence up to the end of the last clip |
| STFT | `n_fft` | `2048` |
| STFT | `win_length` | `2048` |
| STFT | `hop_length` | `375` |
| STFT | `window` | `"flattop"` |
| STFT | `center` | `False` |
| Mel filterbank | `n_mels` | `256` |
| Mel filterbank | `fmin` | `50 Hz` |
| Mel filterbank | `fmax` | `15,000 Hz` |
| Mel filterbank | `htk` | `True` |
| Log-floor | Floor | 20th percentile of each band over the `60 s` up to each clip's end (digital silence left out) |
| Log-floor | Range | `-3` to `45 dB` above the floor → `0` to `100` |
| Log-floor | Frame grid | frame `m` starts at sample `375 m - 250`; before the file, `audio[15750:16000]` |
| Clips | Clip length | `3 s` / `252 frames` |
| Clips | Clip hop | `1.5 s` (50% overlap) |
| Image | Size | `256 × 256 px` |
| Image | Colormap | `inferno` |
| Image | `vmin` / `vmax` | `0.0` / `100.0` |

## References

<a id="ref-harris"></a>
**[5]** Harris, F. J. (1978). "On the use of windows for harmonic analysis with the discrete Fourier transform." *Proceedings of the IEEE*, 66(1):51–83.

<a id="ref-heinzel"></a>
**[6]** Heinzel, G., Rüdiger, A., & Schilling, R. (2002). "Spectrum and spectral density estimation by the Discrete Fourier transform (DFT), including a comprehensive list of window functions and some new flat-top windows." Technical report, Max-Planck-Institut für Gravitationsphysik (Albert-Einstein-Institut), Hannover.

<a id="ref-lostanlen2019"></a>
**[16]** Lostanlen, V., Salamon, J., Cartwright, M., McFee, B., Farnsworth, A., Kelling, S., & Bello, J. P. (2019). "Per-channel energy normalization: Why and how." *IEEE Signal Processing Letters*, 26(1):39–43.

<a id="ref-lostanlenplos"></a>
**[17]** Lostanlen, V., Salamon, J., Farnsworth, A., Kelling, S., & Bello, J. P. (2019). "Robust sound event detection in bioacoustic sensor networks." *PLOS ONE*, 14(10):e0214168.

<a id="ref-mcfee"></a>
**[20]** McFee, B., et al. (2015). "librosa: Audio and Music Signal Analysis in Python." Proceedings of the 14th Python in Science Conference (SciPy).

<a id="ref-wang"></a>
**[29]** Wang, Y., Getreuer, P., Hughes, T., Lyon, R. F., & Saurous, R. A. (2017). "Trainable frontend for robust and far-field keyword spotting." Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 5670–5674.
