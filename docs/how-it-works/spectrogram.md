# Spectrogram Generation

BirdBox converts raw audio into Per-Channel Energy Normalization (PCEN) mel spectrograms before running YOLO detection.
This page documents the complete preprocessing pipeline.
Following these steps exactly lets you reproduce the same spectrogram representation used during both training and inference.

For the full implementation details, you can have a look at [`pcen_inference.py`](https://github.com/birdnet-team/BirdBox/blob/main/src/inference/utils/pcen_inference.py){ target="_blank" rel="noopener noreferrer" }.

## Pipeline at a Glance

The pipeline runs in six sequential stages:

1. Load and normalize audio to a common amplitude range
2. Resample to 32,000 Hz
3. Compute the Short-Time Fourier Transform (STFT)
4. Apply a mel filterbank to compress frequency bins
5. Apply PCEN normalization (with warmup and left-pad strategies)
6. Extract fixed-length clips and render them as 256×256 pixel images

Each step is described in detail below, including all parameter values.

---

## Stage 1: Audio Loading and Normalization

BirdBox reads audio using `soundfile`, with a `librosa` fallback for problematic files. Stereo recordings are mixed down to mono by averaging both channels.

The amplitude is then scaled to the integer range before any spectral processing:

```python
audio = (audio * (2 ** 31)).astype("float32")
```

This maps the normalized float input, typically in [-1.0, 1.0], into the range of a 32-bit signed integer. The scaling is required to match the magnitude assumptions built into the PCEN parameters.

---

## Stage 2: Resampling

All audio is resampled to a fixed sample rate using `librosa.resample`. Files already at the target rate are passed through without modification.

| Parameter | Value |
| :--- | :--- |
| Target sample rate | `32,000 Hz` |

---

## Stage 3: Short-Time Fourier Transform

The STFT is computed using `librosa.stft`[[5]](#ref-mcfee) with the following parameters. Note that the window length equals the FFT size to avoid zero-padding artifacts inside the analysis window.

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

A flat-top window has a very flat passband in the frequency domain. This minimizes amplitude error for tonal signals, which is important for accurately representing bird vocalizations that may be near bin boundaries.

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

## Stage 5: PCEN Normalization

Per-Channel Energy Normalization (PCEN) replaces the simple log-compression step found in traditional mel spectrograms. It applies a time-varying automatic gain control followed by a static compression nonlinearity, making the output more robust to varying background noise levels.

PCEN is computed using `librosa.pcen`:

```python
pcen_output = librosa.pcen(
    melspec, sr=32000, hop_length=375,
    gain=0.75,          # norm_exponent
    bias=1.0,           # delta
    power=0.35,
    time_constant=1.0
)
```

| Parameter | librosa argument | Value | Role |
| :--- | :--- | :--- | :--- |
| Norm exponent | `gain` | `0.75` | Controls strength of the AGC smoothing |
| Delta (bias) | `bias` | `1.0` | Stabilizes division near silence |
| Compression exponent | `power` | `0.35` | Dynamic range compression applied after AGC |
| Time constant | `time_constant` | `1.0` | Smoothing time for the AGC filter (seconds) |
| Hop length | `hop_length` | `375` | Must match the STFT hop to convert time constant correctly |

### Warmup Loop

The PCEN AGC filter requires several frames to reach steady state from a cold start. A warmup sequence is prepended by looping the first frames of the mel spectrogram back onto itself:

```python
loop_length = min(100, melspec.shape[1] // 4)
melspec_looped = np.concatenate([melspec[:, :loop_length], melspec], axis=1)
```

After PCEN is computed on this extended input, the prepended warmup frames are discarded:

```python
pcen_segment = pcen_looped[:, loop_length:]
```

### Left-Pad Strategy

An additional 0.5-second pad is prepended at the segment level before the STFT is computed. This gives the AGC filter extra context for the very start of each segment.

| Parameter | Value |
| :--- | :--- |
| `left_pad_length` | `0.5 s` (16,000 samples at 32 kHz) |
| Pad frames dropped after PCEN | `42` (16,000 // 375) |

The pad frames are dropped from the PCEN output after processing:

```python
pcen_pad_frames = int(0.5 * 32000) // 375  # = 42
pcen_segment = pcen_segment[:, pcen_pad_frames:]
```

---

## Stage 6: Clip Extraction and Image Rendering

### Clip Parameters

Fixed-length clips are extracted from the PCEN output using a sliding window with 50% overlap. Each clip covers exactly 3 seconds of audio.

| Parameter | Value |
| :--- | :--- |
| Clip length | `3 s` (252 frames) |
| Clip hop | `1.5 s` (50% overlap) |
| Output shape per clip | `256 × 252` (mel bands × time frames) |

The 252-frame clip length follows from the hop length: `252 × 375 / 32,000 ≈ 2.953 s`, which BirdBox rounds to 3 seconds.

### Image Rendering

Each PCEN clip is rendered as a 256×256 pixel PNG using matplotlib, with all axes and labels stripped. The image is produced at exactly 100 DPI from a 2.56×2.56-inch figure.

```python
fig = Figure(figsize=(2.56, 2.56), dpi=100)
ax = fig.add_subplot(111)
librosa.display.specshow(
    pcen_data, sr=32000, hop_length=375,
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
| STFT | `n_fft` | `2048` |
| STFT | `win_length` | `2048` |
| STFT | `hop_length` | `375` |
| STFT | `window` | `"flattop"` |
| STFT | `center` | `False` |
| Mel filterbank | `n_mels` | `256` |
| Mel filterbank | `fmin` | `50 Hz` |
| Mel filterbank | `fmax` | `15,000 Hz` |
| Mel filterbank | `htk` | `True` |
| PCEN | `gain` (norm exponent) | `0.75` |
| PCEN | `bias` (delta) | `1.0` |
| PCEN | `power` | `0.35` |
| PCEN | `time_constant` | `1.0 s` |
| PCEN | Left pad | `0.5 s` (42 frames dropped) |
| Clips | Clip length | `3 s` / `252 frames` |
| Clips | Clip hop | `1.5 s` (50% overlap) |
| Image | Size | `256 × 256 px` |
| Image | Colormap | `inferno` |
| Image | `vmin` / `vmax` | `0.0` / `100.0` |

## References

<a id="ref-mcfee"></a>
**[5]** McFee, B., et al. (2015). "librosa: Audio and Music Signal Analysis in Python." Proceedings of the 14th Python in Science Conference (SciPy).
