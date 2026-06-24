```
audio → preprocessing → |STFT|² → mel filterbank → PCEN → PNG
```

### Audio preprocessing

1. **Validation:** `librosa.util.valid_audio(audio)`
2. **Amplitude scaling:** `audio = (audio * 2**31).astype("float32")` — maps float audio to the range `[-2³¹, 2³¹)`
3. **Resampling:** to `32000 Hz` if the input sample rate differs
4. **Left pre-padding:** concatenate the first `int(left_pad_length * sr)` samples (0.5 s) in front of the signal to reduce edge effects

| Parameter | Value |
|-----------|-------|
| Target sample rate | `32000` Hz |
| Left pre-padding | `0.5` s |

### STFT

| Parameter | Value |
|-----------|-------|
| `n_fft` | `2048` |
| `win_length` | `2048` (matches `n_fft` to avoid zero-padding artifacts) |
| `hop_length` | `375` |
| `window` | `"flattop"` |
| `center` | `False` |

### Mel filterbank

| Parameter | Value |
|-----------|-------|
| `S` | Squared STFT magnitude (`abs2_stft`) |
| `sr` | `32000` |
| `n_fft` | `2048` (must match STFT) |
| `n_mels` | `256` |
| `fmin` | `50` Hz |
| `fmax` | `15000` Hz |
| `htk` | `True` |

### PCEN Settings

| librosa argument | Source key | Value |
|------------------|------------|-------|
| `S` | mel power spectrogram | — |
| `sr` | `sr` | `32000` |
| `hop_length` | `hop_length` | `375` |
| `gain` | `pcen_norm_exponent` | `0.75` |
| `bias` | `pcen_delta` | `1.0` |
| `power` | `pcen_power` | `0.35` |
| `time_constant` | `pcen_time_constant` | `1.0` |
| `eps` | `1e-6` |
| `max_size` | `2048` |
| `axis` | `-1` (time axis) |
