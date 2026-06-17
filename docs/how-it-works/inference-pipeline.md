# Inference Pipeline

This guide is a concise summary. For the full technical explanation (STFT, PCEN, box-to-time and frequency conversion, and song reconstruction), see `detect-birds-internals.md`.

## Pipeline Steps (Short)

1. Load and normalize audio (WAV/FLAC/OGG/MP3).
2. Resample to target sample rate if needed.
3. Compute PCEN features in memory-friendly segments.
4. Generate 3-second overlapping clips (50% hop).
5. Render clip spectrogram images.
6. Run YOLO inference per clip.
7. Convert detection coordinates to time and frequency.
8. Merge detections with species-aware song reconstruction.
9. Save outputs as JSON and/or CSV.

## Output Notes

Raw detections are typically used for confidence threshold sweeps with `--no-merge`, while reconstructed
detections are more suitable for direct reporting. The CSV output follows annotation-compatible columns.

For implementation details and callable APIs, see:

- `detect-birds-internals.md`
- ../api/inference.md
