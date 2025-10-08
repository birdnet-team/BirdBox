# Bird Call Detection Inference

This module provides tools to detect bird calls in arbitrary-length audio files using your trained YOLO model.

**Key Feature:** Automatically reconstructs continuous bird songs by merging temporally adjacent detections.

## Quick Start

### Command Line Usage

```bash
# Basic detection (automatically merges detections into songs)
python inference/detect_birds.py \
    --audio path/to/recording.wav \
    --model training_and_validation/final_models/Southern-Sierra-Nevada.pt

# Save results to JSON and CSV
python inference/detect_birds.py \
    --audio recording.wav \
    --model model.pt \
    --output detections.json

# Adjust detection and merging thresholds
python inference/detect_birds.py \
    --audio recording.wav \
    --model model.pt \
    --conf 0.5 \
    --iou 0.3 \
    --song-gap 3.0
```

### Python Library Usage

```python
from inference.detect_birds import BirdCallDetector

# Initialize detector
detector = BirdCallDetector(
    model_path="path/to/model.pt",
    conf_threshold=0.25,  # Minimum confidence for detections
    iou_threshold=0.5     # IoU threshold for merging overlaps
)

# Detect bird calls
detections = detector.detect("audio.wav", output_path="results.json")

# Print summary
detector.print_summary(detections)

# Process results
for det in detections:
    print(f"{det['species']}: {det['time_start']:.2f}s - {det['time_end']:.2f}s "
          f"(confidence: {det['confidence']:.3f})")
```

## How It Works

### Processing Pipeline

1. **Audio Loading**: Loads WAV file and converts to mono if needed
2. **Audio Resampling**: Resamples to target sample rate (typically 32kHz) if needed
3. **PCEN Processing**: 
   - Processes audio in segments for memory efficiency
   - Applies STFT → Mel spectrogram → PCEN normalization (same as training)
   - Uses inference-optimized processing for complete coverage
4. **Clip Extraction**: Creates 3-second clips with 1.5s hop (50% overlap) across entire audio
5. **Spectrogram Generation**: Converts each PCEN clip to a 256x256 spectrogram image
6. **YOLO Inference**: Runs detection on each spectrogram image
7. **Coordinate Conversion**: Converts pixel coordinates to absolute time coordinates
8. **Song Reconstruction**: Merges temporally adjacent detections of same species into continuous songs
9. **Results Output**: Returns reconstructed bird songs with full durations and metadata

### Song Reconstruction

The detector automatically reconstructs continuous bird songs from individual clip detections:

```
Raw detections from overlapping clips:
  71.80s - 72.11s  (clip 1)
  72.50s - 73.20s  (clip 2) 
  73.50s - 75.24s  (clip 3)

Reconstructed song:
  71.80s - 75.24s  (3.44s duration, 3 clips merged)
```

The Low Freq is the minimum of all merged frequencies and the High Freq is the maximum. 

**How it works:**
- Groups detections by species
- Sorts by time
- Merges detections with gaps ≤ `song_gap_threshold` (default: 2 seconds)
- Tracks average and max confidence across merged detections
- Handles songs lasting 20+ seconds!

### Key Features

- ✅ **Handles arbitrary-length audio** (from seconds to hours)
- ✅ **Uses same preprocessing as training** (PCEN, spectrograms)
- ✅ **Sliding window with overlap** to avoid missing calls at boundaries
- ✅ **Automatic duplicate removal** via NMS across time windows
- ✅ **Multiple output formats** (JSON, CSV, console summary, programmatic access)
- ✅ **Model-agnostic** (works with .pt, .onnx, .engine, etc.)

## Detection Output Format

Each detection contains:

```python
{
    'species': 'amerob',              # eBird species code
    'species_id': 2,                  # Numeric class ID
    'confidence': 0.87,               # Detection confidence (0-1)
    'time_start': 12.5,              # Start time in seconds
    'time_end': 14.2,                # End time in seconds
    'freq_low_hz': 2151,             # Low frequency in Hz (50-15000 Hz, integer)
    'freq_high_hz': 5820,            # High frequency in Hz (50-15000 Hz, integer)
    'clip_start': 12.0,              # Which clip this came from
    'clip_end': 15.0
}
```

### JSON Output Example

```json
{
  "audio_file": "recording.wav",
  "model_config": {
    "confidence_threshold": 0.25,
    "iou_threshold": 0.5,
    "dataset": "Southern-Sierra-Nevada"
  },
  "detection_count": 15,
  "detections": [
    {
      "species": "amerob",
      "species_id": 2,
      "confidence": 0.87,
      "time_start": 12.5,
      "time_end": 14.2,
      "freq_low_hz": 2151,
      "freq_high_hz": 5820
    }
  ]
}
```

### CSV Output Format

When you specify an output path, the detector automatically saves both JSON and CSV files:

- **JSON file**: `detections.json` - Full format with all metadata
- **CSV file**: `detections.csv` - Annotations format (matches your training data)

The CSV format matches your `annotations.csv` structure:

```csv
Filename,Start Time (s),End Time (s),Low Freq (Hz),High Freq (Hz),Species eBird Code
recording.wav,12.5,14.2,2151,5820,amerob
recording.wav,25.3,27.8,1890,4560,herthr
```

**CSV fields:**
- `Filename`: Audio file name
- `Start Time (s)`: Detection start time (1 decimal place)
- `End Time (s)`: Detection end time (1 decimal place)  
- `Low Freq (Hz)`: Low frequency (integer)
- `High Freq (Hz)`: High frequency (integer)
- `Species eBird Code`: Species identifier

## Parameters

### Confidence Threshold (`--conf`)
- **Range**: 0.0 - 1.0
- **Default**: 0.25
- **Higher** = fewer but more confident detections
- **Lower** = more detections but may include false positives

### IoU Threshold (`--iou`)
- **Range**: 0.0 - 1.0  
- **Default**: 0.5
- **Higher** = keeps more overlapping detections
- **Lower** = more aggressive duplicate removal
- **Note**: Mainly affects internal NMS, song reconstruction happens separately

### Song Gap Threshold (`--song-gap`)
- **Range**: 0.0 - 10.0 (seconds)
- **Default**: 2.0
- **Higher** = merges detections with larger gaps (more aggressive song reconstruction)
- **Lower** = only merges very close detections (more conservative)
- **Example**: 
  - `--song-gap 1.0` = Only merge detections <1s apart (shorter songs)
  - `--song-gap 5.0` = Merge detections <5s apart (longer songs, may over-merge)

## Advanced Examples

### Filtering by Confidence

```python
detector = BirdCallDetector(model_path="model.pt")
detections = detector.detect("audio.wav")

# Only keep high-confidence detections
high_conf = [d for d in detections if d['confidence'] > 0.8]
```

### Filtering by Species

```python
# Only American Robin
robins = [d for d in detections if d['species'] == 'amerob']

# Multiple species
target_species = ['amerob', 'herthr', 'yelwar']
filtered = [d for d in detections if d['species'] in target_species]
```

### Filtering by Time Range

```python
# First 5 minutes
early_detections = [d for d in detections if d['time_start'] < 300]

# Between 2-5 minutes
middle = [d for d in detections 
          if 120 <= d['time_start'] <= 300]
```

### Batch Processing

```python
from pathlib import Path

detector = BirdCallDetector(model_path="model.pt")
audio_dir = Path("recordings")

for audio_file in audio_dir.glob("*.wav"):
    print(f"Processing {audio_file.name}...")
    detections = detector.detect(
        str(audio_file),
        output_path=f"results/{audio_file.stem}.json"
    )
    print(f"  Found {len(detections)} birds\n")
```

### Export to Custom Format

```python
import csv

detections = detector.detect("audio.wav")

# Export to CSV
with open('detections.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['time_start', 'time_end', 'species', 'confidence'])
    writer.writeheader()
    writer.writerows(detections)

# Export to Audacity labels
with open('labels.txt', 'w') as f:
    for det in detections:
        f.write(f"{det['time_start']}\t{det['time_end']}\t{det['species']}\n")
```

## Performance Tips

1. **GPU Acceleration**: The detector automatically uses GPU if available
2. **Model Format**: Use TensorRT (`.engine`) for fastest inference on NVIDIA GPUs
3. **Confidence Threshold**: Higher thresholds run faster (fewer detections to process)
4. **Batch Size**: For very long files, consider processing in chunks

## Troubleshooting

### Out of Memory
- Process shorter audio segments
- Use a smaller model (e.g., yolo11n instead of yolo11l)
- Reduce PCEN segment length in config

### Missing Detections
- Lower the confidence threshold (`--conf 0.1`)
- Check if the audio matches your training data characteristics

### Too Many False Positives
- Increase confidence threshold (`--conf 0.5`)
- Ensure model was trained on similar audio conditions

## Integration with Your Pipeline

The detector uses the same configuration as your training pipeline:

- **PCEN settings**: From `dataset_conversion/utils/pcen.py`
- **Species mapping**: From `config.ID_TO_EBIRD_CODES`
- **Image size**: From `config.HEIGHT_AND_WIDTH_IN_PIXELS`
- **Clip length**: From `config.CLIP_LENGTH`

This ensures consistency between training and inference.

