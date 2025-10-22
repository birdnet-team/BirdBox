# Training vs Inference PCEN Processing

## Key Difference

The PCEN processing behaves differently for training vs inference due to different requirements around segment boundaries.

## Training: Conservative Boundary Handling

### Why
- Audio is pre-chopped into **independent 60-second chunks**
- Each chunk is processed separately
- Chunks represent **real boundaries** (e.g., file_0, file_1, file_2)
- Must avoid generating clips that would span chunk boundaries

### Behavior
```python
# File structure:
HSN_001_0.wav → 0-60s
HSN_001_1.wav → 60-120s

# Clip generation for HSN_001_0:
✅ Clip at 0.0s → 3.0s
✅ Clip at 1.5s → 4.5s
...
✅ Clip at 55.5s → 58.5s
❌ Clip at 57.0s → 60.0s  (would span into next chunk)
❌ Clip at 58.5s → 61.5s  (would span into next chunk)
```

### Result
- Loses ~1 clip per chunk at boundaries
- For 600s file (10 chunks): ~390 clips instead of 399
- **This is intentional** to prevent data leakage!

### Code
Uses: `dataset_conversion/utils/pcen.py::compute_pcen_long_segment()`

## Inference: Complete Coverage

### Why
- Audio is **one continuous file**
- Segments are just for **memory management** (processing efficiency)
- Segment boundaries are **artificial**, not real data boundaries
- Should generate clips at all valid positions

### Behavior
```python
# File structure:
recording.wav → 0-600s (continuous)

# Clip generation (ignoring internal segment boundaries):
✅ Clip at 0.0s → 3.0s
✅ Clip at 1.5s → 4.5s
...
✅ Clip at 57.0s → 60.0s  ✓ (continuous file, no boundary)
✅ Clip at 58.5s → 61.5s  ✓
...
✅ Clip at 595.5s → 598.5s
✅ Clip at 597.0s → 600.0s  (last possible clip)
```

### Result
- Generates clips at all valid positions
- For 600s file: ~399 clips (as expected)
- **Complete coverage** of the audio

### Code
Uses: `inference/utils/pcen_inference.py::compute_pcen_for_inference()`

## Implementation Details

### Training PCEN (`dataset_conversion/utils/pcen.py`)
```python
# Conservative: stops before segment end
while current_time + config.CLIP_LENGTH <= segment_end_time:
    if clip_start_frame + clip_length_frames <= pcen_segment.shape[1]:
        # Only creates clip if entirely within segment
        create_clip()
```

**Result**: Avoids cross-boundary clips (prevents leakage)

### Inference PCEN (`inference/utils/pcen_inference.py`)
```python
# Pre-calculate all clip times for entire audio
clip_times = []
current_time = 0.0
while current_time + config.CLIP_LENGTH <= total_duration:
    clip_times.append(current_time)
    current_time += clip_hop_seconds

# Then extract them, adding padding around segments
for segment in segments:
    # Add 2s padding before/after for clips at segment edges
    padded_segment = audio[segment_start - 2s : segment_end + 2s]
    
    # Extract all clips that fall in this segment
    for clip_time in clip_times:
        if segment_start <= clip_time < segment_end:
            extract_clip(clip_time)
```

**Result**: Generates all planned clips regardless of internal segment boundaries

## Why This Matters

### For Training
Avoiding cross-boundary clips is **essential** because:
- The same temporal region could appear in multiple chunks
- If `file_0` has clip at 58.5-61.5s and `file_1` has clip at 58.5-61.5s (relative), they're duplicates
- Different chunks can be assigned to different splits
- Cross-boundary clips would create **data leakage** between splits

### For Inference  
Complete coverage is **desirable** because:
- We want to detect all bird calls in the audio
- Missing ~2% of potential detection windows is undesirable
- The file is continuous - no risk of duplication
- Segments are purely for memory management

## Summary Table

| Aspect | Training | Inference |
|--------|----------|-----------|
| **Input** | Pre-chopped 60s chunks | Continuous long file |
| **Boundaries** | Real (between files) | Artificial (memory mgmt) |
| **Clip generation** | Conservative | Complete |
| **Clips for 600s** | ~390 | ~399 |
| **Missing clips** | ~9 at boundaries | 0 |
| **Why** | Prevent leakage | Complete coverage |
| **Code** | `pcen.compute_pcen_long_segment()` | `utils.pcen_inference.compute_pcen_for_inference()` |

## Recommendation

✅ **Keep both implementations**
- Training: Needs conservative boundary handling for data integrity
- Inference: Needs complete coverage for best detection performance
- They serve different purposes and both are correct for their use case

