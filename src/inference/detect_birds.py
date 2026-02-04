#!/usr/bin/env python3
"""
Detect bird calls in arbitrary-length audio files.

This script loads audio files (WAV, FLAC, OGG, MP3), processes them using the same PCEN 
pipeline as training, and detects bird calls using a trained YOLO model. It returns 
timestamped detections with species labels and confidence scores.

Usage:
    python src/inference/detect_birds.py --audio path/to/audio.wav --model path/to/model.pt
    python src/inference/detect_birds.py --audio audio.flac --model model.pt --output-path results
    python src/inference/detect_birds.py --audio audio.mp3 --model model.pt --conf 0.25 --nms-iou 0.5
"""

import os
import sys
import argparse
import json
import tempfile
import shutil
import threading
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
import librosa.display
from ultralytics import YOLO
from tqdm import tqdm

# Try to import file locking (Unix/Linux)
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False
    # Windows fallback
    try:
        import msvcrt
        HAS_MSVCRT = True
    except ImportError:
        HAS_MSVCRT = False

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

# Import inference-specific PCEN processing
try:
    from inference.utils import pcen_inference
except ImportError:
    # If running as script, try relative import
    from utils import pcen_inference


def reconstruct_songs(detections: List[Dict], song_gap_threshold: float) -> List[Dict]:
    """
    Reconstruct continuous bird songs by merging temporally adjacent detections.
    
    Standalone version usable without a BirdCallDetector instance (e.g. in evaluation
    after filtering raw detections by confidence). Matches app/detect_birds workflow:
    filter by confidence first, then merge.
    
    Args:
        detections: List of raw detections (each with time_start, time_end, species_id,
            species, confidence, freq_low_hz, freq_high_hz; optional 'filename' for multi-file)
        song_gap_threshold: Max gap (seconds) between detections to merge into same song
        
    Returns:
        List of merged song segments (with avg_confidence, max_confidence, detections_merged)
    """
    if len(detections) == 0:
        return []
    
    # Group by (filename, species_id) to avoid merging detections across different files.
    # Each file has its own timeline starting at 0, so merging across files would be wrong.
    groups = {}
    for det in detections:
        filename = det.get('filename', '__single_file__')
        species_id = det['species_id']
        key = (filename, species_id)
        if key not in groups:
            groups[key] = []
        groups[key].append(det)
    
    merged_songs = []
    
    for (filename, species_id), group_detections in groups.items():
        group_detections = sorted(group_detections, key=lambda x: x['time_start'])
        current_song = None
        
        for det in group_detections:
            if current_song is None:
                current_song = {
                    'species': det['species'],
                    'species_id': det['species_id'],
                    'time_start': det['time_start'],
                    'time_end': det['time_end'],
                    'avg_confidence': det['confidence'],
                    'max_confidence': det['confidence'],
                    'detections_merged': 1,
                    'freq_low_hz': det['freq_low_hz'],
                    'freq_high_hz': det['freq_high_hz'],
                }
                if filename != '__single_file__':
                    current_song['filename'] = filename
            else:
                gap = det['time_start'] - current_song['time_end']
                if gap <= song_gap_threshold:
                    current_song['time_end'] = max(current_song['time_end'], det['time_end'])
                    current_song['avg_confidence'] = (current_song['avg_confidence'] * current_song['detections_merged'] + det['confidence']) / (current_song['detections_merged'] + 1)
                    current_song['max_confidence'] = max(current_song['max_confidence'], det['confidence'])
                    current_song['detections_merged'] += 1
                    current_song['freq_low_hz'] = min(current_song['freq_low_hz'], det['freq_low_hz'])
                    current_song['freq_high_hz'] = max(current_song['freq_high_hz'], det['freq_high_hz'])
                else:
                    merged_songs.append(current_song)
                    current_song = {
                        'species': det['species'],
                        'species_id': det['species_id'],
                        'time_start': det['time_start'],
                        'time_end': det['time_end'],
                        'avg_confidence': det['confidence'],
                        'max_confidence': det['confidence'],
                        'detections_merged': 1,
                        'freq_low_hz': det['freq_low_hz'],
                        'freq_high_hz': det['freq_high_hz'],
                    }
                    if filename != '__single_file__':
                        current_song['filename'] = filename
        
        if current_song is not None:
            merged_songs.append(current_song)
    
    return sorted(merged_songs, key=lambda x: (x.get('filename', ''), x['time_start']))


class BirdCallDetector:
    """
    Detector for bird calls in audio files.
    
    This class handles the complete pipeline from audio loading to detection,
    using the same processing approach as training.
    """
    
    # Class-level locks for thread/process-safe YOLO model inference
    # YOLO models are not thread-safe when multiple instances run inference simultaneously
    _inference_lock = threading.Lock()  # For thread-level locking (same process)
    _lock_file_path = None  # For process-level locking (file-based, works across processes)
    
    @classmethod
    def _get_lock_file(cls):
        """Get or create the lock file for process-level locking."""
        if cls._lock_file_path is None:
            # Create lock file in system temp directory
            lock_dir = Path(tempfile.gettempdir())
            cls._lock_file_path = lock_dir / "birdbox_yolo_inference.lock"
            # Create the lock file if it doesn't exist
            cls._lock_file_path.touch(exist_ok=True)
        return cls._lock_file_path
    
    # Frequency range constants (same as in dataset_conversion/get_labels.py)
    MAX_FREQ = 15000  # Hz
    MIN_FREQ = 50     # Hz
    
    def __init__(self, model_path: str, species_mapping: str, conf_threshold: float = 0.001, 
                 nms_iou_threshold: float = 0.5, song_gap_threshold: float = 0.1):
        """
        Initialize the bird call detector.
        
        Args:
            model_path: Path to the trained YOLO model (.pt, .onnx, .engine, etc.)
            species_mapping: Dataset name for species mappings (e.g., 'Hawaii', 'Western-US')
            conf_threshold: Confidence threshold for detections (0-1)
            nms_iou_threshold: IoU threshold for NMS (per-clip and across time windows) (0-1)
            song_gap_threshold: Max gap (seconds) between detections to merge into same song (default: 0.1)
        """
        self.model = YOLO(model_path)
        self.model_path = str(model_path)
        self.conf_threshold = conf_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.song_gap_threshold = song_gap_threshold
        self.settings = pcen_inference.get_fft_and_pcen_settings()
        
        # Load species-specific mappings
        self.species_mapping = species_mapping
        self.species_mappings = config.get_species_mapping(species_mapping)
        self.id_to_ebird = self.species_mappings['id_to_ebird']
        self.bird_colors = self.species_mappings['bird_colors']
        
        # PCEN and spectrogram settings (same as training)
        self.colormap = 'inferno'
        self.vmin = 0.0
        self.vmax = 100.0
        self.clip_length = self.species_mappings['clip_length']  # 3 seconds
        self.clip_hop = self.clip_length / 2  # 1.5 seconds (50% overlap)
        self.height_width = self.species_mappings['height_width']  # 256
        self.pcen_segment_length = self.species_mappings['pcen_segment_length']  # 60
        
        # Precompute mel scale range for frequency conversion
        self.max_mel = librosa.hz_to_mel(self.MAX_FREQ, htk=True)
        self.min_mel = librosa.hz_to_mel(self.MIN_FREQ, htk=True)
        self.mel_range = self.max_mel - self.min_mel
        
        print(f"Loaded model: {model_path}")
        print(f"Species mapping: {self.species_mapping}")
        print(f"Species count: {len(self.id_to_ebird)}")
        print(f"Confidence threshold: {conf_threshold}")
        print(f"NMS IoU threshold: {nms_iou_threshold}")
        print(f"Song gap threshold: {song_gap_threshold}s")
    
    def pixels_to_hz(self, y_pixel: float) -> float:
        """
        Convert y-axis pixel coordinate to frequency in Hz.
        
        This reverses the conversion done in dataset_conversion/get_labels.py (BirdBox-Train repository):
        1. Hz → Mel (HTK) → Normalize [0,1] → Invert y-axis → Pixels [0,256]
        
        Args:
            y_pixel: Y-coordinate in pixels (0-256, where 0 is top/high freq)
            
        Returns:
            Frequency in Hz
        """
        image_height = self.height_width  # 256
        
        # Normalize pixel to [0, 1]
        y_normalized = y_pixel / image_height
        
        # Un-invert y-axis (in get_labels.py: y_center = 1 - y_center)
        # Lower pixel values (top of image) = higher frequencies
        y_normalized = 1.0 - y_normalized
        
        # Convert from normalized [0,1] back to mel scale
        mel_value = y_normalized * self.mel_range + self.min_mel
        
        # Convert mel to Hz using HTK scale (same as training)
        freq_hz = librosa.mel_to_hz(mel_value, htk=True)
        
        # Clip to valid range
        freq_hz = np.clip(freq_hz, self.MIN_FREQ, self.MAX_FREQ)
        
        # Round to integer (same as original annotations)
        return int(round(freq_hz))
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file with automatic fallback for compatibility.
        
        Supports multiple formats: WAV, FLAC, OGG, MP3
        Uses soundfile (fast) with librosa fallback for problematic files.
        
        Note: Model was trained on WAV files. Lossy formats (MP3, OGG) may affect 
        detection performance, especially for faint calls or high frequencies.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            audio: Audio signal as numpy array
            sr: Sample rate
            
        Raises:
            Exception: If file cannot be loaded by any method
        """
        print(f"\nLoading audio: {audio_path}")
        
        # Check for lossy formats and warn user
        audio_path_obj = Path(audio_path)
        lossy_formats = {'.mp3', '.ogg'}
        if audio_path_obj.suffix.lower() in lossy_formats:
            print("⚠️  Warning: Lossy audio format detected (.mp3 or .ogg)")
            print("   Model was trained on lossless WAV files. For best results:")
            print("   - Use WAV or FLAC formats")
            print("   - If using MP3/OGG, ensure high bitrate (≥256 kbps)")
            print("   - Be aware of potential performance degradation for faint/distant calls\n")
        
        # Try soundfile first (faster, preferred method)
        try:
            audio, sr = sf.read(audio_path, dtype='float32')
            loading_method = "soundfile"
        except Exception as sf_error:
            # Soundfile failed - try librosa as fallback
            print(f"⚠️  soundfile failed ({sf_error})")
            print("   Attempting to load with librosa fallback...")
            
            try:
                audio, sr = librosa.load(audio_path, sr=None, mono=False, dtype=np.float32)
                loading_method = "librosa"
                print("✓ Successfully loaded using librosa fallback")
            except Exception as librosa_error:
                error_msg = (
                    f"Failed to load audio file with both methods:\n"
                    f"  - soundfile: {sf_error}\n"
                    f"  - librosa: {librosa_error}\n"
                    f"File may be corrupted or in an unsupported format.\n"
                    f"Try re-encoding: ffmpeg -i {audio_path} -c:a flac output.flac"
                )
                raise Exception(error_msg)
        
        # Convert stereo to mono if needed
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        duration = len(audio) / sr
        print(f"Duration: {duration:.2f} seconds")
        print(f"Sample rate: {sr} Hz")
        if loading_method == "librosa":
            print("(Loaded via librosa fallback)")
        
        return audio, sr
    
    def create_spectrogram_image(self, pcen_data: np.ndarray, output_path: str):
        """
        Create a spectrogram image from PCEN data (same as training).
        
        Args:
            pcen_data: PCEN features
            output_path: Where to save the image
        """
        fig, ax = plt.subplots(figsize=(2.56, 2.56), dpi=100)
        
        librosa.display.specshow(
            pcen_data,
            sr=self.settings["sr"],
            hop_length=self.settings["hop_length"],
            ax=ax,
            cmap=self.colormap,
            vmin=self.vmin,
            vmax=self.vmax,
        )
        
        # Remove all axes, labels, and padding (same as training)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close(fig)
    
    def process_audio_to_clips(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """
        Process audio into PCEN clips with sliding window.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            List of clips with PCEN data and timing information
        """
        print("\nProcessing audio with PCEN...")
        
        # Use inference-specific PCEN processing that handles continuous audio
        # (unlike training which must avoid cross-boundary clips between chunks)
        clips, _ = pcen_inference.compute_pcen_for_inference(
            audio, 
            sr, 
            segment_length_seconds=self.pcen_segment_length
        )
        
        return clips
    
    def detect_in_clip(self, clip_data: Dict, temp_dir: Path) -> List[Dict]:
        """
        Run detection on a single clip.
        
        Args:
            clip_data: Dictionary with 'pcen', 'start_time', 'end_time'
            temp_dir: Temporary directory for spectrogram images
            
        Returns:
            List of detections with timing and species information
        """
        # Create spectrogram image
        temp_image = temp_dir / f"temp_{clip_data['start_time']:.1f}s.png"
        self.create_spectrogram_image(clip_data['pcen'], str(temp_image))
        
        # Run inference with locks to prevent interference between concurrent sessions
        # YOLO models are not thread-safe when multiple instances run inference simultaneously
        # Use both thread lock (for same process) and file lock (for different processes)
        with BirdCallDetector._inference_lock:
            # File-based lock works across processes (e.g., multiple Streamlit workers)
            lock_file_path = BirdCallDetector._get_lock_file()
            lock_file = None
            try:
                lock_file = open(lock_file_path, 'w')
                
                # Try to acquire file lock (works across processes)
                if HAS_FCNTL:
                    # Unix/Linux: use fcntl
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                elif HAS_MSVCRT:
                    # Windows: use msvcrt
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
                
                # Run inference
                results = self.model(
                    str(temp_image),
                    conf=self.conf_threshold,
                    iou=self.nms_iou_threshold,
                    verbose=False
                )[0]
                
            except (IOError, OSError, AttributeError):
                # Fallback: if file locking fails, just use the thread lock
                # (works within same process, which is the common case)
                results = self.model(
                    str(temp_image),
                    conf=self.conf_threshold,
                    iou=self.nms_iou_threshold,
                    verbose=False
                )[0]
            finally:
                # Always release the file lock and close file
                if lock_file is not None:
                    try:
                        if HAS_FCNTL:
                            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                        elif HAS_MSVCRT:
                            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                    except (IOError, OSError, AttributeError):
                        pass  # Ignore unlock errors
                    finally:
                        lock_file.close()
        
        detections = []
        
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                # Extract box information
                xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2] in pixels
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Convert pixel coordinates to time coordinates
                # Image is 256x256 pixels representing 3 seconds of audio
                image_width = 256  # pixels
                clip_duration = self.clip_length  # seconds
                
                # x coordinates represent time in the clip
                x1_pixels, y1_pixels, x2_pixels, y2_pixels = xyxy
                
                # Convert x coordinates from pixels to seconds within the clip
                time_start_in_clip = (x1_pixels / image_width) * clip_duration
                time_end_in_clip = (x2_pixels / image_width) * clip_duration
                
                # Convert to absolute time in the audio file
                abs_time_start = clip_data['start_time'] + time_start_in_clip
                abs_time_end = clip_data['start_time'] + time_end_in_clip
                
                # Get species name
                species = self.id_to_ebird.get(cls, f"unknown_{cls}")
                
                # Convert pixel frequencies to Hz
                # Note: y1 (top of box) = high frequency, y2 (bottom of box) = low frequency
                freq_high_hz = self.pixels_to_hz(y1_pixels)
                freq_low_hz = self.pixels_to_hz(y2_pixels)
                
                detections.append({
                    'species': species,
                    'species_id': cls,
                    'confidence': conf,
                    'time_start': abs_time_start,
                    'time_end': abs_time_end,
                    'freq_low_hz': freq_low_hz,
                    'freq_high_hz': freq_high_hz,
                    'clip_start': clip_data['start_time'],
                    'clip_end': clip_data['end_time'],
                })
        
        # Clean up temp image
        temp_image.unlink(missing_ok=True)
        
        return detections
    
    def merge_overlapping_detections(self, detections: List[Dict], merge_mode: str = 'reconstruct') -> List[Dict]:
        """
        Merge detections using different strategies.
        
        Args:
            detections: List of all detections from all clips
            merge_mode: Strategy for merging
                - 'nms': Traditional NMS to remove duplicates (keeps highest confidence)
                - 'reconstruct': Merge temporally adjacent detections to reconstruct songs
            
        Returns:
            Filtered/merged list of detections
        """
        if len(detections) == 0:
            return []
        
        if merge_mode == 'nms':
            return self._merge_with_nms(detections)
        elif merge_mode == 'reconstruct':
            return self._reconstruct_songs(detections)
        else:
            raise ValueError(f"Unknown merge_mode: {merge_mode}")
    
    def _merge_with_nms(self, detections: List[Dict]) -> List[Dict]:
        """
        Traditional NMS: Remove duplicate detections from overlapping windows.
        Keeps the detection with highest confidence.
        """
        # Sort by confidence (highest first)
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        
        for detection in detections:
            should_keep = True
            
            for kept in keep:
                # Only compare detections of the same species
                if detection['species_id'] != kept['species_id']:
                    continue
                
                # Calculate temporal IoU
                time_overlap_start = max(detection['time_start'], kept['time_start'])
                time_overlap_end = min(detection['time_end'], kept['time_end'])
                
                if time_overlap_start < time_overlap_end:
                    overlap_duration = time_overlap_end - time_overlap_start
                    detection_duration = detection['time_end'] - detection['time_start']
                    kept_duration = kept['time_end'] - kept['time_start']
                    
                    intersection = overlap_duration
                    union = detection_duration + kept_duration - intersection
                    iou = intersection / union if union > 0 else 0
                    
                    if iou > self.nms_iou_threshold:
                        should_keep = False
                        break
            
            if should_keep:
                keep.append(detection)
        
        # Sort by time
        keep = sorted(keep, key=lambda x: x['time_start'])
        return keep
    
    def _reconstruct_songs(self, detections: List[Dict]) -> List[Dict]:
        """
        Reconstruct continuous bird songs by merging temporally adjacent detections.
        Delegates to module-level reconstruct_songs for consistency with evaluation.
        """
        return reconstruct_songs(detections, self.song_gap_threshold)
    
    def detect_multiple_files(self, audio_paths: List[str], output_path: str = None, output_format: str = 'json', no_merge: bool = False) -> List[Dict]:
        """
        Detect bird calls in multiple audio files.
        
        Args:
            audio_paths: List of paths to WAV files
            output_path: Optional base path to save results (without extension)
            output_format: Output format - 'json', 'csv', 'txt', or 'all'
            no_merge: If True, return raw (unmerged) detections; add filename to each for later merge.
            
        Returns:
            List of all detections from all files with timing and species information
        """
        all_detections = []
        
        print(f"\nProcessing {len(audio_paths)} audio files...")
        
        for i, audio_path in enumerate(audio_paths, 1):
            print(f"\n{'='*60}")
            print(f"Processing file {i}/{len(audio_paths)}: {Path(audio_path).name}")
            print(f"{'='*60}")
            
            try:
                # Detect in this file
                file_detections = self.detect_single_file(audio_path, no_merge=no_merge)
                
                # Add filename to each detection (needed for multi-file raw → merge in evaluation)
                filename = Path(audio_path).name
                for detection in file_detections:
                    detection['filename'] = filename
                    detection['file_path'] = str(audio_path)
                
                all_detections.extend(file_detections)
                print(f"Found {len(file_detections)} detections in this file")
                
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                continue
        
        print(f"\n{'='*60}")
        print(f"TOTAL DETECTIONS ACROSS ALL FILES: {len(all_detections)}")
        print(f"{'='*60}")
        
        # Save results if output path is specified
        if output_path and all_detections:
            self.save_results(all_detections, output_path, audio_paths[0] if audio_paths else None, output_format)
        
        return all_detections

    def detect_single_file(self, audio_path: str, progress_callback=None, no_merge: bool = False) -> List[Dict]:
        """
        Detect bird calls in a single audio file (renamed from detect method).
        
        Args:
            audio_path: Path to the WAV file
            progress_callback: Optional callback function(current, total, message) for progress updates
            no_merge: If True, return raw (unmerged) detections for later filter-then-merge (e.g. F-score sweep).
            
        Returns:
            List of detections with timing and species information (merged or raw per no_merge)
        """
        # Load audio
        audio, sr = self.load_audio(audio_path)
        
        # Process to clips
        clips = self.process_audio_to_clips(audio, sr)
        
        # Create temporary directory for spectrogram images
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Run detection on each clip
            print(f"\nRunning detection on {len(clips)} clips...")
            all_detections = []
            
            # Use progress callback if provided, otherwise use tqdm
            if progress_callback:
                for i, clip_data in enumerate(clips):
                    clip_detections = self.detect_in_clip(clip_data, temp_dir)
                    all_detections.extend(clip_detections)
                    progress_callback(i + 1, len(clips), f"Detecting bird calls in {self.clip_length} second clips...")
            else:
                for clip_data in tqdm(clips, desc="Detecting"):
                    clip_detections = self.detect_in_clip(clip_data, temp_dir)
                    all_detections.extend(clip_detections)
            
            print(f"\nFound {len(all_detections)} raw detections")
            
            if no_merge:
                return all_detections
            
            # Merge detections (default: reconstruct songs)
            if progress_callback:
                progress_callback(len(clips), len(clips), "Reconstructing bird songs...")
            print("Reconstructing continuous bird songs from detections...")
            final_detections = self.merge_overlapping_detections(all_detections, merge_mode='reconstruct')
            
            print(f"Final count: {len(final_detections)} song segments")
            
            return final_detections
            
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    def detect(self, audio_path: str, output_path: str = None, output_format: str = 'json', no_merge: bool = False) -> List[Dict]:
        """
        Detect bird calls in an audio file or directory.
        
        Supports WAV, FLAC, OGG, and MP3 formats.
        
        Args:
            audio_path: Path to audio file or directory containing audio files
            output_path: Optional base path to save results (without extension)
            output_format: Output format - 'json', 'csv', 'txt', or 'all'
            no_merge: If True, save raw (unmerged) detections for filter-then-merge workflows (e.g. F-score sweep).
            
        Returns:
            List of detections with timing and species information
        """
        # Find all audio files
        audio_files = find_audio_files(audio_path)
        
        if not audio_files:
            print("No audio files found to process")
            return []
        
        if len(audio_files) == 1:
            # Single file - use original logic
            detections = self.detect_single_file(audio_files[0], no_merge=no_merge)
            if output_path:
                self.save_results(detections, output_path, audio_files[0], output_format)
            return detections
        else:
            # Multiple files - use new batch processing
            return self.detect_multiple_files(audio_files, output_path, output_format, no_merge=no_merge)
    
    def _convert_to_json_serializable(self, obj):
        """
        Convert numpy types and other non-JSON-serializable objects to standard Python types.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        else:
            return obj

    def save_detections(self, detections: List[Dict], output_path: str, audio_path: str = None):
        """
        Save detections to JSON file.
        
        Args:
            detections: List of detections
            output_path: Path to save JSON file
            audio_path: Original audio file path (for metadata, optional for multi-file)
        """
        # Determine if this is multi-file output
        is_multi_file = any('filename' in det for det in detections)
        
        if is_multi_file:
            # Multi-file output
            unique_files = list(set(det.get('file_path', 'unknown') for det in detections))
            output = {
                'audio_files': unique_files,
                'file_count': len(unique_files),
                'model_config': {
                    'model': self.model_path,
                    'confidence_threshold': self.conf_threshold,
                    'nms_iou_threshold': self.nms_iou_threshold,
                    'song_gap_threshold': self.song_gap_threshold,
                    'species_mapping': self.species_mapping,
                },
                'detection_count': len(detections),
                'detections': detections
            }
        else:
            # Single file output
            output = {
                'audio_file': str(audio_path) if audio_path else 'unknown',
                'model_config': {
                    'model': self.model_path,
                    'confidence_threshold': self.conf_threshold,
                    'nms_iou_threshold': self.nms_iou_threshold,
                    'song_gap_threshold': self.song_gap_threshold,
                    'species_mapping': self.species_mapping,
                },
                'detection_count': len(detections),
                'detections': detections
            }
        
        # Convert all numpy types to JSON-serializable types
        output = self._convert_to_json_serializable(output)
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nSaved detections to: {output_path}")
    
    def save_detections_csv(self, detections: List[Dict], output_path: str, audio_path: str = None):
        """
        Save detections to CSV file in the same format as annotations.csv.
        
        Args:
            detections: List of detections
            output_path: Path to save CSV file
            audio_path: Original audio file path (for metadata, optional for multi-file)
        """
        import csv
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header (same as annotations.csv, plus Confidence)
            writer.writerow(['Filename', 'Start Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)', 'Species eBird Code', 'Confidence'])
            
            # Write detection data
            for det in detections:
                # Use filename from detection if available (for multi-file), otherwise use audio_path
                if 'filename' in det:
                    filename = det['filename']
                elif audio_path:
                    filename = Path(audio_path).name
                else:
                    filename = 'unknown'
                
                # Merged detections have avg_confidence; raw have confidence
                confidence = det['avg_confidence'] if 'detections_merged' in det else det['confidence']
                
                writer.writerow([
                    filename,
                    f"{det['time_start']:.1f}",
                    f"{det['time_end']:.1f}",
                    det['freq_low_hz'],
                    det['freq_high_hz'],
                    det['species'],
                    f"{confidence:.3f}"
                ])
        
        print(f"\nSaved detections to CSV: {output_path}")
    
    def save_results(self, detections: List[Dict], output_path: str, audio_path: str = None, output_format: str = 'json'):
        """
        Save detections in the specified format(s).
        
        Args:
            detections: List of detections
            output_path: Base path for output files (without extension)
            audio_path: Original audio file path (for metadata, optional for multi-file)
            output_format: Output format - 'json', 'csv', or 'all'
        """
        output_path_obj = Path(output_path)
        
        if output_format == 'json' or output_format == 'all':
            json_path = str(output_path_obj.with_suffix('.json'))
            self.save_detections(detections, json_path, audio_path)
        
        if output_format == 'csv' or output_format == 'all':
            csv_path = str(output_path_obj.with_suffix('.csv'))
            self.save_detections_csv(detections, csv_path, audio_path)
    
    def print_summary(self, detections: List[Dict]):
        """Print a summary of detections."""
        if len(detections) == 0:
            print("\nNo bird calls detected.")
            return
        
        print(f"\n{'='*80}")
        print("DETECTION SUMMARY")
        print(f"{'='*80}")
        
        # Group by species
        species_counts = {}
        for det in detections:
            species = det['species']
            if species not in species_counts:
                species_counts[species] = []
            species_counts[species].append(det)
        
        print(f"\nTotal detections: {len(detections)}")
        print(f"Species detected: {len(species_counts)}")
        
        # Check if these are reconstructed songs (have 'detections_merged' field)
        is_reconstructed = 'detections_merged' in detections[0] if detections else False
        
        print()
        
        for species, dets in sorted(species_counts.items()):
            print(f"{species}: {len(dets)} {'song segments' if is_reconstructed else 'detections'}")
            
            for det in dets[:5]:  # Show first 5 for each species
                duration = det['time_end'] - det['time_start']
                
                if is_reconstructed:
                    print(f"  {det['time_start']:6.2f}s - {det['time_end']:6.2f}s "
                          f"({duration:5.2f}s duration, "
                          f"{det['detections_merged']:2d} clips merged, "
                          f"avg conf: {det['avg_confidence']:.3f}, "
                          f"max conf: {det['max_confidence']:.3f})")
                else:
                    print(f"  {det['time_start']:6.2f}s - {det['time_end']:6.2f}s "
                          f"(confidence: {det['confidence']:.3f})")
            
            if len(dets) > 5:
                print(f"  ... and {len(dets) - 5} more")
            
            # Print statistics for this species
            if is_reconstructed:
                durations = [d['time_end'] - d['time_start'] for d in dets]
                merged_counts = [d['detections_merged'] for d in dets]
                print(f"  Stats: avg duration {sum(durations)/len(durations):.2f}s, "
                      f"avg clips merged {sum(merged_counts)/len(merged_counts):.1f}")
            
            print()


def find_audio_files(audio_path: str) -> List[str]:
    """
    Find all supported audio files in the given path (file or directory).
    
    Supported formats: WAV, FLAC, OGG, MP3
    
    Args:
        audio_path: Path to a single audio file or directory containing audio files
        
    Returns:
        List of paths to audio files
    """
    # Supported audio extensions (soundfile supports these natively)
    SUPPORTED_EXTENSIONS = {'.wav', '.flac', '.ogg', '.mp3'}
    
    audio_path_obj = Path(audio_path)
    
    if audio_path_obj.is_file():
        # Single file
        if audio_path_obj.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [str(audio_path_obj)]
        else:
            print(f"Warning: {audio_path} is not a supported audio file format. Supported: WAV, FLAC, OGG, MP3")
            return []
    
    elif audio_path_obj.is_dir():
        # Directory - find all supported audio files recursively
        audio_files = []
        for ext in SUPPORTED_EXTENSIONS:
            # Search for lowercase
            for audio_file in audio_path_obj.rglob(f'*{ext}'):
                audio_files.append(str(audio_file))
            # Search for uppercase
            for audio_file in audio_path_obj.rglob(f'*{ext.upper()}'):
                audio_files.append(str(audio_file))
        
        audio_files.sort()  # Sort for consistent ordering
        print(f"Found {len(audio_files)} audio files in directory: {audio_path}")
        return audio_files
    
    else:
        print(f"Error: {audio_path} is neither a file nor a directory")
        return []


def ensure_output_directory(output_path: str) -> bool:
    """
    Ensure the output directory exists, creating it automatically if needed.
    
    Args:
        output_path: The output path (may be a file path)
        
    Returns:
        True if directory exists or was created successfully, False if creation failed
    """
    if not output_path:
        return True  # No output path specified, nothing to check
    
    output_dir = Path(output_path).parent
    
    # If the directory already exists, we're good
    if output_dir.exists():
        return True
    
    # Directory doesn't exist, create it automatically
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created output directory: {output_dir}")
        return True
    except Exception as e:
        print(f"✗ Error creating directory: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Detect bird calls in audio files using trained YOLO model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic detection (single file) - supports WAV, FLAC, OGG, MP3
  python src/inference/detect_birds.py --audio recording.wav --model models/Hawaii.pt --species-mapping Hawaii
  
  # Process entire folder of audio files
  python src/inference/detect_birds.py --audio /path/to/audio/folder --model models/Western-US.pt --species-mapping Western-US --output-path results --output-format all
  
  # Process FLAC file
  python src/inference/detect_birds.py --audio recording.flac --model models/Hawaii.pt --species-mapping Hawaii --output-path results --output-format json
  
  # Save results to CSV
  python src/inference/detect_birds.py --audio recording.mp3 --model models/Hawaii.pt --species-mapping Hawaii --output-path results --output-format csv
  
  # Save all formats
  python src/inference/detect_birds.py --audio recording.ogg --model models/Hawaii.pt --species-mapping Hawaii --output-path results --output-format all
  
  # Adjust thresholds
  python src/inference/detect_birds.py --audio audio.wav --model models/Western-US.pt --species-mapping Western-US --conf 0.5 --nms-iou 0.3
        """
    )
    
    parser.add_argument(
        '--audio',
        type=str,
        required=True,
        help='Path to audio file (WAV/FLAC/OGG/MP3) or directory containing audio files'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to the trained model (.pt, .onnx, .engine, etc.)'
    )
    
    parser.add_argument(
        '--species-mapping',
        type=str,
        required=True,
        choices=['Hawaii', 'Hawaii_subset', 'Northeastern-US', 'Northeastern-US_subset', 
                 'Southern-Sierra-Nevada', 'Western-US'],
        help='Dataset/species mapping used to train the model (REQUIRED)'
    )
    
    parser.add_argument(
        '--output-path',
        type=str,
        default='results/all_detections',
        help='Output directory path for detection results.'
    )
    
    parser.add_argument(
        '--output-format',
        type=str,
        choices=['json', 'csv', 'all'],
        default='json',
        help='Output format: json (default), csv, or all formats'
    )
    
    # the default value should work perfectly
    parser.add_argument(
        '--conf',
        type=float,
        default=0.2,
        help='Confidence threshold for detections'
    )
    
    # this value can be further explored
    parser.add_argument(
        '--nms-iou',
        type=float,
        default=0.5,
        help='NMS IoU threshold for per-clip and cross-window NMS (default: 0.5)'
    )
    
    # this value can be further explored
    parser.add_argument(
        '--song-gap',
        type=float,
        default=0.1,
        help='Max gap (seconds) between detections to merge into same song (default: 0.1)'
    )
    
    parser.add_argument(
        '--no-merge',
        action='store_true',
        help='Output raw (unmerged) detections for filter-then-merge workflows (e.g. F-score sweep). Use low --conf (e.g. 0.001).'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.audio).exists():
        print(f"Error: Audio file or directory not found: {args.audio}", file=sys.stderr)
        sys.exit(1)
    
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}", file=sys.stderr)
        sys.exit(1)
    
    # Handle output path (support both old --output and new --output-path)
    output_path = args.output_path if args.output_path is not None else args.output
    
    # Ensure output directory exists (ask user if it needs to be created)
    if output_path and not ensure_output_directory(output_path):
        sys.exit(1)
    
    # Create detector
    detector = BirdCallDetector(
        model_path=args.model,
        species_mapping=args.species_mapping,
        conf_threshold=args.conf,
        nms_iou_threshold=args.nms_iou,
        song_gap_threshold=args.song_gap
    )
    
    # Run detection
    detections = detector.detect(args.audio, output_path, args.output_format, no_merge=args.no_merge)
    
    # Print summary
    detector.print_summary(detections)


if __name__ == '__main__':
    main()

