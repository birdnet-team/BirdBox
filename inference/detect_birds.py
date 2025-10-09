#!/usr/bin/env python3
"""
Detect bird calls in arbitrary-length audio files.

This script loads a WAV file, processes it using the same PCEN pipeline as training,
and detects bird calls using a trained YOLO model. It returns timestamped detections
with species labels and confidence scores.

Usage:
    python inference/detect_birds.py --audio path/to/audio.wav --model path/to/model.pt
    python inference/detect_birds.py --audio audio.wav --model model.pt --output detections.json
    python inference/detect_birds.py --audio audio.wav --model model.pt --conf 0.25 --iou 0.5
"""

import os
import sys
import argparse
import json
import tempfile
import shutil
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

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from dataset_conversion.utils import pcen

# Import inference-specific PCEN processing
try:
    from inference import pcen_inference
except ImportError:
    # If running as script, try relative import
    import pcen_inference


class BirdCallDetector:
    """
    Detector for bird calls in audio files.
    
    This class handles the complete pipeline from audio loading to detection,
    using the same processing approach as training.
    """
    
    # Frequency range constants (same as in dataset_conversion/get_labels.py)
    MAX_FREQ = 15000  # Hz
    MIN_FREQ = 50     # Hz
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.5, song_gap_threshold: float = 2.0):
        """
        Initialize the bird call detector.
        
        Args:
            model_path: Path to the trained YOLO model (.pt, .onnx, .engine, etc.)
            conf_threshold: Confidence threshold for detections (0-1)
            iou_threshold: IoU threshold for NMS across time windows (0-1)
            song_gap_threshold: Max gap (seconds) between detections to merge into same song (default: 2.0)
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.song_gap_threshold = song_gap_threshold
        self.settings = pcen.get_fft_and_pcen_settings()
        
        # PCEN and spectrogram settings (same as training)
        self.colormap = 'inferno'
        self.vmin = 0.0
        self.vmax = 100.0
        self.clip_length = config.CLIP_LENGTH  # 3 seconds
        self.clip_hop = config.CLIP_LENGTH / 2  # 1.5 seconds (50% overlap)
        
        # Precompute mel scale range for frequency conversion
        self.max_mel = librosa.hz_to_mel(self.MAX_FREQ, htk=True)
        self.min_mel = librosa.hz_to_mel(self.MIN_FREQ, htk=True)
        self.mel_range = self.max_mel - self.min_mel
        
        print(f"Loaded model: {model_path}")
        print(f"Using config: {config.DATA_CONFIG_YAML}")
        print(f"Dataset: {config.DATASET_NAME}")
        print(f"Confidence threshold: {conf_threshold}")
        print(f"IoU threshold: {iou_threshold}")
        print(f"Song gap threshold: {song_gap_threshold}s")
    
    def pixels_to_hz(self, y_pixel: float) -> float:
        """
        Convert y-axis pixel coordinate to frequency in Hz.
        
        This reverses the conversion done in dataset_conversion/get_labels.py:
        1. Hz → Mel (HTK) → Normalize [0,1] → Invert y-axis → Pixels [0,256]
        
        Args:
            y_pixel: Y-coordinate in pixels (0-256, where 0 is top/high freq)
            
        Returns:
            Frequency in Hz
        """
        image_height = config.HEIGHT_AND_WIDTH_IN_PIXELS  # 256
        
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
        Load audio file.
        
        Args:
            audio_path: Path to the WAV file
            
        Returns:
            audio: Audio signal as numpy array
            sr: Sample rate
        """
        print(f"\nLoading audio: {audio_path}")
        audio, sr = sf.read(audio_path, dtype='float32')
        
        # Convert stereo to mono if needed
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        duration = len(audio) / sr
        print(f"Duration: {duration:.2f} seconds")
        print(f"Sample rate: {sr} Hz")
        
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
            segment_length_seconds=config.PCEN_SEGMENT_LENGTH
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
        
        # Run inference
        results = self.model(
            str(temp_image),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )[0]
        
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
                species = config.ID_TO_EBIRD_CODES.get(cls, f"unknown_{cls}")
                
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
                    
                    if iou > self.iou_threshold:
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
        
        Uses self.song_gap_threshold to determine when detections are part of same song.
        
        Args:
            detections: List of all detections
            
        Returns:
            List of merged song segments
        """
        if len(detections) == 0:
            return []
        
        # Group by species
        species_groups = {}
        for det in detections:
            species_id = det['species_id']
            if species_id not in species_groups:
                species_groups[species_id] = []
            species_groups[species_id].append(det)
        
        # Merge within each species
        merged_songs = []
        
        for species_id, species_detections in species_groups.items():
            # Sort by start time
            species_detections = sorted(species_detections, key=lambda x: x['time_start'])
            
            # Merge consecutive detections
            current_song = None
            
            for det in species_detections:
                if current_song is None:
                    # Start a new song
                    current_song = {
                        'species': det['species'],
                        'species_id': det['species_id'],
                        'time_start': det['time_start'],
                        'time_end': det['time_end'],
                        'confidence': det['confidence'],
                        'max_confidence': det['confidence'],
                        'detections_merged': 1,
                        'freq_low_hz': det['freq_low_hz'],
                        'freq_high_hz': det['freq_high_hz'],
                    }
                else:
                    # Check if this detection is close enough to merge
                    gap = det['time_start'] - current_song['time_end']
                    
                    if gap <= self.song_gap_threshold:
                        # Merge into current song (expand frequency range to cover both)
                        current_song['time_end'] = max(current_song['time_end'], det['time_end'])
                        current_song['confidence'] = (current_song['confidence'] * current_song['detections_merged'] + det['confidence']) / (current_song['detections_merged'] + 1)
                        current_song['max_confidence'] = max(current_song['max_confidence'], det['confidence'])
                        current_song['detections_merged'] += 1
                        current_song['freq_low_hz'] = min(current_song['freq_low_hz'], det['freq_low_hz'])
                        current_song['freq_high_hz'] = max(current_song['freq_high_hz'], det['freq_high_hz'])
                    else:
                        # Gap too large, save current song and start new one
                        merged_songs.append(current_song)
                        current_song = {
                            'species': det['species'],
                            'species_id': det['species_id'],
                            'time_start': det['time_start'],
                            'time_end': det['time_end'],
                            'confidence': det['confidence'],
                            'max_confidence': det['confidence'],
                            'detections_merged': 1,
                            'freq_low_hz': det['freq_low_hz'],
                            'freq_high_hz': det['freq_high_hz'],
                        }
            
            # Don't forget the last song
            if current_song is not None:
                merged_songs.append(current_song)
        
        # Sort by time
        merged_songs = sorted(merged_songs, key=lambda x: x['time_start'])
        
        return merged_songs
    
    def detect(self, audio_path: str, output_path: str = None, output_format: str = 'json') -> List[Dict]:
        """
        Detect bird calls in an audio file.
        
        Args:
            audio_path: Path to the WAV file
            output_path: Optional base path to save results (without extension)
            output_format: Output format - 'json', 'csv', 'txt', or 'all'
            
        Returns:
            List of detections with timing and species information
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
            
            for clip_data in tqdm(clips, desc="Detecting"):
                clip_detections = self.detect_in_clip(clip_data, temp_dir)
                all_detections.extend(clip_detections)
            
            print(f"\nFound {len(all_detections)} raw detections")
            
            # Merge detections (default: reconstruct songs)
            print("Reconstructing continuous bird songs from detections...")
            final_detections = self.merge_overlapping_detections(all_detections, merge_mode='reconstruct')
            
            print(f"Final count: {len(final_detections)} song segments")
            
            # Save results if output path is specified
            if output_path:
                self.save_results(final_detections, output_path, audio_path, output_format)
            
            return final_detections
            
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)
    
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

    def save_detections(self, detections: List[Dict], output_path: str, audio_path: str):
        """
        Save detections to JSON file.
        
        Args:
            detections: List of detections
            output_path: Path to save JSON file
            audio_path: Original audio file path (for metadata)
        """
        output = {
            'audio_file': str(audio_path),
            'model_config': {
                'confidence_threshold': self.conf_threshold,
                'iou_threshold': self.iou_threshold,
                'dataset': config.DATASET_NAME,
            },
            'detection_count': len(detections),
            'detections': detections
        }
        
        # Convert all numpy types to JSON-serializable types
        output = self._convert_to_json_serializable(output)
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nSaved detections to: {output_path}")
    
    def save_detections_csv(self, detections: List[Dict], output_path: str, audio_path: str):
        """
        Save detections to CSV file in the same format as annotations.csv.
        
        Args:
            detections: List of detections
            output_path: Path to save CSV file
            audio_path: Original audio file path (for metadata)
        """
        import csv
        
        # Get just the filename without path for the CSV
        audio_filename = Path(audio_path).name
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header (same as annotations.csv)
            writer.writerow(['Filename', 'Start Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)', 'Species eBird Code'])
            
            # Write detection data
            for det in detections:
                writer.writerow([
                    audio_filename,
                    f"{det['time_start']:.1f}",
                    f"{det['time_end']:.1f}",
                    det['freq_low_hz'],
                    det['freq_high_hz'],
                    det['species']
                ])
        
        print(f"\nSaved detections to CSV: {output_path}")
    
    def save_detections_raven(self, detections: List[Dict], output_path: str, audio_path: str):
        """
        Save detections to Raven .txt format for visualization.
        
        Args:
            detections: List of detections
            output_path: Path to save Raven .txt file
            audio_path: Original audio file path (for metadata)
        """
        with open(output_path, 'w') as f:
            # Write header (matching example.txt format)
            f.write("Selection\tView\tChannel\tBegin Time (S)\tEnd Time (S)\tLow Freq (Hz)\tHigh Freq (Hz)\tAnnotation\n")
            
            # Write detection data
            for i, det in enumerate(detections, 1):
                f.write(f"{i}\tSpectrogram 1\t1\t{det['time_start']:.1f}\t{det['time_end']:.1f}\t"
                       f"{det['freq_low_hz']}\t{det['freq_high_hz']}\t{det['species']}\n")
        
        print(f"\nSaved detections to Raven format: {output_path}")
    
    def save_results(self, detections: List[Dict], output_path: str, audio_path: str, output_format: str):
        """
        Save detections in the specified format(s).
        
        Args:
            detections: List of detections
            output_path: Base path for output files (without extension)
            audio_path: Original audio file path (for metadata)
            output_format: Output format - 'json', 'csv', 'txt', or 'all'
        """
        output_path_obj = Path(output_path)
        
        if output_format == 'json' or output_format == 'all':
            json_path = str(output_path_obj.with_suffix('.json'))
            self.save_detections(detections, json_path, audio_path)
        
        if output_format == 'csv' or output_format == 'all':
            csv_path = str(output_path_obj.with_suffix('.csv'))
            self.save_detections_csv(detections, csv_path, audio_path)
        
        if output_format == 'txt' or output_format == 'all':
            txt_path = str(output_path_obj.with_suffix('.txt'))
            self.save_detections_raven(detections, txt_path, audio_path)
    
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
                          f"avg conf: {det['confidence']:.3f}, "
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


def ensure_output_directory(output_path: str) -> bool:
    """
    Ensure the output directory exists, asking user for permission to create if needed.
    
    Args:
        output_path: The output path (may be a file path)
        
    Returns:
        True if directory exists or was created successfully, False if user declined
    """
    if not output_path:
        return True  # No output path specified, nothing to check
    
    output_dir = Path(output_path).parent
    
    # If the directory already exists, we're good
    if output_dir.exists():
        return True
    
    # Directory doesn't exist, ask user for permission
    print(f"\nOutput directory does not exist: {output_dir}")
    response = input("Would you like to create this directory? (y/n): ").strip().lower()
    
    if response in ['y', 'yes']:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created directory: {output_dir}")
            return True
        except Exception as e:
            print(f"✗ Error creating directory: {e}")
            return False
    else:
        print("✗ Directory creation declined. Exiting.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Detect bird calls in audio files using trained YOLO model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic detection
  python inference/detect_birds.py --audio recording.wav --model final_models/best.pt
  
  # Save results to JSON (new format)
  python inference/detect_birds.py --audio recording.wav --model best.pt --output-path results --output-format json
  
  # Save results to CSV
  python inference/detect_birds.py --audio recording.wav --model best.pt --output-path results --output-format csv
  
  # Save results to Raven .txt format
  python inference/detect_birds.py --audio recording.wav --model best.pt --output-path results --output-format txt
  
  # Save all formats
  python inference/detect_birds.py --audio recording.wav --model best.pt --output-path results --output-format all
  
  # Adjust thresholds
  python inference/detect_birds.py --audio audio.wav --model best.pt --conf 0.5 --iou 0.3
        """
    )
    
    parser.add_argument(
        '--audio',
        type=str,
        required=True,
        help='Path to the audio file (WAV format)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to the trained model (.pt, .onnx, .engine, etc.)'
    )
    
    parser.add_argument(
        '--output-path',
        type=str,
        default=None,
        help='Base path to save detection results (without extension)'
    )
    
    parser.add_argument(
        '--output-format',
        type=str,
        choices=['json', 'csv', 'txt', 'all'],
        default='json',
        help='Output format: json (default), csv, txt, or all formats'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold for detections (default: 0.25)'
    )
    
    parser.add_argument(
        '--iou',
        type=float,
        default=0.5,
        help='IoU threshold for NMS across time windows (default: 0.5)'
    )
    
    parser.add_argument(
        '--song-gap',
        type=float,
        default=2.0,
        help='Max gap (seconds) between detections to merge into same song (default: 2.0)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.audio).exists():
        print(f"Error: Audio file not found: {args.audio}", file=sys.stderr)
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
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        song_gap_threshold=args.song_gap
    )
    
    # Run detection
    detections = detector.detect(args.audio, output_path, args.output_format)
    
    # Print summary
    detector.print_summary(detections)


if __name__ == '__main__':
    main()

