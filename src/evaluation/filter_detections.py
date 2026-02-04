#!/usr/bin/env python3
"""
Filter bird call detections based on confidence threshold.

This script takes a comprehensive detection results file (generated with very low confidence threshold)
and filters it based on a specified confidence threshold. It can filter using either average confidence
or max confidence from merged detections.

With --raw-detections, the input is raw (unmerged) JSON from detect_birds --no-merge. The script
filters by confidence then merges (reconstruct_songs), producing the same result as running
detect_birds again at that confidence—without running inference a second time.

Usage:
    python src/evaluation/filter_detections.py --input all_detections.json --output-path results/filtered_detections --conf 0.5
    python src/evaluation/filter_detections.py --input all_detections.json --output-path results/filtered_detections --conf 0.3 --use-avg-confidence
    python src/evaluation/filter_detections.py --input all_detections.json --output-path results/filtered_detections --conf 0.7 --format all
    python src/evaluation/filter_detections.py --input raw_detections.json --output-path results/merged_detections --conf 0.25 --format all --raw-detections
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict
import csv

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from inference.detect_birds import reconstruct_songs
except ImportError:
    reconstruct_songs = None


class DetectionFilter:
    """
    Filter bird call detections based on confidence threshold.
    
    This class handles filtering detections from comprehensive detection results
    and saving them in various formats.
    """
    
    def __init__(self, use_max_confidence: bool = True):
        """
        Initialize the detection filter.
        
        Args:
            use_max_confidence: If True, use max_confidence for filtering; if False, use average confidence
        """
        self.use_max_confidence = use_max_confidence
        self.confidence_field = 'max_confidence' if use_max_confidence else 'confidence'
        
        print(f"Using {'max' if use_max_confidence else 'average'} confidence for filtering")
    
    def load_detections(self, input_path: str) -> Dict:
        """
        Load detections from JSON file.
        
        Args:
            input_path: Path to the detections JSON file
            
        Returns:
            Dictionary containing detection data
        """
        print(f"\nLoading detections from: {input_path}")
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        detections = data.get('detections', [])
        print(f"Loaded {len(detections)} total detections")
        
        # Print summary of confidence values (use 'confidence' for raw detections)
        if detections:
            has_merge_field = any(d.get(self.confidence_field) is not None for d in detections)
            conf_key = self.confidence_field if has_merge_field else 'confidence'
            confidences = [det.get(conf_key, 0) for det in detections]
            print(f"Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
            print(f"Mean confidence: {sum(confidences)/len(confidences):.3f}")
        
        return data
    
    def filter_detections(self, data: Dict, conf_threshold: float) -> List[Dict]:
        """
        Filter detections based on confidence threshold.
        
        Args:
            data: Dictionary containing detection data
            conf_threshold: Confidence threshold for filtering
            
        Returns:
            List of filtered detections
        """
        detections = data.get('detections', [])
        
        print(f"\nFiltering detections with confidence threshold: {conf_threshold}")
        print(f"Using field: {self.confidence_field}")
        
        filtered_detections = []
        for det in detections:
            confidence = det.get(self.confidence_field, 0)
            if confidence >= conf_threshold:
                filtered_detections.append(det)
        
        print(f"Filtered from {len(detections)} to {len(filtered_detections)} detections")
        print(f"Retention rate: {len(filtered_detections)/len(detections)*100:.1f}%")
        
        return filtered_detections
    
    def save_filtered_json(self, data: Dict, filtered_detections: List[Dict], output_path: str, conf_threshold: float):
        """
        Save filtered detections to JSON file.
        
        Args:
            data: Original detection data
            filtered_detections: List of filtered detections
            output_path: Path to save JSON file
            conf_threshold: Confidence threshold used for filtering
        """
        # Create filtered output structure
        filtered_data = {
            'audio_files': data.get('audio_files', []),
            'file_count': data.get('file_count', 0),
            'model_config': data.get('model_config', {}),
            'filtering_config': {
                'confidence_threshold': conf_threshold,
                'confidence_field_used': self.confidence_field,
                'use_max_confidence': self.use_max_confidence
            },
            'detection_count': len(filtered_detections),
            'original_detection_count': len(data.get('detections', [])),
            'detections': filtered_detections
        }
        
        with open(output_path, 'w') as f:
            json.dump(filtered_data, f, indent=2)
        
        print(f"Saved filtered detections to JSON: {output_path}")
    
    def save_filtered_csv(self, filtered_detections: List[Dict], output_path: str):
        """
        Save filtered detections to CSV file in the same format as annotations.csv.
        
        Args:
            filtered_detections: List of filtered detections
            output_path: Path to save CSV file
        """
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header (same as annotations.csv)
            writer.writerow(['Filename', 'Start Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)', 'Species eBird Code'])
            
            # Write detection data
            for det in filtered_detections:
                # Use filename from detection if available (for multi-file), otherwise use 'unknown'
                filename = det.get('filename', 'unknown')
                
                writer.writerow([
                    filename,
                    f"{det['time_start']:.1f}",
                    f"{det['time_end']:.1f}",
                    det['freq_low_hz'],
                    det['freq_high_hz'],
                    det['species']
                ])
        
        print(f"Saved filtered detections to CSV: {output_path}")
    
    def save_results(self, data: Dict, filtered_detections: List[Dict], output_path: str, conf_threshold: float, output_format: str = 'json'):
        """
        Save filtered detections in the specified format(s).
        
        Args:
            data: Original detection data
            filtered_detections: List of filtered detections
            output_path: Base path for output files (without extension)
            conf_threshold: Confidence threshold used for filtering
            output_format: Output format - 'json', 'csv', or 'all'
        """
        output_path_obj = Path(output_path)
        
        if output_format == 'json' or output_format == 'all':
            json_path = str(output_path_obj.with_suffix('.json'))
            self.save_filtered_json(data, filtered_detections, json_path, conf_threshold)
        
        if output_format == 'csv' or output_format == 'all':
            csv_path = str(output_path_obj.with_suffix('.csv'))
            self.save_filtered_csv(filtered_detections, csv_path)
    
    def print_summary(self, data: Dict, filtered_detections: List[Dict], conf_threshold: float):
        """Print a summary of filtering results."""
        original_detections = data.get('detections', [])
        
        print(f"\n{'='*80}")
        print("FILTERING SUMMARY")
        print(f"{'='*80}")
        
        print(f"Confidence threshold: {conf_threshold}")
        print(f"Confidence field used: {self.confidence_field}")
        print(f"Original detections: {len(original_detections)}")
        print(f"Filtered detections: {len(filtered_detections)}")
        print(f"Retention rate: {len(filtered_detections)/len(original_detections)*100:.1f}%")
        
        if len(filtered_detections) == 0:
            print("\nNo detections remain after filtering.")
            return
        
        # Group by species
        species_counts = {}
        for det in filtered_detections:
            species = det['species']
            if species not in species_counts:
                species_counts[species] = []
            species_counts[species].append(det)
        
        print(f"\nSpecies detected: {len(species_counts)}")
        
        # Check if these are reconstructed songs (have 'detections_merged' field)
        is_reconstructed = 'detections_merged' in filtered_detections[0] if filtered_detections else False
        
        print()
        
        for species, dets in sorted(species_counts.items()):
            print(f"{species}: {len(dets)} {'song segments' if is_reconstructed else 'detections'}")
            
            # Show confidence statistics for this species
            confidences = [det.get(self.confidence_field, 0) for det in dets]
            avg_conf = sum(confidences) / len(confidences)
            min_conf = min(confidences)
            max_conf = max(confidences)
            
            print(f"  Confidence stats: avg={avg_conf:.3f}, min={min_conf:.3f}, max={max_conf:.3f}")
            
            # Show first few detections
            for det in dets[:3]:  # Show first 3 for each species
                duration = det['time_end'] - det['time_start']
                confidence = det.get(self.confidence_field, 0)
                
                if is_reconstructed:
                    print(f"    {det['time_start']:6.2f}s - {det['time_end']:6.2f}s "
                          f"({duration:5.2f}s duration, "
                          f"{det['detections_merged']:2d} clips merged, "
                          f"conf: {confidence:.3f})")
                else:
                    print(f"    {det['time_start']:6.2f}s - {det['time_end']:6.2f}s "
                          f"(confidence: {confidence:.3f})")
            
            if len(dets) > 3:
                print(f"    ... and {len(dets) - 3} more")
            
            print()


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
        description="Filter bird call detections based on confidence threshold",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter using max confidence (recommended)
  python src/evaluation/filter_detections.py --input all_detections.json --output-path results/filtered_detections --conf 0.5
  
  # Filter using average confidence
  python src/evaluation/filter_detections.py --input all_detections.json --output-path results/filtered_detections --conf 0.3 --use-avg-confidence
  
  # Save in all formats
  python src/evaluation/filter_detections.py --input all_detections.json --output-path results/filtered_detections --conf 0.7 --format all
  
  # Save only CSV (for evaluation)
  python src/evaluation/filter_detections.py --input all_detections.json --output-path results/filtered_detections --conf 0.4 --format csv

  # From raw detections (detect_birds --no-merge): filter then merge, no second inference
  python src/evaluation/filter_detections.py --input raw_detections.json --output-path results/merged --conf 0.25 --format all --raw-detections
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to the comprehensive detections JSON file'
    )
    
    parser.add_argument(
        '--output-path',
        type=str,
        default='results/filtered_detections',
        help='Output directory path for results.'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        required=True,
        help='Confidence threshold for filtering (0.0-1.0)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'csv', 'all'],
        default='json',
        help='Output format: json (default), csv, or all formats'
    )
    
    parser.add_argument(
        '--use-avg-confidence',
        action='store_true',
        help='Use average confidence instead of max confidence for filtering (default: use max confidence)'
    )
    
    parser.add_argument(
        '--raw-detections',
        action='store_true',
        help='Input JSON is raw (unmerged) detections from detect_birds --no-merge. '
             'Filter by confidence then merge (reconstruct_songs) before saving. '
             'Use this to get merged detections at a given confidence without re-running inference.'
    )
    
    parser.add_argument(
        '--song-gap',
        type=float,
        default=None,
        help='Song gap in seconds for merging (only with --raw-detections). '
             'Defaults to model_config.song_gap_threshold from the JSON, or 0.1.'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    if not (0.0 <= args.conf <= 1.0):
        print(f"Error: Confidence threshold must be between 0.0 and 1.0, got: {args.conf}", file=sys.stderr)
        sys.exit(1)
    
    # Ensure output directory exists
    if not ensure_output_directory(args.output_path):
        sys.exit(1)
    
    # Create filter
    filter_obj = DetectionFilter(use_max_confidence=not args.use_avg_confidence)
    
    # Load detections
    data = filter_obj.load_detections(args.input)
    
    if args.raw_detections:
        # Raw (unmerged) workflow: filter by confidence then merge, no second inference
        if reconstruct_songs is None:
            print("Error: --raw-detections requires inference.detect_birds.reconstruct_songs (run from BirdBox with inference installed).", file=sys.stderr)
            sys.exit(1)
        raw_list = data.get('detections', [])
        model_config = data.get('model_config', {})
        song_gap = args.song_gap if args.song_gap is not None else float(model_config.get('song_gap_threshold', 0.1))
        filtered_raw = [d for d in raw_list if d.get('confidence', 0) >= args.conf]
        merged = reconstruct_songs(filtered_raw, song_gap)
        # Ensure filename on each merged segment (single-file raw has no filename on detections)
        if 'audio_file' in data:
            audio_file = Path(data['audio_file']).name
            for det in merged:
                if 'filename' not in det:
                    det['filename'] = audio_file
        filtered_detections = merged
        print(f"Raw detections: filtered at conf>={args.conf} then merged (song_gap={song_gap}s) -> {len(filtered_detections)} segments")
    else:
        # Merged detections: filter by confidence/max_confidence only
        filtered_detections = filter_obj.filter_detections(data, args.conf)
    
    # Save results
    filter_obj.save_results(data, filtered_detections, args.output_path, args.conf, args.format)
    
    # Print summary
    filter_obj.print_summary(data, filtered_detections, args.conf)
    
    print("\n" + "="*80)
    print("FILTERING COMPLETED SUCCESSFULLY")
    print("="*80)


if __name__ == '__main__':
    main()
