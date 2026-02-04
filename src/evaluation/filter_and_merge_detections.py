#!/usr/bin/env python3
"""
Filter raw bird call detections by confidence and merge into song segments.

Expects raw (unmerged) JSON from detect_birds --no-merge. Filters at --conf,
merges at --song-gap (reconstruct_songs), and saves the result—equivalent to
running detect_birds at that confidence without re-running inference.

Usage:
    python src/evaluation/filter_and_merge_detections.py --input raw_detections.json --output-path results/merged_detections --conf 0.25
    python src/evaluation/filter_and_merge_detections.py --input raw_detections.json --output-path results/merged --conf 0.25 --song-gap 0.1 --output-format all
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

from inference.detect_birds import reconstruct_songs


class DetectionFilter:
    """
    Load raw detections and filter-by-confidence then merge (filter-then-merge).
    Used by F-beta analysis and other evaluation tools.
    """

    def __init__(self, use_max_confidence: bool = True):
        """
        Initialize the detection filter.
        For raw detections we always filter by 'confidence'; use_max_confidence
        is kept for API compatibility with FBetaScoreAnalyzer.
        """
        self.use_max_confidence = use_max_confidence
        self.confidence_field = 'max_confidence' if use_max_confidence else 'avg_confidence'

    def load_detections(self, input_path: str) -> Dict:
        """
        Load detections from JSON file (raw unmerged from detect_birds --no-merge).

        Expects optional 'model_config' with e.g. 'model' (path to model), 'song_gap_threshold';
        these are preserved when saving filtered output.

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

        if detections:
            confidences = [d.get('confidence', 0) for d in detections]
            print(f"Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
            print(f"Mean confidence: {sum(confidences)/len(confidences):.3f}")

        return data

    def filter_detections(self, data: Dict, conf_threshold: float, song_gap: float = None) -> List[Dict]:
        """
        Filter raw detections by confidence then merge (reconstruct_songs).
        Returns merged song segments.

        Args:
            data: Dictionary with 'detections' (raw list) and optional 'model_config'
                (may contain 'model', 'song_gap_threshold', etc.).
            conf_threshold: Confidence threshold for filtering
            song_gap: Max gap (seconds) to merge; default from model_config or 0.1

        Returns:
            List of merged detections (song segments)
        """
        raw_list = data.get('detections', [])
        model_config = data.get('model_config', {})
        gap = song_gap if song_gap is not None else float(model_config.get('song_gap_threshold', 0.1))
        filtered_raw = [d for d in raw_list if d.get('confidence', 0) >= conf_threshold]
        merged = reconstruct_songs(filtered_raw, gap)
        if 'audio_file' in data:
            audio_file = Path(data['audio_file']).name
            for det in merged:
                if 'filename' not in det:
                    det['filename'] = audio_file
        return merged

    def save_filtered_json(self, data: Dict, filtered_detections: List[Dict], output_path: str, conf_threshold: float, song_gap: float):
        """Save filtered-and-merged detections to JSON. Preserves model_config (e.g. 'model' path, song_gap_threshold)."""
        filtered_data = {
            'audio_files': data.get('audio_files', []),
            'file_count': data.get('file_count', 0),
            'model_config': data.get('model_config', {}),
            'filtering_config': {
                'confidence_threshold': conf_threshold,
                'song_gap_threshold': song_gap,
            },
            'detection_count': len(filtered_detections),
            'original_detection_count': len(data.get('detections', [])),
            'detections': filtered_detections
        }
        with open(output_path, 'w') as f:
            json.dump(filtered_data, f, indent=2)
        print(f"Saved filtered detections to JSON: {output_path}")

    def save_filtered_csv(self, filtered_detections: List[Dict], output_path: str):
        """Save filtered detections to CSV (same format as annotations.csv)."""
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Filename', 'Start Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)', 'Species eBird Code'])
            for det in filtered_detections:
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

    def save_results(self, data: Dict, filtered_detections: List[Dict], output_path: str, conf_threshold: float, song_gap: float, output_format: str = 'json'):
        """Save filtered detections in the specified format(s)."""
        output_path_obj = Path(output_path)
        if output_format in ('json', 'all'):
            self.save_filtered_json(data, filtered_detections, str(output_path_obj.with_suffix('.json')), conf_threshold, song_gap)
        if output_format in ('csv', 'all'):
            self.save_filtered_csv(filtered_detections, str(output_path_obj.with_suffix('.csv')))

    def print_summary(self, data: Dict, filtered_detections: List[Dict], conf_threshold: float):
        """Print a summary of filtering results."""
        original_detections = data.get('detections', [])
        print(f"\n{'='*80}")
        print("FILTERING SUMMARY")
        print(f"{'='*80}")
        print(f"Confidence threshold: {conf_threshold}")
        print(f"Original detections: {len(original_detections)}")
        print(f"Merged segments: {len(filtered_detections)}")
        if len(original_detections) > 0:
            filtered_count = sum(1 for d in original_detections if d.get('confidence', 0) >= conf_threshold)
            print(f"Detections after filter (before merge): {filtered_count}")
        if len(filtered_detections) == 0:
            print("\nNo detections remain after filtering.")
            return
        species_counts = {}
        for det in filtered_detections:
            species = det['species']
            species_counts.setdefault(species, []).append(det)
        print(f"\nSpecies detected: {len(species_counts)}")
        is_reconstructed = 'detections_merged' in filtered_detections[0] if filtered_detections else False
        print()
        for species, dets in sorted(species_counts.items()):
            print(f"{species}: {len(dets)} {'song segments' if is_reconstructed else 'detections'}")
            confidences = [det[self.confidence_field] for det in dets]
            avg_conf = sum(confidences) / len(confidences)
            min_conf = min(confidences)
            max_conf = max(confidences)
            print(f"  Confidence stats: avg={avg_conf:.3f}, min={min_conf:.3f}, max={max_conf:.3f}")
            for det in dets[:3]:
                duration = det['time_end'] - det['time_start']
                confidence = det[self.confidence_field]
                if is_reconstructed:
                    print(f"    {det['time_start']:6.2f}s - {det['time_end']:6.2f}s "
                          f"({duration:5.2f}s duration, {det.get('detections_merged', 0):2d} clips merged, conf: {confidence:.3f})")
                else:
                    print(f"    {det['time_start']:6.2f}s - {det['time_end']:6.2f}s (confidence: {confidence:.3f})")
            if len(dets) > 3:
                print(f"    ... and {len(dets) - 3} more")
            print()


def ensure_output_directory(output_path: str) -> bool:
    """Ensure the output directory exists."""
    if not output_path:
        return True
    output_dir = Path(output_path).parent
    if output_dir.exists():
        return True
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created output directory: {output_dir}")
        return True
    except Exception as e:
        print(f"✗ Error creating directory: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Filter raw bird call detections by confidence and merge into song segments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/evaluation/filter_and_merge_detections.py --input raw_detections.json --output-path results/merged --conf 0.25
  python src/evaluation/filter_and_merge_detections.py --input raw_detections.json --output-path results/merged --conf 0.25 --song-gap 0.1 --output-format all
        """
    )

    parser.add_argument(
        '--input', 
        type=str, 
        required=True, 
        help='Path to raw detections JSON (from detect_birds --no-merge)'
    )

    parser.add_argument(
        '--output-path', 
        type=str, 
        default='results/merged_detections', 
        help='Output path for results (without extension)'
    )

    parser.add_argument(
        '--conf', 
        type=float, 
        required=True, 
        help='Confidence threshold for filtering (0.0-1.0)'
    )

    parser.add_argument(
        '--song-gap', 
        type=float, 
        default=None, 
        help='Max gap (seconds) to merge detections; default from JSON model_config or 0.1'
    )

    parser.add_argument(
        '--output-format', 
        type=str, 
        choices=['json', 'csv', 'all'], 
        default='json', 
        help='Output format: json (default), csv, or all'
    )

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    if not (0.0 <= args.conf <= 1.0):
        print(f"Error: Confidence threshold must be between 0.0 and 1.0, got: {args.conf}", file=sys.stderr)
        sys.exit(1)
    if not ensure_output_directory(args.output_path):
        sys.exit(1)

    filter_obj = DetectionFilter(use_max_confidence=True)
    data = filter_obj.load_detections(args.input)
    raw_list = data.get('detections', [])
    model_config = data.get('model_config', {})
    song_gap = args.song_gap if args.song_gap is not None else float(model_config.get('song_gap_threshold', 0.1))

    filtered_raw = [d for d in raw_list if d.get('confidence', 0) >= args.conf]
    merged = reconstruct_songs(filtered_raw, song_gap)
    if 'audio_file' in data:
        audio_file = Path(data['audio_file']).name
        for det in merged:
            if 'filename' not in det:
                det['filename'] = audio_file
    print(f"Filtered at conf>={args.conf}, merged (song_gap={song_gap}s) -> {len(merged)} segments")

    filter_obj.save_results(data, merged, args.output_path, args.conf, song_gap, args.output_format)
    filter_obj.print_summary(data, merged, args.conf)

    print("\n" + "="*80)
    print("FILTERING COMPLETED SUCCESSFULLY")
    print("="*80)


if __name__ == '__main__':
    main()
