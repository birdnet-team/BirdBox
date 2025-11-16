#!/usr/bin/env python3
"""
Confusion Matrix Analysis Script for Bird Call Detection Results

This script computes a confusion matrix comparing detection results against
ground truth labels. It matches detections to labels using IoU (Intersection
over Union) and creates a confusion matrix showing how well the model performs
for each species class.

Usage:
    python src/evaluation/confusion_matrix_analysis.py --detections <path_to_detections.csv> --labels <path_to_labels.csv>
    
Example:
    python src/evaluation/confusion_matrix_analysis.py \
        --detections results/filtered_detections/filtered_detections.csv \
        --labels data/labels.csv
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# Add parent directory to path to import utilities
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.utils.confusion_matrix import (
    build_confusion_matrix,
    normalize_filename,
    print_confusion_matrix,
    save_confusion_matrix,
    plot_confusion_matrix
)


class ConfusionMatrixAnalyzer:
    """
    Analyzer for confusion matrix computation from detection results.
    
    This class handles loading detection results and ground truth labels,
    matching them using IoU with the Hungarian algorithm (optimal matching),
    and computing confusion matrices.
    
    Always uses optimal matching for order-independent, reproducible results.
    """
    
    def __init__(self, iou_threshold: float = 0.5, use_2d_iou: bool = True, 
                 include_background: bool = True):
        """
        Initialize the confusion matrix analyzer.
        
        Uses optimal matching (Hungarian algorithm) for order-independent, globally optimal results.
        
        Args:
            iou_threshold: IoU threshold for matching detections to labels
            use_2d_iou: If True, use 2D IoU (time-frequency), otherwise use 1D IoU (time only)
            include_background: If True, include background class for FP/FN
        """
        self.iou_threshold = iou_threshold
        self.use_2d_iou = use_2d_iou
        self.include_background = include_background
        
        print(f"Initialized confusion matrix analyzer with IoU threshold: {iou_threshold}")
        print(f"IoU type: {'2D (time-frequency)' if use_2d_iou else '1D (time only)'}")
        print(f"Include background: {include_background}")
        print(f"Matching method: Optimal (Hungarian)")
    
    @staticmethod
    def normalize_filename(filename: str) -> str:
        """
        Normalize filename by removing extension and path.
        This handles cases where detections use .wav but labels use .flac, etc.
        
        Args:
            filename: Original filename
            
        Returns:
            Filename without extension
        """
        from pathlib import Path
        return Path(filename).stem
    
    def load_detections_csv(self, detections_path: str) -> List[Dict]:
        """
        Load detection results from CSV file.
        
        Args:
            detections_path: Path to the detections CSV file
            
        Returns:
            List of detection dictionaries
        """
        print(f"\nLoading detections from: {detections_path}")
        df = pd.read_csv(detections_path)
        
        # Convert to list of dictionaries
        detections = []
        for _, row in df.iterrows():
            # Skip empty rows
            if pd.isna(row.get('Species eBird Code')):
                continue
                
            detections.append({
                'filename': row['Filename'],
                'time_start': row['Start Time (s)'],
                'time_end': row['End Time (s)'],
                'freq_low_hz': row['Low Freq (Hz)'],
                'freq_high_hz': row['High Freq (Hz)'],
                'species': row['Species eBird Code']
            })
        
        print(f"Loaded {len(detections)} detections")
        return detections
    
    def load_labels_csv(self, labels_path: str) -> List[Dict]:
        """
        Load ground truth labels from CSV file.
        
        Args:
            labels_path: Path to the labels CSV file
            
        Returns:
            List of ground truth label dictionaries
        """
        print(f"\nLoading ground truth labels from: {labels_path}")
        df = pd.read_csv(labels_path)
        
        # Convert to list of dictionaries
        labels = []
        for _, row in df.iterrows():
            # Skip empty rows
            if pd.isna(row.get('Species eBird Code')):
                continue
                
            labels.append({
                'filename': row['Filename'],
                'start_time': row['Start Time (s)'],
                'end_time': row['End Time (s)'],
                'freq_low': row['Low Freq (Hz)'],
                'freq_high': row['High Freq (Hz)'],
                'species': row['Species eBird Code']
            })
        
        print(f"Loaded {len(labels)} ground truth labels")
        return labels
    
    def filter_by_filename(self, detections: List[Dict], labels: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Filter detections and labels to only include files that appear in both lists.
        Uses normalized filenames (without extensions) for matching.
        
        Args:
            detections: List of detection dictionaries
            labels: List of ground truth label dictionaries
            
        Returns:
            Tuple of (filtered_detections, filtered_labels)
        """
        # Get normalized filenames from both lists
        detection_files = set(self.normalize_filename(d['filename']) for d in detections)
        label_files = set(self.normalize_filename(l['filename']) for l in labels)
        
        # Find common files
        common_files = detection_files & label_files
        
        if not common_files:
            print("\nWarning: No common files found between detections and labels!")
            print(f"Detection files: {detection_files}")
            print(f"Label files: {label_files}")
            print("Proceeding with all detections and labels anyway...")
            return detections, labels
        
        # Filter to common files
        filtered_detections = [d for d in detections if self.normalize_filename(d['filename']) in common_files]
        filtered_labels = [l for l in labels if self.normalize_filename(l['filename']) in common_files]
        
        print(f"\nFound {len(common_files)} common file(s)")
        print(f"Filtered to {len(filtered_detections)} detections and {len(filtered_labels)} labels")
        
        return filtered_detections, filtered_labels
    
    def get_species_list(self, detections: List[Dict], labels: List[Dict]) -> List[str]:
        """
        Get sorted list of all unique species from detections and labels.
        
        Args:
            detections: List of detection dictionaries
            labels: List of ground truth label dictionaries
            
        Returns:
            Sorted list of unique species codes
        """
        species_set = set()
        
        for det in detections:
            if 'species' in det and det['species']:
                species_set.add(det['species'])
        
        for label in labels:
            if 'species' in label and label['species']:
                species_set.add(label['species'])
        
        return sorted(list(species_set))
    
    def print_statistics(self, detections: List[Dict], labels: List[Dict], species_list: List[str]):
        """
        Print statistics about detections and labels.
        
        Args:
            detections: List of detection dictionaries
            labels: List of ground truth label dictionaries
            species_list: List of species codes
        """
        print("\n" + "="*80)
        print("DATASET STATISTICS")
        print("="*80)
        
        # Count detections per species
        detection_counts = {}
        for species in species_list:
            count = sum(1 for d in detections if d.get('species') == species)
            detection_counts[species] = count
        
        # Count labels per species
        label_counts = {}
        for species in species_list:
            count = sum(1 for l in labels if l.get('species') == species)
            label_counts[species] = count
        
        print(f"\n{'Species':<15} {'Detections':<15} {'Ground Truth':<15}")
        print("-"*45)
        for species in species_list:
            print(f"{species:<15} {detection_counts.get(species, 0):<15} {label_counts.get(species, 0):<15}")
        print("-"*45)
        print(f"{'Total':<15} {len(detections):<15} {len(labels):<15}")
        print("="*80)
    
    def compute_confusion_matrix(self, detections: List[Dict], labels: List[Dict]) -> Tuple[np.ndarray, List[str]]:
        """
        Compute confusion matrix from detections and labels.
        
        Args:
            detections: List of detection dictionaries
            labels: List of ground truth label dictionaries
            
        Returns:
            Tuple of (confusion_matrix, species_list)
        """
        # Get species list
        species_list = self.get_species_list(detections, labels)
        
        # Build confusion matrix using optimal matching (Hungarian algorithm)
        print("\nBuilding confusion matrix...")
        confusion_matrix = build_confusion_matrix(
            detections=detections,
            labels=labels,
            species_list=species_list,
            iou_threshold=self.iou_threshold,
            use_2d_iou=self.use_2d_iou,
            include_background=self.include_background,
            use_optimal_matching=True  # Always use optimal matching for confusion matrices
        )
        
        return confusion_matrix, species_list
    
    def analyze(self, detections_path: str, labels_path: str, output_path: str = None) -> Tuple[np.ndarray, List[str]]:
        """
        Run full confusion matrix analysis from CSV files.
        
        Args:
            detections_path: Path to detections CSV file
            labels_path: Path to labels CSV file
            output_path: Optional output directory path. If None, results are not saved.
            
        Returns:
            Tuple of (confusion_matrix, species_list)
        """
        # Load detections and labels
        detections = self.load_detections_csv(detections_path)
        labels = self.load_labels_csv(labels_path)
        
        # Filter to common files
        detections, labels = self.filter_by_filename(detections, labels)
        
        if not detections:
            raise ValueError("No detections to analyze!")
        
        if not labels:
            raise ValueError("No labels to analyze!")
        
        # Get species list
        species_list = self.get_species_list(detections, labels)
        print(f"\nFound {len(species_list)} unique species: {', '.join(species_list)}")
        
        # Print statistics
        self.print_statistics(detections, labels, species_list)
        
        # Compute confusion matrix
        confusion_matrix, species_list = self.compute_confusion_matrix(detections, labels)
        
        # Print confusion matrix
        print_confusion_matrix(confusion_matrix, species_list, include_background=self.include_background)
        
        # Save results if output path provided
        if output_path:
            self.save_results(confusion_matrix, species_list, output_path, detections_path, labels_path, 
                            len(detections), len(labels))
        
        return confusion_matrix, species_list
    
    def save_results(self, confusion_matrix: np.ndarray, species_list: List[str], output_path: str,
                    detections_path: str = None, labels_path: str = None,
                    num_detections: int = None, num_labels: int = None):
        """
        Save confusion matrix results to files.
        
        Args:
            confusion_matrix: Confusion matrix numpy array
            species_list: List of species codes
            output_path: Output directory path
            detections_path: Optional path to detections file (for metadata)
            labels_path: Optional path to labels file (for metadata)
            num_detections: Optional number of detections (for metadata)
            num_labels: Optional number of labels (for metadata)
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nCreated output directory: {output_dir}")
        
        # Save confusion matrix
        output_file = output_dir / "confusion_matrix.csv"
        save_confusion_matrix(
            confusion_matrix, 
            species_list, 
            str(output_file),
            include_background=self.include_background
        )
        
        # Also save as a more detailed format with row/column labels
        class_labels = species_list.copy()
        if self.include_background:
            class_labels.append('background')
        
        # Create a nicely formatted version
        df = pd.DataFrame(
            confusion_matrix,
            index=[f"pred_{label}" for label in class_labels],
            columns=[f"true_{label}" for label in class_labels]
        )
        
        detailed_output_file = output_dir / "confusion_matrix_detailed.csv"
        df.to_csv(detailed_output_file)
        print(f"Detailed confusion matrix saved to: {detailed_output_file}")
        
        # Generate PNG visualizations
        print("\nGenerating confusion matrix visualizations...")
        
        # Normalized confusion matrix (shows percentages)
        normalized_png = output_dir / "confusion_matrix_normalized.png"
        plot_confusion_matrix(
            confusion_matrix,
            species_list,
            str(normalized_png),
            include_background=self.include_background,
            normalize=True
        )
        
        # Raw counts confusion matrix
        raw_png = output_dir / "confusion_matrix_raw.png"
        plot_confusion_matrix(
            confusion_matrix,
            species_list,
            str(raw_png),
            include_background=self.include_background,
            normalize=False
        )
        
        # Save metadata if provided
        if detections_path or labels_path:
            metadata = {
                'detections_file': detections_path or 'N/A',
                'labels_file': labels_path or 'N/A',
                'iou_threshold': self.iou_threshold,
                'iou_type': '1D (time only)' if not self.use_2d_iou else '2D (time-frequency)',
                'matching_method': 'Optimal (Hungarian)',
                'include_background': self.include_background,
                'num_detections': num_detections or 'N/A',
                'num_labels': num_labels or 'N/A',
                'num_species': len(species_list),
                'species_list': ', '.join(species_list)
            }
            
            metadata_file = output_dir / "metadata.txt"
            with open(metadata_file, 'w') as f:
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
            print(f"Metadata saved to: {metadata_file}")


def main():
    """Main function to run confusion matrix analysis."""
    parser = argparse.ArgumentParser(
        description='Compute confusion matrix for bird call detection results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python src/evaluation/confusion_matrix_analysis.py \\
        --detections results/filtered_detections/filtered_detections.csv \\
        --labels data/labels.csv
        """
    )
    
    parser.add_argument(
        '--detections',
        type=str,
        required=True,
        help='Path to detections CSV file'
    )
    
    parser.add_argument(
        '--labels',
        type=str,
        required=True,
        help='Path to ground truth labels CSV file'
    )
    
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.5,
        help='IoU threshold for matching detections to labels (default: 0.5)'
    )
    
    parser.add_argument(
        '--use-1d-iou',
        action='store_true',
        help='Use 1D IoU (time only) instead of 2D IoU (time-frequency)'
    )
    
    parser.add_argument(
        '--no-background',
        action='store_true',
        help='Do not include background class in confusion matrix'
    )
    
    parser.add_argument(
        '--output-path',
        type=str,
        default='results/confusion_matrix_analysis',
        help='Output directory path for results.'
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("="*80)
    print("CONFUSION MATRIX ANALYSIS")
    print("="*80)
    print(f"Detections: {args.detections}")
    print(f"Labels: {args.labels}")
    print(f"IoU threshold: {args.iou_threshold}")
    print(f"IoU type: {'1D (time only)' if args.use_1d_iou else '2D (time-frequency)'}")
    print(f"Matching method: Optimal (Hungarian)")
    print(f"Include background: {not args.no_background}")
    print(f"Output directory: {args.output_path}")
    print("="*80)
    
    # Create analyzer
    analyzer = ConfusionMatrixAnalyzer(
        iou_threshold=args.iou_threshold,
        use_2d_iou=not args.use_1d_iou,
        include_background=not args.no_background
    )
    
    # Run analysis
    try:
        analyzer.analyze(args.detections, args.labels, args.output_path)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        
        return 0
    except Exception as e:
        print(f"\nError during analysis: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())

