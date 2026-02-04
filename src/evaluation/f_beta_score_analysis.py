#!/usr/bin/env python3
"""
F-Beta Score Analysis Script for Bird Call Detection Results

This script computes F-beta scores for each bird species class under various confidence thresholds.
It loads detection results from the inference JSON file and compares them against ground truth 
annotations to evaluate model performance across different confidence levels.

The F-beta score allows configurable weighting between precision and recall:
- Beta > 1: Emphasizes recall over precision (e.g., F2-score)
- Beta = 1: Equal weight to precision and recall (F1-score)  
- Beta < 1: Emphasizes precision over recall (e.g., F0.5-score)

For bird detection, F2-score (beta=2) is often preferred as missing a bird is typically 
worse than a false positive.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.filter_detections import DetectionFilter
from evaluation.utils.confusion_matrix import compute_2d_iou

try:
    from inference.detect_birds import reconstruct_songs
except ImportError:
    reconstruct_songs = None  # optional when inference not installed

# Optional dependencies for enhanced plotting
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not available. Basic plotting will be used.")


class FBetaScoreAnalyzer:
    """
    Analyzer for F-beta scores across different confidence thresholds.
    
    This class handles loading detection results, filtering by confidence thresholds,
    and computing F-beta scores for each species class.
    """
    
    def __init__(self, iou_threshold: float = 0.5, beta: float = 1.0, use_max_confidence: bool = True, use_optimal_matching: bool = True):
        """
        Initialize the F-beta score analyzer.
        
        Args:
            iou_threshold: IoU threshold for considering detections as matches
            beta: Beta parameter for F-beta score
            use_max_confidence: If True, use max_confidence for filtering; if False, use average confidence
            use_optimal_matching: If True, use Hungarian algorithm (optimal, order-independent). Recommended for final metrics.
        """
        self.iou_threshold = iou_threshold
        self.beta = beta
        self.use_max_confidence = use_max_confidence
        self.use_optimal_matching = use_optimal_matching
        self.filter = DetectionFilter(use_max_confidence=use_max_confidence)
        
        print(f"Initialized F-beta analyzer with IoU threshold: {iou_threshold}")
        print(f"Using F-{beta} score")
        print(f"Using {'max' if use_max_confidence else 'average'} confidence for filtering")
        print(f"Matching method: {'Optimal (Hungarian)' if use_optimal_matching else 'Greedy (order-dependent)'}")
    
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
    
    def load_detections(self, detections_path: str) -> Dict:
        """
        Load detection results from JSON file.
        
        Args:
            detections_path: Path to the detections JSON file
            
        Returns:
            Dictionary containing detection data
        """
        return self.filter.load_detections(detections_path)
    
    def load_labels(self, labels_path: str) -> List[Dict]:
        """
        Load ground truth labels from CSV file.
        
        Args:
            labels_path: Path to the labels CSV file
            
        Returns:
            List of ground truth labels
        """
        import pandas as pd
        
        print(f"\nLoading ground truth labels from: {labels_path}")
        df = pd.read_csv(labels_path)
        
        # Convert to list of dictionaries
        labels = []
        for _, row in df.iterrows():
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
    
    def filter_detections_by_confidence(self, detections_data: Dict, conf_threshold: float, verbose: bool = False) -> List[Dict]:
        """
        Filter detections by confidence threshold and add filename to each detection.
        
        Args:
            detections_data: Original detection data
            conf_threshold: Confidence threshold for filtering
            verbose: If True, print filtering details
            
        Returns:
            List of filtered detections with filename field added
        """
        from pathlib import Path
        import sys
        import io
        
        # Suppress verbose output from filter unless verbose=True
        if not verbose:
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
        
        filtered = self.filter.filter_detections(detections_data, conf_threshold)
        
        if not verbose:
            sys.stdout = old_stdout
        
        # Add filename to each detection if not present
        # Get filename from top-level audio_file or audio_files field
        if 'audio_file' in detections_data:
            # Single file case
            audio_file = Path(detections_data['audio_file']).name
            for det in filtered:
                if 'filename' not in det:
                    det['filename'] = audio_file
        elif 'audio_files' in detections_data:
            # Multi-file case - each detection should have filename, but check
            for det in filtered:
                if 'filename' not in det:
                    # Try to get from file_path if available
                    if 'file_path' in det:
                        det['filename'] = Path(det['file_path']).name
                    else:
                        det['filename'] = 'unknown'
        
        return filtered
    
    def match_detections_to_labels_optimal(self, detections: List[Dict], labels: List[Dict], verbose: bool = False) -> Dict[str, Dict]:
        """
        Match detections to ground truth labels using optimal matching (Hungarian algorithm).
        Calculate metrics for each class.
        
        This method is ORDER-INDEPENDENT and finds the globally optimal matching.
        
        Args:
            detections: List of filtered detections
            labels: List of ground truth labels
            verbose: If True, print detailed matching statistics
            
        Returns:
            Dictionary with metrics for each class
        """
        # Group detections and labels by normalized filename
        detections_by_file = defaultdict(list)
        labels_by_file = defaultdict(list)
        
        for det in detections:
            filename = det.get('filename', 'unknown')
            normalized_filename = self.normalize_filename(filename)
            detections_by_file[normalized_filename].append(det)
        
        for label in labels:
            filename = label['filename']
            normalized_filename = self.normalize_filename(filename)
            labels_by_file[normalized_filename].append(label)
        
        # Initialize metrics for each class
        class_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        
        # Get all unique species
        detection_species = set(det['species'] for det in detections)
        label_species = set(label['species'] for label in labels)
        all_species = sorted(list(detection_species.union(label_species)))
        
        # Debug: Print file matching info
        all_files = set(detections_by_file.keys()).union(set(labels_by_file.keys()))
        files_with_both = set(detections_by_file.keys()).intersection(set(labels_by_file.keys()))
        
        if verbose or len(files_with_both) == 0:
            print(f"\nFile matching statistics:")
            print(f"  Total unique files (normalized): {len(all_files)}")
            print(f"  Files with detections: {len(detections_by_file)}")
            print(f"  Files with labels: {len(labels_by_file)}")
            print(f"  Files with both: {len(files_with_both)}")
            if len(files_with_both) == 0:
                print(f"  WARNING: No matching files found! Check filename formats.")
                if detections_by_file and labels_by_file:
                    print(f"  Example detection file: {list(detections_by_file.keys())[0]}")
                    print(f"  Example label file: {list(labels_by_file.keys())[0]}")
        
        # Process each file
        for filename in all_files:
            file_detections = detections_by_file.get(filename, [])
            file_labels = labels_by_file.get(filename, [])
            
            # Group by species
            detections_by_species = defaultdict(list)
            labels_by_species = defaultdict(list)
            
            for det in file_detections:
                detections_by_species[det['species']].append(det)
            
            for label in file_labels:
                labels_by_species[label['species']].append(label)
            
            # Process each species using optimal matching
            for species in all_species:
                species_detections = detections_by_species[species]
                species_labels = labels_by_species[species]
                
                if not species_detections and not species_labels:
                    continue
                
                if not species_detections:
                    # Only labels, all are FN
                    class_metrics[species]['fn'] += len(species_labels)
                    continue
                
                if not species_labels:
                    # Only detections, all are FP
                    class_metrics[species]['fp'] += len(species_detections)
                    continue
                
                # Build cost matrix for Hungarian algorithm
                n_det = len(species_detections)
                n_lab = len(species_labels)
                cost_matrix = np.full((n_det, n_lab), 1e6)  # High cost = no match
                
                for i, detection in enumerate(species_detections):
                    for j, label in enumerate(species_labels):
                        # Compute 2D IoU (time-frequency)
                        iou = compute_2d_iou(
                            detection['time_start'], detection['time_end'],
                            detection['freq_low_hz'], detection['freq_high_hz'],
                            label['start_time'], label['end_time'],
                            label['freq_low'], label['freq_high']
                        )
                        
                        # Set cost (negative IoU if above threshold)
                        if iou >= self.iou_threshold:
                            cost_matrix[i, j] = -iou
                
                # Find optimal matching
                det_indices, lab_indices = linear_sum_assignment(cost_matrix)
                
                # Count matches
                matched_labels = set()
                for i, j in zip(det_indices, lab_indices):
                    if cost_matrix[i, j] < 0:  # Valid match
                        class_metrics[species]['tp'] += 1
                        matched_labels.add(j)
                    else:
                        # Detection didn't meet threshold
                        class_metrics[species]['fp'] += 1
                
                # Unmatched detections are FP
                class_metrics[species]['fp'] += n_det - len(det_indices)
                
                # Unmatched labels are FN
                class_metrics[species]['fn'] += n_lab - len(matched_labels)
        
        return dict(class_metrics)
    
    def match_detections_to_labels(self, detections: List[Dict], labels: List[Dict], verbose: bool = False) -> Dict[str, Dict]:
        """
        Match detections to ground truth labels and calculate metrics for each class.
        
        Uses either optimal matching (Hungarian algorithm) or greedy matching based on
        the use_optimal_matching flag set during initialization.
        
        Args:
            detections: List of filtered detections
            labels: List of ground truth labels
            verbose: If True, print detailed matching statistics
            
        Returns:
            Dictionary with metrics for each class
        """
        # Dispatch to optimal or greedy matching
        if self.use_optimal_matching:
            return self.match_detections_to_labels_optimal(detections, labels, verbose)
        else:
            return self.match_detections_to_labels_greedy(detections, labels, verbose)
    
    def match_detections_to_labels_greedy(self, detections: List[Dict], labels: List[Dict], verbose: bool = False) -> Dict[str, Dict]:
        """
        Match detections to ground truth labels using greedy matching (order-dependent).
        Calculate metrics for each class.
        
        Args:
            detections: List of filtered detections
            labels: List of ground truth labels
            verbose: If True, print detailed matching statistics
            
        Returns:
            Dictionary with metrics for each class
        """
        # Group detections and labels by normalized filename (without extension)
        # This handles cases where detections use .wav but labels use .flac
        detections_by_file = defaultdict(list)
        labels_by_file = defaultdict(list)
        
        for det in detections:
            filename = det.get('filename', 'unknown')
            normalized_filename = self.normalize_filename(filename)
            detections_by_file[normalized_filename].append(det)
        
        for label in labels:
            filename = label['filename']
            normalized_filename = self.normalize_filename(filename)
            labels_by_file[normalized_filename].append(label)
        
        # Initialize metrics for each class
        class_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        
        # Get all unique species from both detections and labels
        detection_species = set(det['species'] for det in detections)
        label_species = set(label['species'] for label in labels)
        all_species = sorted(list(detection_species.union(label_species)))
        
        # Debug: Print file matching info (only when verbose or if there's an issue)
        all_files = set(detections_by_file.keys()).union(set(labels_by_file.keys()))
        files_with_both = set(detections_by_file.keys()).intersection(set(labels_by_file.keys()))
        
        if verbose or len(files_with_both) == 0:
            print(f"\nFile matching statistics:")
            print(f"  Total unique files (normalized): {len(all_files)}")
            print(f"  Files with detections: {len(detections_by_file)}")
            print(f"  Files with labels: {len(labels_by_file)}")
            print(f"  Files with both: {len(files_with_both)}")
            if len(files_with_both) == 0:
                print(f"  WARNING: No matching files found! Check filename formats.")
                if detections_by_file and labels_by_file:
                    print(f"  Example detection file: {list(detections_by_file.keys())[0]}")
                    print(f"  Example label file: {list(labels_by_file.keys())[0]}")
        
        # Process each file
        for filename in all_files:
            file_detections = detections_by_file.get(filename, [])
            file_labels = labels_by_file.get(filename, [])
            
            # Group by species
            detections_by_species = defaultdict(list)
            labels_by_species = defaultdict(list)
            
            for det in file_detections:
                detections_by_species[det['species']].append(det)
            
            for label in file_labels:
                labels_by_species[label['species']].append(label)
            
            # Process each species
            for species in all_species:
                species_detections = detections_by_species[species]
                species_labels = labels_by_species[species]
                
                # Track which labels have been matched
                matched_labels = set()
                
                # For each detection, find the best matching label
                for detection in species_detections:
                    best_match = None
                    best_iou = 0
                    
                    for label_idx, label in enumerate(species_labels):
                        if label_idx in matched_labels:
                            continue
                        
                        # Compute 2D IoU (time-frequency)
                        iou = compute_2d_iou(
                            detection['time_start'], detection['time_end'],
                            detection['freq_low_hz'], detection['freq_high_hz'],
                            label['start_time'], label['end_time'],
                            label['freq_low'], label['freq_high']
                        )
                        
                        # Check if this is a valid match using IoU threshold
                        if iou >= self.iou_threshold and iou > best_iou:
                            best_iou = iou
                            best_match = label_idx
                    
                    if best_match is not None:
                        # True positive
                        class_metrics[species]['tp'] += 1
                        matched_labels.add(best_match)
                    else:
                        # False positive
                        class_metrics[species]['fp'] += 1
                
                # Remaining unmatched labels are false negatives
                class_metrics[species]['fn'] += len(species_labels) - len(matched_labels)
        
        return dict(class_metrics)
    
    def calculate_f_beta_score(self, tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
        """
        Calculate precision, recall, and F-beta score.
        
        Args:
            tp: True positives
            fp: False positives  
            fn: False negatives
            
        Returns:
            (precision, recall, f_beta_score)
            
        Note: Returns np.nan for undefined metrics (e.g., precision when TP+FP=0).
              Use np.nanmean() when averaging across classes for macro metrics.
        """
        # Precision is undefined if there are no detections (TP + FP = 0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        # Recall is undefined if there are no labels (TP + FN = 0)
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        
        # F-beta score formula: (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)
        # Undefined if either precision or recall is undefined, or both are 0
        beta_squared = self.beta * self.beta
        if np.isnan(precision) or np.isnan(recall):
            f_beta_score = np.nan
        elif (precision + recall) > 0:
            f_beta_score = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall)
        else:
            f_beta_score = 0.0  # Both precision and recall are 0
        
        return precision, recall, f_beta_score
    
    def analyze_confidence_thresholds(self, detections_path: str, labels_path: str, 
                                    confidence_thresholds: List[float], raw_detections: bool = False) -> pd.DataFrame:
        """
        Analyze F-beta scores across different confidence thresholds for each class.
        
        If raw_detections is True, detections are assumed to be raw (unmerged) from detect_birds --no-merge.
        At each confidence threshold we filter by confidence then merge (filter-then-merge), matching the app workflow.
        
        Args:
            detections_path: Path to detections JSON file
            labels_path: Path to ground truth labels CSV file
            confidence_thresholds: List of confidence thresholds to analyze
            raw_detections: If True, input is raw (unmerged) detections; filter then merge at each threshold.
            
        Returns:
            DataFrame with F-beta scores for each class and confidence threshold
        """
        # Load data
        detections_data = self.load_detections(detections_path)
        labels = self.load_labels(labels_path)
        
        # For raw workflow: get raw list and song_gap from model_config
        raw_list = None
        song_gap_threshold = 0.1
        if raw_detections:
            if reconstruct_songs is None:
                raise RuntimeError("Raw detections mode requires inference.detect_birds.reconstruct_songs (install or run from BirdBox src).")
            raw_list = detections_data.get('detections', [])
            model_config = detections_data.get('model_config', {})
            song_gap_threshold = float(model_config.get('song_gap_threshold', 0.1))
            print(f"Raw detections mode: {len(raw_list)} raw detections, song_gap_threshold={song_gap_threshold}s")
        
        # Results storage
        results = []
        
        print(f"\nAnalyzing F-beta scores (beta={self.beta}) for {len(confidence_thresholds)} confidence thresholds...")
        
        for idx, conf_threshold in enumerate(tqdm(confidence_thresholds, desc="Processing confidence thresholds")):
            # Show verbose output for first threshold to verify everything is working
            verbose = (idx == 0)
            
            if raw_detections and raw_list is not None:
                # Filter then merge (same as app): filter raw by confidence, then merge
                filtered_raw = [d for d in raw_list if d.get('confidence', 0) >= conf_threshold]
                merged = reconstruct_songs(filtered_raw, song_gap_threshold)
                # Ensure filename on each merged segment (single-file raw has no filename on raw dets)
                if 'audio_file' in detections_data:
                    audio_file = Path(detections_data['audio_file']).name
                    for det in merged:
                        if 'filename' not in det:
                            det['filename'] = audio_file
                filtered_detections = merged
            else:
                # Legacy: detections are already merged; filter by confidence/max_confidence
                filtered_detections = self.filter_detections_by_confidence(detections_data, conf_threshold, verbose=verbose)
            
            # Calculate metrics for this confidence threshold
            class_metrics = self.match_detections_to_labels(filtered_detections, labels, verbose=verbose)
            
            # Calculate F-beta scores for each class
            for species, metrics in class_metrics.items():
                precision, recall, f_beta_score = self.calculate_f_beta_score(
                    metrics['tp'], metrics['fp'], metrics['fn']
                )
                
                results.append({
                    'species': species,
                    'confidence_threshold': conf_threshold,
                    'tp': metrics['tp'],
                    'fp': metrics['fp'],
                    'fn': metrics['fn'],
                    'precision': precision,
                    'recall': recall,
                    'f_beta_score': f_beta_score
                })
            
            # Calculate overall metrics using micro-average (sum all TP, FP, FN across classes)
            total_tp = sum(m['tp'] for m in class_metrics.values())
            total_fp = sum(m['fp'] for m in class_metrics.values())
            total_fn = sum(m['fn'] for m in class_metrics.values())
            
            micro_precision, micro_recall, micro_f_beta = self.calculate_f_beta_score(total_tp, total_fp, total_fn)
            
            results.append({
                'species': 'Overall_Micro',
                'confidence_threshold': conf_threshold,
                'tp': total_tp,
                'fp': total_fp,
                'fn': total_fn,
                'precision': micro_precision,
                'recall': micro_recall,
                'f_beta_score': micro_f_beta
            })
            
            # Calculate overall metrics using macro-average (average F-beta scores across classes)
            # Uses nanmean to properly handle undefined metrics (e.g., precision when a species has no detections)
            if class_metrics:  # Only if we have classes to average
                class_f_beta_scores = []
                class_precisions = []
                class_recalls = []
                
                for species, metrics in class_metrics.items():
                    precision, recall, f_beta_score = self.calculate_f_beta_score(
                        metrics['tp'], metrics['fp'], metrics['fn']
                    )
                    class_f_beta_scores.append(f_beta_score)
                    class_precisions.append(precision)
                    class_recalls.append(recall)
                
                # Use nanmean to exclude undefined metrics (NaN) from the average
                macro_precision = np.nanmean(class_precisions)
                macro_recall = np.nanmean(class_recalls)
                macro_f_beta = np.nanmean(class_f_beta_scores)
                
                results.append({
                    'species': 'Overall_Macro',
                    'confidence_threshold': conf_threshold,
                    'tp': None,  # Not meaningful for macro average
                    'fp': None,  # Not meaningful for macro average
                    'fn': None,  # Not meaningful for macro average
                    'precision': macro_precision,
                    'recall': macro_recall,
                    'f_beta_score': macro_f_beta
                })
        
        return pd.DataFrame(results)
    
    def plot_f_beta_curves(self, df: pd.DataFrame, output_dir: str, top_classes: int = 10):
        """
        Create visualizations of F-beta score curves.
        
        Args:
            df: DataFrame with F-beta score results
            output_dir: Directory to save plots
            top_classes: Number of top-performing classes to highlight in plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up the plotting style
        plt.style.use('default')
        if HAS_SEABORN:
            sns.set_palette("husl")
        
        # 1a. Overall F-beta score curve - Micro Average
        plt.figure(figsize=(10, 6))
        micro_data = df[df['species'] == 'Overall_Micro']
        if not micro_data.empty:
            plt.plot(micro_data['confidence_threshold'], micro_data['f_beta_score'], 
                     marker='o', linewidth=2, markersize=6, label=f'Micro F{self.beta}-Score')
            plt.plot(micro_data['confidence_threshold'], micro_data['precision'], 
                     marker='s', linewidth=2, markersize=4, alpha=0.7, label='Precision')
            plt.plot(micro_data['confidence_threshold'], micro_data['recall'], 
                     marker='^', linewidth=2, markersize=4, alpha=0.7, label='Recall')
        
        plt.xlabel('Confidence Threshold')
        plt.ylabel(f'F{self.beta}-Score')
        plt.title(f'Overall Model Performance - Micro Average')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'overall_micro_f{self.beta}_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 1b. Overall F-beta score curve - Macro Average
        plt.figure(figsize=(10, 6))
        macro_data = df[df['species'] == 'Overall_Macro']
        if not macro_data.empty:
            plt.plot(macro_data['confidence_threshold'], macro_data['f_beta_score'], 
                     marker='o', linewidth=2, markersize=6, label=f'Macro F{self.beta}-Score', color='red')
            plt.plot(macro_data['confidence_threshold'], macro_data['precision'], 
                     marker='s', linewidth=2, markersize=4, alpha=0.7, label='Precision', color='orange')
            plt.plot(macro_data['confidence_threshold'], macro_data['recall'], 
                     marker='^', linewidth=2, markersize=4, alpha=0.7, label='Recall', color='green')
        
        plt.xlabel('Confidence Threshold')
        plt.ylabel(f'F{self.beta}-Score')
        plt.title(f'Overall Model Performance - Macro Average')
        if not macro_data.empty:
            plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'overall_macro_f{self.beta}_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 1c. Comparison of Micro vs Macro Average
        plt.figure(figsize=(10, 6))
        if not micro_data.empty and not macro_data.empty:
            plt.plot(micro_data['confidence_threshold'], micro_data['f_beta_score'], 
                     marker='o', linewidth=2, markersize=6, label=f'Micro F{self.beta}-Score', color='blue')
            plt.plot(macro_data['confidence_threshold'], macro_data['f_beta_score'], 
                     marker='s', linewidth=2, markersize=6, label=f'Macro F{self.beta}-Score', color='red')
        
        plt.xlabel('Confidence Threshold')
        plt.ylabel(f'F{self.beta}-Score')
        plt.title(f'Micro vs Macro Average Comparison')
        if not micro_data.empty and not macro_data.empty:
            plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'micro_vs_macro_f{self.beta}_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. F-beta score curves for top classes
        class_data = df[~df['species'].isin(['Overall_Micro', 'Overall_Macro'])]
        
        if not class_data.empty:
            # Find best F-beta score for each class to determine top performers
            best_f_beta_per_class = class_data.groupby('species')['f_beta_score'].max().sort_values(ascending=False)
            top_class_names = best_f_beta_per_class.head(top_classes).index.tolist()
            
            plt.figure(figsize=(12, 8))
            for species in top_class_names:
                class_subset = class_data[class_data['species'] == species]
                plt.plot(class_subset['confidence_threshold'], class_subset['f_beta_score'], 
                        marker='o', linewidth=2, markersize=4, label=species)
            
            plt.xlabel('Confidence Threshold')
            plt.ylabel(f'F{self.beta}-Score')
            plt.title(f'F{self.beta}-Score Curves for Top {top_classes} Performing Species')
            if top_class_names:  # Only add legend if there are classes to show
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'top_species_f{self.beta}_curves.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. F-beta score curves for all classes
            plt.figure(figsize=(16, 10))
            all_species = class_data['species'].unique()
            
            # Use a colormap to generate distinct colors for all classes
            colors = plt.cm.tab20(np.linspace(0, 1, len(all_species)))
            
            for i, species in enumerate(all_species):
                class_subset = class_data[class_data['species'] == species]
                plt.plot(class_subset['confidence_threshold'], class_subset['f_beta_score'], 
                        linewidth=1.5, alpha=0.7, color=colors[i], label=species)
            
            plt.xlabel('Confidence Threshold')
            plt.ylabel(f'F{self.beta}-Score')
            plt.title(f'F{self.beta}-Score Curves for All Species')
            if len(all_species) > 0:  # Only add legend if there are classes to show
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            plt.grid(True, alpha=0.3)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'all_species_f{self.beta}_curves.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 4. Heatmap of F-beta scores
            pivot_data = class_data.pivot(index='species', columns='confidence_threshold', values='f_beta_score')
            
            if not pivot_data.empty:
                plt.figure(figsize=(12, max(8, len(pivot_data.index) * 0.4)))
                if HAS_SEABORN:
                    sns.heatmap(pivot_data, annot=False, cmap='viridis', cbar_kws={'label': f'F{self.beta}-Score'})
                else:
                    # Fallback to basic matplotlib heatmap
                    im = plt.imshow(pivot_data.values, cmap='viridis', aspect='auto')
                    plt.colorbar(im, label=f'F{self.beta}-Score')
                    plt.xticks(range(len(pivot_data.columns)), [f'{x:.1f}' for x in pivot_data.columns])
                    plt.yticks(range(len(pivot_data.index)), pivot_data.index)
                plt.title(f'F{self.beta}-Score Heatmap: Species vs Confidence Thresholds')
                plt.xlabel('Confidence Threshold')
                plt.ylabel('Species')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'f{self.beta}_score_heatmap.png'), dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"Plots saved to {output_dir}")
    
    def find_optimal_thresholds(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Find optimal confidence threshold for each species based on F-beta score.
        
        Args:
            df: DataFrame with F-beta score results
            
        Returns:
            DataFrame with optimal thresholds for each species
        """
        class_data = df[~df['species'].isin(['Overall_Micro', 'Overall_Macro'])]
        optimal_thresholds = []
        
        for species in class_data['species'].unique():
            class_subset = class_data[class_data['species'] == species]
            # Skip species with all-NaN f_beta_scores
            valid_scores = class_subset['f_beta_score'].dropna()
            if valid_scores.empty:
                continue
            best_idx = class_subset['f_beta_score'].idxmax(skipna=True)
            if pd.notna(best_idx):
                best_row = class_subset.loc[best_idx]
                optimal_thresholds.append(best_row)
        
        # Add overall optimal thresholds for both micro and macro
        micro_data = df[df['species'] == 'Overall_Micro']
        if not micro_data.empty:
            valid_micro = micro_data['f_beta_score'].dropna()
            if not valid_micro.empty:
                best_micro = micro_data.loc[micro_data['f_beta_score'].idxmax(skipna=True)]
                optimal_thresholds.append(best_micro)
        
        macro_data = df[df['species'] == 'Overall_Macro']
        if not macro_data.empty:
            valid_macro = macro_data['f_beta_score'].dropna()
            if not valid_macro.empty:
                best_macro = macro_data.loc[macro_data['f_beta_score'].idxmax(skipna=True)]
                optimal_thresholds.append(best_macro)
        
        return pd.DataFrame(optimal_thresholds)
    
    def print_summary(self, df: pd.DataFrame):
        """Print a summary of the F-beta analysis results."""
        print(f"\n{'='*80}")
        print("F-BETA SCORE ANALYSIS SUMMARY")
        print(f"{'='*80}")
        
        # Micro-average performance
        micro_data = df[df['species'] == 'Overall_Micro']
        if not micro_data.empty:
            valid_micro = micro_data['f_beta_score'].dropna()
            if not valid_micro.empty:
                best_micro = micro_data.loc[micro_data['f_beta_score'].idxmax(skipna=True)]
                
                print(f"\nBest Overall Performance (Micro-Average):")
                print(f"  Confidence Threshold: {best_micro['confidence_threshold']:.2f}")
                print(f"  F{self.beta}-Score: {best_micro['f_beta_score']:.4f}")
                print(f"  Precision: {best_micro['precision']:.4f}")
                print(f"  Recall: {best_micro['recall']:.4f}")
        
        # Macro-average performance
        macro_data = df[df['species'] == 'Overall_Macro']
        if not macro_data.empty:
            valid_macro = macro_data['f_beta_score'].dropna()
            if not valid_macro.empty:
                best_macro = macro_data.loc[macro_data['f_beta_score'].idxmax(skipna=True)]
                
                print(f"\nBest Overall Performance (Macro-Average):")
                print(f"  Confidence Threshold: {best_macro['confidence_threshold']:.2f}")
                print(f"  F{self.beta}-Score: {best_macro['f_beta_score']:.4f}")
                print(f"  Precision: {best_macro['precision']:.4f}")
                print(f"  Recall: {best_macro['recall']:.4f}")
        
        # Top 5 species by best F-beta score
        class_data = df[~df['species'].isin(['Overall_Micro', 'Overall_Macro'])]
        if not class_data.empty:
            # Use max with skipna to handle NaN values
            best_f_beta_per_class = class_data.groupby('species')['f_beta_score'].max().dropna().sort_values(ascending=False)
            
            if not best_f_beta_per_class.empty:
                print(f"\nTop 5 Species by F{self.beta}-Score:")
                for i, (species, f_beta_score) in enumerate(best_f_beta_per_class.head(5).items(), 1):
                    print(f"  {i}. {species}: {f_beta_score:.4f}")
                
                print(f"\nBottom 5 Species by F{self.beta}-Score:")
                for i, (species, f_beta_score) in enumerate(best_f_beta_per_class.tail(5).items(), 1):
                    print(f"  {i}. {species}: {f_beta_score:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze F-beta scores for bird call detection results across confidence thresholds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic F1-score analysis with default confidence range
  python src/evaluation/f_beta_score_analysis.py --detections detections.json --labels labels.csv
  
  # F2-score analysis (emphasizes recall)
  python src/evaluation/f_beta_score_analysis.py --detections detections.json --labels labels.csv --beta 2.0
  
  # Custom confidence range with finer steps
  python src/evaluation/f_beta_score_analysis.py --detections detections.json --labels labels.csv --conf-range 0.05 0.95 0.05
  
  # Use average confidence instead of max confidence
  python src/evaluation/f_beta_score_analysis.py --detections detections.json --labels labels.csv --use-avg-confidence
  
  # Specify custom output path
  python src/evaluation/f_beta_score_analysis.py --detections detections.json --labels labels.csv --output-path results/my_analysis
        """
    )
    
    parser.add_argument(
        '--detections',
        type=str,
        required=True,
        help='Path to the detection results JSON file (must include confidence scores for threshold analysis)'
    )
    
    parser.add_argument(
        '--labels',
        type=str,
        required=True,
        help='Path to the ground truth labels CSV file (filenames will be matched without extensions)'
    )
    
    parser.add_argument(
        '--conf-range',
        nargs=3,
        type=float,
        metavar=('MIN', 'MAX', 'STEP'),
        default=[0.01, 1.0, 0.01],
        help='Confidence threshold range: min max step (default: 0.01 1.0 0.01)'
    )
    
    parser.add_argument(
        '--beta',
        type=float,
        default=1.0,
        help='Beta parameter for F-beta score (default: 1.0 for F1-score)'
    )
    
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.5,
        help='IoU threshold for considering detections as matches (default: 0.5)'
    )
    
    parser.add_argument(
        '--use-avg-confidence',
        action='store_true',
        help='Use average confidence instead of max confidence for filtering (default: use max confidence)'
    )
    
    parser.add_argument(
        '--no-optimal-matching',
        action='store_true',
        help='Use greedy matching instead of optimal matching (Hungarian algorithm). '
             'Not recommended: greedy matching is order-dependent and suboptimal. '
             'Default: use optimal matching for reproducible results.'
    )
    
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip generating plots'
    )
    
    parser.add_argument(
        '--raw-detections',
        action='store_true',
        help='Input JSON is raw (unmerged) detections from detect_birds --no-merge. '
             'At each confidence threshold: filter by confidence then merge (matches app workflow).'
    )
    
    parser.add_argument(
        '--output-path',
        type=str,
        default='results/f_beta_score_analysis',
        help='Output directory path for results.'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.detections).exists():
        print(f"Error: Detections file not found: {args.detections}", file=sys.stderr)
        sys.exit(1)
    
    if not Path(args.labels).exists():
        print(f"Error: Labels file not found: {args.labels}", file=sys.stderr)
        sys.exit(1)
    
    if not (0.0 <= args.conf_range[0] <= args.conf_range[1] <= 1.0):
        print(f"Error: Invalid confidence range: {args.conf_range}", file=sys.stderr)
        sys.exit(1)
    
    if args.conf_range[2] <= 0:
        print(f"Error: Step size must be positive: {args.conf_range[2]}", file=sys.stderr)
        sys.exit(1)
    
    # Create analyzer
    analyzer = FBetaScoreAnalyzer(
        iou_threshold=args.iou_threshold,
        beta=args.beta,
        use_max_confidence=not args.use_avg_confidence,
        use_optimal_matching=not args.no_optimal_matching
    )
    
    # Generate confidence thresholds
    conf_thresholds = np.arange(args.conf_range[0], args.conf_range[1] + args.conf_range[2], args.conf_range[2])
    conf_thresholds = [round(t, 3) for t in conf_thresholds]
    
    print(f"Will analyze {len(conf_thresholds)} confidence thresholds: {conf_thresholds}")
    
    # Run analysis
    try:
        results_df = analyzer.analyze_confidence_thresholds(
            args.detections, args.labels, conf_thresholds, raw_detections=args.raw_detections
        )
        
        # Create output directory
        output_dir = Path(args.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving all results to: {output_dir}")
        
        # Save results to CSV
        csv_path = output_dir / f'f{args.beta}_score_analysis.csv'
        results_df.to_csv(csv_path, index=False)
        print(f"Saved results to CSV: {csv_path}")
        
        # Save as JSON
        json_path = output_dir / f'f{args.beta}_score_analysis.json'
        results_df.to_json(json_path, orient='records', indent=2)
        print(f"Saved results to JSON: {json_path}")
        
        # Find and save optimal thresholds
        optimal_df = analyzer.find_optimal_thresholds(results_df)
        optimal_csv_path = output_dir / 'optimal_thresholds.csv'
        optimal_df.to_csv(optimal_csv_path, index=False)
        print(f"Saved optimal thresholds to: {optimal_csv_path}")
        
        # Generate plots (save directly in the output directory)
        if not args.no_plot:
            analyzer.plot_f_beta_curves(results_df, str(output_dir), top_classes=12)
        
        # Print summary
        analyzer.print_summary(results_df)
        
        print("\n" + "="*80)
        print("F-BETA SCORE ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*80)
        
    except Exception as e:
        print(f"\nError during analysis: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
