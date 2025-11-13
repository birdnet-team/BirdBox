#!/usr/bin/env python3
"""
Confusion matrix computation utilities for bird call detection evaluation.

This module provides functions to match detections to ground truth labels,
compute confusion matrices, and visualize the results.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict
from scipy.optimize import linear_sum_assignment


def compute_1d_iou(start1: float, end1: float, start2: float, end2: float) -> float:
    """
    Compute 1D Intersection over Union (IoU) for time intervals.
    
    Args:
        start1: Start time of first interval
        end1: End time of first interval
        start2: Start time of second interval
        end2: End time of second interval
        
    Returns:
        IoU value between 0 and 1
    """
    # Calculate intersection
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection = max(0, intersection_end - intersection_start)
    
    # Calculate union
    union_start = min(start1, start2)
    union_end = max(end1, end2)
    union = union_end - union_start
    
    if union == 0:
        return 0.0
    
    return intersection / union


def compute_2d_iou(time_start1: float, time_end1: float, freq_low1: float, freq_high1: float,
                   time_start2: float, time_end2: float, freq_low2: float, freq_high2: float) -> float:
    """
    Compute 2D Intersection over Union (IoU) for time-frequency boxes.
    
    Args:
        time_start1: Start time of first box
        time_end1: End time of first box
        freq_low1: Low frequency of first box
        freq_high1: High frequency of first box
        time_start2: Start time of second box
        time_end2: End time of second box
        freq_low2: Low frequency of second box
        freq_high2: High frequency of second box
        
    Returns:
        IoU value between 0 and 1
    """
    # Compute time overlap
    time_intersection_start = max(time_start1, time_start2)
    time_intersection_end = min(time_end1, time_end2)
    time_intersection = max(0, time_intersection_end - time_intersection_start)
    
    # Compute frequency overlap
    freq_intersection_low = max(freq_low1, freq_low2)
    freq_intersection_high = min(freq_high1, freq_high2)
    freq_intersection = max(0, freq_intersection_high - freq_intersection_low)
    
    # Compute intersection area
    intersection_area = time_intersection * freq_intersection
    
    # Compute areas of both boxes
    area1 = (time_end1 - time_start1) * (freq_high1 - freq_low1)
    area2 = (time_end2 - time_start2) * (freq_high2 - freq_low2)
    
    # Compute union area
    union_area = area1 + area2 - intersection_area
    
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area


def has_time_overlap(start1: float, end1: float, start2: float, end2: float) -> bool:
    """
    Check if two time intervals overlap.
    
    Args:
        start1: Start time of first interval
        end1: End time of first interval
        start2: Start time of second interval
        end2: End time of second interval
        
    Returns:
        True if intervals overlap, False otherwise
    """
    return start1 < end2 and start2 < end1


def match_detections_to_labels_optimal(detections: List[Dict], labels: List[Dict],
                                       iou_threshold: float = 0.5,
                                       use_2d_iou: bool = True) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """
    Match detections to ground truth labels using optimal bipartite matching (Hungarian algorithm).
    
    This method finds the globally optimal matching that maximizes the total IoU across all matches.
    Unlike the greedy approach, this is ORDER-INDEPENDENT and gives consistent results.
    
    OPTIMIZED VERSION: Groups by filename for efficiency.
    
    Args:
        detections: List of detection dictionaries
        labels: List of ground truth label dictionaries
        iou_threshold: Minimum IoU to consider a match
        use_2d_iou: If True, use 2D IoU (time-frequency), otherwise use 1D IoU (time only)
        
    Returns:
        Tuple of (matches, unmatched_detections, unmatched_labels)
        - matches: List of (detection_idx, label_idx, iou) tuples
        - unmatched_detections: List of detection indices with no match
        - unmatched_labels: List of label indices with no match
    """
    from pathlib import Path
    
    matches = []
    matched_detections = set()
    matched_labels = set()
    
    # Group labels by filename for efficiency
    labels_by_file = defaultdict(list)
    print("Grouping labels by filename...")
    for label_idx, label in enumerate(labels):
        filename = Path(label['filename']).stem
        labels_by_file[filename].append((label_idx, label))
    
    # Group detections by filename
    detections_by_file = defaultdict(list)
    print("Grouping detections by filename...")
    for det_idx, detection in enumerate(detections):
        filename = Path(detection['filename']).stem
        detections_by_file[filename].append((det_idx, detection))
    
    print(f"Processing {len(detections_by_file)} unique files with optimal matching...")
    
    # Process each file separately
    file_count = 0
    for filename in detections_by_file.keys():
        file_count += 1
        if file_count % 100 == 0:
            print(f"  Processed {file_count}/{len(detections_by_file)} files...")
        
        file_detections = detections_by_file[filename]
        file_labels = labels_by_file.get(filename, [])
        
        if not file_labels:
            continue
        
        n_det = len(file_detections)
        n_lab = len(file_labels)
        
        # Build cost matrix (negative IoU for maximization, since linear_sum_assignment minimizes)
        cost_matrix = np.full((n_det, n_lab), 1e6)  # High cost = no match
        
        for i, (det_idx, detection) in enumerate(file_detections):
            det_start = detection.get('time_start', detection.get('start_time'))
            det_end = detection.get('time_end', detection.get('end_time'))
            
            for j, (label_idx, label) in enumerate(file_labels):
                # Compute IoU
                if use_2d_iou:
                    iou = compute_2d_iou(
                        det_start, det_end,
                        detection.get('freq_low_hz', detection.get('freq_low')),
                        detection.get('freq_high_hz', detection.get('freq_high')),
                        label.get('start_time'),
                        label.get('end_time'),
                        label.get('freq_low'),
                        label.get('freq_high')
                    )
                else:
                    iou = compute_1d_iou(
                        det_start, det_end,
                        label.get('start_time'),
                        label.get('end_time')
                    )
                
                # Set cost (negative IoU if above threshold, high cost otherwise)
                if iou >= iou_threshold:
                    cost_matrix[i, j] = -iou
        
        # Find optimal matching using Hungarian algorithm
        det_indices, lab_indices = linear_sum_assignment(cost_matrix)
        
        # Extract valid matches (those that meet the threshold)
        for i, j in zip(det_indices, lab_indices):
            if cost_matrix[i, j] < 0:  # Valid match (IoU >= threshold)
                det_idx = file_detections[i][0]
                label_idx = file_labels[j][0]
                iou = -cost_matrix[i, j]
                matches.append((det_idx, label_idx, iou))
                matched_detections.add(det_idx)
                matched_labels.add(label_idx)
    
    print(f"Matched {len(matches)} detection-label pairs")
    
    # Find unmatched detections and labels
    unmatched_detections = [i for i in range(len(detections)) if i not in matched_detections]
    unmatched_labels = [i for i in range(len(labels)) if i not in matched_labels]
    
    print(f"Unmatched detections: {len(unmatched_detections)}")
    print(f"Unmatched labels: {len(unmatched_labels)}")
    
    return matches, unmatched_detections, unmatched_labels


def match_detections_to_labels(detections: List[Dict], labels: List[Dict], 
                               iou_threshold: float = 0.5, 
                               use_2d_iou: bool = True) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """
    Match detections to ground truth labels using IoU.
    
    OPTIMIZED VERSION: Groups by filename and uses early exit conditions.
    
    Args:
        detections: List of detection dictionaries
        labels: List of ground truth label dictionaries
        iou_threshold: Minimum IoU to consider a match
        use_2d_iou: If True, use 2D IoU (time-frequency), otherwise use 1D IoU (time only)
        
    Returns:
        Tuple of (matches, unmatched_detections, unmatched_labels)
        - matches: List of (detection_idx, label_idx, iou) tuples
        - unmatched_detections: List of detection indices with no match
        - unmatched_labels: List of label indices with no match
    """
    from pathlib import Path
    
    matches = []
    matched_detections = set()
    matched_labels = set()
    
    # OPTIMIZATION: Group labels by filename for fast lookup
    # This avoids comparing detections and labels from different files
    labels_by_file = defaultdict(list)
    
    print("Grouping labels by filename...")
    for label_idx, label in enumerate(labels):
        filename = Path(label['filename']).stem
        labels_by_file[filename].append((label_idx, label))
    
    print(f"Processing {len(detections)} detections...")
    
    # Process detections in their original order to maintain deterministic behavior
    # This ensures results match the original non-optimized implementation
    for det_idx, detection in enumerate(detections):
        if det_idx % 1000 == 0 and det_idx > 0:
            print(f"  Processed {det_idx}/{len(detections)} detections...")
        
        if det_idx in matched_detections:
            continue
        
        # Get filename for this detection
        filename = Path(detection['filename']).stem
        
        # Get labels for the same file
        file_labels = labels_by_file.get(filename, [])
        
        if not file_labels:
            # No labels for this file, detection is unmatched
            continue
        
        best_match = None
        best_iou = 0
        
        # Get detection time bounds
        det_start = detection.get('time_start', detection.get('start_time'))
        det_end = detection.get('time_end', detection.get('end_time'))
        
        # Only compare with labels from the same file
        for label_idx, label in file_labels:
            if label_idx in matched_labels:
                continue
            
            # Compute IoU
            if use_2d_iou:
                iou = compute_2d_iou(
                    det_start, det_end,
                    detection.get('freq_low_hz', detection.get('freq_low')), 
                    detection.get('freq_high_hz', detection.get('freq_high')),
                    label.get('start_time'), 
                    label.get('end_time'),
                    label.get('freq_low'), 
                    label.get('freq_high')
                )
            else:
                iou = compute_1d_iou(
                    det_start, det_end, 
                    label.get('start_time'), 
                    label.get('end_time')
                )
            
            # Check if this is a valid match
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_match = label_idx
        
        if best_match is not None:
            matches.append((det_idx, best_match, best_iou))
            matched_detections.add(det_idx)
            matched_labels.add(best_match)
    
    print(f"Matched {len(matches)} detection-label pairs")
    
    # Find unmatched detections and labels
    unmatched_detections = [i for i in range(len(detections)) if i not in matched_detections]
    unmatched_labels = [i for i in range(len(labels)) if i not in matched_labels]
    
    print(f"Unmatched detections: {len(unmatched_detections)}")
    print(f"Unmatched labels: {len(unmatched_labels)}")
    
    return matches, unmatched_detections, unmatched_labels


def build_confusion_matrix(detections: List[Dict], labels: List[Dict], 
                          species_list: List[str],
                          iou_threshold: float = 0.5,
                          use_2d_iou: bool = True,
                          include_background: bool = True,
                          use_optimal_matching: bool = False) -> np.ndarray:
    """
    Build a confusion matrix from detections and labels.
    
    Args:
        detections: List of detection dictionaries
        labels: List of ground truth label dictionaries
        species_list: List of species codes in order
        iou_threshold: Minimum IoU to consider a match
        use_2d_iou: If True, use 2D IoU (time-frequency), otherwise use 1D IoU (time only)
        include_background: If True, include background class for FP/FN
        use_optimal_matching: If True, use Hungarian algorithm (optimal, order-independent).
                            If False, use greedy matching (faster, order-dependent).
        
    Returns:
        Confusion matrix as numpy array of shape (n_classes, n_classes)
        Rows are predictions, columns are ground truth
        If include_background=True, last row/column is background
    """
    # Create species to index mapping
    species_to_idx = {species: idx for idx, species in enumerate(species_list)}
    n_classes = len(species_list)
    
    if include_background:
        n_classes += 1
        background_idx = n_classes - 1
    
    # Initialize confusion matrix
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    # Match detections to labels
    if use_optimal_matching:
        print("\nUsing optimal matching (Hungarian algorithm)...")
        matches, unmatched_detections, unmatched_labels = match_detections_to_labels_optimal(
            detections, labels, iou_threshold, use_2d_iou
        )
    else:
        print("\nUsing greedy matching...")
        matches, unmatched_detections, unmatched_labels = match_detections_to_labels(
            detections, labels, iou_threshold, use_2d_iou
        )
    
    # Process matches (true positives or class confusion)
    for det_idx, label_idx, iou in matches:
        pred_species = detections[det_idx].get('species')
        true_species = labels[label_idx].get('species')
        
        if pred_species in species_to_idx and true_species in species_to_idx:
            pred_idx = species_to_idx[pred_species]
            true_idx = species_to_idx[true_species]
            confusion_matrix[pred_idx, true_idx] += 1
    
    # Process unmatched detections (false positives)
    if include_background:
        for det_idx in unmatched_detections:
            pred_species = detections[det_idx].get('species')
            if pred_species in species_to_idx:
                pred_idx = species_to_idx[pred_species]
                confusion_matrix[pred_idx, background_idx] += 1
    
    # Process unmatched labels (false negatives)
    if include_background:
        for label_idx in unmatched_labels:
            true_species = labels[label_idx].get('species')
            if true_species in species_to_idx:
                true_idx = species_to_idx[true_species]
                confusion_matrix[background_idx, true_idx] += 1
    
    return confusion_matrix


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


def print_confusion_matrix(confusion_matrix: np.ndarray, class_labels: List[str], 
                          include_background: bool = True):
    """
    Print a formatted confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix as numpy array
        class_labels: List of class labels
        include_background: If True, include background class
    """
    # Add background to labels if needed
    labels = class_labels.copy()
    if include_background:
        labels.append('background')
    
    # Calculate column widths
    max_label_len = max(len(label) for label in labels)
    col_width = max(max_label_len, 8)
    
    print("\n" + "="*100)
    print("CONFUSION MATRIX")
    print("="*100)
    print("Rows: Predicted class")
    print("Columns: True class")
    print("-"*100)
    
    # Print header
    print(f"{'':>{col_width}} | ", end="")
    for label in labels:
        print(f"{label:>{col_width}}", end=" ")
    print()
    print("-"*100)
    
    # Print rows
    for i, label in enumerate(labels):
        print(f"{label:>{col_width}} | ", end="")
        for j in range(len(labels)):
            print(f"{confusion_matrix[i, j]:>{col_width}}", end=" ")
        print()
    
    print("="*100)


def save_confusion_matrix(confusion_matrix: np.ndarray, class_labels: List[str], 
                         output_path: str, include_background: bool = True):
    """
    Save confusion matrix to a CSV file.
    
    Args:
        confusion_matrix: Confusion matrix as numpy array
        class_labels: List of class labels
        output_path: Path to save the CSV file
        include_background: If True, include background class
    """
    # Add background to labels if needed
    labels = class_labels.copy()
    if include_background:
        labels.append('background')
    
    # Create DataFrame
    df = pd.DataFrame(confusion_matrix, index=labels, columns=labels)
    
    # Save to CSV
    df.to_csv(output_path)
    print(f"\nConfusion matrix saved to: {output_path}")


def plot_confusion_matrix(confusion_matrix: np.ndarray, class_labels: List[str],
                         output_path: str, include_background: bool = True,
                         normalize: bool = True, figsize: tuple = (10, 8)):
    """
    Plot and save confusion matrix as a PNG file with a heatmap visualization.
    
    Args:
        confusion_matrix: Confusion matrix as numpy array
        class_labels: List of class labels
        output_path: Path to save the PNG file
        include_background: If True, include background class
        normalize: If True, normalize the confusion matrix by row (predicted class)
        figsize: Figure size as (width, height) tuple
    """
    import matplotlib.pyplot as plt
    
    # Try to import seaborn for better styling
    try:
        import seaborn as sns
        has_seaborn = True
    except ImportError:
        has_seaborn = False
    
    # Add background to labels if needed
    labels = class_labels.copy()
    if include_background:
        labels.append('background')
    
    # Normalize if requested
    if normalize:
        # Normalize by column (each true class sums to 1)
        # This shows: for each true class, what percentage was predicted as each class
        col_sums = confusion_matrix.sum(axis=0, keepdims=True)
        # Avoid division by zero
        col_sums[col_sums == 0] = 1
        cm_normalized = confusion_matrix.astype(float) / col_sums
    else:
        cm_normalized = confusion_matrix.astype(float)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    if has_seaborn:
        sns.heatmap(cm_normalized, annot=True, fmt='.2f' if normalize else '.0f',
                   cmap='Blues', xticklabels=labels, yticklabels=labels,
                   cbar=True, square=True, ax=ax, vmin=0, vmax=1 if normalize else None)
    else:
        im = ax.imshow(cm_normalized, cmap='Blues', aspect='auto', 
                      vmin=0, vmax=1 if normalize else None)
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Add text annotations
        for i in range(len(labels)):
            for j in range(len(labels)):
                if normalize:
                    text = f'{cm_normalized[i, j]:.2f}'
                else:
                    text = f'{int(cm_normalized[i, j])}'
                ax.text(j, i, text, ha='center', va='center',
                       color='white' if cm_normalized[i, j] > 0.5 else 'black')
        
        # Set ticks
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    
    # Set labels and title
    ax.set_xlabel('True Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Class', fontsize=12, fontweight='bold')
    
    title = 'Confusion Matrix'
    if normalize:
        title += ' (Normalized)'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Rotate x-axis labels if needed
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix plot saved to: {output_path}")

