#!/usr/bin/env python3
"""
F-beta score computation utilities for bird call detection evaluation.

This module provides functions to compute F-beta scores from confusion matrices
and detection results.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def compute_f_beta_scores(confusion_matrix: np.ndarray, class_labels: List[str], 
                         beta: float = 1.0) -> Dict[str, Dict[str, float]]:
    """
    Compute F-beta scores for each class from a confusion matrix.
    
    Args:
        confusion_matrix: Normalized confusion matrix
        class_labels: List of class labels
        beta: Beta parameter for F-beta score (default: 1.0 for F1-score)
        
    Returns:
        Dictionary with F-beta scores for each class
    """
    n_classes = len(class_labels)
    results = {}
    
    for i, species in enumerate(class_labels):
        if species == 'background':  # Skip the background class
            continue
            
        # Extract TP, FP, FN from confusion matrix
        # Matrix is [pred_idx, true_idx], so:
        # TP = confusion_matrix[i, i] (pred=species, true=species)
        # FP = sum of all predictions of this species - TP (row sum - TP)
        # FN = sum of all true instances of this species - TP (column sum - TP)
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[i, :].sum() - tp  # All predictions of species i - TP (row)
        fn = confusion_matrix[:, i].sum() - tp  # All true instances of species i - TP (column)
        
        # Compute precision, recall, and F-beta score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F-beta score formula: (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)
        if precision + recall > 0:
            f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
        else:
            f_beta = 0.0
        
        results[species] = {
            'precision': precision,
            'recall': recall,
            'f_beta': f_beta,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        }
    
    return results


def compute_macro_f_beta_score(f_beta_scores: Dict[str, Dict[str, float]], 
                              beta: float = 1.0) -> float:
    """
    Compute macro-averaged F-beta score.
    
    Args:
        f_beta_scores: Dictionary with F-beta scores for each class
        beta: Beta parameter for F-beta score
        
    Returns:
        Macro-averaged F-beta score
    """
    if not f_beta_scores:
        return 0.0
    
    f_beta_values = [scores['f_beta'] for scores in f_beta_scores.values()]
    return np.mean(f_beta_values)


def compute_weighted_f_beta_score(confusion_matrix: np.ndarray, class_labels: List[str], 
                                 beta: float = 1.0) -> float:
    """
    Compute weighted F-beta score based on class frequencies.
    
    Args:
        confusion_matrix: Normalized confusion matrix
        class_labels: List of class labels
        beta: Beta parameter for F-beta score
        
    Returns:
        Weighted F-beta score
    """
    n_classes = len(class_labels)
    
    # Get class frequencies (sum of each column = true class frequencies)
    class_frequencies = np.sum(confusion_matrix, axis=0)
    total_samples = np.sum(class_frequencies)
    
    if total_samples == 0:
        return 0.0
    
    # Compute F-beta for each class
    f_beta_scores = compute_f_beta_scores(confusion_matrix, class_labels, beta)
    
    # Weight by class frequency
    weighted_sum = 0.0
    for i, species in enumerate(class_labels):
        if species != 'background' and species in f_beta_scores:  # Skip background class
            weight = class_frequencies[i] / total_samples
            weighted_sum += weight * f_beta_scores[species]['f_beta']
    
    return weighted_sum


def compute_micro_f_beta_score(confusion_matrix: np.ndarray, class_labels: List[str], 
                              beta: float = 1.0) -> float:
    """
    Compute micro-averaged F-beta score.
    
    Args:
        confusion_matrix: Normalized confusion matrix
        class_labels: List of class labels
        beta: Beta parameter for F-beta score
        
    Returns:
        Micro-averaged F-beta score
    """
    n_classes = len(class_labels)
    
    # Aggregate TP, FP, FN across all classes
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for i, species in enumerate(class_labels):
        if species != 'background':  # Skip background class
            total_tp += confusion_matrix[i, i]
            total_fp += confusion_matrix[i, :].sum() - confusion_matrix[i, i]  # All predictions - TP (row)
            total_fn += confusion_matrix[:, i].sum() - confusion_matrix[i, i]  # All true instances - TP (column)
    
    # Compute micro-averaged precision and recall
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    
    # Compute micro F-beta score
    if micro_precision + micro_recall > 0:
        micro_f_beta = (1 + beta**2) * (micro_precision * micro_recall) / (beta**2 * micro_precision + micro_recall)
    else:
        micro_f_beta = 0.0
    
    return micro_f_beta


def print_f_beta_summary(f_beta_scores: Dict[str, Dict[str, float]], 
                        macro_f_beta: float, weighted_f_beta: float, micro_f_beta: float,
                        beta: float = 1.0):
    """
    Print a summary of F-beta scores.
    
    Args:
        f_beta_scores: Dictionary with F-beta scores for each class
        macro_f_beta: Macro-averaged F-beta score
        weighted_f_beta: Weighted F-beta score
        micro_f_beta: Micro-averaged F-beta score
        beta: Beta parameter used
    """
    print("\n" + "="*80)
    print(f"F-{beta} SCORE SUMMARY")
    print("="*80)
    
    # Per-class scores
    print(f"\nPer-class F-{beta} scores:")
    print("-" * 80)
    print(f"{'Species':<15} {'Precision':<10} {'Recall':<10} {'F-{beta}':<10} {'TP':<6} {'FP':<6} {'FN':<6}")
    print("-" * 80)
    
    for species, scores in f_beta_scores.items():
        print(f"{species:<15} {scores['precision']:<10.3f} {scores['recall']:<10.3f} "
              f"{scores['f_beta']:<10.3f} {scores['true_positives']:<6.0f} "
              f"{scores['false_positives']:<6.0f} {scores['false_negatives']:<6.0f}")
    
    # Overall scores
    print("\n" + "-" * 80)
    print("Overall F-beta scores:")
    print("-" * 80)
    print(f"Macro-averaged F-{beta}: {macro_f_beta:.3f}")
    print(f"Weighted F-{beta}:       {weighted_f_beta:.3f}")
    print(f"Micro-averaged F-{beta}: {micro_f_beta:.3f}")
    
    # Interpretation
    print(f"\nInterpretation:")
    print(f"- Macro F-{beta}: Average F-{beta} across all classes (treats all classes equally)")
    print(f"- Weighted F-{beta}: F-{beta} weighted by class frequency")
    print(f"- Micro F-{beta}: F-{beta} computed from aggregated TP/FP/FN across all classes")


def save_f_beta_results(f_beta_scores: Dict[str, Dict[str, float]], 
                       macro_f_beta: float, weighted_f_beta: float, micro_f_beta: float,
                       beta: float, output_path: str):
    """
    Save F-beta results to a CSV file.
    
    Args:
        f_beta_scores: Dictionary with F-beta scores for each class
        macro_f_beta: Macro-averaged F-beta score
        weighted_f_beta: Weighted F-beta score
        micro_f_beta: Micro-averaged F-beta score
        beta: Beta parameter used
        output_path: Path to save the results
    """
    # Prepare data for CSV
    data = []
    
    # Per-class results
    for species, scores in f_beta_scores.items():
        data.append({
            'Species': species,
            'Precision': scores['precision'],
            'Recall': scores['recall'],
            f'F_{beta}': scores['f_beta'],
            'True_Positives': scores['true_positives'],
            'False_Positives': scores['false_positives'],
            'False_Negatives': scores['false_negatives']
        })
    
    # Overall results
    data.append({
        'Species': 'MACRO_AVERAGE',
        'Precision': np.nan,
        'Recall': np.nan,
        f'F_{beta}': macro_f_beta,
        'True_Positives': np.nan,
        'False_Positives': np.nan,
        'False_Negatives': np.nan
    })
    
    data.append({
        'Species': 'WEIGHTED_AVERAGE',
        'Precision': np.nan,
        'Recall': np.nan,
        f'F_{beta}': weighted_f_beta,
        'True_Positives': np.nan,
        'False_Positives': np.nan,
        'False_Negatives': np.nan
    })
    
    data.append({
        'Species': 'MICRO_AVERAGE',
        'Precision': np.nan,
        'Recall': np.nan,
        f'F_{beta}': micro_f_beta,
        'True_Positives': np.nan,
        'False_Positives': np.nan,
        'False_Negatives': np.nan
    })
    
    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"\nF-beta results saved to: {output_path}")
