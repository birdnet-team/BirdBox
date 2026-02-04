"""
Bird call detection evaluation module.

This module provides tools to evaluate the predictions of bird call detection models.
It includes functions to compute F-beta scores, confusion matrices, and other evaluation metrics.
"""

from .f_beta_score_analysis import FBetaScoreAnalyzer
from .filter_and_merge_detections import DetectionFilter
from .confusion_matrix_analysis import ConfusionMatrixAnalyzer

__all__ = ['FBetaScoreAnalyzer', 'DetectionFilter', 'ConfusionMatrixAnalyzer']
