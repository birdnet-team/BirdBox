"""
Bird call detection inference module.

This module provides tools to detect bird calls in arbitrary-length audio files
using trained YOLO models.
"""

from .detect_birds import BirdCallDetector
from .utils import pcen_inference

__all__ = ['BirdCallDetector', 'pcen_inference']

