# SPDX-FileCopyrightText: 2025-2026 BirdNET-Team
# SPDX-License-Identifier: MIT
"""
Bird call detection inference module.

This module provides tools to detect bird calls in arbitrary-length audio files
using trained YOLO models.
"""

from .detect_birds import BirdCallDetector, merge_detections, reconstruct_songs
from .utils import pcen_inference

__all__ = ['BirdCallDetector', 'merge_detections', 'reconstruct_songs', 'pcen_inference']
