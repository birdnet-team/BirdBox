# SPDX-FileCopyrightText: 2025-2026 BirdNET-Team
# SPDX-License-Identifier: MIT
"""
Bird call detection inference module.

This module provides tools to detect bird calls in arbitrary-length audio files
using trained YOLO models.
"""

from .detect_birds import BirdCallDetector, merge_detections, reconstruct_songs
from .utils import log_floor

__all__ = ['BirdCallDetector', 'merge_detections', 'reconstruct_songs', 'log_floor']
