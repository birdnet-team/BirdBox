# SPDX-FileCopyrightText: 2026 BirdNET-Team
# SPDX-License-Identifier: MIT
"""Shared helpers for the unit tests. pytest.ini puts src/ on the import path."""

import pytest


def make_det(time_start, time_end, freq_low, freq_high, confidence=0.5, species='bird',
             species_id=0, clip_start=None, clip_end=None, **extra):
    """A raw detection as BirdCallDetector produces it (clip_* only when given)."""
    det = {
        'species': species,
        'species_id': species_id,
        'time_start': float(time_start),
        'time_end': float(time_end),
        'freq_low_hz': float(freq_low),
        'freq_high_hz': float(freq_high),
        'confidence': float(confidence),
    }
    if clip_start is not None:
        det['clip_start'] = float(clip_start)
        det['clip_end'] = float(clip_end if clip_end is not None else clip_start + 3.0)
    det.update(extra)
    return det


@pytest.fixture
def det():
    return make_det
