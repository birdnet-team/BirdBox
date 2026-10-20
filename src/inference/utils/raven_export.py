# SPDX-FileCopyrightText: 2026 BirdNET-Team
# SPDX-License-Identifier: MIT
"""
Helpers for exporting BirdBox detections as Raven Selection Tables.

A Raven table has no filename column, so detections from several recordings
are written as one table per recording (``raven/<filename>.txt``). Times keep
millisecond precision: calls are often shorter than 0.3 s, and one decimal
would move their boxes by up to 50 ms on each side.
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional

RAVEN_FIELDNAMES = [
    'Selection',
    'View',
    'Channel',
    'Begin Time (S)',
    'End Time (S)',
    'Low Freq (Hz)',
    'High Freq (Hz)',
    'Annotation',
]


def build_raven_rows(detections: List[Dict]) -> List[Dict]:
    """Raven Selection Table rows for the detections of one recording, sorted by start time."""
    return [
        {
            'Selection': index,
            'View': 'Spectrogram 1',
            'Channel': 1,
            'Begin Time (S)': f"{det['time_start']:.3f}",
            'End Time (S)': f"{det['time_end']:.3f}",
            'Low Freq (Hz)': det['freq_low_hz'],
            'High Freq (Hz)': det['freq_high_hz'],
            'Annotation': det['species'],
        }
        for index, det in enumerate(sorted(detections, key=lambda x: x['time_start']), start=1)
    ]


def _write_table(detections: List[Dict], path: Path) -> None:
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=RAVEN_FIELDNAMES, delimiter='\t')
        writer.writeheader()
        writer.writerows(build_raven_rows(detections))


def save_raven_tables(
    detections: List[Dict],
    single_file_path: Path,
    multi_file_dir: Path,
    multi_file: Optional[bool] = None,
) -> Path:
    """
    Write detections as Raven Selection Tables.

    A multi-file run is written as one table per recording,
    ``multi_file_dir/<filename>.txt``; a single-file run as one table at
    ``single_file_path``. ``multi_file`` None treats the run as multi-file when
    any detection carries a ``filename``.

    Returns:
        The directory (multi-file) or the file (single file) that was written.
    """
    if multi_file is None:
        multi_file = any('filename' in det for det in detections)
    if not multi_file:
        _write_table(detections, Path(single_file_path))
        return Path(single_file_path)

    multi_file_dir = Path(multi_file_dir)
    multi_file_dir.mkdir(parents=True, exist_ok=True)
    grouped: Dict[str, List[Dict]] = {}
    for det in detections:
        grouped.setdefault(det.get('filename', 'unknown'), []).append(det)
    for source_filename, group_detections in grouped.items():
        _write_table(group_detections, multi_file_dir / f"{source_filename}.txt")
    return multi_file_dir
