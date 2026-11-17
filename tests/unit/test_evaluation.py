# SPDX-FileCopyrightText: 2026 BirdNET-Team
# SPDX-License-Identifier: MIT
"""Matching detections to labels (F-beta and confusion matrix tools) and the species mappings."""

import pytest

import config
from evaluation.utils.confusion_matrix import (
    compute_1d_iou,
    compute_2d_iou,
    match_detections_to_labels_optimal,
)


def box(t0, t1, f0, f1, filename='rec.wav', species='bird'):
    """A detection."""
    return {'filename': filename, 'species': species, 'time_start': t0, 'time_end': t1,
            'freq_low_hz': f0, 'freq_high_hz': f1}


def label(t0, t1, f0, f1, filename='rec.wav', species='bird'):
    """A label, with the fields of the annotation CSV."""
    return {'filename': filename, 'species': species, 'start_time': t0, 'end_time': t1,
            'freq_low': f0, 'freq_high': f1}


def test_iou():
    assert compute_1d_iou(0, 2, 1, 3) == pytest.approx(1 / 3)
    assert compute_1d_iou(0, 1, 2, 3) == 0.0
    assert compute_2d_iou(0, 2, 0, 2, 1, 3, 1, 3) == pytest.approx(1 / 7)
    assert compute_2d_iou(0, 1, 0, 1, 0, 1, 0, 1) == 1.0
    assert compute_2d_iou(0, 1, 0, 1, 0, 1, 2, 3) == 0.0


def test_optimal_matching_maximises_total_iou():
    # Detection 0 overlaps both labels and fits label 0 best (IoU 0.53 against 0.41), but
    # detection 1 only fits label 0 (0.8). Pairing each with a different label gives the
    # larger total IoU, so detection 0 goes to label 1.
    labels = [label(0.0, 1.0, 1000, 2000), label(1.0, 2.0, 1000, 2000)]
    detections = [box(0.1, 1.7, 1000, 2000), box(0.0, 0.8, 1000, 2000)]
    matches, unmatched_dets, unmatched_labels = match_detections_to_labels_optimal(
        detections, labels, iou_threshold=0.3)
    assert sorted((d, l) for d, l, _iou in matches) == [(0, 1), (1, 0)]
    assert unmatched_dets == [] and unmatched_labels == []


def test_matching_respects_files_and_threshold():
    labels = [label(0, 1, 1000, 2000, filename='a.wav')]
    detections = [box(0, 1, 1000, 2000, filename='b.wav'), box(0.9, 2, 1000, 2000, filename='a.wav')]
    matches, unmatched_dets, unmatched_labels = match_detections_to_labels_optimal(
        detections, labels, iou_threshold=0.25)
    assert matches == []
    assert sorted(unmatched_dets) == [0, 1] and unmatched_labels == [0]


def test_species_mappings():
    just_bird = config.get_species_mapping('Just-Bird')
    assert just_bird['id_to_ebird'] == {0: 'bird'}
    labels = config.get_species_mapping('Birds-With-Labels')
    ids = labels['id_to_ebird']
    assert sorted(ids) == list(range(633))
    assert ids[0] == '????'
    assert all(code in labels['ebird_to_name'] for code in ids.values())
    assert set(labels['bird_colors']) == set(ids)
    assert (labels['clip_length'], labels['height_width']) == (3, 256)
    # A copy: changing it leaves the configuration alone
    labels['id_to_ebird'][0] = 'changed'
    assert config.get_species_mapping('Birds-With-Labels')['id_to_ebird'][0] == '????'
    with pytest.raises(ValueError):
        config.get_species_mapping('All-In-One')


@pytest.mark.parametrize('path, mapping', [
    ('models/Just-Bird_fp32.pt', 'Just-Bird'),
    ('docs/models/Just-Bird_fp16.onnx', 'Just-Bird'),
    ('Birds-With-Labels_int8.tflite', 'Birds-With-Labels'),
    ('/x/birds_with_labels.engine', 'Birds-With-Labels'),
])
def test_mapping_from_the_model_file_name(path, mapping):
    assert config.get_species_mapping_for_model(path) == mapping


def test_unknown_model_file_name():
    with pytest.raises(ValueError):
        config.get_species_mapping_for_model('models/All-In-One.pt')
