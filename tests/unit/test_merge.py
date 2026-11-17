# SPDX-FileCopyrightText: 2026 BirdNET-Team
# SPDX-License-Identifier: MIT
"""Merging the boxes of overlapping windows: the settings, the song merge and window fusion."""

import numpy as np
import pytest

from inference import detect_birds as db
from conftest import make_det


# --- Settings per species mapping -------------------------------------------------------

def test_frequency_iou():
    assert db._frequency_iou(100, 200, 100, 200) == 1.0
    assert db._frequency_iou(100, 200, 300, 400) == 0.0
    assert db._frequency_iou(100, 200, 200, 300) == 0.0          # touching bands do not overlap
    assert db._frequency_iou(100, 200, 100, 300) == pytest.approx(0.5)
    assert db._frequency_iou(100, 100, 100, 100) == 0.0          # zero width


def test_defaults_per_mapping():
    assert db.resolve_merge_method('Just-Bird', None) == 'fusion'
    assert db.resolve_merge_method('Birds-With-Labels', 'auto') == 'reconstruct'
    assert db.resolve_merge_method('Birds-With-Labels', 'fusion') == 'fusion'
    with pytest.raises(ValueError):
        db.resolve_merge_method('Just-Bird', 'nms')

    assert db.resolve_conf('Just-Bird') == 0.15
    assert db.resolve_conf('Birds-With-Labels') == 0.2
    assert db.resolve_conf('Just-Bird', 0.3) == 0.3

    assert db.resolve_phrase_gap('Just-Bird', 'fusion', None) == 0.05
    assert db.resolve_phrase_gap('Birds-With-Labels', 'fusion', None) == 0.0
    assert db.resolve_phrase_gap('Just-Bird', 'fusion', -1) == 0.0
    assert db.resolve_phrase_gap('Just-Bird', 'reconstruct', 0.3) is None

    assert db.resolve_freq_iou_threshold('Just-Bird', None) == 0.6
    assert db.resolve_freq_iou_threshold('Just-Bird', -1) is None    # negative turns the gate off
    assert db.resolve_freq_iou_threshold('Just-Bird', 0.3) == 0.3
    assert db.resolve_freq_iou_threshold('Birds-With-Labels', 0.3) is None


def test_merge_settings_from_model_config():
    # A raw detections JSON of Just-Bird: fusion with the default phrase gap
    settings = db.resolve_merge_settings({'species_mapping': 'Just-Bird'})
    assert settings == {'merge_method': 'fusion', 'song_gap_threshold': 0.1,
                        'freq_iou_threshold': 0.6, 'phrase_gap': 0.05}
    # Values stored in the JSON win over the defaults, explicit arguments over both
    stored = {'species_mapping': 'Just-Bird', 'merge_method': 'reconstruct', 'song_gap_threshold': 0.3}
    assert db.resolve_merge_settings(stored)['merge_method'] == 'reconstruct'
    assert db.resolve_merge_settings(stored)['song_gap_threshold'] == 0.3
    assert db.resolve_merge_settings(stored, song_gap=0.5, merge_method='fusion')['merge_method'] == 'fusion'
    assert db.resolve_merge_settings({'species_mapping': 'Birds-With-Labels'})['phrase_gap'] is None


# --- The song merge (reconstruct) -------------------------------------------------------

def test_song_merge_joins_within_the_gap():
    dets = [make_det(0.0, 1.0, 1000, 2000, 0.4), make_det(1.05, 2.0, 1000, 2000, 0.8),
            make_det(2.5, 3.0, 1000, 2000, 0.6)]
    songs = db.reconstruct_songs(dets, song_gap_threshold=0.1)
    assert [(s['time_start'], s['time_end']) for s in songs] == [(0.0, 2.0), (2.5, 3.0)]
    first = songs[0]
    assert first['detections_merged'] == 2
    assert first['max_confidence'] == 0.8
    assert first['avg_confidence'] == pytest.approx(0.6)


def test_song_merge_keeps_species_and_files_apart():
    dets = [make_det(0.0, 1.0, 1000, 2000), make_det(0.5, 1.5, 1000, 2000, species='amro', species_id=3),
            make_det(0.5, 1.5, 1000, 2000, filename='b.wav')]
    songs = db.reconstruct_songs(dets, song_gap_threshold=1.0)
    assert len(songs) == 3
    assert {s.get('filename') for s in songs} == {None, 'b.wav'}


def test_song_merge_with_frequency_gate_keeps_bands_apart():
    # A low and a high call at the same time: one box without the gate, two with it
    dets = [make_det(0.0, 0.5, 1000, 2000), make_det(0.1, 0.6, 6000, 8000)]
    assert len(db.reconstruct_songs(dets, 0.1)) == 1
    gated = db.reconstruct_songs(dets, 0.1, freq_iou_threshold=0.6)
    assert sorted((s['freq_low_hz'], s['freq_high_hz']) for s in gated) == [(1000, 2000), (6000, 8000)]


def test_song_merge_of_nothing():
    assert db.reconstruct_songs([], 0.1) == []
    assert db.fuse_window_detections([]) == []


# --- Window fusion -----------------------------------------------------------------------

def test_fusion_keeps_a_call_from_its_most_central_window():
    # Windows 0-3 s and 1.5-4.5 s of a 6 s file both see a call at 2.0-2.4 s (centre 2.2 s).
    # Window 0 owns everything before 2.25 s, so only its box is kept.
    first = make_det(2.0, 2.4, 3000, 5000, 0.5, clip_start=0.0)
    second = make_det(1.95, 2.45, 2800, 5200, 0.9, clip_start=1.5)
    fused = db.fuse_window_detections([first, second], file_end=6.0)
    assert len(fused) == 1
    assert (fused[0]['time_start'], fused[0]['time_end']) == (2.0, 2.4)
    assert fused[0]['max_confidence'] == 0.5


def test_fusion_owns_file_start_and_end():
    # A box in the first quarter of the first window, and one in the last quarter of the last
    start = make_det(0.1, 0.3, 3000, 5000, clip_start=0.0)
    end = make_det(5.7, 5.9, 3000, 5000, clip_start=3.0)
    assert len(db.fuse_window_detections([start, end], file_end=6.0)) == 2
    # Not the last window: its last quarter belongs to the next one
    assert db.fuse_window_detections([end], file_end=9.0) == []


def test_fusion_averages_near_duplicates_of_one_window():
    a = make_det(1.0, 2.0, 3000, 5000, 0.9, clip_start=0.0)
    b = make_det(1.1, 2.1, 3100, 5100, 0.3, clip_start=0.0)
    fused = db.fuse_window_detections([a, b], file_end=3.0)
    assert len(fused) == 1
    box = fused[0]
    assert box['time_start'] == pytest.approx((0.9 * 1.0 + 0.3 * 1.1) / 1.2)
    assert box['max_confidence'] == 0.9
    assert box['detections_merged'] == 2


def test_fusion_drops_a_box_inside_a_larger_one():
    phrase = make_det(0.5, 2.5, 2000, 6000, 0.6, clip_start=0.0)
    syllable = make_det(1.0, 1.3, 3000, 4000, 0.8, clip_start=0.0)
    fused = db.fuse_window_detections([phrase, syllable], file_end=3.0)
    assert [(f['time_start'], f['time_end']) for f in fused] == [(0.5, 2.5)]


def test_fusion_joins_a_call_split_across_windows():
    # Two boxes of one band that overlap by 0.1 s: one call, joined at phrase gap 0
    a = make_det(1.5, 2.2, 3000, 5000, 0.7, clip_start=0.0)
    b = make_det(2.1, 2.9, 3000, 5000, 0.5, clip_start=1.5)
    assert len(db.fuse_window_detections([a, b], phrase_gap=0, file_end=4.5)) == 1


def test_phrase_gap_joins_close_calls_of_one_band_only():
    calls = [make_det(0.2, 0.4, 3000, 5000, clip_start=0.0),
             make_det(0.43, 0.6, 3000, 5000, clip_start=0.0),       # 0.03 s later, same band
             make_det(0.43, 0.6, 7000, 9000, clip_start=0.0)]       # same time, another band
    assert len(db.fuse_window_detections(calls, phrase_gap=0, file_end=3.0)) == 3
    joined = db.fuse_window_detections(calls, phrase_gap=0.05, file_end=3.0)
    assert sorted((b['time_start'], b['time_end'], b['freq_low_hz']) for b in joined) == [
        (0.2, 0.6, 3000), (0.43, 0.6, 7000)]


def random_raw_detections(seed, n=60, duration=12.0):
    """Raw detections of overlapping 3 s windows every 1.5 s, in a few frequency bands."""
    rng = np.random.default_rng(seed)
    dets = []
    for _ in range(n):
        clip = 1.5 * rng.integers(0, int((duration - 3.0) / 1.5) + 1)
        start = clip + rng.uniform(0, 2.6)
        length = rng.uniform(0.05, min(0.6, clip + 3.0 - start))
        low = rng.choice([1500, 3000, 4500, 6000]) + rng.uniform(-150, 150)
        dets.append(make_det(start, start + length, low, low + rng.uniform(800, 1600),
                             rng.uniform(0.05, 1.0), clip_start=clip))
    return dets, duration


@pytest.mark.parametrize('seed', range(8))
def test_a_larger_phrase_gap_only_merges(seed):
    """Every box at a smaller gap lies inside one box at the larger gap: boxes never split
    or move as the gap grows (the link join)."""
    dets, end = random_raw_detections(seed)
    previous = db.fuse_window_detections(dets, phrase_gap=0, file_end=end)
    for gap in (0.02, 0.05, 0.1, 0.3, 0.6):
        boxes = db.fuse_window_detections(dets, phrase_gap=gap, file_end=end)
        assert len(boxes) <= len(previous)
        for small in previous:
            assert any(big['time_start'] <= small['time_start'] + 1e-9
                       and small['time_end'] <= big['time_end'] + 1e-9
                       and big['freq_low_hz'] <= small['freq_low_hz'] + 1e-9
                       and small['freq_high_hz'] <= big['freq_high_hz'] + 1e-9
                       for big in boxes), (gap, small)
        previous = boxes


@pytest.mark.parametrize('seed', range(8))
def test_no_two_boxes_of_one_band_still_join(seed):
    """After the join no two boxes overlap by the join's 0.05 s at frequency IoU >= 0.8."""
    dets, end = random_raw_detections(seed)
    boxes = db.fuse_window_detections(dets, phrase_gap=0, file_end=end)
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            overlap = min(a['time_end'], b['time_end']) - max(a['time_start'], b['time_start'])
            fiou = db._frequency_iou(a['freq_low_hz'], a['freq_high_hz'], b['freq_low_hz'], b['freq_high_hz'])
            assert not (overlap >= -db.FUSION_JOIN_GAP and fiou >= db.FUSION_JOIN_FREQ_IOU), (a, b)


def test_file_ends_from_raw():
    dets = [make_det(0, 1, 1, 2, clip_start=0.0), make_det(4, 5, 1, 2, clip_start=3.0),
            make_det(1, 2, 1, 2, clip_start=0.0, filename='b.wav')]
    assert db.file_ends_from_raw(dets) == {'__single_file__': 6.0, 'b.wav': 3.0}


def test_merge_detections_dispatches():
    dets = [make_det(0.2, 0.4, 3000, 5000, clip_start=0.0), make_det(0.43, 0.6, 3000, 5000, clip_start=0.0)]
    assert len(db.merge_detections(dets, 'fusion', 0.1, phrase_gap=0, file_end=3.0)) == 2
    assert len(db.merge_detections(dets, 'reconstruct', 0.1)) == 1
