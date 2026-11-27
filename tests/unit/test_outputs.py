# SPDX-FileCopyrightText: 2026 BirdNET-Team
# SPDX-License-Identifier: MIT
"""The output formats: JSON with algorithm metadata, simplified CSV, Raven tables, xeno-canto Annota-JSON."""

import csv
import json

import pytest

import config
from inference.detect_birds import BirdCallDetector
from inference.utils.raven_export import RAVEN_FIELDNAMES, build_raven_rows, save_raven_tables
from inference.utils.xeno_canto_export import build_xeno_canto_json
from conftest import make_det


def merged(time_start, time_end, species='amro', **extra):
    """A merged detection as the song merge and window fusion return it."""
    det = make_det(time_start, time_end, 2000, 4000, species=species, **extra)
    conf = det.pop('confidence')
    det.update(avg_confidence=conf, max_confidence=conf + 0.1, detections_merged=2)
    return det


def detector(**settings):
    """A BirdCallDetector with only the attributes the writers read; no model is loaded."""
    d = BirdCallDetector.__new__(BirdCallDetector)
    d.model_path = 'models/Just-Bird_fp32.pt'
    d.conf_threshold = 0.15
    d.nms_iou_threshold = 0.7
    d.song_gap_threshold = 0.1
    d.species_mapping = 'Just-Bird'
    d.merge_method = 'fusion'
    d.freq_iou_threshold = 0.6
    d.phrase_gap = 0.05
    d.settings = {'floor_window': 60.0}
    d.species_mappings = config.get_species_mapping('Just-Bird')
    d.__dict__.update(settings)
    return d


def test_json_states_the_settings(tmp_path):
    path = tmp_path / 'out.json'
    detector().save_detections([merged(1.0, 2.0)], str(path), 'rec.wav')
    out = json.loads(path.read_text())
    assert out['audio_file'] == 'rec.wav'
    assert out['detection_count'] == 1
    assert out['model_config'] == {
        'model': 'models/Just-Bird_fp32.pt', 'confidence_threshold': 0.15, 'nms_iou_threshold': 0.7,
        'song_gap_threshold': 0.1, 'species_mapping': 'Just-Bird', 'merge_method': 'fusion',
        'floor_window_seconds': 60.0, 'freq_iou_threshold': 0.6, 'phrase_gap': 0.05}


def test_json_names_the_segment_floor_and_lists_files(tmp_path):
    path = tmp_path / 'out.json'
    dets = [merged(1.0, 2.0, filename='a.wav', file_path='/x/a.wav'),
            merged(1.0, 2.0, filename='b.wav', file_path='/x/b.wav')]
    detector(settings={'floor_window': None}, phrase_gap=None).save_detections(dets, str(path))
    out = json.loads(path.read_text())
    assert out['model_config']['floor_window_seconds'] == 'segment'
    assert 'phrase_gap' not in out['model_config']
    assert sorted(out['audio_files']) == ['/x/a.wav', '/x/b.wav'] and out['file_count'] == 2


def test_simplified_csv(tmp_path):
    path = tmp_path / 'out.csv'
    raw = make_det(0.25, 0.5, 3000, 5000, 0.42)
    detector().save_detections_csv([merged(1.04, 2.26), raw], str(path), '/data/rec.wav')
    rows = list(csv.reader(path.open()))
    assert rows[0] == ['Filename', 'Start Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)',
                       'Species eBird Code', 'Confidence']
    assert rows[1] == ['rec.wav', '1.0', '2.3', '2000', '4000', 'amro', '0.500']   # merged: average, whole Hz
    assert rows[2][-1] == '0.420'                                                     # raw: its own


def test_raven_rows_are_sorted_with_millisecond_times():
    rows = build_raven_rows([merged(2.0, 2.5), merged(0.1234, 0.3456, species='houfin', freq_low_hz=3375.55)])
    assert [r['Selection'] for r in rows] == [1, 2]
    assert rows[0]['Begin Time (S)'] == '0.123' and rows[0]['End Time (S)'] == '0.346'
    assert rows[0]['Annotation'] == 'houfin'
    assert rows[0]['Low Freq (Hz)'] == 3376                     # whole Hz
    assert list(rows[0]) == RAVEN_FIELDNAMES


def test_raven_one_table_per_recording(tmp_path):
    dets = [merged(1, 2, filename='a.wav'), merged(3, 4, filename='a.wav'), merged(1, 2, filename='b.wav')]
    written = save_raven_tables(dets, tmp_path / 'single.txt', tmp_path / 'raven')
    assert written == tmp_path / 'raven'
    assert sorted(p.name for p in written.iterdir()) == ['a.wav.txt', 'b.wav.txt']
    table = list(csv.DictReader((written / 'a.wav.txt').open(), delimiter='\t'))
    assert len(table) == 2
    single = save_raven_tables([merged(1, 2)], tmp_path / 'single.txt', tmp_path / 'raven2')
    assert single == tmp_path / 'single.txt' and single.is_file()


def test_xeno_canto_annota_json():
    mappings = config.get_species_mapping('Birds-With-Labels')
    dets = [merged(1.0, 2.5, species='amerob'),      # named from AviList
            merged(3.0, 3.5, species='scbwoo5'),     # not in AviList: named from the species mapping
            merged(4.0, 4.5, species='bird'),        # no species: class Aves
            merged(5.0, 5.5, species='unknown')]
    out = build_xeno_canto_json(dets, audio_path='/data/XC123456.wav', species_mappings=mappings)
    first, second, third, fourth = out['annotations']
    assert third['scientific_name'] == fourth['scientific_name'] == 'Aves'
    assert first['sound_file'] == 'XC123456.wav'
    assert first['xc_nr'] == '123456'
    assert first['scientific_name'] == 'Turdus migratorius'
    assert second['scientific_name'] == 'Celeus grammicus'
    assert (first['start_time'], first['end_time']) == (1.0, 2.5)
    assert 'confidence: 0.500' in first['annotation_remarks']
    assert out['scope'][0]['taxon_coverage'] == 'Aves, Celeus grammicus, Turdus migratorius'


@pytest.mark.parametrize('formats, files', [
    (['json-with-algorithm-metadata'], {'with_algorithm_metadata.json'}),
    (['simplified-csv', 'raven-selection-table'], {'simplified.csv', 'raven_selection_table.txt'}),
])
def test_save_results_writes_the_requested_formats(tmp_path, formats, files):
    detector().save_results([merged(1, 2)], str(tmp_path), 'rec.wav', output_formats=formats)
    assert {p.name for p in tmp_path.iterdir()} == files


def test_pixels_to_hz_inverts_the_mel_axis():
    d = BirdCallDetector.__new__(BirdCallDetector)
    d.height_width = 256
    import librosa
    d.min_mel = librosa.hz_to_mel(d.MIN_FREQ, htk=True)
    d.mel_range = librosa.hz_to_mel(d.MAX_FREQ, htk=True) - d.min_mel
    assert d.pixels_to_hz(0) == 15000          # top row: the highest frequency
    assert d.pixels_to_hz(256) == 50
    values = [d.pixels_to_hz(y) for y in range(0, 257, 16)]
    assert values == sorted(values, reverse=True)
    middle = librosa.mel_to_hz(d.min_mel + d.mel_range / 2, htk=True)
    assert d.pixels_to_hz(128) == round(float(middle))


def test_find_audio_files_lists_each_file_once(tmp_path):
    """Suffixes match in any case and no file is listed twice."""
    from inference.detect_birds import find_audio_files
    (tmp_path / 'sub').mkdir()
    for name in ('a.wav', 'b.Wav', 'sub/c.FLAC', 'notes.txt'):
        (tmp_path / name).touch()
    found = find_audio_files(str(tmp_path), verbose=False)
    assert found == sorted(str(tmp_path / n) for n in ('a.wav', 'b.Wav', 'sub/c.FLAC'))


def test_save_results_creates_the_output_directory(tmp_path):
    from evaluation.filter_and_merge_detections import DetectionFilter
    out = tmp_path / 'new' / 'run'
    DetectionFilter().save_results({'audio_file': 'x.wav', 'detections': []}, [], str(out), 0.15, 0.1)
    assert (out / 'simplified.csv').is_file()


def test_filtered_json_keeps_the_recording_names(tmp_path):
    from evaluation.filter_and_merge_detections import DetectionFilter
    f = DetectionFilter()
    for raw, key, value in (({'audio_file': '/x/a.wav'}, 'audio_file', '/x/a.wav'),
                            ({'audio_files': ['/x/a.wav', '/x/b.wav'], 'file_count': 2}, 'file_count', 2)):
        path = tmp_path / 'out.json'
        f.save_filtered_json({**raw, 'model_config': {}, 'detections': []}, [], str(path), 0.15, 0.1)
        out = json.loads(path.read_text())
        assert out[key] == value
        assert ('audio_files' in out) == ('audio_files' in raw)
