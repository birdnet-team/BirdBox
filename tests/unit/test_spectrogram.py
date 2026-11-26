# SPDX-FileCopyrightText: 2026 BirdNET-Team
# SPDX-License-Identifier: MIT
"""The model's input: log-floor mel features of 3 s clips, every 1.5 s, with the rolling floor."""

import numpy as np
import pytest

import config
from inference.utils import log_floor as pi

SR = 32000


def soundscape(seconds, seed=0, start=0.0):
    """Noise with a chirp every 2 s, as float32 audio in [-1, 1]."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR + start
    audio = 0.01 * rng.standard_normal(t.size)
    phase = t % 2.0
    chirp = (phase < 0.3) * 0.1 * np.sin(2 * np.pi * (3000 + 4000 * phase) * phase)
    return (audio + chirp).astype(np.float32)


def features(audio):
    clips, sr = pi.compute_features_for_inference(audio, SR, verbose=False)
    assert sr == SR
    return {round(c['start_time'], 3): c['features'] for c in clips}


@pytest.fixture(scope='module')
def long_file():
    audio = soundscape(70.0)
    return audio, features(audio)


def test_settings_match_training():
    s = pi.get_spectrogram_settings()
    assert (s['sr'], s['n_fft'], s['hop_length'], s['n_mels'], s['fmin'], s['fmax']) == (
        32000, 2048, 375, 256, 50, 15000)
    assert (s['floor_percentile'], s['floor_lo_db'], s['floor_hi_db']) == (20, -3.0, 45.0)
    assert s['floor_window'] == config.FLOOR_WINDOW_SECONDS


def test_clip_grid():
    assert pi._plan_clip_times(7.5, 3) == [0.0, 1.5, 3.0, 4.5]
    assert pi._plan_clip_times(2.9, 3) == []


def test_padding_completes_the_last_window():
    audio, end = pi._pad_to_clip_length(np.ones(4 * SR, np.float32), SR, 3)
    assert len(audio) == int(4.5 * SR) and end == 4.0
    assert not audio[4 * SR:].any()
    same, end = pi._pad_to_clip_length(np.ones(int(4.5 * SR), np.float32), SR, 3)
    assert len(same) == int(4.5 * SR) and end is None
    short, end = pi._pad_to_clip_length(np.ones(SR, np.float32), SR, 3)
    assert len(short) == 3 * SR and end == 1.0
    assert pi.trailing_silence_seconds(4.0, SR, 3) == pytest.approx(0.5)


def test_log_floor_scale():
    s = pi.get_spectrogram_settings()
    # A band at its floor is drawn at -lo / (hi - lo) of the scale (3 dB of 48), 10 dB above at 13 / 48
    mel = np.ones((2, 100))
    mel[1, 50:] = 10.0                       # band 1: louder by 10 dB in half its frames
    out = pi.log_floor(mel, s)
    assert out.dtype == np.float32
    assert out[0] == pytest.approx(np.full(100, 300 / 48))
    assert out[1, 50:] == pytest.approx(np.full(50, 1300 / 48))


def test_log_floor_ignores_digital_silence():
    s = pi.get_spectrogram_settings()
    mel = np.ones((1, 100))
    padded = np.concatenate([mel, np.zeros((1, 100))], axis=1)
    assert pi.log_floor(padded, s)[:, :100] == pytest.approx(pi.log_floor(mel, s))


def test_clips_cover_the_file(long_file):
    _audio, clips = long_file
    # 0 to 67.5 s: the file is padded to 70.5 s so that the last window holds its end
    assert sorted(clips) == [1.5 * i for i in range(46)]
    for clip in clips.values():
        assert clip.shape == (256, 252)
        assert clip.dtype == np.float32
        assert np.isfinite(clip).all()


def test_no_audio_after_a_clip_changes_it(long_file):
    """The rolling floor only looks back: other audio after 40 s leaves every clip that ends
    by then unchanged."""
    audio, clips = long_file
    other = np.concatenate([audio[:40 * SR], soundscape(30.0, seed=1, start=40.0) * 3])
    changed = features(other)
    for start, clip in clips.items():
        if start + 3.0 <= 40.0:
            np.testing.assert_allclose(changed[start], clip, atol=1e-3, err_msg=f'clip at {start} s')
    assert not np.allclose(changed[60.0], clips[60.0], atol=1e-3)


def test_independent_of_the_recording_level(long_file):
    """The same recording 20 dB quieter gives the same picture."""
    audio, clips = long_file
    quiet = features(audio * 0.1)
    for start in (0.0, 30.0, 66.0):
        np.testing.assert_allclose(quiet[start], clips[start], atol=1e-2)


def test_a_short_file_gives_one_clip():
    clips = features(soundscape(1.0))
    assert list(clips) == [0.0]


def test_floor_window_from_the_environment(monkeypatch):
    monkeypatch.delenv('BIRDBOX_FLOOR_WINDOW', raising=False)
    assert config._floor_window_seconds() == 60.0
    monkeypatch.setenv('BIRDBOX_FLOOR_WINDOW', '30')
    assert config._floor_window_seconds() == 30.0
    monkeypatch.setenv('BIRDBOX_FLOOR_WINDOW', 'Segment')
    assert config._floor_window_seconds() is None
    monkeypatch.setenv('BIRDBOX_FLOOR_WINDOW', '-5')
    with pytest.raises(ValueError):
        config._floor_window_seconds()
