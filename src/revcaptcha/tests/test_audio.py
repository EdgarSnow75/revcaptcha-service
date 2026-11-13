from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from revcaptcha.audio import LONGEST_DUR, SHORTEST_DUR, AudioError, load_audio


def make_tone(tmp_path, duration, sr, channels=1):
    n = int(duration * sr)
    t = np.arange(n) / sr
    mono = 0.5 * np.sin(2 * np.pi * 440 * t)
    data = mono if channels == 1 else np.column_stack([mono] * channels)
    path = tmp_path / f"tone_{sr}hz_{channels}ch.wav"
    sf.write(path, data, sr)
    return path


def make_loud_tone(tmp_path, duration, sr=48000, channels=1):
    n = int(duration * sr)
    t = np.arange(n) / sr
    mono = 0.9 * np.sign(np.sin(2 * np.pi * 440 * t))
    data = mono if channels == 1 else np.column_stack([mono] * channels)
    path = tmp_path / f"tone_{sr}hz_{channels}ch.wav"
    sf.write(path, data, sr)
    return path


def make_text(tmp_path):
    file_path = Path(f"{tmp_path}/dummy.wav")
    file_path.write_text("Hello")
    # assert file_path.read_text() == "Hello"
    return file_path


@pytest.mark.parametrize("duration, rate, channel", [(1, 16000, 1), (1, 16000, 2)])
def test_make_tone(tmp_path, duration, rate, channel):
    audio_path = make_tone(tmp_path, duration=duration, sr=rate, channels=channel)
    audio_info = sf.info(audio_path)
    assert audio_info.channels == channel


@pytest.mark.parametrize("duration, rate, channel", [(1, 16000, 1), (1, 16000, 2)])
def test_load_audio_basic_16khz(tmp_path, duration, rate, channel):
    audio_path = make_tone(tmp_path, duration, rate, channel)
    audio = load_audio(audio_path)
    assert audio.size == 16000
    assert audio.ndim == 1
    assert len(audio) == 16000


@pytest.mark.parametrize("duration, rate, channel", [(1, 48000, 1), (1, 48000, 2)])
def test_load_audio_basic_48khz(tmp_path, duration, rate, channel):
    audio_path = make_tone(tmp_path, duration, rate, channel)
    audio = load_audio(audio_path)
    assert abs(len(audio) - 16000) <= 1
    assert audio.ndim == 1


@pytest.mark.parametrize("duration, rate, channel", [(2, 44100, 1), (2, 44100, 2)])
def test_load_audio_2s_44khz(tmp_path, duration, rate, channel):
    audio_path = make_tone(tmp_path, duration, rate, channel)
    audio = load_audio(audio_path)
    assert abs(len(audio) - 32000) <= 1
    assert audio.ndim == 1


@pytest.mark.parametrize("duration, rate, channel", [(2, 44100, 1), (2, 44100, 2)])
def test_values_min_max(tmp_path, duration, rate, channel):
    audio_path = make_tone(tmp_path, duration, rate, channel)
    audio = load_audio(audio_path)
    assert audio.max() <= 1
    assert audio.min() >= -1


@pytest.mark.parametrize("duration, rate, channel", [(1, 48000, 1), (1, 48000, 2)])
def test_load_audio_clipping(tmp_path, duration, rate, channel):
    audio_path = make_loud_tone(tmp_path, duration, rate, channel)
    audio = load_audio(audio_path)
    assert audio.max() <= 1
    assert audio.min() >= -1


@pytest.mark.parametrize("duration, rate, channel", [(0.2, 16000, 1), (0.49, 32000, 2)])
def test_load_audio_shortest_duration(tmp_path, duration, rate, channel):
    audio_path = make_tone(tmp_path, duration, rate, channel)
    with pytest.raises(
        AudioError,
        match=f"The audio is too short. The audio must be between {SHORTEST_DUR}s and {LONGEST_DUR}s",
    ):
        load_audio(audio_path)


@pytest.mark.parametrize("duration, rate, channel", [(15.1, 16000, 1), (20.0, 32000, 2)])
def test_load_audio_longest_duration(tmp_path, duration, rate, channel):
    audio_path = make_tone(tmp_path, duration, rate, channel)
    with pytest.raises(
        AudioError,
        match=f"The audio is too long. The audio must be between {SHORTEST_DUR}s and {LONGEST_DUR}s",
    ):
        load_audio(audio_path)


@pytest.mark.parametrize(
    "duration, rate, channel, samples", [(0.5, 16000, 1, 8000), (15.0, 32000, 2, 240000)]
)
def test_load_audio_exact_durations(tmp_path, duration, rate, channel, samples):
    audio_path = make_tone(tmp_path, duration, rate, channel)
    audio = load_audio(audio_path)
    assert audio.size == samples
    assert audio.ndim == 1
    assert len(audio) == samples


def test_load_audio_wrong_path(tmp_path):
    path = tmp_path / "missing.wav"
    with pytest.raises(AudioError, match="Audio File not found:"):
        load_audio(path)


def test_load_audio_unreadable_file(tmp_path):
    path = make_text(tmp_path)
    with pytest.raises(AudioError) as excinfo:
        load_audio(path)
    assert isinstance(excinfo.value.__cause__, sf.SoundFileError)
