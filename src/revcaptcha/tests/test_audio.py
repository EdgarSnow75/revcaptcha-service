from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from revcaptcha.audio import load_audio


def make_tone(tmp_path, duration, sr, channels=1):
    n = int(duration * sr)
    t = np.arange(n) / sr
    mono = 0.5 * np.sin(2 * np.pi * 440 * t)
    data = mono if channels == 1 else np.column_stack([mono] * channels)
    path = tmp_path / f"tone_{sr}hz_{channels}ch.wav"
    sf.write(path, data, sr)
    return path


# def test_tmp_path(tmp_path):
#     file_path = Path(f"{tmp_path}/word.txt")
#     file_path.write_text("Hello")
#     assert file_path.read_text() == "Hello"
#     print(tmp_path)


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
    assert audio.size == pytest.approx(16000)
    assert audio.ndim == 1
    assert len(audio) == pytest.approx(16000)


@pytest.mark.parametrize("duration, rate, channel", [(2, 44100, 1), (2, 44100, 2)])
def test_load_audio_2s_44khz(tmp_path, duration, rate, channel):
    audio_path = make_tone(tmp_path, duration, rate, channel)
    audio = load_audio(audio_path)
    assert audio.size == pytest.approx(32000)
    assert audio.ndim == 1
    assert len(audio) == pytest.approx(32000)
