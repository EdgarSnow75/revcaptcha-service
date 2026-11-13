import numpy as np
import soundfile as sf

import pytest
from pathlib import Path


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
