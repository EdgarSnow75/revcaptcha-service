"""Load the audio with sampling to 16khz mono-channel"""

from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

TARGET_RATE = 16000
SHORTEST_DUR = 0.5
LONGEST_DUR = 15.0


class AudioError(Exception):
    """Raised when there is an audio error: missing files, duration shorter than 0.5s or longer than 15s or file is unreadable"""


def load_audio(path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise AudioError(f"Audio File not found: {path}")
    try:
        audio_info = sf.info(path)
    except sf.SoundFileError as err:
        raise AudioError(f"Audio file cannot be read: {path}") from err
    if audio_info.duration < SHORTEST_DUR:
        raise AudioError(
            f"The audio is too short. The audio must be between {SHORTEST_DUR}s and {LONGEST_DUR}s"
        )
    elif audio_info.duration > LONGEST_DUR:
        raise AudioError(
            f"The audio is too long. The audio must be between {SHORTEST_DUR}s and {LONGEST_DUR}s"
        )
    audio, sr = sf.read(path, dtype="float32")
    audio = convert_to_mono(audio)
    if sr != TARGET_RATE:
        audio = resample_khz(audio, sr)
    audio = np.clip(audio, a_max=1, a_min=-1)
    return audio


def convert_to_mono(audio):
    if audio.ndim >= 2:
        return np.mean(audio, axis=1)
    return audio


def resample_khz(audio, sample_rate):
    out = soxr.resample(audio, sample_rate, TARGET_RATE)
    return out


def check_samples(samples):
    if not isinstance(samples, np.ndarray):
        raise ValueError("The samples is not a ndarray")
    elif samples.ndim != 1:
        raise ValueError("The sample's ndim is not 1-D")
    elif samples.dtype != "float32":
        raise ValueError("The sample's ndtype is not float32")
    elif samples.size == 0:
        raise ValueError("The sample array is empty")


def main(): ...


if __name__ == "__main__":
    main()
