import time
from datetime import UTC, datetime

import numpy as np


class MicError(Exception):
    """Raised when there are issues with the microphone"""


def keep_recorded(recording: np.ndarray, seconds: float) -> np.array:
    sliced_recording = recording[: round(seconds * 16000)]
    sliced_recording = sliced_recording.reshape(-1)
    return sliced_recording


def check_microphone() -> None:
    import sounddevice as sd

    try:
        sd.check_input_settings(samplerate=16000, channels=1, dtype="float32")
    except sd.PortAudioError as err:
        raise MicError("The input device is not supported") from err


def record_answer(max_seconds: float = 15.0) -> tuple[np.ndarray, datetime]:
    import sounddevice as sd

    out = np.zeros((round(max_seconds * 16000), 1), dtype=np.float32)
    audio = sd.rec(out=out, samplerate=16000)
    start = time.monotonic()
    input("Please press Enter to stop once done")
    end = time.monotonic()
    finish_time = datetime.now(UTC)
    sd.stop()
    elapsed_time = end - start
    audio = keep_recorded(audio, elapsed_time)
    return audio, finish_time
