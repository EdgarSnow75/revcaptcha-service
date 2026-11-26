import pytest
import numpy as np

from revcaptcha.mic import MicError, check_microphone, keep_recorded


@pytest.mark.parametrize(
    "recording, seconds, expected",
    (
        [np.zeros(15 * 16000, dtype=np.float32), 1.5, 24000],
        [np.zeros(15 * 16000, dtype=np.float32), 20, 240000],
    ),
)
def test_keep_recorded(recording, seconds, expected):
    result = keep_recorded(recording, seconds)
    assert result.shape[0] == expected
    assert result.ndim == 1


def test_check_input():
    assert check_microphone() == None
