import pytest

from revcaptcha.asr import WhisperTranscriber


@pytest.fixture(scope="session")
def call_model():
    return WhisperTranscriber()
