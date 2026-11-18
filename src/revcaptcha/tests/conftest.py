import pytest

from pathlib import Path

from revcaptcha.asr import WhisperTranscriber
from revcaptcha.speaker import CamppEmbedder


@pytest.fixture(scope="session")
def call_whisper_model():
    return WhisperTranscriber()


@pytest.fixture(scope="session")
def call_embed_model():
    model_dir = Path(__file__).resolve().parents[3] / "models" / "campplus"
    if not model_dir.exists():
        pytest.skip("Model directory does not exits")
    return CamppEmbedder(model_dir)
