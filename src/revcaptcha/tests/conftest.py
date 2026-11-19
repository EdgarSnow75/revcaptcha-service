from pathlib import Path

import pytest

from revcaptcha.asr import WhisperTranscriber
from revcaptcha.speaker import CamppEmbedder


@pytest.fixture(scope="session")
def whisper_model_fixture():
    return WhisperTranscriber()


@pytest.fixture(scope="session")
def embedder_model_fixture():
    model_dir = Path(__file__).resolve().parents[3] / "models" / "campplus"
    if not model_dir.exists():
        pytest.skip("Model directory does not exits")
    return CamppEmbedder(model_dir)
