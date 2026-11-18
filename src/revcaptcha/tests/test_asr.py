from pathlib import Path

import pytest

from revcaptcha.asr import Transcriber
from revcaptcha.audio import load_audio
from revcaptcha.tests.fakes import FakeTranscriber
from revcaptcha.text import normalize_transcript


def test_check_fake_trans():
    def run(t: Transcriber, samples) -> str:
        return t.transcribe(samples)

    fake = FakeTranscriber("A O T 5 4")
    result = run(fake, "H G K 2 O")
    assert result == "A O T 5 4"


@pytest.mark.slow
@pytest.mark.parametrize(
    "file_name, expected",
    [
        ("004ae714_seven.wav", ["7"]),
        ("00b01445_five.wav", ["5"]),
        ("012c8314_eight.wav", ["8"]),
    ],
)
def test_whiper_transcriber(call_whisper_model, file_name, expected):
    path = Path(__file__).parent / "data" / file_name
    # print(path)
    audio = load_audio(path)
    transcript = call_whisper_model.transcribe(audio)
    assert normalize_transcript(transcript) == expected
