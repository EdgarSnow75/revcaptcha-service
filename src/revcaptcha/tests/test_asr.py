from revcaptcha.asr import Transcriber
from revcaptcha.tests.fakes import FakeTranscriber


def test_check_fake_trans():
    def run(t: Transcriber, samples) -> str:
        return t.transcribe(samples)

    fake = FakeTranscriber("A O T 5 4")
    result = run(fake, "H G K 2 O")
    assert result == "A O T 5 4"
