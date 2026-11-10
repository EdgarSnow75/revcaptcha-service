import pytest

from revcaptcha.text import normalize_transcript


@pytest.mark.parametrize(
    "raw, transcript",
    [("Pee Bee SEE", ["P", "B", "C"]), ("ZEE K 1 AITCH 8", ["Z", "K", "1", "H", "8"])],
)
@pytest.mark.parametrize("raw_2, transcript_2", [("Pee BEE See", ["PB", "B", "CC"])])
def test_canonical_r1(raw, raw_2, transcript, transcript_2):
    assert normalize_transcript(raw) == transcript
    assert normalize_transcript(raw_2) != transcript_2


@pytest.mark.parametrize(
    "raw, transcript",
    [("P B C", ["B", "P", "Z"]), ("Z K 1 AITCH 8", ["X", "T", "2", "L", "T"])],
)
def test_canonical_r2(raw, transcript):
    assert normalize_transcript(raw) != transcript
