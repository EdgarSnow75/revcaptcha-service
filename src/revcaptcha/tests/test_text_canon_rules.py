import pytest

from revcaptcha.challenge import ALPHABET
from revcaptcha.text import NORMALIZATION_MAP, normalize_transcript


@pytest.mark.parametrize(
    "raw, transcript",
    [("Pee Bee SEE", ["P", "B", "C"]), ("ZEE K 1 AITCH 8", ["Z", "K", "1", "H", "8"])],
)
@pytest.mark.parametrize("raw_2, transcript_2", [("Pee BEE See", ["PB", "B", "CC"])])
def test_canonical_r1(raw, raw_2, transcript, transcript_2):
    assert normalize_transcript(raw) == transcript
    assert normalize_transcript(raw_2) != transcript_2
    for key, word in enumerate(NORMALIZATION_MAP.values()):
        assert len(word) == 1


@pytest.mark.parametrize(
    "raw, transcript",
    [("P B C", ["B", "P", "Z"]), ("Z K 1 AITCH 8", ["X", "T", "2", "L", "T"])],
)
@pytest.mark.parametrize("spoken_word", NORMALIZATION_MAP.keys())
def test_canonical_r2(raw, transcript, spoken_word):
    assert normalize_transcript(raw) != transcript
    assert spoken_word not in set(ALPHABET)
