import pytest

from revcaptcha.text import normalize_transcript


@pytest.mark.parametrize(
    "n, expected",
    [
        ("up, left!", ["UP", "LEFT"]),
        ("kay gee em eight em", ["K", "G", "M", "8", "M"]),
        ("5G5LB", ["5", "G", "5", "L", "B"]),
        ("won to for", ["1", "2", "4"]),
        ("seven", ["7"]),
    ],
)
def test_normal_string(n, expected):
    assert normalize_transcript(n) == expected


@pytest.mark.parametrize(
    "n, expected",
    [
        ("Five, G, five, L, B.", ["5", "G", "5", "L", "B"]),
        ("....", []),
        (",,,...on! ...,,,left--!-", ["ON", "LEFT"]),
        ("Double-u, 3, Z.", ["W", "3", "Z"]),
        ("five,G", ["5", "G"]),
    ],
)
def test_punctuation_removal(n, expected):
    assert normalize_transcript(n) == expected


@pytest.mark.parametrize(
    "n, expected",
    [("up     left", ["UP", "LEFT"]), ("    ", []), ("on,  .left.", ["ON", "LEFT"])],
)
def test_whitespace_removal(n, expected):
    assert normalize_transcript(n) == expected


@pytest.mark.parametrize("n, expected", [("", []), (" .... ", []), ("         ", [])])
def test_empty_string(n, expected):
    assert normalize_transcript(n) == expected


@pytest.mark.parametrize(
    "n, expected",
    [
        ("up,,, left!!!!", ["UP", "LEFT"]),
        ("seven. .six,,,", ["7", "6"]),
        ("on- left", ["ON", "LEFT"]),
    ],
)
def test_whitespace_replacement_on_punctuation(n, expected):
    assert normalize_transcript(n) == expected


@pytest.mark.parametrize(
    "n, expected",
    [
        ("left,up seven", ["LEFT", "UP", "7"]),
        ("don't up", ["DON", "T", "UP"]),
        ("down-up right", ["DOWN", "UP", "RIGHT"]),
        ("5 G 5 L B ครับ", ["5", "G", "5", "L", "B", "ครับ"]),
    ],
)
def test_edgecases(n, expected):
    assert normalize_transcript(n) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("3 double u Z", ["3", "W", "Z"]),
        ("K, double you, 7", ["K", "W", "7"]),
        ("A B double u", ["A", "B", "W"]),
    ],
)
def test_correct_replacement(raw, expected):
    assert normalize_transcript(raw) == expected
