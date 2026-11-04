import pytest

from revcaptcha.text import normalize_transcript


@pytest.mark.parametrize(
    "n, expected",
    [
        ("up, left!", ["UP", "LEFT"]),
        ("DOWN! OFF.", ["DOWN", "OFF"]),
        ("on off", ["ON", "OFF"]),
    ],
)
def test_normal_string(n, expected):
    assert normalize_transcript(n) == expected


@pytest.mark.parametrize(
    "n, expected",
    [
        ("up,,, left!!!!", ["UP", "LEFT"]),
        ("....", []),
        (",,,...on! ...,,,left--!-", ["ON", "LEFT"]),
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
        ("seven. .six,,,", ["SEVEN", "SIX"]),
        ("on- left", ["ON", "LEFT"]),
    ],
)
def test_whitespace_replacement_on_punctuation(n, expected):
    assert normalize_transcript(n) == expected


@pytest.mark.parametrize(
    "n, expected",
    [
        ("left,up seven", ["LEFTUP", "SEVEN"]),
        ("don't up", ["DONT", "UP"]),
        ("down-up right", ["DOWNUP", "RIGHT"]),
    ],
)
def test_edgecases_fail(n, expected):
    assert normalize_transcript(n) == expected
