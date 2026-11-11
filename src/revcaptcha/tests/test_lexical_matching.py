import pytest

from revcaptcha.text import lexical_match


@pytest.mark.parametrize(
    "transcript, code, outcome",
    [
        ("5G5LB", "5G5LB", True),
        ("Five, G, five, L, B.", "5G5LB", True),
        ("5 G 5 B L", "5G5LB", False),
        ("5 G 5 L", "5G5LB", False),
        ("5G5LB please", "5G5LB", False),
        ("won to for K M", "124KM", True),
        ("S, E, V, E, N.", "SEVEN", True),
        ("seven", "SEVEN", False),
        ("", "5G5LB", False),
    ],
)
def test_normal_match(transcript, code, outcome):
    assert lexical_match(transcript, code) == outcome
