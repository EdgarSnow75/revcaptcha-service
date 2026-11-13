import random
from datetime import UTC, datetime, timedelta

import pytest

from revcaptcha.challenge import (
    ALPHABET,
    CODE_LENGTH,
    NUMBERS,
    create_challenge,
    is_expired,
    is_valid_code,
)

T0 = datetime(2026, 11, 11, 0, tzinfo=UTC)


def test_alphabet_len():
    assert len(ALPHABET) == 25
    assert len(NUMBERS) == 10


def test_code_len():
    assert CODE_LENGTH == 5


def test_no_two_chara():
    expected = set("ABCDEFGHIJKLMNPQRSTUVWXYZ")
    assert set(ALPHABET) == expected
    nums_expected = set("0123456789")
    assert set(NUMBERS) == nums_expected


@pytest.mark.parametrize(
    "raw_input, results",
    (
        ["5G5LB", True],
        ["A1B2C", True],
        ["HIJK7", True],
        ["AB12C", False],
        ["12ABC", False],
        ["ABC12", False],
        ["55GLB", False],
        ["5GOLB", False],
        ["5g5lb", False],
        ["5G5L", False],
        ["5G5LBX", False],
        ["5G5L๕", False],
    ),
)
def test_is_valid_code(raw_input, results):
    assert is_valid_code(raw_input) == results


def test_answer_window_is_10():
    c = create_challenge(rng=random.Random(7), now=T0)
    assert c.expires_at - c.issued_at == timedelta(seconds=10)


@pytest.mark.parametrize("n, expected", [(10000, True), (100000, True)])
def test_code_validity(n, expected):
    starts_with_digit = 0
    duplicate_chara = False
    rng = random.Random(7)
    for _ in range(n):
        c = create_challenge(rng, T0)
        assert is_valid_code(c.code) == expected
        if c.code[0] in NUMBERS:
            starts_with_digit += 1
        if len(c.code) != len(set(c.code)):
            duplicate_chara = True
    assert duplicate_chara
    assert 22.0 <= ((starts_with_digit / n) * 100) <= 27.0


def test_is_expired():
    c = create_challenge(rng=random.Random(7), now=T0)
    tick = timedelta(microseconds=1)
    assert is_expired(c, c.expires_at - tick) == False
    assert is_expired(c, c.expires_at + tick) == True
    assert is_expired(c, c.expires_at) == False


def test_naive_time():
    with pytest.raises(ValueError):
        create_challenge(random.Random(7), datetime(2026, 11, 11, 9, 0))


def test_same_code_same_seed():
    c = create_challenge(random.Random(7), T0)
    d = create_challenge(random.Random(7), T0)
    assert c.code == d.code
