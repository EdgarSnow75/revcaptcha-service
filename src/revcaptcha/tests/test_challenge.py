from revcaptcha.challenge import ALPHABET, CODE_LENGTH


def test_alphabet_len():
    assert len(ALPHABET) == 35


def test_code_len():
    assert CODE_LENGTH == 5


def test_no_two_chara():
    expected = set("ABCDEFGHIJKLMNPQRSTUVWXYZ0123456789")
    assert set(ALPHABET) == expected
