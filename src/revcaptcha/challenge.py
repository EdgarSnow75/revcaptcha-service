"""Generate randomized alphanumeric challenges of 5 letters or more using a decent randomizer"""

import secrets, random

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

ALPHABET = "ABCDEFGHIJKLMNPQRSTUVWXYZ"
NUMBERS = "0123456789"
CODE_LENGTH = 5
ANSWER_WINDOW = timedelta(seconds=10)
T0 = datetime(2026, 11, 11, 0, tzinfo=timezone.utc)
random.seed(7)

RANDOM_LIST = []


@dataclass(frozen=True)
class Challenge:
    id: str
    code: str
    issued_at: datetime
    expires_at: datetime


def create_challenge(rng, now: datetime) -> Challenge:
    if not has_timezone(now):
        raise ValueError("Now is naive time. Does not have a timezone.")

    created_code = ""
    # Old logic
    # digit_repeat = True
    # last_chara = ""
    # while len(created_code) < CODE_LENGTH:
    #     if digit_repeat:
    #         chosen_word = rng.choice(ALPHABET + NUMBERS)
    #         if chosen_word == last_chara:
    #             continue
    #         if chosen_word in NUMBERS:
    #             digit_repeat = False
    #         created_code += chosen_word
    #     else:
    #         chosen_word = rng.choice(ALPHABET)
    #         if chosen_word == last_chara:
    #             continue
    #         digit_repeat = True
    #         created_code += chosen_word
    #     last_chara = created_code[-1]
    while True:
        created_code = "".join(rng.choices(ALPHABET + NUMBERS, k=CODE_LENGTH))
        if is_valid_code(created_code):
            break

    expires_at = now + ANSWER_WINDOW
    challenge_id = secrets.token_urlsafe(16)

    new_challenge = Challenge(challenge_id, created_code, now, expires_at)

    return new_challenge


def is_valid_code(code: str) -> bool:
    if len(code) != CODE_LENGTH:
        return False

    for word in code:
        if word not in ALPHABET and word not in NUMBERS:
            return False

    last_character = code[0]
    for i in range(1, len(code)):
        if last_character == code[i]:
            return False
        last_character = code[i]

    for i in range(len(code) - 1):
        if code[i] in NUMBERS and code[i + 1] in NUMBERS:
            return False

    return True


def has_timezone(now: datetime) -> bool:
    return now.tzinfo is not None and now.tzinfo.utcoffset(now) is not None


def is_expired(challenge: Challenge, now: datetime) -> bool:
    return now > challenge.expires_at


def main():
    for _ in range(10000):
        c = create_challenge(random.Random(), T0)
        if is_valid_code(c.code):
            RANDOM_LIST.append(c.code)
        else:
            return print(f"The code is not valid: {c.code}")
    print(RANDOM_LIST)
    # c = create_challenge(random.Random(), T0)
    # d = create_challenge(random.Random(), T0)
    # print(c.id)
    # print(d.id)


if __name__ == "__main__":
    main()
