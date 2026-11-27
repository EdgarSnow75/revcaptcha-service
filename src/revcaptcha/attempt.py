from datetime import datetime

from revcaptcha.challenge import Challenge
from revcaptcha.verify import VerifyResult, verify_attempt


def run_attempt(
    *, profile, transcriber, embedder, issue, record, threshold
) -> tuple[Challenge, datetime, VerifyResult]:
    challenge = issue()
    samples, arrived_at = record()
    result = verify_attempt(
        challenge,
        samples=samples,
        profile=profile,
        transcriber=transcriber,
        embedder=embedder,
        now=arrived_at,
        threshold=threshold,
    )
    return challenge, arrived_at, result
