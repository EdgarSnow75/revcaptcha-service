import time
from dataclasses import dataclass
from enum import StrEnum, auto

from revcaptcha.challenge import is_expired
from revcaptcha.scoring import DEFAULT_THRESHOLD
from revcaptcha.text import lexical_match
from revcaptcha.scoring import cosine_score, speaker_accepts


class Outcome(StrEnum):
    ACCEPTED = auto()
    EXPIRED = auto()
    GATE_REJECTED = auto()
    SPEAKER_REJECTED = auto()


@dataclass(frozen=True)
class VerifyResult:
    outcome: Outcome
    transcript: str | None
    score: float | None
    threshold: float
    asr_seconds: float | None
    speaker_seconds: float | None

    @property
    def accepted(self) -> bool:
        return self.outcome == Outcome.ACCEPTED


def verify_attempt(
    challenge,
    *,
    samples,
    profile,
    transcriber,
    embedder,
    now,
    threshold=DEFAULT_THRESHOLD,
) -> VerifyResult:
    if is_expired(challenge, now):
        return VerifyResult(
            Outcome.EXPIRED,
            transcript=None,
            score=None,
            threshold=threshold,
            asr_seconds=None,
            speaker_seconds=None,
        )
    transcription_start_time = time.perf_counter()
    transcript = transcriber.transcribe(samples)
    transcription_total_time = time.perf_counter() - transcription_start_time
    if not lexical_match(transcript, challenge.code):
        return VerifyResult(
            Outcome.GATE_REJECTED,
            transcript=transcript,
            score=None,
            threshold=threshold,
            asr_seconds=transcription_total_time,
            speaker_seconds=None,
        )
    embedding_start_time = time.perf_counter()
    embedding = embedder.embed(samples)
    embedding_total_time = time.perf_counter() - embedding_start_time
    score = cosine_score(profile, embedding)
    if speaker_accepts(score, threshold):
        return VerifyResult(
            Outcome.ACCEPTED,
            transcript,
            score,
            threshold,
            asr_seconds=transcription_total_time,
            speaker_seconds=embedding_total_time,
        )
    return VerifyResult(
        Outcome.SPEAKER_REJECTED,
        transcript,
        score,
        threshold,
        asr_seconds=transcription_total_time,
        speaker_seconds=embedding_total_time,
    )


def main(): ...
