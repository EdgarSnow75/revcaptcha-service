import dataclasses
import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from revcaptcha.challenge import ANSWER_WINDOW, Challenge
from revcaptcha.tests.fakes import FakeEmbedder, FakeTranscriber
from revcaptcha.verify import Outcome, VerifyResult, verify_attempt

T0 = datetime(2026, 11, 23, 0, tzinfo=UTC)


@pytest.mark.parametrize("outcomes", list(Outcome))
def test_accepted(outcomes):
    result = VerifyResult(
        outcome=outcomes,
        transcript=None,
        score=None,
        threshold=0.57,
        asr_seconds=None,
        speaker_seconds=None,
    )
    assert result.accepted == (outcomes == Outcome.ACCEPTED)


def test_json_dump():
    s = json.dumps({"s": Outcome.GATE_REJECTED})
    assert s == '{"s": "gate_rejected"}'


def test_frozen_true():
    result = VerifyResult(
        outcome=Outcome.ACCEPTED,
        transcript=None,
        score=None,
        threshold=0.57,
        asr_seconds=None,
        speaker_seconds=None,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.score = 0.98


@pytest.mark.parametrize(
    "challenge, sample, code, embed, embed_count, transc_count, outcome",
    (
        [
            Challenge(
                id="t1",
                code="5G5LB",
                issued_at=T0,
                expires_at=T0 + timedelta(seconds=8),
            ),
            np.zeros(16000, dtype=np.float32),
            "5 G B L 5",
            (0.6, 0.8),
            0,
            0,
            Outcome.EXPIRED,
        ],
        [
            Challenge(
                id="t1",
                code="5G5LB",
                issued_at=T0,
                expires_at=T0 + ANSWER_WINDOW,
            ),
            np.zeros(16000, dtype=np.float32),
            "Five, G, five, L, B.",
            (1, 0),
            1,
            1,
            Outcome.ACCEPTED,
        ],
        [
            Challenge(
                id="t1",
                code="5G5LB",
                issued_at=T0,
                expires_at=T0 + ANSWER_WINDOW,
            ),
            np.zeros(16000, dtype=np.float32),
            "five gee five el bee",
            (1, 0),
            1,
            1,
            Outcome.ACCEPTED,
        ],
        [
            Challenge(
                id="t1",
                code="5G5LB",
                issued_at=T0,
                expires_at=T0 + ANSWER_WINDOW,
            ),
            np.zeros(16000, dtype=np.float32),
            "5G5LB",
            (-1, 0),
            1,
            1,
            Outcome.SPEAKER_REJECTED,
        ],
    ),
)
def test_verify_simple(
    challenge, sample, code, embed, embed_count, transc_count, outcome
):
    embedder = FakeEmbedder(embed)
    transcriber = FakeTranscriber(code)
    result = verify_attempt(
        challenge,
        samples=sample,
        profile=np.array([0.6, 0.8], dtype="float32"),
        transcriber=transcriber,
        embedder=embedder,
        now=T0 + timedelta(seconds=9),
    )
    assert result.outcome == outcome
    assert embedder.calls == embed_count
    if embedder.calls > 0:
        assert result.speaker_seconds is not None
    assert transcriber.calls == transc_count
    if transcriber.calls > 0:
        assert result.asr_seconds is not None


@pytest.mark.parametrize(
    "challenge, sample, code, embed, embed_count, transc_count, outcome",
    (
        [
            Challenge(
                id="t1",
                code="5G5LB",
                issued_at=T0,
                expires_at=T0 + ANSWER_WINDOW,
            ),
            np.zeros(16000, dtype=np.float32),
            "5 G 5 L P",
            (0.6, 0.8),
            0,
            1,
            Outcome.GATE_REJECTED,
        ],
    ),
)
def test_verify_gate_rejected(
    challenge, sample, code, embed, embed_count, transc_count, outcome
):
    embedder = FakeEmbedder(embed)
    transcriber = FakeTranscriber(code)
    result = verify_attempt(
        challenge,
        samples=sample,
        profile=np.array([0.6, 0.8], dtype="float32"),
        transcriber=transcriber,
        embedder=embedder,
        now=T0 + timedelta(seconds=9),
    )
    assert result.outcome == outcome
    assert embedder.calls == embed_count
    assert result.score is None
    assert result.transcript is not None


@pytest.mark.parametrize(
    "challenge, sample, code, embed, embed_count, transc_count, outcome, score",
    (
        [
            Challenge(
                id="t1",
                code="5G5LB",
                issued_at=T0,
                expires_at=T0 + ANSWER_WINDOW,
            ),
            np.zeros(16000, dtype=np.float32),
            "Five, G, five, L, B.",
            (1, 0),
            1,
            1,
            Outcome.ACCEPTED,
            0.6,
        ],
        [
            Challenge(
                id="t1",
                code="5G5LB",
                issued_at=T0,
                expires_at=T0 + ANSWER_WINDOW,
            ),
            np.zeros(16000, dtype=np.float32),
            "5G5LB",
            (-1, 0),
            1,
            1,
            Outcome.SPEAKER_REJECTED,
            -0.6,
        ],
    ),
)
def test_verify_score(
    challenge, sample, code, embed, embed_count, transc_count, outcome, score
):
    embedder = FakeEmbedder(embed)
    transcriber = FakeTranscriber(code)
    result = verify_attempt(
        challenge,
        samples=sample,
        profile=np.array([0.6, 0.8], dtype="float32"),
        transcriber=transcriber,
        embedder=embedder,
        now=T0 + timedelta(seconds=9),
    )
    assert result.outcome == outcome
    assert embedder.calls == embed_count
    assert transcriber.calls == transc_count
    assert result.score == pytest.approx(score)


@pytest.mark.parametrize(
    "challenge, sample, code, embed, embed_count, transc_count, outcome, threshold",
    (
        [
            Challenge(
                id="t1",
                code="5G5LB",
                issued_at=T0,
                expires_at=T0 + ANSWER_WINDOW,
            ),
            np.zeros(16000, dtype=np.float32),
            "Five, G, five, L, B.",
            (1, 0),
            1,
            1,
            Outcome.SPEAKER_REJECTED,
            0.9,
        ],
    ),
)
def test_verify_strict_threshold(
    challenge, sample, code, embed, embed_count, transc_count, outcome, threshold
):
    embedder = FakeEmbedder(embed)
    transcriber = FakeTranscriber(code)
    result = verify_attempt(
        challenge,
        samples=sample,
        profile=np.array([0.6, 0.8], dtype="float32"),
        transcriber=transcriber,
        embedder=embedder,
        now=T0 + timedelta(seconds=9),
        threshold=threshold,
    )
    assert result.outcome == outcome
    assert embedder.calls == embed_count
    assert transcriber.calls == transc_count
    assert result.threshold == threshold


@pytest.mark.parametrize(
    "challenge, sample, code, embed, embed_count, transc_count, outcome, finish_time",
    (
        [
            Challenge(
                id="t1",
                code="5G5LB",
                issued_at=T0,
                expires_at=T0 + ANSWER_WINDOW,
            ),
            np.zeros(16000, dtype=np.float32),
            "Five, G, five, L, B.",
            (1, 0),
            1,
            1,
            Outcome.ACCEPTED,
            timedelta(seconds=10),
        ],
        [
            Challenge(
                id="t1",
                code="5G5LB",
                issued_at=T0,
                expires_at=T0 + ANSWER_WINDOW,
            ),
            np.zeros(16000, dtype=np.float32),
            "Five, G, five, L, B.",
            (1, 0),
            0,
            0,
            Outcome.EXPIRED,
            timedelta(microseconds=10000001),
        ],
    ),
)
def test_verify_exact_time(
    challenge, sample, code, embed, embed_count, transc_count, outcome, finish_time
):
    embedder = FakeEmbedder(embed)
    transcriber = FakeTranscriber(code)
    result = verify_attempt(
        challenge,
        samples=sample,
        profile=np.array([0.6, 0.8], dtype="float32"),
        transcriber=transcriber,
        embedder=embedder,
        now=T0 + finish_time,
    )
    assert result.outcome == outcome
    assert embedder.calls == embed_count
    assert transcriber.calls == transc_count
    if transcriber.calls == 0:
        assert result.asr_seconds is None
    if embedder.calls == 0:
        assert result.speaker_seconds is None
