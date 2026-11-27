from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from revcaptcha.attempt import run_attempt
from revcaptcha.challenge import ANSWER_WINDOW, Challenge
from revcaptcha.tests.fakes import FakeEmbedder, FakeTranscriber
from revcaptcha.verify import Outcome

T0 = datetime(2026, 11, 27, 8, 0, tzinfo=UTC)
PROFILE = np.array([0.6, 0.8], dtype=np.float32)
ONE_SILENCE = np.zeros(16000, dtype=np.float32)


# Helper Funcs
def issue():
    return Challenge(id="t1", code="5G7LB", issued_at=T0, expires_at=T0 + ANSWER_WINDOW)


def record_at_4():
    return ONE_SILENCE, T0 + timedelta(seconds=4)


def record_at_11():
    return ONE_SILENCE, T0 + timedelta(seconds=11)


@pytest.mark.parametrize(
    "record, outcome, transcriber_count, embedder_count",
    ([record_at_4, Outcome.ACCEPTED, 1, 1], [record_at_11, Outcome.EXPIRED, 0, 0]),
)
def test_run_attempt(record, outcome, transcriber_count, embedder_count):
    transcriber = FakeTranscriber("5G7LB")
    embedder = FakeEmbedder([1, 0])
    issued_challenge, _, result = run_attempt(
        profile=PROFILE,
        transcriber=transcriber,
        embedder=embedder,
        issue=issue,
        record=record,
        threshold=0.5387,
    )
    assert result.outcome == outcome
    assert issued_challenge.code == "5G7LB"
    assert transcriber.calls == transcriber_count
    assert embedder.calls == embedder_count


def test_issue_before_record():
    order = []

    def issue():
        order.append("issue")
        return Challenge(
            id="t1", code="5G7LB", issued_at=T0, expires_at=T0 + ANSWER_WINDOW
        )

    def record():
        order.append("record")
        return ONE_SILENCE, T0 + timedelta(seconds=4)

    transcriber = FakeTranscriber("5G7LB")
    embedder = FakeEmbedder([1, 0])
    run_attempt(
        profile=PROFILE,
        transcriber=transcriber,
        embedder=embedder,
        issue=issue,
        record=record,
        threshold=0.5387,
    )

    assert order == ["issue", "record"]
