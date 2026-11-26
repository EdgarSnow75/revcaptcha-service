import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from revcaptcha.speaker import MODEL_NAME
from revcaptcha.store import StoreError, append_attempt, load_profile, save_profile

FOLDER = "profiles"
PROFILE = np.array([0.6, 0.8], dtype=np.float32)
LONG_PROFILE = np.array([1.2, 1.6], dtype=np.float32)
CLIPS = ["e1.wav", "e2.wav", "e3.wav", "e4.wav", "e5.wav"]
NAME_CASES = [r"..\..\notes", "../../notes", "edgar!"]


@pytest.mark.parametrize(
    "folder, user, profile, model, clips",
    [(FOLDER, "edgar", PROFILE, MODEL_NAME, CLIPS)],
)
def test_save_profile_simple(tmp_path, folder, user, profile, model, clips):
    path = tmp_path / folder
    save_profile(path, user, profile=profile, model=model, clips=clips)
    user_profile, clip_len = load_profile(path, user, model=MODEL_NAME)
    assert np.array_equal(user_profile, profile)
    assert user_profile.dtype == np.float32
    assert clip_len == 5


@pytest.mark.parametrize(
    "folder, user, profile, model, clips",
    [(FOLDER, "edgar", PROFILE, "ecapa-tdnn", CLIPS)],
)
def test_load_wrong_model(tmp_path, folder, user, profile, model, clips):
    path = tmp_path / folder
    save_profile(path, user, profile=profile, model=MODEL_NAME, clips=clips)
    with pytest.raises(StoreError):
        load_profile(path, user, model=model)


@pytest.mark.parametrize(
    "folder, user, profile, model, clips",
    [(FOLDER, "edgar", LONG_PROFILE, MODEL_NAME, CLIPS)],
)
def test_load_wrong_len(tmp_path, folder, user, profile, model, clips):
    path = tmp_path / folder
    save_profile(path, user, profile=profile, model=model, clips=clips)
    with pytest.raises(StoreError):
        load_profile(path, user, model=model)


@pytest.mark.parametrize(
    "folder, user, profile, model, clips",
    [(FOLDER, "snow", PROFILE, MODEL_NAME, CLIPS)],
)
def test_load_wrong_user(tmp_path, folder, user, profile, model, clips):
    path = tmp_path / folder
    save_profile(path, "edgar", profile=profile, model=model, clips=clips)
    with pytest.raises(StoreError):
        load_profile(path, user, model=model)


@pytest.mark.parametrize("user", NAME_CASES)
def test_invalid_user_names_save(tmp_path, user):
    path = tmp_path / "a" / FOLDER
    with pytest.raises(StoreError):
        save_profile(path, user, profile=PROFILE, model=MODEL_NAME, clips=CLIPS)
    assert not (tmp_path / "notes.json").exists()


@pytest.mark.parametrize("user", NAME_CASES)
def test_invalid_user_names_load(tmp_path, user):
    path = tmp_path / "a" / FOLDER
    save_profile(tmp_path, "notes", profile=PROFILE, model=MODEL_NAME, clips=CLIPS)
    saved = save_profile(path, "edgar", profile=PROFILE, model=MODEL_NAME, clips=CLIPS)
    saved.rename(path / "edgar!.json")
    with pytest.raises(StoreError):
        load_profile(path, user, model=MODEL_NAME)


def test_append_attempt(tmp_path):
    t = datetime(2026, 11, 24, 10, 15, 9, tzinfo=UTC)
    first = {
        "at": t.isoformat(),
        "user": "edgar",
        "code": "5G5LB",
        "answer_seconds": 4.2,
        "outcome": "accepted",
        "transcript": "5G5LB.",
        "score": 0.7123,
        "threshold": 0.5387,
        "asr_seconds": 1.42,
        "speaker_seconds": 0.19,
    }
    second = {
        "at": (t + timedelta(minutes=1)).isoformat(),
        "user": "edgar",
        "code": "K7M3P",
        "answer_seconds": 6.8,
        "outcome": "gate_rejected",
        "transcript": "K 7 M 3 B",
        "score": None,
        "threshold": 0.5387,
        "asr_seconds": 1.37,
        "speaker_seconds": None,
    }
    log = tmp_path / "data" / "attempts.jsonl"
    append_attempt(log, first)
    append_attempt(log, second)
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == first
    assert json.loads(lines[1]) == second
    assert datetime.fromisoformat(json.loads(lines[0])["at"]) == t
