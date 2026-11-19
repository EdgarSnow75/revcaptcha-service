from pathlib import Path

import numpy as np
import pytest

from revcaptcha.audio import load_audio
from revcaptcha.scoring import (
    build_profile,
    cosine_score,
    speaker_accepts,
    threshold_for,
)


@pytest.mark.parametrize(
    "target1, target2, test, expected",
    [
        (np.array([3.0, 4.0]), np.array([4.0, 3.0]), np.array([1.0, 0.0]), 0.707),
        (np.array([3.0, 4.0]), np.array([4.0, 3.0]), np.array([2.0, 2.0]), 1.0),
    ],
)
def test_simple_scoring(target1, target2, test, expected):
    profile = build_profile([target1, target2])
    score = cosine_score(profile, test)
    assert score == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize(
    "target1, target2, target3,, target4",
    [
        (
            np.array([3.0, 4.0]),
            np.array([4.0, 3.0]),
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
        ),
        (
            np.array([3.0, 4.0, 1.0]),
            np.array([4.0, 3.0, 2.0]),
            np.array([1.0, 0.0, 3.0]),
            np.array([0.0, 1.0, 3.5]),
        ),
        (
            np.array([3.0, 4.0, 1.0, 2.0, 0.0]),
            np.array([4.0, 3.0, 2.2, 1.5, 1.4]),
            np.array([1.0, 0.0, 1.0, 1.1, 1.4]),
            np.array([0.0, 1.0, 3.0, 5.0, 4.0]),
        ),
    ],
)
def test_profile_len1(target1, target2, target3, target4):
    profile = build_profile([target1, target2, target3, target4])
    assert np.linalg.norm(profile) == pytest.approx(1, abs=1e-3)


@pytest.mark.parametrize(
    "target, test, expected",
    [
        (np.array([3.0, 4.0]), np.array([3.0, 4.0]), 1.0),
        (np.array([3.0, 4.0]), np.array([-3.0, -4.0]), -1.0),
        (np.array([5.0, 0.0]), np.array([0.0, 5.0]), 0.0),
    ],
)
def test_cosine_scoring(target, test, expected):
    score = cosine_score(target, test)
    assert score == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize(
    "targets",
    [
        ([]),
        ([np.array([1.0, 0.0]), np.array([1.0, 0.0, 0.0])]),
        ([np.array([1.0, 0.0]), np.array([-1.0, 0.0])]),
    ],
)
def test_profile_error_handling(targets):
    with pytest.raises(ValueError):
        build_profile(targets)


@pytest.mark.parametrize(
    "enrollment_size, expected,",
    [
        (5, 0.5387),
        (10, 0.5669),
    ],
)
def test_valid_thresholds(enrollment_size, expected):
    assert threshold_for(enrollment_size) == expected


@pytest.mark.parametrize(
    "enrollment_size",
    [
        (7),
        (10.1),
        (4.98),
    ],
)
def test_invalid_thresholds(enrollment_size):
    with pytest.raises(ValueError):
        threshold_for(enrollment_size)


@pytest.mark.parametrize(
    "score, threshold, expected",
    [
        (0.5387, 0.5387, True),
        (0.5386, 0.5387, False),
    ],
)
def test_score_accepting(score, threshold, expected):
    assert speaker_accepts(score, threshold) == expected


@pytest.mark.slow
@pytest.mark.parametrize(
    "cen_audio1, cen_audio2, cen_audio3, cen_audio4, cen_audio5, target_audio, non_target_audio",
    [
        (
            "cce7416f_down.wav",
            "cce7416f_eight.wav",
            "cce7416f_five.wav",
            "cce7416f_four.wav",
            "cce7416f_go.wav",
            "cce7416f_left.wav",
            "c50f55b8_down.wav",
        ),
    ],
)
def test_real_embeddings(
    embedder_model_fixture,
    cen_audio1,
    cen_audio2,
    cen_audio3,
    cen_audio4,
    cen_audio5,
    target_audio,
    non_target_audio,
):
    cen_audio1 = load_audio(Path(__file__).parent / "data" / cen_audio1)
    cen_audio2 = load_audio(Path(__file__).parent / "data" / cen_audio2)
    cen_audio3 = load_audio(Path(__file__).parent / "data" / cen_audio3)
    cen_audio4 = load_audio(Path(__file__).parent / "data" / cen_audio4)
    cen_audio5 = load_audio(Path(__file__).parent / "data" / cen_audio5)
    target_audio = load_audio(Path(__file__).parent / "data" / target_audio)
    non_target_audio = load_audio(Path(__file__).parent / "data" / non_target_audio)

    cen_embed1 = embedder_model_fixture.embed(cen_audio1)
    cen_embed2 = embedder_model_fixture.embed(cen_audio2)
    cen_embed3 = embedder_model_fixture.embed(cen_audio3)
    cen_embed4 = embedder_model_fixture.embed(cen_audio4)
    cen_embed5 = embedder_model_fixture.embed(cen_audio5)
    target_embed = embedder_model_fixture.embed(target_audio)
    non_target_embed = embedder_model_fixture.embed(non_target_audio)
    profile = build_profile([cen_embed1, cen_embed2, cen_embed3, cen_embed4, cen_embed5])
    target_score = cosine_score(profile, target_embed)
    non_target_score = cosine_score(profile, non_target_embed)
    assert target_score > non_target_score
