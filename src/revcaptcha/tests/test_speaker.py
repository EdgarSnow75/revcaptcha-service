from pathlib import Path

import numpy as np
import pytest

from revcaptcha.audio import load_audio
from revcaptcha.speaker import Embedder
from revcaptcha.tests.fakes import FakeEmbedder


def test_check_fake_embed():
    def run(embedder: Embedder, samples: np.ndarray) -> np.ndarray:
        return embedder.embed(samples)

    embedder = FakeEmbedder([1, 2, 3])
    result = run(embedder, np.zeros(10))
    assert np.array_equal(result, [1, 2, 3])


@pytest.mark.slow
@pytest.mark.parametrize("sample, expected", [("cce7416f_down.wav", (512,))])
def test_embed_shape(embedder_model_fixture, sample, expected):
    sample = Path(__file__).parent / "data" / sample
    assert embedder_model_fixture.embed(load_audio(sample)).shape == expected
    assert embedder_model_fixture.embed(load_audio(sample)).dtype == "float32"


@pytest.mark.slow
@pytest.mark.parametrize(
    "sample1, sample2",
    [
        (
            "cce7416f_eight.wav",
            "cce7416f_eight.wav",
        )
    ],
)
def test_embed_twice(embedder_model_fixture, sample1, sample2):
    sample1 = Path(__file__).parent / "data" / sample1
    sample2 = Path(__file__).parent / "data" / sample2
    embed1 = embedder_model_fixture.embed(load_audio(sample1))
    embed2 = embedder_model_fixture.embed(load_audio(sample2))
    assert np.allclose(embed1, embed2)


@pytest.mark.slow
@pytest.mark.parametrize(
    "cen_sample1, cen_sample2, test_sample_t, test_sample_f",
    [
        (
            "cce7416f_down.wav",
            "cce7416f_eight.wav",
            "cce7416f_five.wav",
            "c50f55b8_down.wav",
        ),
    ],
)
def test_cosine_similarity(
    embedder_model_fixture, cen_sample1, cen_sample2, test_sample_t, test_sample_f
):
    cen_sample1 = Path(__file__).parent / "data" / cen_sample1
    cen_sample2 = Path(__file__).parent / "data" / cen_sample2
    test_sample_t = Path(__file__).parent / "data" / test_sample_t
    test_sample_f = Path(__file__).parent / "data" / test_sample_f
    cen_embed1 = embedder_model_fixture.embed(load_audio(cen_sample1))
    cen_embed2 = embedder_model_fixture.embed(load_audio(cen_sample2))
    test_embed_t = embedder_model_fixture.embed(load_audio(test_sample_t))
    test_embed_f = embedder_model_fixture.embed(load_audio(test_sample_f))
    centriod_embed = (cen_embed1 + cen_embed2) / 2
    cosine_1 = np.dot(centriod_embed, test_embed_t) / (
        np.linalg.norm(centriod_embed) * np.linalg.norm(test_embed_t)
    )
    cosine_2 = np.dot(centriod_embed, test_embed_f) / (
        np.linalg.norm(centriod_embed) * np.linalg.norm(test_embed_f)
    )
    assert cosine_1 > cosine_2
