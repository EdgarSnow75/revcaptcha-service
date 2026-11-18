import numpy as np
import pytest
import torch

from revcaptcha.speaker import CamppEmbedder, Embedder
from revcaptcha.tests.fakes import FakeEmbedder


def test_check_fake_embed():
    def run(embedder: Embedder, samples: np.ndarray) -> np.ndarray:
        return embedder.embed(samples)

    embedder = FakeEmbedder()
    result = run(embedder, np.zeros(10))
    assert np.array_equal(result, [1, 2, 3])


@pytest.mark.slow
@pytest.mark.parametrize(
    "sample, expected", [("../../data/samples/edgar_VKR2E_mono.wav", (512,))]
)
def test_embed_shape(call_embed_model, sample, expected):
    assert call_embed_model.embed(sample).shape == expected
    assert call_embed_model.embed(sample).dtype == torch.float32


@pytest.mark.slow
@pytest.mark.parametrize(
    "sample1, sample2",
    [
        (
            "../../data/samples/edgar_VKR2E_mono.wav",
            "../../data/samples/edgar_VKR2E_mono.wav",
        )
    ],
)
def test_embed_twice(call_embed_model, sample1, sample2):
    embed1 = call_embed_model.embed(sample1)
    embed2 = call_embed_model.embed(sample2)
    assert np.allclose(embed1, embed2)


@pytest.mark.slow
@pytest.mark.parametrize(
    "cen_sample1, cen_sample2, test_sample_t, test_sample_f",
    [
        (
            "./tests/data/cce7416f_down.wav",
            "./tests/data/cce7416f_eight.wav",
            "./tests/data/cce7416f_five.wav",
            "./tests/data/c50f55b8_down.wav",
        ),
        # (
        #     "./data/cce7416f_down.wav",
        #     "./data/cce7416f_eight.wav",
        #     "./data/c50f55b8_down.wav",
        #     False,
        # ),
    ],
)
def test_cosine_similarity(
    call_embed_model, cen_sample1, cen_sample2, test_sample_t, test_sample_f
):
    cen_embed1 = call_embed_model.embed(cen_sample1)
    cen_embed2 = call_embed_model.embed(cen_sample2)
    test_embed_t = call_embed_model.embed(test_sample_t)
    test_embed_f = call_embed_model.embed(test_sample_f)
    centriod_embed = (cen_embed1 + cen_embed2) / 2
    cosine_1 = np.dot(centriod_embed, test_embed_t) / (
        np.linalg.norm(centriod_embed) * np.linalg.norm(test_embed_t)
    )
    cosine_2 = np.dot(centriod_embed, test_embed_f) / (
        np.linalg.norm(centriod_embed) * np.linalg.norm(test_embed_f)
    )
    assert cosine_1 > cosine_2
