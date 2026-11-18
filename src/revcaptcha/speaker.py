from typing import Protocol

import numpy as np
import torch

from revcaptcha.audio import check_samples, load_audio


class Embedder(Protocol):
    def embed(self, samples: np.ndarray) -> np.ndarray: ...


class CamppEmbedder:
    def __init__(self, model_dir):
        import wespeaker

        self.model = wespeaker.load_model(model_dir)
        self.model.set_device("cpu")

    def embed(self, samples) -> np.ndarray:
        audio = load_audio(samples)
        check_samples(audio)
        audio = torch.from_numpy(audio).unsqueeze(0)
        sample_rate = 16000
        embedding = self.model.extract_embedding_from_pcm(audio, sample_rate)
        return embedding

        # audio = load_audio(samples)
        # check_samples(samples)
        # samples = torch.from_numpy(samples).unsqueeze(0)
        # sample_rate = 16000
        # embedding = self.model.extract_embedding_from_pcm(samples, sample_rate)
        # return embedding


# embedder = CamppEmbedder("../../models/campplus")
# embedder.embed("../../data/samples/edgar_VKR2E_mono.wav")
