from typing import Protocol

import numpy as np

from revcaptcha.audio import check_samples

MODEL_NAME = "wespeaker-voxceleb-campplus"


class Embedder(Protocol):
    def embed(self, samples: np.ndarray) -> np.ndarray: ...


class CamppEmbedder:
    def __init__(self, model_dir):
        import wespeaker

        self.model = wespeaker.load_model(model_dir)
        self.model.set_device("cpu")

    def embed(self, samples) -> np.ndarray:
        check_samples(samples)
        import torch

        audio = torch.from_numpy(samples).unsqueeze(0)
        sample_rate = 16000
        audio = audio * 32768  # Scale back
        embedding = self.model.extract_embedding_from_pcm(audio, sample_rate)
        embedding = embedding.numpy()
        return embedding
