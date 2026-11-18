import numpy as np


class FakeTranscriber:
    def __init__(self, samples):
        self.samples = samples

    def transcribe(self, samples: str) -> str:
        return self.samples


class FakeEmbedder:
    def __init__(self):
        self.samples = np.array([1, 2, 3])

    def embed(self, samples: np.ndarray) -> np.ndarray:
        return self.samples
