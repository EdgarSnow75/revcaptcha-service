import numpy as np


class FakeTranscriber:
    def __init__(self, text):
        self.text = text

    def transcribe(self, samples: np.ndarray) -> str:
        return self.text


class FakeEmbedder:
    def __init__(self, vc):
        self.vc = np.array(vc)

    def embed(self, samples: np.ndarray) -> np.ndarray:
        return self.vc
