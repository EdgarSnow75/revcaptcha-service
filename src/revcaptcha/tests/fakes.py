import numpy as np


class FakeTranscriber:
    def __init__(self, text):
        self.text = text
        self.calls = 0

    def transcribe(self, samples: np.ndarray) -> str:
        self.calls += 1
        return self.text


class FakeEmbedder:
    def __init__(self, vc):
        self.vc = np.array(vc)
        self.calls = 0

    def embed(self, samples: np.ndarray) -> np.ndarray:
        self.calls += 1
        return self.vc
