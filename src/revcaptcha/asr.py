from typing import Protocol


class Transcriber(Protocol):
    def __init__(self, samples):
        super().__init__()
        self.samples = samples

    def transcribe(self, samples: str) -> str: ...
