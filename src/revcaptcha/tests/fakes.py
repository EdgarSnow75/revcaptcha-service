class FakeTranscriber:
    def __init__(self, samples):
        self.samples = samples

    def transcribe(self, samples: str) -> str:
        return self.samples
