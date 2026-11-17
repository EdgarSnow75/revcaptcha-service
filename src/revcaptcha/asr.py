from typing import Protocol

from revcaptcha.audio import check_samples


class Transcriber(Protocol):
    def __init__(self, samples):
        super().__init__()
        self.samples = samples

    def transcribe(self, samples: str) -> str: ...


class WhisperTranscriber:
    def __init__(self, device_name="cpu", model_name="small"):
        import whisper

        self.model = whisper.load_model(model_name, device_name)

    def transcribe(self, samples):
        check_samples(samples)
        data = self.model.transcribe(
            samples, language="en", task="transcribe", fp16=False
        )
        text = data["text"].strip()
        return text
