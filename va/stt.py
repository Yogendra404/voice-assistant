"""Speech-to-text with local Whisper (faster-whisper). Free, no API key."""
from dataclasses import dataclass
from faster_whisper import WhisperModel
from . import config
_model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
@dataclass
class Transcript:
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    @property
    def is_empty(self) -> bool:
        return not self.text.strip()
    @property
    def is_unclear(self) -> bool:
        if self.is_empty:
            return True
        if self.no_speech_prob is not None and self.no_speech_prob > config.NO_SPEECH_THRESHOLD:
            return True
        if self.avg_logprob is not None and self.avg_logprob < config.LOGPROB_THRESHOLD:
            return True
        return False
def transcribe(audio_path: str) -> Transcript:
    segments, _info = _model.transcribe(audio_path, beam_size=1)
    segs = list(segments)
    text = " ".join(s.text.strip() for s in segs).strip()
    logprobs = [s.avg_logprob for s in segs]
    nospeech = [s.no_speech_prob for s in segs]
    return Transcript(
        text=text,
        avg_logprob=sum(logprobs) / len(logprobs) if logprobs else None,
        no_speech_prob=max(nospeech) if nospeech else None,
    )
