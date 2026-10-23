"""Speech-to-text with local Whisper (faster-whisper). Free, no API key."""
import wave
from dataclasses import dataclass

import numpy as np
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


def _load_wav(path: str) -> np.ndarray:
    """Fallback decoder for 16-bit WAV files (no 'av' library needed)."""
    with wave.open(path, "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw != 2:
        raise ValueError("Fallback decoder only supports 16-bit WAV files")
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    if sr != 16000:
        n = int(len(audio) * 16000 / sr)
        audio = np.interp(np.linspace(0, len(audio) - 1, n),
                          np.arange(len(audio)), audio).astype(np.float32)
    return audio


def transcribe(audio_path: str) -> Transcript:
    try:
        segments, _info = _model.transcribe(audio_path, beam_size=1)
        segs = list(segments)
    except TypeError:
        # av/faster-whisper version mismatch: decode WAV ourselves
        if not audio_path.lower().endswith(".wav"):
            raise
        segments, _info = _model.transcribe(_load_wav(audio_path), beam_size=1)
        segs = list(segments)

    text = " ".join(s.text.strip() for s in segs).strip()
    logprobs = [s.avg_logprob for s in segs]
    nospeech = [s.no_speech_prob for s in segs]
    return Transcript(
        text=text,
        avg_logprob=sum(logprobs) / len(logprobs) if logprobs else None,
        no_speech_prob=max(nospeech) if nospeech else None,
    )