"""Text-to-speech: edge-tts (online, natural voice) with offline pyttsx3 fallback."""
import asyncio
import os
from . import config


def _offline(text: str, out_path: str) -> str:
    import pyttsx3
    wav_path = os.path.splitext(out_path)[0] + ".wav"
    engine = pyttsx3.init()
    engine.save_to_file(text, wav_path)
    engine.runAndWait()
    return wav_path


def synthesize(text: str, out_path: str, fmt: str = "mp3") -> str:
    try:
        import edge_tts

        async def _go():
            await edge_tts.Communicate(text, config.TTS_VOICE).save(out_path)

        asyncio.run(_go())
        if os.path.getsize(out_path) == 0:
            raise RuntimeError("empty audio")
        return out_path
    except Exception:
        return _offline(text, out_path)