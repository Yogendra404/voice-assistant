"""Text-to-speech with edge-tts. Free, no API key (needs internet)."""
import asyncio
import edge_tts
from . import config
def synthesize(text: str, out_path: str, fmt: str = "mp3") -> str:
    async def _go():
        await edge_tts.Communicate(text, config.TTS_VOICE).save(out_path)
    asyncio.run(_go())
    return out_path
