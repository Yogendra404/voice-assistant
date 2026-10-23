import os
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_PATH, encoding="utf-8-sig")

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")      # tiny / base / small
LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-AriaNeural")

NO_SPEECH_THRESHOLD = 0.6
LOGPROB_THRESHOLD = -1.0


@lru_cache(maxsize=1)
def get_client():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(f"GEMINI_API_KEY is missing. Looked for .env at: {ENV_PATH} (exists: {ENV_PATH.exists()})")
    from google import genai
    from google.genai import types
    return genai.Client(api_key=key, http_options=types.HttpOptions(timeout=20000))  # 20 s