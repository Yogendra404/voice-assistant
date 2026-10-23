import os
from functools import lru_cache
from dotenv import load_dotenv
load_dotenv()
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")      # tiny / base / small
LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-AriaNeural")
NO_SPEECH_THRESHOLD = 0.6
LOGPROB_THRESHOLD = -1.0
@lru_cache(maxsize=1)
def get_client():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to your .env file.")
    from google import genai
    return genai.Client(api_key=key)
