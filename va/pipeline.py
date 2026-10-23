"""STT -> LLM -> TTS, with per-stage timing and graceful degradation."""
import os
import time
import uuid
from dataclasses import dataclass, field

from . import llm, stt, tts

FALLBACK_REPLY = "Sorry, I couldn't catch that. Could you please say it again a bit more clearly?"
ERROR_REPLY = "Sorry, something went wrong on my side. Please try again in a moment."


@dataclass
class Result:
    transcript: str = ""
    reply: str = ""
    audio_path: str | None = None
    timings: dict = field(default_factory=dict)  # seconds
    unclear: bool = False
    errors: list = field(default_factory=list)


def run(audio_path: str, history: list | None = None, out_dir: str = "outputs") -> Result:
    os.makedirs(out_dir, exist_ok=True)
    res = Result()
    t_start = time.perf_counter()

    # ---- 1. STT ----
    t = time.perf_counter()
    transcript = None
    try:
        transcript = stt.transcribe(audio_path)
        res.transcript = transcript.text
        res.unclear = transcript.is_unclear
    except Exception as e:  # bad file, network, API error...
        res.errors.append(f"STT failed: {e}")
        res.unclear = True
    res.timings["stt"] = time.perf_counter() - t

    # ---- 2. LLM ----
    t = time.perf_counter()
    if transcript is None or transcript.is_empty:
        res.reply = ERROR_REPLY if res.errors else FALLBACK_REPLY  # nothing to send to the LLM
    else:
        try:
            res.reply = llm.respond(transcript.text, history, unclear=res.unclear)
        except Exception as e:
            res.errors.append(f"LLM failed: {e}")
            res.reply = ERROR_REPLY
    res.timings["llm"] = time.perf_counter() - t

    # ---- 3. TTS ----
    t = time.perf_counter()
    try:
        out = os.path.join(out_dir, f"reply_{uuid.uuid4().hex[:8]}.mp3")
        res.audio_path = tts.synthesize(res.reply, out)
    except Exception as e:
        res.errors.append(f"TTS failed: {e}")
    res.timings["tts"] = time.perf_counter() - t

    res.timings["total"] = time.perf_counter() - t_start
    return res


def bottleneck(timings: dict) -> str:
    stages = {k: v for k, v in timings.items() if k != "total"}
    name = max(stages, key=stages.get)
    share = 100 * stages[name] / timings["total"] if timings["total"] else 0
    return f"{name.upper()} ({share:.0f}% of total)"