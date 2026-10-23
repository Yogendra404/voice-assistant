"""Generate test clips in samples/ (clear, noisy, quiet/mumbled, silence).
Uses Windows' built-in offline voice, so no API is needed.
Run from the project root:  python scripts/make_test_audio.py
"""
import os, wave
import numpy as np
import pyttsx3

os.makedirs("samples", exist_ok=True)
TEXT = "Hi there! Can you tell me a fun fact about the planet Mars?"

engine = pyttsx3.init()
engine.save_to_file(TEXT, "samples/_clean.wav")
engine.runAndWait()

with wave.open("samples/_clean.wav") as w:
    params, frames = w.getparams(), w.readframes(w.getnframes())
if params.sampwidth != 2:
    raise SystemExit("Unexpected audio format; record your own test clips instead.")
sig = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
os.remove("samples/_clean.wav")


def save(name, data):
    data = np.clip(data, -32768, 32767).astype(np.int16)
    with wave.open(f"samples/{name}.wav", "wb") as w:
        w.setparams(params)
        w.writeframes(data.tobytes())
    print("saved", name)


rng = np.random.default_rng(0)
save("01_clear", sig)
save("02_noisy", sig + rng.normal(0, 3500, sig.shape))
save("03_very_quiet_mumble", sig * 0.03 + rng.normal(0, 300, sig.shape))
save("04_silence", np.zeros(params.framerate * 3))