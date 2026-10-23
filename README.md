# 🎙️ Voice Assistant: STT → LLM → TTS

A complete voice-in, voice-out pipeline. Upload (or record) audio → it is transcribed →
an LLM answers → the answer is spoken back as an mp3.

**🎥 Demo video:** <PASTE YOUR LINK HERE>

## Architecture
audio file → Whisper STT (+confidence) → GPT LLM (+history) → OpenAI TTS → reply.mp3

   | Stage | Model |
   |---|---|
   | STT | `faster-whisper` (Whisper `base`, local CPU) |
   | LLM | Gemini (`gemini-3.1-flash-lite`) |
   | TTS | `edge-tts` (`en-US-AriaNeural`), offline `pyttsx3` fallback |

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add your OPENAI_API_KEY
python app.py               # web UI at http://127.0.0.1:7860
python cli.py samples/01_clear.wav
```

## Test audio & benchmark
```bash
python scripts/make_test_audio.py   # clear / noisy / quiet / silent clips in samples/
python benchmark.py                 # writes latency_results.md
[Demo screenshot](docs/demo_screenshot.png)
```

## Latency results
<!-- paste the table from latency_results.md -->

**Where the time goes:** <!-- e.g. LLM ~55%, TTS ~30%, STT ~15% -->

**Ideas to cut latency:** stream LLM tokens and start TTS per sentence, cap `max_tokens`,
keep replies short.

## Handling unclear / noisy audio
- Whisper's `no_speech_prob` and `avg_logprob` flag unclear transcripts (thresholds in `va/config.py`).
- Empty transcript or silence → no LLM call; the assistant speaks a polite "please repeat".
- Low-confidence transcript → the LLM is told it may be wrong and asks for clarification.
- Every stage is wrapped in try/except, so errors produce a spoken fallback instead of a crash.

| Test clip | Behaviour observed |
|---|---|
| 01_clear | |
| 02_noisy | |
| 03_very_quiet_mumble | |
| 04_silence | |

## Example inputs and outputs

| Input (audio you speak) | Assistant reply (audio) |
|---|---|
| [01_clear.wav](examples/inputs/01_clear.wav) | [01_clear_reply.mp3](examples/outputs/01_clear_reply.mp3) |
| [02_noisy.wav](examples/inputs/02_noisy.wav) | [02_noisy_reply.mp3](examples/outputs/02_noisy_reply.mp3) |
| [03_very_quiet_mumble.wav](examples/inputs/03_very_quiet_mumble.wav) | [03_very_quiet_mumble_reply.mp3](examples/outputs/03_very_quiet_mumble_reply.mp3) |
| [04_silence.wav](examples/inputs/04_silence.wav) | [04_silence_reply.mp3](examples/outputs/04_silence_reply.mp3) |

Create a file named `.env` in the project folder:

GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-3.1-flash-lite
WHISPER_MODEL=base
TTS_VOICE=en-US-AriaNeural