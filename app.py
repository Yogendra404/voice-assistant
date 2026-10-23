"""Gradio UI: upload or record audio, get a spoken reply + latency breakdown."""

import os
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

import gradio as gr
from va.pipeline import run, bottleneck


def respond(audio_path, history):
    history = history or []
    if audio_path is None:
        return "", "", None, "Please upload or record some audio first.", history
    res = run(audio_path, history)
    if res.transcript and not res.errors:
        history = history + [{"role": "user", "content": res.transcript},
                             {"role": "assistant", "content": res.reply}]
    t = res.timings
    report = (f"STT: {t['stt']:.2f}s | LLM: {t['llm']:.2f}s | TTS: {t['tts']:.2f}s | "
              f"TOTAL: {t['total']:.2f}s\nBottleneck: {bottleneck(t)}")
    if res.unclear:
        report += "\n⚠️ Low-confidence / unclear audio detected."
    if res.errors:
        report += "\n" + "\n".join(res.errors)
    return res.transcript, res.reply, res.audio_path, report, history


with gr.Blocks(title="Voice Assistant") as demo:
    gr.Markdown("# 🎙️ Voice Assistant (STT → LLM → TTS)")
    state = gr.State([])
    audio_in = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Your voice")
    btn = gr.Button("Send", variant="primary")
    transcript = gr.Textbox(label="Transcript (STT)")
    reply = gr.Textbox(label="Assistant reply (LLM)")
    audio_out = gr.Audio(label="Spoken reply (TTS)", autoplay=True)
    timing = gr.Textbox(label="Latency report", lines=3)
    btn.click(respond, [audio_in, state], [transcript, reply, audio_out, timing, state])

if __name__ == "__main__":
        demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)