"""Run every file in samples/ through the pipeline several times; report median latency."""
import glob
import statistics as st
from va.pipeline import run

RUNS = 3  # runs per file (after one warm-up)

files = sorted(f for ext in ("wav", "mp3", "m4a", "webm") for f in glob.glob(f"samples/*.{ext}"))
if not files:
    raise SystemExit("Put some audio files in samples/ first.")

print("Warm-up run...")
run(files[0])

rows = []
for f in files:
    stt_t, llm_t, tts_t, tot_t, unclear, transcript = [], [], [], [], False, ""
    for _ in range(RUNS):
        r = run(f)
        t = r.timings
        stt_t.append(t["stt"]); llm_t.append(t["llm"])
        tts_t.append(t["tts"]); tot_t.append(t["total"])
        unclear, transcript = r.unclear, r.transcript[:40]
    rows.append((f, st.median(stt_t), st.median(llm_t), st.median(tts_t), st.median(tot_t), unclear, transcript))
    print(f"{f}: median total {rows[-1][4]:.2f}s  unclear={unclear}")

lines = [f"Median of {RUNS} runs per file (after warm-up)\n",
         "| File | STT (s) | LLM (s) | TTS (s) | Total (s) | Unclear? | Transcript |",
         "|---|---|---|---|---|---|---|"]
for f, a, b, c, d, u, tx in rows:
    lines.append(f"| {f} | {a:.2f} | {b:.2f} | {c:.2f} | {d:.2f} | {u} | {tx} |")
lines.append(f"| **Average** | {st.mean(r[1] for r in rows):.2f} | {st.mean(r[2] for r in rows):.2f} | "
             f"{st.mean(r[3] for r in rows):.2f} | {st.mean(r[4] for r in rows):.2f} | | |")
open("latency_results.md", "w").write("\n".join(lines))
print("\n".join(lines))