"""Usage: python cli.py path/to/audio.wav"""
import sys
from va.pipeline import run, bottleneck

if len(sys.argv) < 2:
    sys.exit("Usage: python cli.py <audio_file>")

res = run(sys.argv[1])
print("Transcript:", res.transcript or "(empty)")
print("Reply     :", res.reply)
print("Audio     :", res.audio_path)
print("Unclear   :", res.unclear)
for k, v in res.timings.items():
    print(f"  {k:>5}: {v:.2f}s")
print("Bottleneck:", bottleneck(res.timings))
for e in res.errors:
    print("ERROR:", e)