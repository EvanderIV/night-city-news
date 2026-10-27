"""One-off: narrate the entire latest script in the male 'bright and loud'
(dramatic intro) voice — every line, real text — and stitch into one file.

Run with the MAIN venv; it shells out to the Chatterbox venv for synthesis.
"""
import json
import subprocess
import sys
from pathlib import Path

from config_loader import load_config
from paths import PROJECT_ROOT, SEGMENTS_DIR, FINAL_DIR, VOICE_DIR
from voice.synthesizer import _resolve_venv_python

cfg = load_config()
cb = cfg["chatterbox"]
intro = cb["presets"]["intro"]

# The "bright and loud" voice = the intro preset's dramatic reference + settings,
# but applied to EVERY line with the line's own text (not the forced greeting).
ref = str(PROJECT_ROOT / intro["reference"])
gen = {
    "exaggeration": intro.get("exaggeration", 1.2),
    "cfg_weight": intro.get("cfg_weight", 0.3),
    "temperature": intro.get("temperature", 0.6),
    "repetition_penalty": 1.2, "min_p": 0.05, "top_p": 1.0,
}

script = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
for old in SEGMENTS_DIR.glob("narration_*.wav"):
    old.unlink(missing_ok=True)

jobs, outs = [], []
for i, turn in enumerate(script):
    line = turn.get("line", "").strip()
    if not line:
        continue
    out = SEGMENTS_DIR / f"narration_{i:03d}.wav"
    outs.append(out)
    jobs.append({"text": line, "ref": ref, "out": str(out), "gen_kwargs": gen})

spec = {"model_dir": str(PROJECT_ROOT / cb["model_dir"]), "device": "cpu",
        "mode": "full", "quantize_t3": False, "gen_kwargs": gen, "jobs": jobs}

venv_py = _resolve_venv_python(cb["venv_python"])
helper = VOICE_DIR / "chatterbox_synth.py"
print(f"Narrating {len(jobs)} lines through {venv_py.name}...", flush=True)
proc = subprocess.run([str(venv_py), str(helper)], input=json.dumps(spec),
                      capture_output=True, text=True)
ok = {ln[3:].strip() for ln in proc.stdout.splitlines() if ln.startswith("OK ")}
print(f"rendered OK: {len(ok)}/{len(jobs)}", flush=True)

# Stitch into one file with a short pause between lines.
from pydub import AudioSegment
gap = AudioSegment.silent(duration=350)
full = AudioSegment.empty()
for out in outs:
    if str(out) in ok and out.exists():
        full += AudioSegment.from_file(out) + gap
FINAL_DIR.mkdir(parents=True, exist_ok=True)
dest = FINAL_DIR / "male_narration_full.mp3"
full.export(dest, format="mp3")
print(f"DONE -> {dest} ({len(full)/1000:.1f}s)", flush=True)
