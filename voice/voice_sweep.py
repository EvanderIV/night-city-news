"""Audition a voice across the levers that matter for zero-shot cloning.

Renders ONE test line for a voice across a small matrix of:
  - reference composition (all clips / a few longest / single longest clip)
  - generation params (cfg_weight, temperature, exaggeration)
  - seed
...so you can A/B by ear and lock the best combo. All takes render in a single
Chatterbox helper call (model loads once; uses [chatterbox].device, e.g. GPU).

Usage (MAIN venv):
    python voice/voice_sweep.py [voice] ["test line"]
    # defaults: voice=skye, a calm narration line

Outputs -> output/voice_sweep/<voice>__<label>.wav
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_loader import get_logger, load_config  # noqa: E402
from paths import EMBEDDINGS_DIR, OUTPUT_DIR, PROJECT_ROOT, TRAINING_DATA_DIR, VOICE_DIR  # noqa: E402
from voice.cloner import _concatenate_clips, _discover_clips  # noqa: E402
from voice.synthesizer import _resolve_venv_python  # noqa: E402

log = get_logger("sweep")

DEFAULT_LINE = "Good morning. Here's a look at what's happening around the city today."
_BASE_GEN = {"repetition_penalty": 1.2, "min_p": 0.05, "top_p": 1.0}


def _longest_clips(voice: str, n: int) -> list[Path]:
    """The ``n`` longest clips for a voice (longest = most prosodic context)."""
    from pydub import AudioSegment

    clips = _discover_clips(TRAINING_DATA_DIR / voice)
    durations = []
    for c in clips:
        try:
            durations.append((len(AudioSegment.from_file(c)), c))
        except Exception:
            pass
    durations.sort(reverse=True)
    return [c for _, c in durations[:n]]


def _alt_reference(voice: str, kind: str, sweep_dir: Path) -> Path:
    """Build (cleaned) alternate references: 'single' / 'few'. 'all' uses the
    existing voice reference."""
    if kind == "all":
        return EMBEDDINGS_DIR / f"{voice}_reference.wav"
    n = 1 if kind == "single" else 3
    out = sweep_dir / f"_ref_{voice}_{kind}.wav"
    _concatenate_clips(_longest_clips(voice, n), out, clean=True)
    return out


def build_matrix(voice: str, sweep_dir: Path) -> list[dict]:
    """The sweep grid. Same seed across param variants makes them comparable."""
    refs = {k: _alt_reference(voice, k, sweep_dir) for k in ("all", "single", "few")}
    S = 1  # fixed seed for the comparable param/reference variants
    grid = [
        # label,            ref,      exag, cfg,  temp, seed
        ("all_base",        "all",    0.5,  0.5,  0.8,  S),   # current defaults
        ("all_calm",        "all",    0.5,  0.5,  0.6,  S),   # lower temp = steadier
        ("all_tight",       "all",    0.4,  0.7,  0.6,  S),   # closer to reference
        ("all_loose",       "all",    0.5,  0.3,  0.6,  S),   # freer / more natural
        ("single_calm",     "single", 0.5,  0.5,  0.6,  S),   # 1 longest clip ref
        ("few_calm",        "few",    0.5,  0.5,  0.6,  S),   # 3 longest clips ref
        ("all_calm_seed7",  "all",    0.5,  0.5,  0.6,  7),   # seed variation
        ("all_calm_seed42", "all",    0.5,  0.5,  0.6,  42),
    ]
    jobs = []
    for label, refk, exag, cfg, temp, seed in grid:
        jobs.append({
            "label": label,
            "ref": str(refs[refk].resolve()),
            "out": str((sweep_dir / f"{voice}__{label}.wav").resolve()),
            "gen_kwargs": {"exaggeration": exag, "cfg_weight": cfg, "temperature": temp, **_BASE_GEN},
            "seed": seed,
        })
    return jobs


def main(argv: list[str]) -> int:
    voice = argv[0] if argv else "skye"
    line = argv[1] if len(argv) > 1 else DEFAULT_LINE
    ref = EMBEDDINGS_DIR / f"{voice}_reference.wav"
    if not ref.exists():
        log.error("No reference for '%s' (%s). Build it first.", voice, ref.name)
        return 1

    cfg = load_config()
    cb = cfg["chatterbox"]
    sweep_dir = OUTPUT_DIR / "voice_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_matrix(voice, sweep_dir)
    for j in jobs:
        j["text"] = line
    spec = {
        "model_dir": str(PROJECT_ROOT / cb["model_dir"]),
        "device": cb.get("device", "auto"),
        "mode": "full",
        "quantize_t3": False,
        "gen_kwargs": {},
        "jobs": [{k: j[k] for k in ("text", "ref", "out", "gen_kwargs", "seed")} for j in jobs],
    }

    venv_py = _resolve_venv_python(cb["venv_python"])
    helper = VOICE_DIR / "chatterbox_synth.py"
    log.info("Sweeping %d takes for '%s' on %s ...", len(jobs), voice, spec["device"])
    proc = subprocess.run([str(venv_py), str(helper)], input=json.dumps(spec),
                          capture_output=True, text=True)
    ok = {ln[3:].strip() for ln in proc.stdout.splitlines() if ln.startswith("OK ")}
    if proc.returncode != 0:
        log.error("Helper exited %s: %s", proc.returncode, proc.stderr[-400:])

    print(f"\nLine: {line!r}")
    print(f"Takes in {sweep_dir} ([ok]/[--] = rendered or failed):")
    for j in jobs:
        mark = "[ok]" if j["out"] in ok else "[--]"
        print(f"  {mark} {Path(j['out']).name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
