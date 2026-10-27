"""Render a saved script to a plain concatenated audio file.

Synthesizes every turn (honoring each turn's "speaker" and optional "preset",
including [voices] registry voices like skye/johnny) and stitches the raw
segments back-to-back. NO intro music bed, NO radio grain, NO outro jingle - just
the clean voices joined with the natural inter-line pause.

Usage (run with the MAIN venv; it shells out to the Chatterbox venv):
    python render_script.py [path/to/script.json] [--wav]

With no path, uses the most recent output/scripts/script_*.json. Output goes to
output/final/<script-stem>_clean.mp3 (or .wav with --wav).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from config_loader import get_logger, load_config
from paths import FINAL_DIR, SCRIPTS_DIR
from voice.synthesizer import synthesize_script

log = get_logger("render")


def _pick_script(argv: list[str]) -> Path | None:
    for a in argv:
        if not a.startswith("-"):
            return Path(a)
    scripts = sorted(SCRIPTS_DIR.glob("script_*.json"))
    return scripts[-1] if scripts else None


def main(argv: list[str]) -> int:
    fmt = "wav" if "--wav" in argv else "mp3"
    path = _pick_script(argv)
    if path is None or not path.exists():
        log.error("No script found. Pass a path or create output/scripts/script_*.json.")
        return 1

    script = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(script, list) or not script:
        log.error("%s is not a non-empty list of turns.", path.name)
        return 1

    config = load_config()
    log.info("Rendering %d turns from %s ...", len(script), path.name)
    segments = synthesize_script(script, config)

    from pydub import AudioSegment

    full = AudioSegment.empty()
    used = 0
    for seg in segments:
        seg = Path(seg)
        if not seg.exists():
            continue
        try:
            full += AudioSegment.from_file(seg)
            used += 1
        except Exception as exc:
            log.warning("Skipping unreadable segment %s (%s).", seg.name, exc)

    if used == 0:
        log.error("No segments rendered; nothing to write.")
        return 1

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    dest = FINAL_DIR / f"{path.stem}_clean.{fmt}"
    full.export(dest, format=fmt)
    log.info("DONE -> %s (%.1fs, %d/%d segments).", dest, len(full) / 1000, used, len(segments))
    print(dest)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
