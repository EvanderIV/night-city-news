"""Final broadcast assembly.

Stitches processed voice segments together with an intro music bed, topic-switch
swooshes, and an FM-station outro jingle, then writes the dated final broadcast
to ``output/final/broadcast_YYYY-MM-DD.mp3``.

Assembly:
  1. Spoken body: voice segments in order, with a swoosh sting layered in
     wherever the broadcast switches topics (detected from transition cue
     phrases in the script).
  2. Intro bed: a non-looping music bed from ``background/`` plays under the
     opening (intro/hype) lines, then fades out over the next few seconds once
     the intro sequence ends.
  3. Outro: a random "prejingle" smoothing sting, then exactly ``prejingle_gap_ms``
     of silence, then a random jingle from a randomly-picked FM frequency folder.

Every asset is optional: if a directory is empty the mixer logs a warning and
continues without that element.
"""
from __future__ import annotations

import random
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_loader import get_logger  # noqa: E402
from paths import (  # noqa: E402
    BACKGROUND_DIR,
    FINAL_DIR,
    JINGLES_DIR,
    PREJINGLE_DIR,
    TRANSITIONS_DIR,
)

log = get_logger("mixer")

_AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}

# Swooshes mark MAJOR section changes (e.g. the news block -> the weather), not
# story-to-story hand-offs within the news. We classify each line into a coarse
# section and only swoosh when entering one of _SWOOSH_SECTIONS from a different
# section. Markers are matched as substrings (case-insensitive) and chosen to be
# distinctive enough to rarely appear inside an ordinary satirical news story.
_SPONSOR_MARKERS = (
    "brought to you by", "sponsored by", "this segment is brought",
    "this broadcast is brought", "a word from our sponsor", "today's sponsor",
    "our sponsor",
)
_WEATHER_MARKERS = (
    "commut", "forecast", "temperature", "overcast", "precipitation",
    "humidity", "wind chill", "miles per hour", "traffic",
    "the morning drive", "the evening drive",
)
_SIGNOFF_MARKERS = (
    "thanks for joining", "thank you for joining", "thank you for sharing",
    "back tomorrow", "see you tomorrow", "until next time", "until tomorrow",
    "that's all for", "that wraps",
)
# Which sections get a swoosh when first entered. The news block does NOT (so
# story-to-story hand-offs stay swoosh-free); the outro jingle covers the close.
_SWOOSH_SECTIONS = frozenset({"weather", "sponsor"})


def _section_of(line: str, city: str = "") -> str:
    """Classify a line into a coarse broadcast section.

    Order matters (sponsor read can mention weather, etc.). Anything that isn't
    clearly weather/sponsor/sign-off is treated as part of the news block.
    """
    t = line.lower()
    if any(k in t for k in _SPONSOR_MARKERS):
        return "sponsor"
    if any(k in t for k in _WEATHER_MARKERS) or (city and city in t):
        return "weather"
    if any(k in t for k in _SIGNOFF_MARKERS):
        return "signoff"
    return "news"


def _list_assets(directory: Path) -> list[Path]:
    """All audio files directly in ``directory`` (case-insensitive extension).

    Case-insensitivity matters: the jingle files ship as ``.Ogg`` and the
    deployment target (Debian) has a case-sensitive filesystem.
    """
    if not directory.is_dir():
        return []
    files = [
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in _AUDIO_EXTS
    ]
    files.sort()
    return files


def _turn_index(seg_path: Path) -> int | None:
    """Recover the script turn index encoded in a segment filename.

    Names look like ``segment_003_male_host.wav`` (or ``processed_segment_003_…``
    after post-processing). Returns the int index, or None if unparseable.
    """
    m = re.search(r"segment_(\d+)", seg_path.stem)
    return int(m.group(1)) if m else None


def _last_intro_turn(script: list[dict[str, Any]] | None) -> int:
    """Index of the last opening turn (intro/hype preset). Defaults to 0."""
    if not script:
        return 0
    last = 0
    for i, turn in enumerate(script):
        if (turn.get("preset") or "").strip().lower() in ("intro", "hype"):
            last = i
    return last


def _pick_outro_jingle() -> Path | None:
    """Pick a random FM frequency folder, then a random jingle within it.

    The frequency is chosen fairly (uniformly over the folders, NOT over the
    pooled files) so a station with more jingles isn't over-represented.
    """
    if not JINGLES_DIR.is_dir():
        return None
    fm_folders = [
        d for d in JINGLES_DIR.iterdir()
        if d.is_dir() and d.resolve() != PREJINGLE_DIR.resolve()
    ]
    random.shuffle(fm_folders)  # try folders in random order; skip any empty one
    for folder in fm_folders:
        jingles = _list_assets(folder)
        if jingles:
            log.info("Outro jingle: %s / %s.", folder.name, random.choice(jingles).name)
            return random.choice(jingles)
    return None


def mix_broadcast(
    segment_paths: list[Path],
    config: dict[str, Any],
    script: list[dict[str, Any]] | None = None,
    out_dir: Path = FINAL_DIR,
    timeline: list[dict[str, Any]] | None = None,
    out_name: str | None = None,
    out_format: str | None = None,
) -> Path:
    """Assemble the final broadcast from processed voice segments.

    ``script`` (the validated dialogue turns) lets the mixer find the intro
    boundary (for the bed) and detect topic switches (for the swooshes). It is
    optional; without it the bed and swooshes are skipped.

    If ``timeline`` (a list) is given, it is populated with absolute-timed cues
    ({kind, start_ms, end_ms, ...}) describing what plays when - the video
    compositor consumes this so the picture matches this exact audio. ``out_name``
    overrides the output filename stem and ``out_format`` the container, so the
    video's clean audio can be written next to (not over) the radio broadcast.

    Returns the path to the written file.
    """
    from pydub import AudioSegment

    out_dir.mkdir(parents=True, exist_ok=True)
    audio_cfg = config.get("audio", {})
    out_format = out_format or audio_cfg.get("output_format", "mp3")

    bed_volume = float(audio_cfg.get("intro_bed_volume", audio_cfg.get("background_volume", 0.18)))
    bed_fade_ms = int(audio_cfg.get("intro_bed_fade_ms", 5000))
    bed_min_ms = int(audio_cfg.get("intro_bed_min_ms", 8000))
    prejingle_gap_ms = int(audio_cfg.get("prejingle_gap_ms", 1000))
    swoosh_gain_db = float(audio_cfg.get("swoosh_gain_db", -2.0))
    max_swooshes = int(audio_cfg.get("max_swooshes", 8))
    swoosh_min_gap_ms = int(audio_cfg.get("swoosh_min_gap_ms", 6000))

    script_by_idx = {i: t for i, t in enumerate(script)} if script else {}
    intro_end_turn = _last_intro_turn(script)
    swooshes = _list_assets(TRANSITIONS_DIR)
    weather_city = config.get("weather", {}).get("location", "").lower()

    # --- Build the spoken body, layering a swoosh at each major SECTION change
    # (news -> weather -> sponsor), not at every story-to-story hand-off. ---
    body = AudioSegment.empty()
    intro_end_ms: int | None = None
    swoosh_count = 0
    last_swoosh_ms = -10 ** 9  # so the first eligible change always qualifies
    prev_section: str | None = None

    loaded = 0
    for seg_path in segment_paths:
        seg_path = Path(seg_path)
        if not seg_path.exists():
            log.warning("Segment missing, skipping: %s", seg_path)
            continue

        idx = _turn_index(seg_path)
        turn = script_by_idx.get(idx) if idx is not None else None

        # Section for this segment (drives swoosh + the video shot choice).
        if idx is not None and idx <= intro_end_turn:
            section = "intro"
        elif turn is not None:
            section = _section_of(turn.get("line", ""), weather_city)
        else:
            section = "news"

        # Swoosh BEFORE this segment when it ENTERS a swoosh-worthy section
        # (weather/sponsor) that differs from the previous section - one swoosh
        # per section entry, subject to the cap and minimum spacing.
        if turn is not None and idx is not None and idx > intro_end_turn:
            if (
                swooshes and section in _SWOOSH_SECTIONS and section != prev_section
                and swoosh_count < max_swooshes
                and len(body) - last_swoosh_ms >= swoosh_min_gap_ms
            ):
                try:
                    sw_start = len(body)
                    body += AudioSegment.from_file(random.choice(swooshes)).apply_gain(swoosh_gain_db)
                    swoosh_count += 1
                    last_swoosh_ms = len(body)
                    if timeline is not None:
                        timeline.append({"kind": "swoosh", "start_ms": sw_start, "end_ms": len(body)})
                    log.info("Swoosh -> entering '%s' section.", section)
                except Exception as exc:
                    log.debug("Swoosh load failed (%s).", exc)
            prev_section = section

        try:
            seg_start = len(body)
            body += AudioSegment.from_file(seg_path)
            loaded += 1
            if timeline is not None:
                timeline.append({
                    "kind": "segment",
                    "start_ms": seg_start,
                    "end_ms": len(body),
                    "turn_index": idx,
                    "speaker": (turn or {}).get("speaker", "male_host"),
                    "section": section,
                    "line": (turn or {}).get("line", ""),
                })
        except Exception as exc:
            log.warning("Could not load segment %s (%s); skipping.", seg_path.name, exc)
            continue

        # Mark where the intro sequence ends (for the bed fade).
        if idx is not None and idx == intro_end_turn:
            intro_end_ms = len(body)

    if loaded == 0:
        log.error("No voice segments could be loaded; producing silent placeholder.")
        body = AudioSegment.silent(duration=2000)
    if swoosh_count:
        log.info("Inserted %d topic-switch swoosh(es).", swoosh_count)

    # --- Intro music bed (non-looping; stays at full level through the intro,
    # then fades to zero over the next few seconds AFTER the intro ends). ---
    # The fade always begins at `hold_ms` (the intro end), never earlier, so the
    # bed is unmistakably present for the whole open. A floor (`bed_min_ms`)
    # guards against an under-tagged intro collapsing the bed too early.
    bed_path = _pick_random(BACKGROUND_DIR)
    hold_ms = max(intro_end_ms or 0, bed_min_ms)
    if bed_path and hold_ms and bed_volume > 0:
        try:
            bed = AudioSegment.from_file(bed_path)
            bed = bed[: hold_ms + bed_fade_ms]  # cap length; never loop
            # fade_out touches the LAST N ms, so N = whatever material sits past
            # the intro end -> the fade starts exactly at hold_ms.
            fade_len = min(bed_fade_ms, max(0, len(bed) - hold_ms))
            bed = bed.apply_gain(20.0 * _safe_log10(bed_volume))
            if fade_len > 0:
                bed = bed.fade_out(fade_len)
            body = body.overlay(bed)  # overlays at position 0 (the show open)
            log.info(
                "Intro bed '%s': full under %.1fs of intro, then %.1fs fade-out.",
                bed_path.name, hold_ms / 1000, fade_len / 1000,
            )
        except Exception as exc:
            log.warning("Intro bed mix failed (%s); continuing without it.", exc)
    elif not bed_path:
        log.warning("No intro bed found in %s; skipping bed.", BACKGROUND_DIR)

    broadcast = body

    # --- Outro: prejingle, then the FM jingle starts prejingle_gap_ms after the
    # prejingle BEGINS (so the two overlap), via overlay rather than concat. ---
    prejingle = _pick_random(PREJINGLE_DIR)
    jingle = _pick_outro_jingle()
    pre_audio = _safe_load(prejingle, AudioSegment)
    jin_audio = _safe_load(jingle, AudioSegment)
    outro_start = len(broadcast)
    if pre_audio is not None and jin_audio is not None:
        total = max(len(pre_audio), prejingle_gap_ms + len(jin_audio))
        outro = AudioSegment.silent(duration=total)
        outro = outro.overlay(pre_audio, position=0)
        outro = outro.overlay(jin_audio, position=prejingle_gap_ms)
        broadcast += outro
    elif pre_audio is not None:
        broadcast += pre_audio
    elif jin_audio is not None:
        broadcast += jin_audio
    else:
        log.warning("No outro jingle/prejingle found under %s.", JINGLES_DIR)
    if timeline is not None and len(broadcast) > outro_start:
        timeline.append({"kind": "outro", "start_ms": outro_start, "end_ms": len(broadcast)})

    # Tiny safety fade to avoid an end-of-file click (keeps the jingle intact).
    broadcast = broadcast.fade_out(min(150, len(broadcast)))

    stem = out_name or f"broadcast_{date.today().isoformat()}"
    out_path = out_dir / f"{stem}.{out_format}"
    broadcast.export(out_path, format=out_format)
    log.info("Final broadcast written: %s (%.1fs).", out_path, len(broadcast) / 1000)
    return out_path


def _pick_random(directory: Path) -> Path | None:
    assets = _list_assets(directory)
    return random.choice(assets) if assets else None


def _safe_load(path: Path | None, AudioSegment):
    """Load an audio file, returning None (with a debug log) on failure/absence."""
    if path is None:
        return None
    try:
        return AudioSegment.from_file(path)
    except Exception as exc:
        log.debug("Audio load failed for %s (%s).", path, exc)
        return None


def _safe_log10(x: float) -> float:
    import math

    return math.log10(max(x, 1e-6))


if __name__ == "__main__":
    import glob
    import json

    from config_loader import load_config
    from paths import SCRIPTS_DIR, SEGMENTS_DIR

    cfg = load_config()
    segs = sorted(Path(p) for p in glob.glob(str(SEGMENTS_DIR / "*segment_*.wav")))
    if not segs:
        print("No segments in", SEGMENTS_DIR, "- run synthesis first.")
        raise SystemExit(1)
    # Pair with the most recent script (for bed boundary + swoosh detection).
    script_files = sorted(SCRIPTS_DIR.glob("script_*.json"))
    scr = json.loads(script_files[-1].read_text(encoding="utf-8")) if script_files else None
    print("Mixed ->", mix_broadcast(segs, cfg, script=scr))
