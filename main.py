"""Night City Morning Broadcast - top-level orchestrator.

Usage:
    python main.py            Run the full generation pipeline.
    python main.py --play     Skip generation; play the most recent broadcast.
    python main.py --setup    Run only the one-time voice cloning step.
    python main.py --dry-run  Generate + save the script only (no TTS/audio).

Every stage is wrapped so a single failure logs and (where sensible) degrades
gracefully instead of crashing the whole overnight job.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

from config_loader import get_logger, load_config
from paths import (
    FEMALE_TRAINING_DIR,
    FINAL_DIR,
    MALE_TRAINING_DIR,
    SCRIPTS_DIR,
    SEGMENTS_DIR,
    ensure_runtime_dirs,
)

log = get_logger("main")


# --- Setup (voice cloning) -------------------------------------------------

def run_setup(config: dict[str, Any]) -> None:
    """Clone both host voices and write their ids back into config.toml."""
    from voice.cloner import clone_voice

    log.info("=== Voice setup ===")
    male_id = clone_voice("male_host", MALE_TRAINING_DIR, config, force=True)
    female_id = clone_voice("female_host", FEMALE_TRAINING_DIR, config, force=True)

    _persist_voice_ids(male_id, female_id)
    log.info("Voice setup complete. male=%s female=%s", male_id, female_id)


def _persist_voice_ids(male_id: str, female_id: str) -> None:
    """Write the cloned voice ids back into config.toml (simple line rewrite).

    We avoid a full TOML serializer to preserve comments; we just replace the
    two known keys' values.
    """
    from paths import CONFIG_FILE

    try:
        text = CONFIG_FILE.read_text(encoding="utf-8")
        lines = text.splitlines()
        # Use TOML *literal* strings (single quotes) so Windows backslashes are
        # not treated as escape sequences (e.g. "C:\Users" -> invalid \U escape).
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("male_voice_id"):
                lines[i] = f"male_voice_id = '{male_id}'"
            elif stripped.startswith("female_voice_id"):
                lines[i] = f"female_voice_id = '{female_id}'"
        CONFIG_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
        log.info("Saved voice ids to %s.", CONFIG_FILE.name)
    except OSError as exc:
        log.error("Could not write voice ids to config (%s). Set them manually.", exc)


def _ensure_voices(config: dict[str, Any]) -> None:
    """Clone voices on the fly if embeddings/ids don't exist yet."""
    engine = config.get("tts", {}).get("engine", "xtts").lower()
    if engine == "piper":
        return  # pre-trained; nothing to clone

    tts_cfg = config.get("tts", {})
    male_ok = bool(tts_cfg.get("male_voice_id"))
    female_ok = bool(tts_cfg.get("female_voice_id"))

    # For xtts a "voice id" is a reference WAV path; verify it still exists.
    if engine == "xtts":
        male_ok = male_ok and Path(tts_cfg["male_voice_id"]).exists()
        female_ok = female_ok and Path(tts_cfg["female_voice_id"]).exists()

    if not (male_ok and female_ok):
        log.info("Voice ids missing/stale; running cloning setup first.")
        run_setup(config)
        # Reload config so the freshly written ids are visible this run.
        config.update(load_config())


# --- Content + script ------------------------------------------------------

def generate_content(config: dict[str, Any]) -> tuple[str, Any]:
    """Produce content per the configured mode. Returns (effective_mode, data).

    Babylon mode falls back to simulation automatically if scraping fails.
    """
    mode = config.get("mode", {}).get("content_source", "simulation").lower()

    if mode == "babylon":
        from content.babylon_scraper import fetch_headlines

        articles = fetch_headlines()
        if articles:
            log.info("Babylon mode: %d articles.", len(articles))
            return "babylon", articles
        log.warning("Babylon mode failed; falling back to simulation.")
        mode = "simulation"

    from content.city_simulation import generate_city_stats

    stats = generate_city_stats()
    log.info("Simulation mode: city stats generated for %s.", stats["date"])
    return "simulation", stats


def fetch_weather(config: dict[str, Any]) -> dict[str, Any] | None:
    """Fetch the real commute-weather segment (None if disabled or it fails)."""
    try:
        from content.weather import fetch_commute_weather

        return fetch_commute_weather(config)
    except Exception as exc:
        log.error("Weather stage failed (%s); broadcast will omit it.", exc)
        return None


def fetch_sponsored(config: dict[str, Any], mode: str) -> dict[str, Any] | None:
    """Babylon-only: a real Bee 'Sponsored' article from yesterday, else None.

    When present it replaces the broadcast's invented sponsor read. Any failure
    returns None so the run continues with the normal generated sponsor segment.
    """
    if mode != "babylon":
        return None
    try:
        from content.babylon_scraper import fetch_sponsored_yesterday

        return fetch_sponsored_yesterday()
    except Exception as exc:
        log.error("Sponsored stage failed (%s); using normal sponsor read.", exc)
        return None


def build_script(
    config: dict[str, Any],
    mode: str,
    content: Any,
    weather: dict[str, Any] | None = None,
    sponsored: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    from content.script_writer import generate_script

    return generate_script(
        content, mode=mode, config=config, weather=weather, sponsored=sponsored
    )


def save_script(script: list[dict[str, str]]) -> Path:
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SCRIPTS_DIR / f"script_{date.today().isoformat()}.txt"
    lines = []
    for turn in script:
        speaker = turn["speaker"].replace("_", " ").title()
        lines.append(f"{speaker}: {turn['line']}")
    out_path.write_text("\n\n".join(lines), encoding="utf-8")
    # Also keep the raw JSON next to it for re-synthesis.
    (out_path.with_suffix(".json")).write_text(
        json.dumps(script, indent=2), encoding="utf-8"
    )
    log.info("Script saved: %s", out_path)
    return out_path


# --- Audio stages ----------------------------------------------------------

def synthesize(config: dict[str, Any], script: list[dict[str, str]]) -> list[Path]:
    from voice.synthesizer import synthesize_script

    return synthesize_script(script, config)


def postprocess(config: dict[str, Any], segments: list[Path]) -> list[Path]:
    if not config.get("audio", {}).get("radio_filter", True):
        log.info("radio_filter disabled; skipping segment post-processing.")
        return segments

    from audio.postprocessor import process_segment

    processed: list[Path] = []
    for seg in segments:
        out = SEGMENTS_DIR / f"processed_{Path(seg).name}"
        try:
            process_segment(seg, out)
            processed.append(out)
        except Exception as exc:
            log.error("Post-process failed for %s (%s); using raw segment.", seg, exc)
            processed.append(Path(seg))
    return processed


def mix(
    config: dict[str, Any],
    segments: list[Path],
    script: list[dict[str, str]] | None = None,
) -> Path:
    from audio.mixer import mix_broadcast

    mixed = mix_broadcast(segments, config, script=script)

    if config.get("audio", {}).get("radio_filter", True):
        from audio.postprocessor import process_final

        mastered = mixed.with_name(mixed.stem + "_master" + mixed.suffix)
        try:
            process_final(mixed, mastered)
            mastered.replace(mixed)  # overwrite with mastered version
        except Exception as exc:
            log.error("Final mastering failed (%s); keeping unmastered mix.", exc)
    return mixed


# --- Pipelines -------------------------------------------------------------

def run_pipeline(config: dict[str, Any], dry_run: bool = False, video: bool = False) -> None:
    """Full generation pipeline (stages 2-9 of the spec)."""
    log.info("=== Broadcast generation START (dry_run=%s) ===", dry_run)

    if not dry_run:
        try:
            _ensure_voices(config)
        except Exception as exc:
            log.error("Voice initialization failed (%s); aborting run.", exc)
            return

    # Content + script (always run).
    try:
        mode, content = generate_content(config)
        weather = fetch_weather(config)
        sponsored = fetch_sponsored(config, mode)
        script = build_script(config, mode, content, weather, sponsored)
        save_script(script)
    except Exception as exc:
        log.error("Content/script stage failed hard (%s); aborting.", exc)
        return

    if dry_run:
        log.info("Dry run complete - script saved, skipping TTS and audio.")
        return

    # Synthesis.
    try:
        segments = synthesize(config, script)
    except Exception as exc:
        log.error("Synthesis stage failed (%s); aborting.", exc)
        return
    if not segments:
        log.error("No audio segments produced; aborting.")
        return

    # Post-process + mix.
    try:
        processed = postprocess(config, segments)
        final_path = mix(config, processed, script)
    except Exception as exc:
        log.error("Audio assembly failed (%s); aborting.", exc)
        return

    log.info("=== Broadcast generation COMPLETE: %s ===", final_path)

    # Optional high-fidelity video presentation (built from the RAW segments).
    if video or config.get("video", {}).get("enabled", False):
        try:
            video_path = build_video_outputs(config, script, segments, mode, content, sponsored)
            if video_path:
                log.info("=== Video presentation COMPLETE: %s ===", video_path)
        except Exception as exc:
            log.error("Video stage failed (%s); audio broadcast is unaffected.", exc)


def build_video_outputs(
    config: dict[str, Any],
    script: list[dict[str, str]],
    raw_segments: list[Path],
    mode: str,
    content: Any,
    sponsored: dict[str, Any] | None,
) -> Path | None:
    """Render the MP4 video presentation from the RAW (non-radio) segments.

    Uses the mixer to build a clean audio mix (capturing a synced cue timeline),
    masters it with a slight bass boost (no radio grain), then composites the
    news-desk video against that timeline.
    """
    from datetime import date

    from audio.mixer import mix_broadcast
    from audio.postprocessor import master_for_video
    from paths import VIDEO_CACHE_DIR
    from video.compositor import build_video

    stamp = date.today().isoformat()
    timeline: list[dict[str, Any]] = []
    raw_mix = mix_broadcast(
        raw_segments, config, script=script, timeline=timeline,
        out_name=f"broadcast_{stamp}_videomix", out_format="wav",
    )
    video_audio = VIDEO_CACHE_DIR / f"video_audio_{stamp}.wav"
    master_for_video(
        raw_mix, video_audio,
        bass_gain_db=float(config.get("video", {}).get("bass_boost_db", 3.0)),
    )
    articles = content if (mode == "babylon" and isinstance(content, list)) else []
    out_path = FINAL_DIR / f"broadcast_{stamp}.mp4"
    return build_video(timeline, video_audio, script, articles, sponsored, config, out_path)


def run_play(config: dict[str, Any]) -> None:
    from playback.player import play

    ok = play(None, config)
    if not ok:
        log.error("Playback did not succeed.")


# --- CLI -------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Night City Morning Broadcast")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--play", action="store_true", help="Play the most recent broadcast and exit.")
    group.add_argument("--setup", action="store_true", help="Run voice cloning setup only.")
    group.add_argument("--dry-run", action="store_true", help="Generate and save the script only.")
    parser.add_argument("--video", action="store_true",
                        help="Also render the MP4 video presentation (news-desk composite).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ensure_runtime_dirs()
    args = parse_args(argv)

    try:
        config = load_config()
    except Exception as exc:
        log.error("Could not load config: %s", exc)
        return 1

    try:
        if args.play:
            run_play(config)
        elif args.setup:
            run_setup(config)
        else:
            run_pipeline(config, dry_run=args.dry_run, video=args.video)
    except Exception as exc:  # final safety net - never crash the cron job
        log.exception("Unhandled error in pipeline: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
