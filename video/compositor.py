"""Compose the MP4 video presentation from the audio timeline.

Given the cue timeline emitted by the mixer (so the picture matches the clean
video audio exactly), this builds one ffmpeg "shot" per cue - host footage with
a lower-third and an optional over-the-shoulder story thumbnail, or a full-screen
sponsor/weather/end card - then concatenates them and muxes the audio.

News-desk composite layout (see video/cards.py for the graphics).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_loader import get_logger  # noqa: E402
from paths import VIDEO_CACHE_DIR  # noqa: E402
from video import assets, cards  # noqa: E402

log = get_logger("video.compositor")

BOX_W_RATIO, BOX_H_RATIO = 0.34, 0.34


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("ffmpeg not found on PATH.")
    return exe


def _run(cmd: list[str]) -> None:
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed ({res.returncode}): {res.stderr.decode('utf-8', 'replace')[-600:]}"
        )


def _scale_cover(stream: str, w: int, h: int, fps: int) -> str:
    return (f"[{stream}]scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h},setsar=1,fps={fps}")


def _render_host_shot(host_src: Path, overlay_png: Path, dur: float,
                      out: Path, w: int, h: int, fps: int, crf: int, preset: str) -> None:
    """A shot of host footage (image or video, looped) under an overlay PNG."""
    ff = _ffmpeg()
    if assets.is_video_source(host_src):
        in_host = ["-stream_loop", "-1", "-i", str(host_src)]
    else:
        in_host = ["-loop", "1", "-i", str(host_src)]
    fc = (f"{_scale_cover('0:v', w, h, fps)}[bg];"
          f"[bg][1:v]overlay=0:0:format=auto,format=yuv420p[v]")
    cmd = [ff, "-y", *in_host, "-loop", "1", "-i", str(overlay_png),
           "-filter_complex", fc, "-map", "[v]", "-t", f"{dur:.3f}",
           "-r", str(fps), "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-an", str(out)]
    _run(cmd)


def _render_card_shot(card_img: Path, dur: float, out: Path,
                      w: int, h: int, fps: int, crf: int, preset: str) -> None:
    """A full-screen card shot (sponsor / end)."""
    ff = _ffmpeg()
    fc = f"{_scale_cover('0:v', w, h, fps)},format=yuv420p[v]"
    cmd = [ff, "-y", "-loop", "1", "-i", str(card_img),
           "-filter_complex", fc, "-map", "[v]", "-t", f"{dur:.3f}",
           "-r", str(fps), "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-an", str(out)]
    _run(cmd)


def build_video(
    timeline: list[dict[str, Any]],
    audio_path: Path,
    script: list[dict[str, Any]],
    articles: list[dict[str, Any]],
    sponsored: dict[str, Any] | None,
    config: dict[str, Any],
    out_path: Path,
) -> Path | None:
    """Render the MP4. Returns the output path, or None if nothing to render."""
    vcfg = config.get("video", {})
    w = int(vcfg.get("width", 1920))
    h = int(vcfg.get("height", 1080))
    fps = int(vcfg.get("fps", 30))
    crf = int(vcfg.get("crf", 20))
    preset = str(vcfg.get("preset", "veryfast"))
    male_name = str(vcfg.get("male_host_name", "Male Host"))
    female_name = str(vcfg.get("female_host_name", "Female Host"))
    show_name = str(vcfg.get("show_name", "Morning Broadcast"))
    city = config.get("weather", {}).get("location", "")
    box_w, box_h = int(w * BOX_W_RATIO), int(h * BOX_H_RATIO)

    if not timeline:
        log.error("Empty timeline; cannot build video.")
        return None

    shots_dir = VIDEO_CACHE_DIR / "shots"
    if shots_dir.exists():
        shutil.rmtree(shots_dir, ignore_errors=True)
    shots_dir.mkdir(parents=True, exist_ok=True)

    # Pre-resolve host placeholders/footage once.
    host_src = {sp: assets.resolve_host_source(sp, w, h) for sp in ("male_host", "female_host")}

    shots: list[tuple[str, dict[str, Any], float, float]] = []  # (kind, cue, start_ms, end_ms)
    pending_start: float | None = None
    for cue in timeline:
        kind = cue["kind"]
        if kind == "swoosh":
            pending_start = cue["start_ms"]  # next shot wipes in over the swoosh
            continue
        start = pending_start if pending_start is not None else cue["start_ms"]
        pending_start = None
        shots.append((kind, cue, start, cue["end_ms"]))

    log.info("Building %d video shots at %dx%d@%dfps...", len(shots), w, h, fps)
    current_story: dict[str, Any] | None = None
    list_file = shots_dir / "concat.txt"
    lines: list[str] = []

    for i, (kind, cue, start_ms, end_ms) in enumerate(shots):
        dur = max(0.2, (end_ms - start_ms) / 1000.0)
        out = shots_dir / f"shot_{i:03d}.mp4"
        try:
            if kind == "outro":
                _render_card_shot(assets.end_card(show_name, w, h), dur, out, w, h, fps, crf, preset)
            else:
                section = cue.get("section", "news")
                speaker = cue.get("speaker", "male_host")
                line = cue.get("line", "")
                if section == "sponsor":
                    _render_card_shot(assets.sponsor_image(sponsored, w, h), dur, out,
                                      w, h, fps, crf, preset)
                else:
                    name = male_name if speaker == "male_host" else female_name
                    box_img: Path | None = None
                    caption = kicker = None
                    if section == "weather":
                        box_img = assets.weather_card(city, box_w, box_h)
                        kicker = "WEATHER"
                    elif section == "news":
                        match = assets.match_article(line, articles)
                        if match:
                            current_story = match
                        if current_story:
                            box_img = assets.article_image(current_story, box_w, box_h)
                            caption = current_story.get("title")
                            kicker = "BABYLON BEE"
                    overlay = shots_dir / f"overlay_{i:03d}.png"
                    cards.segment_overlay(w, h, name, show_name, box_image=box_img,
                                          caption=caption, accent=cards.ACCENT,
                                          kicker=kicker or "").save(overlay)
                    _render_host_shot(host_src[speaker], overlay, dur, out, w, h, fps, crf, preset)
            lines.append(f"file '{out.as_posix()}'")
        except Exception as exc:
            log.error("Shot %d (%s) failed (%s); skipping.", i, kind, exc)

    if not lines:
        log.error("No shots rendered; aborting video.")
        return None
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _ffmpeg(), "-y",
        "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart",
        str(out_path),
    ]
    try:
        _run(cmd)
    except Exception as exc:
        log.error("Final video mux failed (%s).", exc)
        return None
    log.info("Video presentation written: %s", out_path)
    return out_path
