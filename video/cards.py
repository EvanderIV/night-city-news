"""PIL rendering for the video presentation.

All on-screen text and graphics (placeholder host frames, the over-the-shoulder
thumbnail box, the lower-third name bar, sponsor/weather/end cards) are rendered
here with Pillow and saved as PNGs, then composited by ffmpeg. Doing the text in
Pillow avoids ffmpeg ``drawtext`` font-path issues across Windows/Debian.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

# Palette (dark studio look).
BG_TOP = (13, 21, 48)
BG_BOTTOM = (6, 9, 18)
WHITE = (240, 244, 255)
MUTED = (150, 160, 185)
ACCENT_NEUTRAL = (230, 180, 60)  # amber
# Single brand accent for ALL on-screen graphics (lower third, box border,
# kickers, cards). Kept constant - it does NOT change with the speaker.
ACCENT = ACCENT_NEUTRAL

_FONT_CANDIDATES = [
    "C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    try:  # Pillow >= 10 renders a scalable default
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def _vertical_gradient(w: int, h: int, top: Sequence[int], bottom: Sequence[int]) -> Image.Image:
    base = Image.new("RGB", (w, h), tuple(top))
    draw = ImageDraw.Draw(base)
    for y in range(h):
        t = y / max(1, h - 1)
        draw.line(
            [(0, y), (w, y)],
            fill=tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3)),
        )
    return base


def studio_bg(w: int, h: int) -> Image.Image:
    return _vertical_gradient(w, h, BG_TOP, BG_BOTTOM)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if draw.textlength(trial, font=font) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def placeholder_host(w: int, h: int, label: str, accent: Sequence[int]) -> Image.Image:
    """A stand-in 'host on camera' frame until real footage is supplied."""
    img = studio_bg(w, h)
    draw = ImageDraw.Draw(img)
    # Silhouette: shoulders + head.
    cx, cy = w // 2, int(h * 0.62)
    r = int(h * 0.16)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(28, 38, 70))           # head
    sw, sh = int(w * 0.34), int(h * 0.30)
    draw.rounded_rectangle([cx - sw, cy + r - 10, cx + sw, h], radius=sw // 2, fill=(28, 38, 70))
    draw.rectangle([0, h - 14, w, h], fill=tuple(accent))                        # accent strip
    # Left-aligned so the label never collides with the top-right thumbnail box.
    x = int(w * 0.06)
    draw.text((x, int(h * 0.12)), label, font=_font(int(h * 0.05)), fill=WHITE)
    sub = "placeholder — drop real footage in video/assets/hosts/"
    draw.text((x, int(h * 0.20)), sub, font=_font(int(h * 0.026)), fill=MUTED)
    return img


def _rounded_cover(src: Image.Image, box_w: int, box_h: int, radius: int) -> Image.Image:
    """Resize ``src`` to cover box_w x box_h, center-crop, with rounded corners."""
    src = src.convert("RGB")
    scale = max(box_w / src.width, box_h / src.height)
    resized = src.resize((max(1, int(src.width * scale)), max(1, int(src.height * scale))))
    left = (resized.width - box_w) // 2
    top = (resized.height - box_h) // 2
    cropped = resized.crop((left, top, left + box_w, top + box_h))
    mask = Image.new("L", (box_w, box_h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, box_w - 1, box_h - 1], radius=radius, fill=255)
    out = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    out.paste(cropped, (0, 0), mask)
    return out


def segment_overlay(
    w: int, h: int, host_name: str, show_name: str,
    box_image: str | Path | None = None, caption: str | None = None,
    accent: Sequence[int] = ACCENT_NEUTRAL, kicker: str = "",
) -> Image.Image:
    """Transparent overlay: optional over-the-shoulder thumbnail + lower third."""
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # --- Over-the-shoulder thumbnail box (top-right) ---
    if box_image is not None:
        bw, bh = int(w * 0.34), int(h * 0.34)
        bx, by = int(w * 0.60), int(h * 0.10)
        # Drop shadow + frame.
        draw.rounded_rectangle([bx - 6, by - 6, bx + bw + 6, by + bh + 6], radius=18,
                               fill=(0, 0, 0, 150))
        try:
            thumb = _rounded_cover(Image.open(box_image), bw, bh, 14)
            img.paste(thumb, (bx, by), thumb)
        except Exception:
            draw.rounded_rectangle([bx, by, bx + bw, by + bh], radius=14, fill=(20, 28, 52, 255))
        draw.rounded_rectangle([bx, by, bx + bw, by + bh], radius=14,
                               outline=tuple(accent) + (255,), width=4)
        if kicker:
            kf = _font(int(h * 0.026))
            draw.rectangle([bx, by - int(h * 0.05), bx + draw.textlength(kicker, font=kf) + 24,
                            by - 4], fill=tuple(accent) + (255,))
            draw.text((bx + 12, by - int(h * 0.05) + 4), kicker, font=kf, fill=(10, 12, 20))
        if caption:
            cf = _font(int(h * 0.028))
            cap_lines = _wrap(draw, caption, cf, bw)[:2]
            cy = by + bh + 12
            for ln in cap_lines:
                draw.text((bx, cy), ln, font=cf, fill=WHITE)
                cy += int(h * 0.038)

    # --- Lower third (bottom-left) ---
    bar_h = int(h * 0.12)
    bar_y = h - bar_h - int(h * 0.06)
    bar_w = int(w * 0.46)
    draw.rectangle([0, bar_y, 14, bar_y + bar_h], fill=tuple(accent))
    draw.rectangle([14, bar_y, bar_w, bar_y + bar_h], fill=(10, 14, 28, 220))
    nf = _font(int(bar_h * 0.42))
    sf = _font(int(bar_h * 0.26))
    draw.text((40, bar_y + int(bar_h * 0.16)), host_name, font=nf, fill=WHITE)
    draw.text((40, bar_y + int(bar_h * 0.62)), show_name.upper(), font=sf, fill=tuple(accent))
    return img


def info_card(w: int, h: int, title: str, subtitle: str = "", accent: Sequence[int] = ACCENT_NEUTRAL) -> Image.Image:
    """A full-screen card (sponsor / weather / end card / fallback thumbnail)."""
    img = studio_bg(w, h)
    draw = ImageDraw.Draw(img)
    tf = _font(int(h * 0.10))
    lines = _wrap(draw, title, tf, int(w * 0.8))
    total_h = len(lines) * int(h * 0.12)
    y = (h - total_h) // 2 - (int(h * 0.04) if subtitle else 0)
    for ln in lines:
        draw.text(((w - draw.textlength(ln, font=tf)) / 2, y), ln, font=tf, fill=WHITE)
        y += int(h * 0.12)
    if subtitle:
        sf = _font(int(h * 0.045))
        draw.text(((w - draw.textlength(subtitle, font=sf)) / 2, y + int(h * 0.02)),
                  subtitle, font=sf, fill=tuple(accent))
    draw.rectangle([0, h - 16, w, h], fill=tuple(accent))
    return img
