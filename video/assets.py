"""Asset resolution for the video build.

Locates real host footage / sponsor cards when present, otherwise generates
placeholders; downloads + caches real Babylon Bee article thumbnails; and maps
a spoken line back to the article it's about (to choose the over-the-shoulder
image).
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_loader import get_logger  # noqa: E402
from paths import HOST_VIDEO_DIR, SPONSOR_CARD_DIR, VIDEO_CACHE_DIR  # noqa: E402
from video import cards  # noqa: E402

log = get_logger("video.assets")

_VIDEO_EXTS = (".mp4", ".mov", ".webm", ".m4v")
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
)
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is",
    "are", "was", "were", "be", "by", "at", "as", "it", "its", "his", "her",
    "their", "that", "this", "they", "you", "your", "we", "our", "from", "after",
    "over", "into", "out", "up", "new", "more", "has", "have", "but", "not",
    "who", "what", "how", "now", "all", "just", "about", "than", "then",
}


def _cache() -> Path:
    VIDEO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return VIDEO_CACHE_DIR


def resolve_host_source(speaker: str, w: int, h: int) -> Path:
    """Real host clip/still if present in video/assets/hosts/, else a placeholder."""
    if HOST_VIDEO_DIR.is_dir():
        for ext in _VIDEO_EXTS + _IMAGE_EXTS:
            cand = HOST_VIDEO_DIR / f"{speaker}{ext}"
            if cand.exists():
                return cand
    # Placeholder still (generated once, reused).
    label = "MALE HOST" if speaker == "male_host" else "FEMALE HOST"
    out = _cache() / f"placeholder_{speaker}_{w}x{h}.png"
    if not out.exists():
        cards.placeholder_host(w, h, label, cards.ACCENT).save(out)
        log.info("Generated placeholder host frame: %s", out.name)
    return out


def is_video_source(path: Path) -> bool:
    return path.suffix.lower() in _VIDEO_EXTS


def download_image(url: str) -> Path | None:
    """Download an image to the cache (by URL hash). Returns path or None."""
    if not url:
        return None
    key = hashlib.md5(url.encode("utf-8")).hexdigest()[:16]
    # Reuse any previously cached copy regardless of extension.
    for existing in _cache().glob(f"thumb_{key}.*"):
        return existing
    try:
        import httpx
        from PIL import Image

        with httpx.Client(timeout=20.0, headers={"User-Agent": _BROWSER_UA},
                          follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.content
        ext = ".jpg"
        ct = resp.headers.get("content-type", "").lower()
        if "png" in ct:
            ext = ".png"
        elif "webp" in ct:
            ext = ".webp"
        out = _cache() / f"thumb_{key}{ext}"
        out.write_bytes(data)
        Image.open(out).verify()  # validate it's a real image
        return out
    except Exception as exc:
        log.warning("Thumbnail download failed for %s (%s).", url[:80], exc)
        return None


def article_image(article: dict[str, Any], w: int, h: int) -> Path:
    """Local image path for an article: real thumbnail, else a titled card."""
    img = download_image(str(article.get("image", "")))
    if img is not None:
        return img
    title = str(article.get("title", "Top Story"))
    key = hashlib.md5(title.encode("utf-8")).hexdigest()[:12]
    out = _cache() / f"card_story_{key}.png"
    if not out.exists():
        cards.info_card(w, h, title, "BABYLON BEE", cards.ACCENT_NEUTRAL).save(out)
    return out


def sponsor_image(sponsored: dict[str, Any] | None, w: int, h: int) -> Path:
    """Sponsor card: real sponsored-article image, a dropped-in PNG, or placeholder."""
    if sponsored and sponsored.get("image"):
        img = download_image(str(sponsored["image"]))
        if img is not None:
            return img
    if SPONSOR_CARD_DIR.is_dir():  # any real sponsor PNGs the user dropped in
        for ext in _IMAGE_EXTS:
            for cand in sorted(SPONSOR_CARD_DIR.glob(f"*{ext}")):
                return cand
    out = _cache() / f"card_sponsor_{w}x{h}.png"
    if not out.exists():
        cards.info_card(w, h, "A WORD FROM OUR SPONSOR", "SPONSORED", cards.ACCENT_NEUTRAL).save(out)
    return out


def weather_card(city: str, w: int, h: int) -> Path:
    key = hashlib.md5(city.encode("utf-8")).hexdigest()[:10]
    out = _cache() / f"card_weather_{key}.png"
    if not out.exists():
        cards.info_card(w, h, "COMMUTE WEATHER", city or "Your Area", cards.ACCENT).save(out)
    return out


def end_card(show_name: str, w: int, h: int) -> Path:
    out = _cache() / f"card_end_{w}x{h}.png"
    if not out.exists():
        cards.info_card(w, h, "THANKS FOR WATCHING", show_name.upper(), cards.ACCENT_NEUTRAL).save(out)
    return out


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS}


def match_article(line: str, articles: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Best article a spoken line is about, by title-token overlap (>=2), else None."""
    line_tokens = _tokens(line)
    if not line_tokens or not articles:
        return None
    best, best_score = None, 0
    for art in articles:
        score = len(line_tokens & _tokens(str(art.get("title", ""))))
        if score > best_score:
            best, best_score = art, score
    return best if best_score >= 2 else None
