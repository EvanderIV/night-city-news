"""Babylon Bee story fetcher for the "babylon" content mode.

Primary source is the site's **RSS feed** (https://babylonbee.com/feed) - the
robust, syndication-friendly way to automate this: a stable RSS 2.0 document
with per-item title / link / description, immune to homepage redesigns. If the
feed is unreachable, falls back to scraping `/news/...` links off the homepage.

Returns a list of ``{"title": str, "summary": str, "url": str}`` dicts. On total
failure returns ``[]`` and logs the error so ``main.py`` can fall back to
simulation mode. Results are cached on disk for 12 hours to avoid hammering the
site.

Run standalone to test::

    python content/babylon_scraper.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_loader import get_logger  # noqa: E402
from paths import BABYLON_CACHE_FILE  # noqa: E402

log = get_logger("babylon")

BASE_URL = "https://babylonbee.com"
FEED_URL = "https://babylonbee.com/feed"
CACHE_TTL_SECONDS = 12 * 60 * 60  # 12 hours
REQUEST_TIMEOUT = 20.0
USER_AGENT = (
    "Mozilla/5.0 (compatible; NightCityBroadcast/1.0; overnight news aggregator)"
)
MAX_ARTICLES = 15
# If the RSS feed yields fewer than this, top up from the homepage (the Bee's
# /feed is sometimes sparse while the homepage still lists many stories).
MIN_ARTICLES = 6

# --- Sponsored-segment feed (the site's category=sponsored API) -------------
# The Bee drops a native "Sponsored" listicle ~weekly. When one landed
# yesterday we use it as the broadcast's sponsor read instead of a mock one.
# The endpoint is Laravel CSRF-protected: GET a page to receive an XSRF-TOKEN +
# session cookie, then echo the (URL-decoded) token back in the x-xsrf-token
# header on the POST. No Cloudflare clearance cookie is required.
SPONSORED_PAGE_URL = "https://babylonbee.com/news/categories/sponsored"
LOAD_ARTICLES_URL = "https://babylonbee.com/loadArticles"
SPONSORED_TAKE = 12
# A browser-like UA keeps the CSRF/Cloudflare layer happy on the JSON endpoint.
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
)


def _load_cache(path: Path = BABYLON_CACHE_FILE) -> list[dict[str, str]] | None:
    """Return cached articles if the cache exists and is still fresh."""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        fetched_at = float(payload.get("fetched_at", 0))
        articles = payload.get("articles", [])
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        log.warning("Babylon cache unreadable (%s); ignoring.", exc)
        return None

    age = time.time() - fetched_at
    if age > CACHE_TTL_SECONDS:
        log.info("Babylon cache expired (%.1f h old).", age / 3600)
        return None
    if not isinstance(articles, list) or not articles:
        return None
    log.info("Using cached Babylon articles (%.1f h old).", age / 3600)
    return articles


def _save_cache(articles: list[dict[str, str]], path: Path = BABYLON_CACHE_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump({"fetched_at": time.time(), "articles": articles}, fh, indent=2)
        tmp.replace(path)
    except OSError as exc:
        log.warning("Could not write Babylon cache: %s", exc)
        tmp.unlink(missing_ok=True)


def _clean_summary(description_html: str) -> str:
    """Turn an RSS item's HTML description into a plain-text summary.

    The Bee's ``<description>`` holds a thumbnail ``<img>`` plus the article's
    lede in a ``<p>``. We drop the markup and collapse whitespace.
    """
    from bs4 import BeautifulSoup

    text = BeautifulSoup(description_html or "", "html.parser").get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    # The feed sometimes appends a trailing "[...]"/"Read more" remnant.
    return text


def _extract_image(item, description_html: str) -> str:
    """Best-effort thumbnail URL for a feed item (for the video version).

    Tries an <enclosure>/<media:content> image, then the first <img> in the
    description HTML. Returns "" if none found.
    """
    from bs4 import BeautifulSoup

    enc = item.find("enclosure")
    if enc and enc.get("url") and "image" in (enc.get("type", "") or "").lower():
        return enc["url"].strip()
    for tag in ("media:content", "media:thumbnail"):
        media = item.find(tag)
        if media and media.get("url"):
            return media["url"].strip()
    img = BeautifulSoup(description_html or "", "html.parser").find("img")
    if img and img.get("src"):
        return img["src"].strip()
    return ""


def _fetch_from_feed(client) -> list[dict[str, str]]:
    """Parse the RSS feed into article dicts. Raises on network/parse error."""
    from bs4 import BeautifulSoup

    resp = client.get(FEED_URL)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "xml")

    articles: list[dict[str, str]] = []
    for item in soup.find_all("item"):
        title = (item.title.get_text(strip=True) if item.title else "").strip()
        url = (item.link.get_text(strip=True) if item.link else "").strip()
        description_html = item.description.get_text() if item.description else ""
        summary = _clean_summary(description_html)
        if not title or not url:
            continue
        articles.append({
            "title": title, "summary": summary, "url": url,
            "image": _extract_image(item, description_html),
        })
        if len(articles) >= MAX_ARTICLES:
            break
    return articles


def _parse_articles(html: str) -> list[dict[str, str]]:
    """Fallback: extract /news/ story links from the homepage HTML."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    seen: set[str] = set()
    articles: list[dict[str, str]] = []
    for link in soup.select('a[href^="/news/"]'):
        href = link.get("href", "").strip()
        slug = href[len("/news/"):]
        # Skip section/category links (e.g. /news/categories/...); want real slugs.
        if not slug or "/" in slug.strip("/") or slug.startswith("categories"):
            continue
        title = link.get_text(strip=True)
        if not title or len(title) < 12:
            continue
        url = BASE_URL + href
        if url in seen:
            continue
        seen.add(url)
        articles.append({"title": title, "summary": "", "url": url})
        if len(articles) >= MAX_ARTICLES:
            break
    return articles


def _fetch_from_html(client) -> list[dict[str, str]]:
    resp = client.get(BASE_URL)
    resp.raise_for_status()
    return _parse_articles(resp.text)


def _load_articles_api(client, category: str, take: int) -> list[dict]:
    """Run the Laravel CSRF handshake and return raw loadArticles items.

    ``category`` is a Bee category slug ("latest", "sponsored", ...). Raises on
    network/CSRF/parse failure so the caller can degrade gracefully.
    """
    import urllib.parse

    # Step 1: GET the homepage so the server sets XSRF-TOKEN + session cookies
    # (httpx stores them automatically). We always use the homepage because some
    # POST categories (e.g. "latest") aren't real category PAGES and 404 on GET,
    # even though they're valid in the POST body. Referer stays cosmetic.
    referer = f"{BASE_URL}/news/categories/{category}" if category else BASE_URL
    page = client.get(BASE_URL)
    page.raise_for_status()
    xsrf = client.cookies.get("XSRF-TOKEN")
    if not xsrf:
        raise RuntimeError("no XSRF-TOKEN cookie returned (Cloudflare challenge?)")

    # Step 2: POST with the URL-decoded token echoed in the header.
    headers = {
        "accept": "application/json, text/plain, */*",
        "content-type": "application/json",
        "origin": BASE_URL,
        "referer": referer,
        "x-requested-with": "XMLHttpRequest",
        "x-xsrf-token": urllib.parse.unquote(xsrf),
    }
    body = {"category": category, "sort": "desc", "skip": 0, "take": take, "isAuthor": False}
    resp = client.post(LOAD_ARTICLES_URL, json=body, headers=headers)
    resp.raise_for_status()
    articles = resp.json().get("articles", [])
    if not isinstance(articles, list):
        raise RuntimeError("unexpected loadArticles response shape")
    return articles


def _api_to_article(a: dict) -> dict[str, str]:
    """Convert a raw loadArticles item to our {title, summary, url, image} shape."""
    path = str(a.get("path", "")).strip()
    return {
        "title": str(a.get("title", "")).strip(),
        "summary": _clean_summary(str(a.get("shortDescription", ""))),
        "url": (BASE_URL + path) if path.startswith("/") else path,
        "image": str(a.get("imagePath", "")).strip(),
    }


def fetch_sponsored_yesterday() -> dict[str, str] | None:
    """Return the Babylon Bee 'Sponsored' article published yesterday, or None.

    The Bee posts a sponsored listicle only ~weekly, so most days this returns
    None and the broadcast uses its normal generated sponsor read. Any failure
    (network, CSRF, parse) also returns None - this feature never blocks a run.
    """
    from datetime import date, timedelta

    yesterday = (date.today() - timedelta(days=1)).isoformat()
    try:
        import httpx

        with httpx.Client(
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": BROWSER_UA},
            follow_redirects=True,
        ) as client:
            articles = _load_articles_api(client, "sponsored", SPONSORED_TAKE)
    except Exception as exc:
        log.warning("Babylon sponsored fetch failed (%s); using normal sponsor read.", exc)
        return None

    for art in articles:
        published = str(art.get("published_on", ""))[:10]  # "YYYY-MM-DD"
        category = str(art.get("primaryCategory", "")).strip().lower()
        if published == yesterday and category == "sponsored":
            title = str(art.get("title", "")).strip()
            path = str(art.get("path", "")).strip()
            item = {
                "title": title,
                "summary": _clean_summary(str(art.get("shortDescription", ""))),
                "url": (BASE_URL + path) if path.startswith("/") else path,
                "published_on": published,
                "image": str(art.get("imagePath", "")).strip(),
            }
            log.info("Babylon: sponsored article from yesterday -> %r", title)
            return item

    log.info("Babylon: no sponsored article published yesterday (%s).", yesterday)
    return None


def fetch_headlines(use_cache: bool = True) -> list[dict[str, str]]:
    """Fetch Babylon Bee stories: the loadArticles "latest" JSON API first
    (clean titles + summaries + images), then the RSS feed, then homepage HTML.

    Returns ``{"title", "summary", "url", "image"}`` dicts. On total failure
    returns ``[]`` and logs the error so the caller can fall back to simulation.
    """
    if use_cache:
        cached = _load_cache()
        if cached:
            return cached

    import httpx

    articles: list[dict[str, str]] = []

    # 1) loadArticles "latest" API - the cleanest, richest source.
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT, headers={"User-Agent": BROWSER_UA},
                          follow_redirects=True) as api:
            for raw in _load_articles_api(api, "latest", MAX_ARTICLES):
                art = _api_to_article(raw)
                if art["title"] and art["url"]:
                    articles.append(art)
                if len(articles) >= MAX_ARTICLES:
                    break
        if articles:
            log.info("Babylon: %d stories from latest API.", len(articles))
    except Exception as exc:
        log.warning("Babylon latest API failed (%s); trying RSS feed.", exc)

    # 2) RSS feed - supplement if the API was thin, or stand in if it failed.
    if len(articles) < MIN_ARTICLES:
        try:
            with httpx.Client(timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT},
                              follow_redirects=True) as client:
                seen = {a["url"] for a in articles}
                for art in _fetch_from_feed(client):
                    if art["url"] not in seen:
                        articles.append(art)
                        seen.add(art["url"])
                        if len(articles) >= MAX_ARTICLES:
                            break
        except Exception as exc:
            log.warning("Babylon RSS feed failed (%s).", exc)

    # 3) Homepage HTML - last resort if we still have nothing.
    if not articles:
        try:
            with httpx.Client(timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT},
                              follow_redirects=True) as client:
                articles = _fetch_from_html(client)
                if articles:
                    log.info("Babylon: %d stories from homepage HTML.", len(articles))
        except Exception as exc:
            log.error("Babylon homepage HTML failed (%s).", exc)

    if not articles:
        log.error("Babylon produced no stories. Falling back to simulation.")
        return []

    _save_cache(articles)
    return articles


if __name__ == "__main__":
    results = fetch_headlines(use_cache=False)
    print(f"Fetched {len(results)} stories\n")
    for art in results:
        print(f"- {art['title']}")
        if art["summary"]:
            print(f"    {art['summary'][:140]}")
        print(f"    {art['url']}")

    print("\n=== Sponsored (published yesterday?) ===")
    sponsored = fetch_sponsored_yesterday()
    print(sponsored or "None (broadcast will use its normal sponsor read)")
