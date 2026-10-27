"""Real consumer-product catalog for the fallback sponsor read.

When no real Babylon Bee "Sponsored" article ran, the broadcast does a short,
YouTuber-style mock sponsor read. To keep the facts accurate (real slogans and
founding years, not LLM hallucinations), we feed the script writer a random
sample of verified brands from ``assets/product_catalog.json``.

Each entry: ``{brand, slogan, founded, category, does}`` - ``does`` is the
product's real function, used to anchor the satirical second line.

Run standalone to preview a sample::

    python content/product_catalog.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_loader import get_logger  # noqa: E402
from paths import PRODUCT_CATALOG  # noqa: E402

log = get_logger("catalog")

_cache: list[dict[str, Any]] | None = None


def load_catalog() -> list[dict[str, Any]]:
    """Load (and memoize) the product catalog. Returns [] on any failure."""
    global _cache
    if _cache is not None:
        return _cache
    try:
        with PRODUCT_CATALOG.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        products = data.get("products", []) if isinstance(data, dict) else data
        _cache = [p for p in products if isinstance(p, dict) and p.get("brand")]
    except Exception as exc:
        log.warning("Product catalog unavailable (%s); fallback sponsor will improvise.", exc)
        _cache = []
    return _cache


def sample_products(n: int = 6) -> list[dict[str, Any]]:
    """Return up to ``n`` random brands for the LLM to pick a sponsor from."""
    catalog = load_catalog()
    if not catalog:
        return []
    return random.sample(catalog, min(n, len(catalog)))


if __name__ == "__main__":
    cat = load_catalog()
    print(f"Catalog: {len(cat)} products\n")
    for p in sample_products(6):
        print(f"- {p['brand']} | \"{p['slogan']}\" (since {p['founded']}) - {p['does']}")
