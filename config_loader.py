"""Configuration loading and shared logging setup.

Kept tiny and dependency-light so every module can import it. Uses the stdlib
``tomllib`` on Python 3.11+ and falls back to the ``tomli`` backport otherwise.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - 3.10 and older
    import tomli as tomllib  # type: ignore[no-redef]

from paths import CONFIG_FILE, LOG_FILE, ensure_runtime_dirs

_LOGGER_CONFIGURED = False


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load and lightly validate ``config.toml``.

    Returns the parsed config as a nested dict. Raises ``FileNotFoundError``
    with an actionable message if the file is missing.
    """
    config_path = Path(path) if path is not None else CONFIG_FILE
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {config_path}. Copy the example "
            "config.toml from the repo root and edit it."
        )
    with config_path.open("rb") as fh:
        config = tomllib.load(fh)

    # Validate the handful of fields the pipeline cannot run without.
    required_sections = ("mode", "llm", "tts", "city", "audio", "playback")
    missing = [s for s in required_sections if s not in config]
    if missing:
        raise ValueError(
            f"config.toml is missing required sections: {', '.join(missing)}"
        )
    return config


def get_logger(stage: str = "broadcast") -> logging.Logger:
    """Return the shared project logger, configuring handlers once.

    All stages log to ``logs/broadcast.log`` *and* stderr with a timestamp and
    the stage name, satisfying the spec's error-handling requirements.
    """
    global _LOGGER_CONFIGURED
    logger = logging.getLogger("night_city")

    if not _LOGGER_CONFIGURED:
        ensure_runtime_dirs()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        logger.propagate = False
        _LOGGER_CONFIGURED = True

    # A child logger keeps the stage name in every record.
    return logger.getChild(stage)
