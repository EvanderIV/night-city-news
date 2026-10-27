"""Central path resolution for the Night City Broadcast project.

Every module imports from here so that no file path is hardcoded relative to
a working directory. Paths are derived from this file's location, which is the
project root.
"""
from __future__ import annotations

from pathlib import Path

# This file lives at the project root.
PROJECT_ROOT = Path(__file__).resolve().parent

CONFIG_FILE = PROJECT_ROOT / "config.toml"

CONTENT_DIR = PROJECT_ROOT / "content"
VOICE_DIR = PROJECT_ROOT / "voice"
AUDIO_DIR = PROJECT_ROOT / "audio"
VIDEO_DIR = PROJECT_ROOT / "video"
PLAYBACK_DIR = PROJECT_ROOT / "playback"
ASSETS_DIR = PROJECT_ROOT / "assets"

# Video presentation assets (drop real host clips / sponsor cards here).
VIDEO_ASSETS_DIR = VIDEO_DIR / "assets"
HOST_VIDEO_DIR = VIDEO_ASSETS_DIR / "hosts"        # male_host.mp4 / female_host.mp4
SPONSOR_CARD_DIR = VIDEO_ASSETS_DIR / "sponsors"   # sponsor placeholder PNGs

# Voice training data and saved embeddings.
TRAINING_DATA_DIR = VOICE_DIR / "training_data"
MALE_TRAINING_DIR = TRAINING_DATA_DIR / "male_host"
FEMALE_TRAINING_DIR = TRAINING_DATA_DIR / "female_host"
EMBEDDINGS_DIR = VOICE_DIR / "embeddings"

# Audio assets.
BACKGROUND_DIR = AUDIO_DIR / "assets" / "background"
JINGLES_DIR = AUDIO_DIR / "assets" / "jingles"
# Outro jingles are grouped into per-FM-frequency subfolders of JINGLES_DIR;
# the "prejingle" subfolder holds the smoothing sting that precedes the jingle.
PREJINGLE_DIR = JINGLES_DIR / "prejingle"
# Swoosh stings layered in when the broadcast switches topics.
TRANSITIONS_DIR = AUDIO_DIR / "assets" / "transitions"

# Output tree (gitignored).
OUTPUT_DIR = PROJECT_ROOT / "output"
SCRIPTS_DIR = OUTPUT_DIR / "scripts"
SEGMENTS_DIR = OUTPUT_DIR / "segments"
FINAL_DIR = OUTPUT_DIR / "final"
# Scratch for the video build: downloaded thumbnails, generated cards, shot clips.
VIDEO_CACHE_DIR = OUTPUT_DIR / "video_cache"
CITY_STATE_FILE = OUTPUT_DIR / "city_state.json"
BABYLON_CACHE_FILE = OUTPUT_DIR / "babylon_cache.json"

# Logs.
LOGS_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOGS_DIR / "broadcast.log"

# Emergency fallback script.
FALLBACK_SCRIPT = ASSETS_DIR / "fallback_script.json"

# Real consumer-brand facts for the YouTuber-style fallback sponsor read.
PRODUCT_CATALOG = ASSETS_DIR / "product_catalog.json"


def ensure_runtime_dirs() -> None:
    """Create all output/log directories that the pipeline writes to."""
    for directory in (
        OUTPUT_DIR,
        SCRIPTS_DIR,
        SEGMENTS_DIR,
        FINAL_DIR,
        VIDEO_CACHE_DIR,
        LOGS_DIR,
        EMBEDDINGS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
