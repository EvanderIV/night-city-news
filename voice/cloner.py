"""One-time voice cloning setup.

Turns a directory of training MP3s into a reusable "voice id" for the
configured TTS engine:

- ``xtts``       -> concatenates the clips, stores a speaker reference WAV +
                    a saved speaker embedding under ``voice/embeddings/``.
- ``piper``      -> cloning is not applicable (pre-trained voices); logs a
                    warning and returns the configured Piper model name.
- ``elevenlabs`` -> uploads the clips to the ElevenLabs cloning API and
                    returns the created voice id.

Exposes ``clone_voice(name, audio_dir, config) -> voice_id``. Embeddings are
reused on subsequent runs (no re-cloning) unless ``force=True``.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_loader import get_logger  # noqa: E402
from paths import EMBEDDINGS_DIR  # noqa: E402

log = get_logger("cloner")


def _discover_clips(audio_dir: Path) -> list[Path]:
    """Find training audio in a directory (filenames are not predictable).

    If a ``curated/`` subfolder exists and contains clips, ONLY those are used.
    This lets you hand-pick clean, consistent, music-free reference audio --
    which matters far more than quantity for XTTS zero-shot cloning -- without
    moving the rest of your clip library out of the way.
    """
    patterns = ("*.mp3", "*.wav", "*.m4a", "*.flac", "*.ogg")

    def _gather(directory: Path) -> list[Path]:
        found: list[Path] = []
        for pattern in patterns:
            found.extend(Path(p) for p in glob.glob(str(directory / pattern)))
        found.sort()
        return found

    curated_dir = audio_dir / "curated"
    if curated_dir.is_dir():
        curated = _gather(curated_dir)
        if curated:
            log.info("Using %d curated clip(s) from %s.", len(curated), curated_dir)
            return curated
        log.warning("curated/ exists but is empty; falling back to all clips.")

    return _gather(audio_dir)


# XTTS clones best from a modest reference (~30-120s). Concatenating hundreds
# of clips is slow and actually degrades quality, so we stop once we reach this.
MAX_REFERENCE_MS = 120_000  # 120 seconds


def _prep_clip(segment, target_dbfs: float, highpass_hz: int, silence_thresh: float):
    """Clean one clip: mono/22.05k, high-pass rumble, trim silence, level it.

    Returns the processed segment, or None if it's essentially silent.
    """
    from pydub import AudioSegment, effects, silence

    seg = segment.set_channels(1).set_frame_rate(22050)
    seg = effects.high_pass_filter(seg, highpass_hz)
    lead = silence.detect_leading_silence(seg, silence_threshold=silence_thresh)
    trail = silence.detect_leading_silence(seg.reverse(), silence_threshold=silence_thresh)
    end = len(seg) - trail
    seg = seg[lead:end] if end > lead else seg
    if len(seg) < 150:  # nothing but silence after trimming
        return None
    if seg.dBFS != float("-inf"):  # RMS-normalize each clip to a common level
        seg = seg.apply_gain(target_dbfs - seg.dBFS)
    return seg + AudioSegment.silent(duration=120)  # tiny gap so words don't merge


def _concatenate_clips(
    clips: list[Path],
    out_path: Path,
    max_ms: int = MAX_REFERENCE_MS,
    clean: bool = False,
    target_dbfs: float = -20.0,
    highpass_hz: int = 85,
    silence_thresh: float = -45.0,
) -> Path:
    """Concatenate variable-length training clips into one reference WAV.

    Zero-shot cloning handles short reference audio best, so we accumulate clips
    only up to ``max_ms`` and stop early. With ``clean=True`` each clip is also
    high-passed, silence-trimmed, and loudness-normalized to a common level - a
    much more consistent reference when the source clips vary in level / have
    dead air / low rumble (typical of ripped game audio).
    """
    from pydub import AudioSegment, effects

    combined = AudioSegment.empty()
    used = 0
    for clip in clips:
        if len(combined) >= max_ms:
            break
        try:
            segment = AudioSegment.from_file(clip)
            if clean:
                prepped = _prep_clip(segment, target_dbfs, highpass_hz, silence_thresh)
                if prepped is None:
                    continue
                segment = prepped
            combined += segment
            used += 1
        except Exception as exc:
            log.warning("Skipping unreadable training clip %s (%s).", clip.name, exc)

    if len(combined) == 0:
        raise RuntimeError("No usable audio after concatenating training clips.")

    combined = combined[:max_ms].set_channels(1).set_frame_rate(22050)
    if clean:
        combined = effects.normalize(combined, headroom=1.0)  # gentle peak ceiling
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.export(out_path, format="wav")
    log.info(
        "Concatenated %d/%d clips%s -> %s (%.1fs).",
        used, len(clips), " (cleaned)" if clean else "",
        out_path.name, len(combined) / 1000,
    )
    return out_path


def _clone_xtts(name: str, clips: list[Path], force: bool) -> str:
    """Create/reuse an XTTS speaker reference. Returns the reference WAV path."""
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    reference_wav = EMBEDDINGS_DIR / f"{name}_reference.wav"

    if reference_wav.exists() and not force:
        log.info("Reusing existing XTTS reference for '%s': %s", name, reference_wav)
        return str(reference_wav)

    _concatenate_clips(clips, reference_wav)

    # Pre-warm / validate the model is loadable so failures surface at setup,
    # not at 2 AM. The synthesizer loads the model again at render time.
    try:
        from TTS.api import TTS  # noqa: F401

        log.info("XTTS available; reference ready for '%s'.", name)
    except Exception as exc:
        log.warning(
            "Coqui TTS not importable yet (%s). Reference WAV saved anyway; "
            "install TTS before running synthesis.",
            exc,
        )
    return str(reference_wav)


def _clone_elevenlabs(name: str, clips: list[Path], config: dict[str, Any]) -> str:
    """Upload clips to ElevenLabs cloning API. Returns the new voice id."""
    from elevenlabs.client import ElevenLabs

    api_key = config.get("tts", {}).get("elevenlabs_api_key", "")
    if not api_key:
        raise RuntimeError("engine=elevenlabs requires [tts].elevenlabs_api_key.")

    client = ElevenLabs(api_key=api_key)
    log.info("Uploading %d clips to ElevenLabs for voice '%s'...", len(clips), name)
    voice = client.clone(
        name=f"NightCity_{name}",
        description=f"Night City Broadcast cloned voice: {name}",
        files=[str(c) for c in clips],
    )
    voice_id = getattr(voice, "voice_id", None) or getattr(voice, "id", None)
    if not voice_id:
        raise RuntimeError("ElevenLabs clone() returned no voice id.")
    log.info("ElevenLabs voice created for '%s': %s", name, voice_id)
    return str(voice_id)


def clone_voice(
    name: str,
    audio_dir: Path | str,
    config: dict[str, Any],
    force: bool = False,
) -> str:
    """Clone (or reuse) a voice for ``name`` from MP3s in ``audio_dir``.

    Returns an engine-specific voice id:
      - xtts: path to the speaker reference WAV
      - piper: the configured Piper model name (no cloning)
      - elevenlabs: the ElevenLabs voice id
    """
    audio_dir = Path(audio_dir)
    engine = config.get("tts", {}).get("engine", "xtts").lower()

    if engine == "piper":
        tts_cfg = config.get("tts", {})
        model = (
            tts_cfg.get("piper_male_model")
            if "male" in name
            else tts_cfg.get("piper_female_model")
        )
        log.warning(
            "engine=piper uses pre-trained voices; skipping cloning for '%s'. "
            "Using model '%s'.",
            name,
            model,
        )
        return str(model or "")

    clips = _discover_clips(audio_dir)
    if not clips:
        raise FileNotFoundError(
            f"No training audio found in {audio_dir}. Add MP3 clips and rerun "
            "`python main.py --setup`."
        )
    log.info("Found %d training clips for '%s' in %s.", len(clips), name, audio_dir)

    if engine == "xtts":
        return _clone_xtts(name, clips, force)
    if engine == "elevenlabs":
        return _clone_elevenlabs(name, clips, config)
    raise ValueError(f"Unknown TTS engine: {engine!r}")


if __name__ == "__main__":
    from config_loader import load_config
    from paths import MALE_TRAINING_DIR, FEMALE_TRAINING_DIR

    cfg = load_config()
    print("male_host  ->", clone_voice("male_host", MALE_TRAINING_DIR, cfg))
    print("female_host->", clone_voice("female_host", FEMALE_TRAINING_DIR, cfg))
