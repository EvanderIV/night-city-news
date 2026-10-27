"""Text-to-speech synthesis.

Renders a broadcast script (a list of ``(speaker, line)`` turns) to one WAV
file per turn in ``output/segments/``, returning the ordered list of paths.

- Maps ``"male_host"`` / ``"female_host"`` to the voice ids stored in config.
- Inserts a brief natural pause (200-400ms) of trailing silence per segment.
- Chunks long lines at sentence boundaries to respect XTTS token limits.
- On a per-segment failure, substitutes ~1s of silence and continues so the
  broadcast still assembles.

Supported engines: ``xtts`` (Coqui), ``piper`` (system binary), ``elevenlabs``,
and ``chatterbox`` (out-of-process, with per-line delivery presets).
"""
from __future__ import annotations

import json
import random
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_loader import get_logger  # noqa: E402
from paths import PROJECT_ROOT, SEGMENTS_DIR, VOICE_DIR  # noqa: E402

log = get_logger("synth")

SAMPLE_RATE = 22050
PAUSE_MIN_MS = 200
PAUSE_MAX_MS = 400
MAX_CHARS_PER_CHUNK = 250  # conservative for XTTS token limits


# --- Text normalization (spell out symbols TTS mishandles) -----------------
# Chatterbox (and most TTS) misreads "$20" - it ignores or mangles the symbol
# rather than saying "twenty dollars". We rewrite currency to plain English
# before synthesis so it's spoken correctly.

_ONES = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety")
_SCALES = ((10 ** 12, "trillion"), (10 ** 9, "billion"), (10 ** 6, "million"),
           (10 ** 3, "thousand"))

# $20 | $1,000 | $1.50 | $1.5 million | $20.99
_CURRENCY_RE = re.compile(
    r"\$\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?"
    r"(?:[  ]?(million|billion|trillion|thousand))?",
    re.IGNORECASE,
)


def _under_1000_words(x: int) -> list[str]:
    words: list[str] = []
    if x >= 100:
        words += [_ONES[x // 100], "hundred"]
        x %= 100
    if x >= 20:
        tens = _TENS[x // 10]
        if x % 10:
            tens += "-" + _ONES[x % 10]
        words.append(tens)
    elif x > 0:
        words.append(_ONES[x])
    return words


def _int_to_words(n: int) -> str:
    """Whole number to words, e.g. 1234 -> 'one thousand two hundred thirty-four'."""
    if n == 0:
        return "zero"
    parts: list[str] = []
    for scale, name in _SCALES:
        if n >= scale:
            parts += _under_1000_words(n // scale)
            parts.append(name)
            n %= scale
    parts += _under_1000_words(n)
    return " ".join(parts)


def _digit_words(digits: str) -> str:
    return " ".join(_ONES[int(d)] for d in digits)


def _cents_phrase(cents: int) -> str:
    return f"{_int_to_words(cents)} cent{'' if cents == 1 else 's'}"


def _currency_repl(m: "re.Match[str]") -> str:
    whole = int(m.group(1).replace(",", ""))
    frac = m.group(2)
    mag = m.group(3)

    if mag:  # "$1.5 million" -> "one point five million dollars"
        words = _int_to_words(whole)
        if frac:
            words += " point " + _digit_words(frac)
        return f"{words} {mag.lower()} dollars"

    if frac is not None and len(frac) == 2:  # standard cents
        cents = int(frac)
        if whole == 0:
            return _cents_phrase(cents)
        unit = "dollar" if whole == 1 else "dollars"
        base = f"{_int_to_words(whole)} {unit}"
        return base if cents == 0 else f"{base} and {_cents_phrase(cents)}"

    if frac is not None:  # unusual decimal length -> read the digits out
        return f"{_int_to_words(whole)} point {_digit_words(frac)} dollars"

    unit = "dollar" if whole == 1 else "dollars"
    return f"{_int_to_words(whole)} {unit}"


def normalize_for_tts(text: str) -> str:
    """Rewrite tokens the TTS engine mishandles (currency) into spoken English."""
    return _CURRENCY_RE.sub(_currency_repl, text)


# --- Helpers ---------------------------------------------------------------

def _sentence_chunks(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> list[str]:
    """Split a long line into <= max_chars chunks at sentence boundaries."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_chars:
            current += " " + sentence
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)

    # Any single sentence still too long: hard-split on whitespace.
    final: list[str] = []
    for chunk in chunks:
        while len(chunk) > max_chars:
            cut = chunk.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            final.append(chunk[:cut].strip())
            chunk = chunk[cut:].strip()
        if chunk:
            final.append(chunk)
    return final or [text[:max_chars]]


def _silence(duration_ms: int):
    from pydub import AudioSegment

    return AudioSegment.silent(duration=duration_ms, frame_rate=SAMPLE_RATE)


def _append_pause(wav_path: Path) -> None:
    """Append a short natural pause to the end of a rendered segment."""
    from pydub import AudioSegment

    try:
        seg = AudioSegment.from_file(wav_path)
        seg += _silence(random.randint(PAUSE_MIN_MS, PAUSE_MAX_MS))
        seg.export(wav_path, format="wav")
    except Exception as exc:
        log.warning("Could not append pause to %s (%s).", wav_path.name, exc)


# --- Engine backends -------------------------------------------------------

class _XTTSBackend:
    """Lazy-loaded Coqui XTTS v2 model, shared across all segments."""

    def __init__(self) -> None:
        self._tts = None

    def _model(self):
        if self._tts is None:
            from TTS.api import TTS

            log.info("Loading XTTS v2 model (first segment only)...")
            self._tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        return self._tts

    def render(self, text: str, voice_id: str, out_path: Path) -> None:
        from pydub import AudioSegment

        chunks = _sentence_chunks(text)
        combined = AudioSegment.empty()
        for i, chunk in enumerate(chunks):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                self._model().tts_to_file(
                    text=chunk,
                    speaker_wav=voice_id,  # reference WAV from cloner
                    language="en",
                    file_path=str(tmp_path),
                )
                combined += AudioSegment.from_file(tmp_path)
            finally:
                tmp_path.unlink(missing_ok=True)
            if i < len(chunks) - 1:
                combined += _silence(120)  # tiny gap between chunks
        combined.export(out_path, format="wav")


class _PiperBackend:
    """Piper TTS via the system ``piper`` binary."""

    def __init__(self) -> None:
        if shutil.which("piper") is None:
            raise RuntimeError(
                "engine=piper but the 'piper' binary is not on PATH. See README."
            )

    def render(self, text: str, voice_id: str, out_path: Path) -> None:
        # voice_id is a Piper model name/path (e.g. en_US-ryan-high).
        cmd = ["piper", "--model", voice_id, "--output_file", str(out_path)]
        subprocess.run(cmd, input=text.encode("utf-8"), check=True)


class _ElevenLabsBackend:
    def __init__(self, config: dict[str, Any]) -> None:
        from elevenlabs.client import ElevenLabs

        api_key = config.get("tts", {}).get("elevenlabs_api_key", "")
        if not api_key:
            raise RuntimeError("engine=elevenlabs requires [tts].elevenlabs_api_key.")
        self._client = ElevenLabs(api_key=api_key)

    def render(self, text: str, voice_id: str, out_path: Path) -> None:
        audio = self._client.text_to_speech.convert(
            voice_id=voice_id,
            model_id="eleven_multilingual_v2",
            text=text,
            output_format="mp3_44100_128",
        )
        # Stream chunks to a temp mp3, then transcode to wav for uniformity.
        from pydub import AudioSegment

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            for chunk in audio:
                tmp.write(chunk)
        try:
            AudioSegment.from_file(tmp_path).export(out_path, format="wav")
        finally:
            tmp_path.unlink(missing_ok=True)


def _make_backend(config: dict[str, Any]):
    engine = config.get("tts", {}).get("engine", "xtts").lower()
    if engine == "xtts":
        return _XTTSBackend()
    if engine == "piper":
        return _PiperBackend()
    if engine == "elevenlabs":
        return _ElevenLabsBackend(config)
    raise ValueError(f"Unknown TTS engine: {engine!r}")


# --- Chatterbox engine (out-of-process, preset-driven) ---------------------

_GEN_KEYS = ("exaggeration", "cfg_weight", "temperature", "repetition_penalty", "min_p", "top_p")


def _resolve_venv_python(configured: str) -> Path:
    """Resolve the Chatterbox venv interpreter cross-platform.

    The config may carry a Windows (``Scripts/python.exe``) or Linux
    (``bin/python``) path. If the configured one is absent, try the other
    OS's convention so the same config.toml works on Windows and Debian.
    """
    p = PROJECT_ROOT / configured
    if p.exists():
        return p
    venv_dir = (PROJECT_ROOT / configured).parent.parent  # .../.venv-chatterbox
    for cand in (venv_dir / "bin" / "python", venv_dir / "Scripts" / "python.exe",
                 venv_dir / "bin" / "python3"):
        if cand.exists():
            return cand
    return p  # original; caller logs the missing-interpreter error


def _base_reference(speaker: str, cb_cfg: dict[str, Any], voices: dict[str, Any]) -> Any:
    """The speaker's normal reference WAV: host defaults, else a [voices] entry."""
    if speaker == "male_host":
        return cb_cfg.get("male_reference")
    if speaker == "female_host":
        return cb_cfg.get("female_reference")
    # A registry voice (skye, johnny, ...); fall back to the female host if a
    # turn names an unknown speaker, so synthesis never breaks on a typo.
    return voices.get(speaker) or cb_cfg.get("female_reference")


def _resolve_preset(
    turn: dict[str, Any], cb_cfg: dict[str, Any], voices: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Resolve a turn into a Chatterbox render job (ref / text / seed / params).

    Speaker -> reference: the two hosts use ``[chatterbox].*_reference``; any other
    ``speaker`` is looked up in the ``[voices]`` registry. A turn may carry an
    optional ``"preset"`` naming a ``[chatterbox.presets.*]`` entry for delivery
    overrides (and a per-speaker reference variant via ``[voices].<speaker>_<preset>``,
    e.g. ``skye_whisper``).
    """
    voices = voices or {}
    speaker = turn.get("speaker", "male_host")
    presets = cb_cfg.get("presets", {})
    preset_name = turn.get("preset") or ""
    # intro/hype are the male host's signature (the dramatic reference); ignore
    # them on any other speaker so a mistag can't pull the wrong voice.
    if preset_name in ("intro", "hype") and speaker != "male_host":
        preset_name = ""
    preset = presets.get(preset_name, {})

    # Reference, most specific first:
    #   1. the preset's explicit reference override (e.g. intro/hype dramatic)
    #   2. a per-speaker preset variant in [voices] (e.g. "skye_whisper")
    #   3. the speaker's base reference (host defaults or a [voices] entry)
    reference = preset.get("reference")
    if not reference and preset_name:
        reference = voices.get(f"{speaker}_{preset_name}")
    if not reference:
        reference = _base_reference(speaker, cb_cfg, voices)

    # Generation params: global defaults overlaid with any preset overrides.
    gen = {k: cb_cfg[k] for k in _GEN_KEYS if k in cb_cfg}
    gen.update({k: preset[k] for k in _GEN_KEYS if k in preset})

    # Text: preset may force a fixed line (e.g. the locked intro greeting).
    text = normalize_for_tts(preset.get("text") or turn.get("line", "").strip())

    # Seed: pick one at random from the preset's pool (for day-to-day variation).
    seeds = preset.get("seeds")
    seed = random.choice(seeds) if seeds else None

    return {"text": text, "reference": reference, "gen_kwargs": gen, "seed": seed}


def _synthesize_chatterbox(
    script: list[dict[str, str]],
    config: dict[str, Any],
    out_dir: Path,
) -> list[Path]:
    """Render the whole script via the out-of-process Chatterbox helper.

    Builds one job spec (one job per turn, with per-line preset applied), runs
    the helper once (model loads a single time) in the isolated venv, then adds
    the natural inter-segment pause. Failed/missing segments become ~1s silence.
    """
    cb_cfg = config.get("chatterbox", {})
    voices = config.get("voices", {})
    venv_python = _resolve_venv_python(cb_cfg.get("venv_python", ".venv-chatterbox/Scripts/python.exe"))
    model_dir = PROJECT_ROOT / cb_cfg.get("model_dir", "models/chatterbox")
    helper = VOICE_DIR / "chatterbox_synth.py"

    if not venv_python.exists():
        log.error("Chatterbox venv python not found at %s; emitting silence.", venv_python)
    if not model_dir.exists():
        log.error("Chatterbox model dir not found at %s; emitting silence.", model_dir)

    intro_available = "intro" in cb_cfg.get("presets", {})
    first_handled = False

    jobs: list[dict[str, Any]] = []
    out_paths: list[Path] = []
    for idx, turn in enumerate(script):
        if not turn.get("line", "").strip() and not (turn.get("preset")):
            continue
        speaker = turn.get("speaker", "male_host")
        out_path = out_dir / f"segment_{idx:03d}_{speaker}.wav"
        out_paths.append(out_path)

        # Guarantee the signature open: if the broadcast's first spoken line is
        # an untagged male greeting, apply the "intro" preset ourselves. Weaker
        # (local) models often omit the preset tag the script writer requests.
        eff_turn = turn
        if not first_handled:
            first_handled = True
            if intro_available and speaker == "male_host" and not turn.get("preset"):
                eff_turn = {**turn, "preset": "intro"}
                log.info("Auto-applied 'intro' preset to the opening male line.")

        r = _resolve_preset(eff_turn, cb_cfg, voices)
        ref = PROJECT_ROOT / r["reference"] if r["reference"] else None
        job = {
            "text": r["text"],
            "ref": str(ref) if ref else "",
            "out": str(out_path),
            "gen_kwargs": r["gen_kwargs"],
        }
        if r["seed"] is not None:
            job["seed"] = int(r["seed"])
        jobs.append(job)

    spec = {
        "model_dir": str(model_dir),
        "device": cb_cfg.get("device", "auto"),
        "mode": "full",
        "quantize_t3": False,
        "gen_kwargs": {k: cb_cfg[k] for k in _GEN_KEYS if k in cb_cfg},
        "jobs": jobs,
    }

    ok: set[str] = set()
    if venv_python.exists() and model_dir.exists() and jobs:
        try:
            proc = subprocess.run(
                [str(venv_python), str(helper)],
                input=json.dumps(spec),
                capture_output=True,
                text=True,
                timeout=60 * 60,
            )
            for ln in proc.stdout.splitlines():
                if ln.startswith("OK "):
                    ok.add(ln[3:].strip())
                elif ln.startswith("ERR "):
                    log.error("Chatterbox: %s", ln)
            if proc.returncode != 0:
                log.error("Chatterbox helper exited %s: %s", proc.returncode, proc.stderr[-500:])
        except Exception as exc:
            log.error("Chatterbox helper invocation failed (%s).", exc)

    # Add pauses to rendered segments; substitute silence for any that failed.
    for p in out_paths:
        if str(p) in ok and p.exists():
            _append_pause(p)
        else:
            log.warning("Chatterbox segment missing (%s); inserting silence.", p.name)
            _silence(1000).export(p, format="wav")

    log.info("Synthesized %d Chatterbox segments to %s.", len(out_paths), out_dir)
    return out_paths


# --- Public entry point ----------------------------------------------------

def synthesize_script(
    script: list[dict[str, str]],
    config: dict[str, Any],
    out_dir: Path = SEGMENTS_DIR,
) -> list[Path]:
    """Render each dialogue turn to a WAV. Returns ordered segment paths.

    ``script`` items are ``{"speaker": "...", "line": "..."}`` dicts. Speakers
    map to ``[tts].male_voice_id`` / ``female_voice_id`` in config.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clear stale segments from previous runs - both the raw renders and the
    # post-processed copies, so a shorter new script can't leave old tail
    # segments behind for the mixer/test path to pick up.
    for pattern in ("segment_*.wav", "processed_segment_*.wav"):
        for old in out_dir.glob(pattern):
            old.unlink(missing_ok=True)

    # Chatterbox runs out-of-process and renders the whole script in one model
    # load, with per-line delivery presets — handled on its own path.
    if config.get("tts", {}).get("engine", "xtts").lower() == "chatterbox":
        return _synthesize_chatterbox(script, config, out_dir)

    tts_cfg = config.get("tts", {})
    voice_map = {
        "male_host": tts_cfg.get("male_voice_id", ""),
        "female_host": tts_cfg.get("female_voice_id", ""),
    }

    backend = _make_backend(config)
    paths: list[Path] = []

    for idx, turn in enumerate(script):
        speaker = turn.get("speaker", "male_host")
        line = normalize_for_tts(turn.get("line", "").strip())
        voice_id = voice_map.get(speaker, "")
        out_path = out_dir / f"segment_{idx:03d}_{speaker}.wav"

        if not line:
            continue
        if not voice_id:
            log.warning(
                "No voice id configured for speaker '%s'; segment %d is silence.",
                speaker,
                idx,
            )
            _silence(1000).export(out_path, format="wav")
            paths.append(out_path)
            continue

        try:
            backend.render(line, voice_id, out_path)
            _append_pause(out_path)
        except Exception as exc:
            # Spec: substitute ~1s of silence on failure and keep going.
            log.error("TTS failed on segment %d (%s); inserting silence.", idx, exc)
            _silence(1000).export(out_path, format="wav")

        paths.append(out_path)

    log.info("Synthesized %d segments to %s.", len(paths), out_dir)
    return paths


if __name__ == "__main__":
    from config_loader import load_config

    cfg = load_config()
    demo = [
        {"speaker": "male_host", "line": "Good morning, Night City. The chrome is bright today."},
        {"speaker": "female_host", "line": "Bright and overpriced, choom. Let's get into it."},
    ]
    for p in synthesize_script(demo, cfg):
        print(p)
