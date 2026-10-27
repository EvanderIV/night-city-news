"""Radio broadcast audio effect chain (ffmpeg).

Applies a lo-fi radio treatment to voice segments and to the final mix using
ffmpeg via subprocess. The chain, in order:

1. Bandpass filter (300-3400 Hz, telephone/radio range)
2. Soft compression (even out levels)
3. Slight distortion / saturation (warm analog feel)
4. Subtle background static/hiss layer
5. Normalize (peak at -3 dBFS)

``process_segment`` is aggressive (per-line voice). ``process_final`` is
gentler (wider band, more headroom) for the fully mixed broadcast.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_loader import get_logger  # noqa: E402

log = get_logger("postprocess")


def _ffmpeg_bin() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("ffmpeg not found on PATH. Install it (see README).")
    return exe


# --- Filtergraphs ----------------------------------------------------------

def _segment_filtergraph() -> str:
    """Aggressive radio chain for individual voice lines."""
    return ",".join(
        [
            # 1. Bandpass: high-pass then low-pass = 300-3400 Hz.
            "highpass=f=300",
            "lowpass=f=3400",
            # 2. Soft compression.
            "acompressor=threshold=-18dB:ratio=4:attack=5:release=120:makeup=4",
            # 3. Slight saturation/distortion (gentle overdrive).
            "aexciter=amount=1.2",
            # 4. Subtle static/hiss mixed in (very low level).
            "anoisesrc=color=pink:amplitude=0.004[hiss]",  # noqa - see _build_cmd
        ]
    )


def _final_filtergraph() -> str:
    """Gentler chain for the fully mixed broadcast (wider band, more headroom)."""
    return ",".join(
        [
            "highpass=f=180",
            "lowpass=f=6500",
            "acompressor=threshold=-20dB:ratio=2.5:attack=10:release=200:makeup=2",
            "aexciter=amount=0.6",
        ]
    )


def _run(cmd: list[str]) -> None:
    log.debug("ffmpeg: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (code {result.returncode}): "
            f"{result.stderr.decode('utf-8', 'replace')[-500:]}"
        )


def process_segment(input_path: Path | str, output_path: Path | str) -> Path:
    """Apply the aggressive radio chain to one voice segment.

    Adds a faint pink-noise hiss bed, then bandpass + compression + saturation,
    and normalizes the peak to -3 dBFS.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # We mix in synthesized hiss as a second input so its level is independent
    # of the voice. The voice gets the radio band + comp + saturation, then we
    # amix with quiet pink noise and normalize.
    filtergraph = (
        "[0:a]highpass=f=300,lowpass=f=3400,"
        "acompressor=threshold=-18dB:ratio=4:attack=5:release=120:makeup=4,"
        "aexciter=amount=1.2[voice];"
        "[1:a]volume=0.012,highpass=f=300,lowpass=f=3400[hiss];"
        "[voice][hiss]amix=inputs=2:duration=first:dropout_transition=0,"
        "dynaudnorm=p=0.9,alimiter=limit=0.708"  # ~ -3 dBFS peak ceiling
    )

    cmd = [
        _ffmpeg_bin(),
        "-y",
        "-i", str(input_path),
        "-f", "lavfi", "-i", "anoisesrc=color=pink:amplitude=0.5",
        "-filter_complex", filtergraph,
        "-ar", "44100",
        str(output_path),
    ]
    try:
        _run(cmd)
    except Exception as exc:
        log.error("process_segment failed for %s (%s); copying unprocessed.",
                  input_path.name, exc)
        shutil.copyfile(input_path, output_path)
    return output_path


def process_final(input_path: Path | str, output_path: Path | str) -> Path:
    """Apply the gentler mastering chain to the fully mixed broadcast."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    filtergraph = _final_filtergraph() + ",dynaudnorm=p=0.85,alimiter=limit=0.708"
    cmd = [
        _ffmpeg_bin(),
        "-y",
        "-i", str(input_path),
        "-af", filtergraph,
        "-ar", "44100",
        str(output_path),
    ]
    try:
        _run(cmd)
    except Exception as exc:
        log.error("process_final failed (%s); copying unprocessed.", exc)
        shutil.copyfile(input_path, output_path)
    return output_path


def master_for_video(input_path: Path | str, output_path: Path | str,
                     bass_gain_db: float = 3.0) -> Path:
    """Clean master for the VIDEO mix: NO radio band-limiting or hiss, just a
    very slight low-shelf bass boost, gentle leveling, and a peak limiter.

    The video presentation wants full-range, high-fidelity audio (the opposite of
    the lo-fi radio chain), with a touch of low-end warmth.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Low-shelf lift below ~110 Hz for a subtle bass boost; keep full bandwidth.
    filtergraph = (
        f"bass=g={bass_gain_db}:f=110:width_type=q:w=0.7,"
        "dynaudnorm=p=0.9:m=10,"
        "alimiter=limit=0.95"
    )
    cmd = [
        _ffmpeg_bin(),
        "-y",
        "-i", str(input_path),
        "-af", filtergraph,
        "-ar", "48000",  # 48k pairs with video/AAC
        str(output_path),
    ]
    try:
        _run(cmd)
    except Exception as exc:
        log.error("master_for_video failed (%s); copying unprocessed.", exc)
        shutil.copyfile(input_path, output_path)
    return output_path


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python audio/postprocessor.py <input.wav> <output.wav>")
        raise SystemExit(1)
    out = process_segment(sys.argv[1], sys.argv[2])
    print("Wrote", out)
