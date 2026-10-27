"""Speaker playback.

Plays the finished broadcast through the configured output method:
  - ``aplay``    : Linux ALSA (WAV only; transcodes MP3 first if needed)
  - ``mpg123``   : cross-platform MP3 CLI player
  - ``pipewire`` : PipeWire/PulseAudio via ``paplay``
  - ``sonos``    : streams to a Sonos room via the ``soco`` library

Logs playback start/end times and never crashes on a missing file.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_loader import get_logger  # noqa: E402
from paths import FINAL_DIR  # noqa: E402

log = get_logger("playback")


def find_latest_broadcast(directory: Path = FINAL_DIR) -> Path | None:
    """Return the most recently modified broadcast file, or None."""
    candidates: list[Path] = []
    for pattern in ("broadcast_*.mp3", "broadcast_*.wav"):
        candidates.extend(directory.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _ensure_wav(path: Path) -> Path:
    """Transcode to a temp WAV if needed (for aplay). Returns path to use."""
    if path.suffix.lower() == ".wav":
        return path
    from pydub import AudioSegment

    tmp = Path(tempfile.gettempdir()) / (path.stem + "_play.wav")
    AudioSegment.from_file(path).export(tmp, format="wav")
    return tmp


def _play_subprocess(cmd: list[str], tool: str) -> None:
    if shutil.which(cmd[0]) is None:
        raise RuntimeError(f"Playback tool '{cmd[0]}' not found on PATH.")
    log.info("Playing via %s: %s", tool, " ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(
            f"{tool} exited {result.returncode}: "
            f"{result.stderr.decode('utf-8', 'replace')[-300:]}"
        )


def _play_aplay(path: Path, volume: int) -> None:
    _play_subprocess(["aplay", str(_ensure_wav(path))], "aplay")


def _play_mpg123(path: Path, volume: int) -> None:
    # mpg123 -f scales samples; map 0-100 volume to 0-32768.
    scale = int(max(0, min(100, volume)) / 100 * 32768)
    _play_subprocess(["mpg123", "-q", "-f", str(scale), str(path)], "mpg123")


def _play_pipewire(path: Path, volume: int) -> None:
    vol = max(0.0, min(1.0, volume / 100))
    _play_subprocess(
        ["paplay", f"--volume={int(vol * 65536)}", str(_ensure_wav(path))],
        "paplay",
    )


def _play_sonos(path: Path, config: dict[str, Any]) -> None:
    """Play on a Sonos room. Requires the file be reachable; we host it briefly.

    soco streams from a URL, so we spin up a tiny local HTTP server rooted at
    the file's directory and hand Sonos the LAN URL.
    """
    import http.server
    import socket
    import socketserver
    import threading

    import soco

    room = config.get("playback", {}).get("sonos_room", "")
    if not room:
        raise RuntimeError("playback.method=sonos requires playback.sonos_room.")

    device = next((z for z in soco.discover() or [] if z.player_name == room), None)
    if device is None:
        raise RuntimeError(f"Sonos room '{room}' not found on the network.")

    serve_dir = path.parent
    handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(  # noqa: E731
        *a, directory=str(serve_dir), **k
    )
    httpd = socketserver.TCPServer(("0.0.0.0", 0), handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    # Determine our LAN IP as seen toward the Sonos device.
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((device.ip_address, 1400))
        local_ip = s.getsockname()[0]
    finally:
        s.close()

    url = f"http://{local_ip}:{port}/{path.name}"
    volume = config.get("playback", {}).get("volume", 80)
    try:
        device.volume = int(max(0, min(100, volume)))
        log.info("Sonos: streaming %s to '%s'.", url, room)
        device.play_uri(url)
        # Keep the HTTP server alive until playback finishes.
        time.sleep(2)
        while device.get_current_transport_info().get("current_transport_state") == "PLAYING":
            time.sleep(2)
    finally:
        httpd.shutdown()


def play(path: Path | str | None, config: dict[str, Any]) -> bool:
    """Play ``path`` (or the latest broadcast if None) via the configured method.

    Returns True on success, False on a handled error. Never raises.
    """
    playback_cfg = config.get("playback", {})
    method = playback_cfg.get("method", "mpg123").lower()
    volume = int(playback_cfg.get("volume", 80))

    target = Path(path) if path else find_latest_broadcast()
    if target is None or not target.exists():
        log.error("No broadcast file to play (looked for %s).", target or FINAL_DIR)
        return False

    start = time.time()
    log.info("Playback START %s | method=%s", target.name, method)
    try:
        if method == "aplay":
            _play_aplay(target, volume)
        elif method == "mpg123":
            _play_mpg123(target, volume)
        elif method == "pipewire":
            _play_pipewire(target, volume)
        elif method == "sonos":
            _play_sonos(target, config)
        else:
            log.error("Unknown playback method '%s'.", method)
            return False
    except Exception as exc:
        log.error("Playback failed (%s).", exc)
        return False

    log.info("Playback END %s | %.1fs elapsed.", target.name, time.time() - start)
    return True


if __name__ == "__main__":
    from config_loader import load_config

    cfg = load_config()
    ok = play(None, cfg)
    raise SystemExit(0 if ok else 1)
