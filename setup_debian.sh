#!/usr/bin/env bash
# =============================================================================
# Night City Morning Broadcast - Debian 13 (headless) deployment
# =============================================================================
# One-shot, idempotent installer: system packages, both virtualenvs, the local
# LLM (Ollama + model), the Chatterbox weights, runtime dirs, and (optionally)
# the scheduled generate/play systemd user timers. Safe to re-run.
#
# Usage:
#   bash setup_debian.sh [options]
#
# Options:
#   --no-apt      Skip the apt system-package step (use if you lack sudo).
#   --no-ollama   Skip installing Ollama / pulling the model.
#   --no-model    Skip downloading the Chatterbox weights.
#   --no-timers   Skip creating the systemd user timers.
#   --yes, -y     Don't prompt; assume yes.
#   --help, -h    Show this help.
#
# Targets the chatterbox TTS engine (the configured default). No GPU assumed -
# the Chatterbox venv installs CPU torch and [chatterbox].device="auto" resolves
# to CPU on this box.
# =============================================================================
set -euo pipefail

# --- Resolve project root (this script's directory) --------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Options -----------------------------------------------------------------
DO_APT=1; DO_OLLAMA=1; DO_MODEL=1; DO_TIMERS=1; ASSUME_YES=0
for arg in "$@"; do
  case "$arg" in
    --no-apt) DO_APT=0 ;;
    --no-ollama) DO_OLLAMA=0 ;;
    --no-model) DO_MODEL=0 ;;
    --no-timers) DO_TIMERS=0 ;;
    -y|--yes) ASSUME_YES=1 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg (see --help)"; exit 2 ;;
  esac
done

# --- Pretty logging ----------------------------------------------------------
if [ -t 1 ]; then C_B=$'\033[1m'; C_G=$'\033[32m'; C_Y=$'\033[33m'; C_R=$'\033[31m'; C_0=$'\033[0m'
else C_B=""; C_G=""; C_Y=""; C_R=""; C_0=""; fi
step() { echo; echo "${C_B}==> $*${C_0}"; }
ok()   { echo "${C_G}  ✓ $*${C_0}"; }
warn() { echo "${C_Y}  ! $*${C_0}"; }
die()  { echo "${C_R}  ✗ $*${C_0}" >&2; exit 1; }

VENV=".venv"
CBVENV=".venv-chatterbox"
PY="$VENV/bin/python"
CBPY="$CBVENV/bin/python"

# --- Preflight ---------------------------------------------------------------
step "Preflight"
[ -f main.py ] && [ -f config.toml ] || die "Run this from the project root (main.py/config.toml not found)."
if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; else SUDO=""; fi
command -v curl >/dev/null 2>&1 || warn "curl not found yet (apt step will install it)."
ok "Project root: $SCRIPT_DIR"

# --- 1. System packages ------------------------------------------------------
APT_PKGS="python3 python3-venv python3-pip git curl ca-certificates ffmpeg libsndfile1 mpg123 fonts-dejavu-core"
if [ "$DO_APT" -eq 1 ]; then
  step "System packages (apt)"
  if command -v apt-get >/dev/null 2>&1; then
    $SUDO apt-get update -qq
    # shellcheck disable=SC2086
    $SUDO apt-get install -y $APT_PKGS
    ok "Installed: $APT_PKGS"
  else
    warn "apt-get not found; skipping (not Debian/Ubuntu?). Ensure these exist: $APT_PKGS"
  fi
else
  warn "Skipping apt step (--no-apt). Required: $APT_PKGS"
fi

command -v python3 >/dev/null 2>&1 || die "python3 not available."
PYVER="$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
ok "python3 = $PYVER"
python3 -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3,11) else 1)' \
  || die "Python >= 3.11 required (for tomllib). Found $PYVER."
command -v ffmpeg >/dev/null 2>&1 && ok "ffmpeg present" || warn "ffmpeg missing (needed for audio mix + video)."

# --- 2. Read settings from config.toml --------------------------------------
step "Reading config.toml"
read_cfg() { python3 - "$@" <<'PY'
import sys, tomllib
cfg = tomllib.load(open("config.toml","rb"))
cur = cfg
for k in sys.argv[1].split("."):
    cur = cur.get(k, {}) if isinstance(cur, dict) else {}
print(cur if not isinstance(cur, dict) else (sys.argv[2] if len(sys.argv) > 2 else ""))
PY
}
LLM_MODEL="$(read_cfg llm.model)"
GEN_TIME="$(read_cfg schedule.generate_time)"; GEN_TIME="${GEN_TIME:-02:00}"
PLAY_TIME="$(read_cfg schedule.playback_time)"; PLAY_TIME="${PLAY_TIME:-07:00}"
[ -n "$LLM_MODEL" ] || die "Could not read [llm].model from config.toml."
ok "LLM model: $LLM_MODEL | generate @ $GEN_TIME | play @ $PLAY_TIME"

# --- 3. Main venv (light deps; chatterbox engine, so no heavy XTTS stack) -----
step "Main venv ($VENV)"
[ -d "$VENV" ] || python3 -m venv "$VENV"
"$PY" -m pip install --upgrade -q pip
"$PY" -m pip install -q \
  "pydub>=0.25.1" "ffmpeg-python>=0.2.0" "numpy>=1.24.0" \
  "httpx>=0.27.0" "beautifulsoup4>=4.12.0" "lxml>=4.9" \
  "soco>=0.30.0" "Pillow>=10.0.0"
ok "Main venv ready (lxml = RSS xml parser; Pillow = video cards)."

# --- 4. Chatterbox venv (isolated; CPU torch 2.6 + chatterbox-tts) -----------
step "Chatterbox venv ($CBVENV)"
[ -d "$CBVENV" ] || python3 -m venv "$CBVENV"
"$CBPY" -m pip install --upgrade -q pip
if "$CBPY" -c 'import torch' 2>/dev/null; then
  ok "torch already installed ($("$CBPY" -c 'import torch;print(torch.__version__)'))"
else
  echo "  installing CPU torch 2.6.0 (this is a large download)..."
  "$CBPY" -m pip install -q torch==2.6.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cpu
fi
if "$CBPY" -c 'import chatterbox' 2>/dev/null; then
  ok "chatterbox-tts already installed"
else
  echo "  installing chatterbox-tts + psutil (pulls transformers 5.2, librosa, ...)..."
  "$CBPY" -m pip install -q chatterbox-tts psutil
fi
ok "Chatterbox venv ready (do NOT downgrade transformers here)."

# --- 5. Ollama + model -------------------------------------------------------
if [ "$DO_OLLAMA" -eq 1 ]; then
  step "Ollama + LLM model"
  if command -v ollama >/dev/null 2>&1; then
    ok "Ollama already installed"
  else
    echo "  installing Ollama (systemd service)..."
    curl -fsSL https://ollama.com/install.sh | sh
  fi
  # Make sure the server is reachable before pulling.
  if ! curl -fsS -m 3 http://localhost:11434/api/tags >/dev/null 2>&1; then
    warn "Ollama server not responding; trying to start it..."
    $SUDO systemctl start ollama 2>/dev/null || (nohup ollama serve >/dev/null 2>&1 &) || true
    sleep 3
  fi
  if ollama list 2>/dev/null | grep -q "${LLM_MODEL%%:*}"; then
    ok "Model already pulled: $LLM_MODEL"
  else
    echo "  pulling $LLM_MODEL (several GB; one time)..."
    ollama pull "$LLM_MODEL"
  fi
else
  warn "Skipping Ollama (--no-ollama). Ensure '$LLM_MODEL' is available at [llm].ollama_host."
fi

# --- 6. Chatterbox weights ---------------------------------------------------
CB_DIR="models/chatterbox"
CB_FILES="tokenizer.json conds.pt ve.safetensors s3gen.safetensors t3_cfg.safetensors"
if [ "$DO_MODEL" -eq 1 ]; then
  step "Chatterbox weights ($CB_DIR)"
  have_all=1
  for f in $CB_FILES; do [ -s "$CB_DIR/$f" ] || have_all=0; done
  if [ "$have_all" -eq 1 ]; then
    ok "All weight files already present"
  else
    [ -f download_chatterbox.sh ] || die "download_chatterbox.sh missing."
    bash download_chatterbox.sh
  fi
else
  warn "Skipping Chatterbox weight download (--no-model)."
fi

# --- 7. Voice references -----------------------------------------------------
step "Voice references (voice/embeddings/)"
CORE_REFS="male_host_reference.wav female_host_reference.wav male_dramatic_reference.wav"
missing=""
for r in $CORE_REFS; do [ -s "voice/embeddings/$r" ] || missing="$missing $r"; done
if [ -z "$missing" ]; then
  ok "Core host references present"
else
  warn "Missing core references:$missing"
  warn "These are gitignored. Copy them from your dev machine, e.g.:"
  warn "    scp dev:night-city-news/voice/embeddings/*.wav voice/embeddings/"
  warn "Or copy voice/training_data/<host>/curated/ and rebuild with the cloner."
fi

# --- 8. Runtime directories --------------------------------------------------
step "Runtime directories"
"$PY" -c "from paths import ensure_runtime_dirs; ensure_runtime_dirs()" && ok "output/, logs/, etc. ready"

# --- 9. systemd user timers (scheduled generate + play) ----------------------
if [ "$DO_TIMERS" -eq 1 ]; then
  step "systemd user timers"
  if command -v systemctl >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1; then
    UNIT_DIR="$HOME/.config/systemd/user"
    mkdir -p "$UNIT_DIR"
    WD="$SCRIPT_DIR"
    cat > "$UNIT_DIR/ncn-generate.service" <<EOF
[Unit]
Description=Night City Broadcast - generate
[Service]
Type=oneshot
WorkingDirectory=$WD
ExecStart=$WD/$VENV/bin/python main.py
EOF
    cat > "$UNIT_DIR/ncn-generate.timer" <<EOF
[Unit]
Description=Night City Broadcast - generate (daily $GEN_TIME)
[Timer]
OnCalendar=*-*-* $GEN_TIME:00
Persistent=true
[Install]
WantedBy=timers.target
EOF
    cat > "$UNIT_DIR/ncn-play.service" <<EOF
[Unit]
Description=Night City Broadcast - play
[Service]
Type=oneshot
WorkingDirectory=$WD
ExecStart=$WD/$VENV/bin/python main.py --play
EOF
    cat > "$UNIT_DIR/ncn-play.timer" <<EOF
[Unit]
Description=Night City Broadcast - play (daily $PLAY_TIME)
[Timer]
OnCalendar=*-*-* $PLAY_TIME:00
Persistent=true
[Install]
WantedBy=timers.target
EOF
    systemctl --user daemon-reload
    systemctl --user enable --now ncn-generate.timer ncn-play.timer
    loginctl enable-linger "$USER" 2>/dev/null \
      && ok "Lingering enabled (timers run without an active login)" \
      || warn "Could not enable linger; run: sudo loginctl enable-linger $USER"
    ok "Timers installed: generate @ $GEN_TIME, play @ $PLAY_TIME"
  else
    warn "No systemd --user session (headless without lingering?)."
    warn "Re-run after 'sudo loginctl enable-linger $USER' and a fresh login,"
    warn "or schedule via cron (see docs/debian_deployment.md §8)."
  fi
else
  warn "Skipping systemd timers (--no-timers). See docs/debian_deployment.md §8."
fi

# --- 10. Smoke tests ---------------------------------------------------------
step "Smoke tests"
"$PY" -c "import pydub, httpx, bs4, lxml, PIL, numpy; print('  main venv imports OK')" \
  && ok "main venv" || warn "main venv import check failed"
"$CBPY" -c "import torch; from chatterbox.tts import ChatterboxTTS; print('  chatterbox OK | cuda=%s' % torch.cuda.is_available())" \
  && ok "chatterbox venv" || warn "chatterbox import check failed"
if [ "$DO_OLLAMA" -eq 1 ]; then
  ollama list 2>/dev/null | grep -q "${LLM_MODEL%%:*}" && ok "model present in Ollama" || warn "model not found in 'ollama list'"
fi

# --- Done --------------------------------------------------------------------
step "Done"
cat <<EOF
${C_G}Setup complete.${C_0}

Next:
  1. Confirm core voice references exist in voice/embeddings/ (see warnings above).
  2. Dry run (script only, no audio):   $PY main.py --dry-run
  3. Full run (script -> voice -> mp3): $PY main.py
  4. Playback the latest broadcast:     $PY main.py --play

Logs: logs/broadcast.log   |   Timers: systemctl --user list-timers 'ncn-*'
Details + audio/cron notes: docs/debian_deployment.md
EOF
