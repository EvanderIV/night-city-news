# Deploying on Debian 13 (Trixie)

Target: a headless Debian 13 server, **~9 GB free RAM** (idles ~6.15 / 15.5 GiB),
no GPU, **lots of time** (CPU generation, ≤20 h window). The pipeline runs
overnight: generate at ~02:00, play at ~07:00.

The project was developed on Windows; the two virtualenvs and the model weights
are **not** portable — recreate them natively on Debian (Linux wheels). Source
files, `config.toml`, `assets/`, and the curated voice reference WAVs are
portable.

---

## 0. Quick start (automated)

```bash
git clone <your-repo-url> night-city-broadcast
cd night-city-broadcast
# copy the (gitignored) Chatterbox reference WAVs from your dev machine:
scp dev:night-city-news/voice/embeddings/*.wav voice/embeddings/
bash setup_debian.sh
```

`setup_debian.sh` is idempotent and does everything in §2–§9: apt packages, both
venvs, Ollama + the configured model, the Chatterbox weights, runtime dirs, and
the scheduled generate/play **systemd user timers** (times read from
`[schedule]`). Flags: `--no-apt`, `--no-ollama`, `--no-model`, `--no-timers`,
`-y`. It finishes with smoke tests and prints next steps. The rest of this doc is
the manual reference for what the script automates and how to tune it.

---

## 1. Architecture recap (what runs where)

Three sequential stages, so peak RAM is the **largest single stage**, never the
sum:

| Stage | Engine | Approx RAM | Notes |
|---|---|---|---|
| Script generation | Ollama, `qwen2.5:14b-instruct-q3_K_S` | ~7.8 GB | unloads after (`keep_alive=0`) |
| Voice synthesis | Chatterbox in `.venv-chatterbox` | ~4.7 GB | own venv (torch 2.6 pins) |
| Post-process + mix | ffmpeg + `pydub` in `.venv` | < 0.5 GB | |

With a 6.15 GB idle baseline: script gen peaks ~14 GB / 15.5 (≈1.5 GB headroom),
then the model unloads before synthesis. **Two separate venvs** are required —
the Chatterbox stack (`torch==2.6`, `transformers==5.2`) conflicts with the main
venv, and the main pipeline shells out to the Chatterbox venv.

---

## 2. System packages

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git curl ffmpeg \
                    libsndfile1 mpg123
```

- **ffmpeg** — provides the shared libraries (`libavcodec`, `libavformat`, …)
  that `pydub` *and* torchcodec need. On Linux this is the whole story — none of
  the Windows DLL-copying dance is required.
- **libsndfile1** — `soundfile` (Chatterbox output) needs it.
- **mpg123** — the broadcast player (matches `[playback].method`). For
  PipeWire/Pulse output use `pipewire`/`paplay` instead (see §6).
- Debian 13 ships Python 3.13, so `tomllib` is built in (no `tomli` needed).

---

## 3. Get the code + clone references

```bash
git clone <your-repo-url> night-city-broadcast
cd night-city-broadcast
```

The `output/`, `models/`, `voice/embeddings/`, and training-data dirs are
gitignored, so copy these from your dev machine (or regenerate):

- **`voice/embeddings/*.wav`** — the Chatterbox references: `male_host_reference.wav`
  (stoic), `male_dramatic_reference.wav` (the intro voice), `female_host_reference.wav`.
  These are plain WAVs — `scp` them over. (Or copy `voice/training_data/**/curated/`
  and re-run the reference build.)
- **`models/chatterbox/`** — fetch on the server with `bash download_chatterbox.sh`
  (curl, resumable; works on Linux unchanged).

---

## 4. Main venv (`.venv`)

The broadcast engine is **chatterbox**, so the main venv does NOT need the heavy
XTTS/torch stack — only the light deps:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install "pydub>=0.25.1" "ffmpeg-python>=0.2.0" \
    "numpy>=1.24.0" "httpx>=0.27.0" "beautifulsoup4>=4.12.0" "soco>=0.30.0"
```

(If you ever switch `[tts].engine` back to `xtts`, install the full
`requirements.txt` instead — but that adds a multi-GB torch download.)

---

## 5. Chatterbox venv (`.venv-chatterbox`)

```bash
python3 -m venv .venv-chatterbox
.venv-chatterbox/bin/python -m pip install --upgrade pip
# CPU torch first (the exact pins Chatterbox requires)
.venv-chatterbox/bin/python -m pip install torch==2.6.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cpu
# Chatterbox itself (pulls transformers 5.2, librosa, soundfile, etc.) + psutil
.venv-chatterbox/bin/python -m pip install chatterbox-tts psutil
```

> Do **not** downgrade `transformers` here — Chatterbox pins `transformers==5.2`
> and works with it. (The `transformers<5` downgrade from the Windows notes
> applied only to the *XTTS* stack, which this deployment doesn't use.)
>
> torch 2.6's `torchaudio` writes WAVs via the soundfile backend (so
> `libsndfile1` from §2 is what matters); no torchcodec / DLL setup is needed on
> Linux. Verify the stack:
> `.venv-chatterbox/bin/python -c "from chatterbox.tts import ChatterboxTTS; print('ok')"`

The synthesizer auto-detects `bin/python` vs `Scripts/python.exe`, so the
`[chatterbox].venv_python` value in `config.toml` works as-is on both OSes.

---

## 6. Ollama + the model

```bash
curl -fsSL https://ollama.com/install.sh | sh   # installs + starts a systemd service
ollama pull qwen2.5:14b-instruct-q3_K_S          # ~6.7 GB download
```

The pipeline talks to `http://localhost:11434` and sets `keep_alive=0` per
request, so the 14B unloads from RAM right after each call (freeing it before
the synthesis stage). No extra Ollama config needed.

**Model choice / RAM:** `q3_K_S` measured ~7.8 GB resident at `num_ctx=6144`.
If your free RAM is tighter than measured, drop `[llm].model` to
`qwen2.5:7b-instruct-q6_K` (~6.4 GB) — see the model note in `config.toml`.
Quality scales mostly from the best-of-N + refinement knobs below, not the base
model.

---

## 7. config.toml for the server

Already set for this deployment; double-check:

```toml
[llm]
provider = "ollama"
model = "qwen2.5:14b-instruct-q3_K_S"
candidates = 3          # best-of-N: more candidates = better + slower
refine_passes = 2       # critique-and-rewrite passes per candidate
ollama_num_ctx = 6144
ollama_keep_alive = 0   # unload after each call (frees RAM for synthesis)

[playback]
method = "mpg123"       # installed via apt; or "pipewire" (paplay) / "aplay"
```

**Using your 20 h budget:** raise `candidates` and/or `refine_passes`. Rough
LLM-call count = `candidates * (1 + refine_passes) + 1`. At 3/2 that's ~10 calls;
a CPU 14B-Q3 runs each in single-digit minutes, so ~1–2 h total — you have room
to push `candidates` to 5–6 if you want the judge choosing from more takes.

---

## 8. Audio output from a scheduled job (the tricky part)

A bare cron job has **no audio session**, so `mpg123`/`paplay` won't reach the
speakers. Two clean options:

**A. systemd *user* timer (recommended)** — runs in your logged-in user's
session, which owns the audio device:

```ini
# ~/.config/systemd/user/ncn-play.service
[Service]
Type=oneshot
WorkingDirectory=%h/night-city-broadcast
ExecStart=%h/night-city-broadcast/.venv/bin/python main.py --play
```
```ini
# ~/.config/systemd/user/ncn-play.timer
[Timer]
OnCalendar=*-*-* 07:00:00
Persistent=true
[Install]
WantedBy=timers.target
```
```bash
systemctl --user enable --now ncn-play.timer
loginctl enable-linger $USER   # let user timers run without an active login
```

Mirror this with an `ncn-generate.{service,timer}` at `02:00` running
`main.py` (no `--play`). Generation needs no audio session, so it can also be a
plain root/system cron if you prefer.

**B. cron with the audio env exported** — if you stick with crontab, the play
job must export the session bus:

```cron
0 2 * * *  cd /home/USER/night-city-broadcast && .venv/bin/python main.py >> logs/broadcast.log 2>&1
0 7 * * *  XDG_RUNTIME_DIR=/run/user/$(id -u) cd /home/USER/night-city-broadcast && .venv/bin/python main.py --play >> logs/broadcast.log 2>&1
```

The systemd user timer (A) is more reliable for audio; prefer it.

---

## 9. Smoke test

```bash
# 1. Script only (local LLM, no audio) — should produce output/scripts/*.txt
.venv/bin/python main.py --dry-run

# 2. Chatterbox import sanity
.venv-chatterbox/bin/python -c "from chatterbox.tts import ChatterboxTTS; print('cb ok')"

# 3. Full pipeline (script -> voices -> mixed mp3 in output/final/)
.venv/bin/python main.py

# 4. Playback
.venv/bin/python main.py --play
```

Watch `logs/broadcast.log`. Every stage degrades rather than crashing (LLM
failure -> fallback script; TTS failure -> 1s silence; missing assets -> skipped).

---

## 10. RAM guardrails

- The 14B + the 9 GB baseline peak ~14 GB of 15.5 during generation. If other
  services spike, the OOM killer could trigger. To harden: lower `[llm].model`
  to the 7B-Q6, or lower `ollama_num_ctx` (less KV cache). Confirm live with
  `ollama ps` (shows the loaded model's footprint) while a generation runs.
- `keep_alive=0` is essential here — it guarantees the LLM frees its RAM before
  the Chatterbox stage, so the two big consumers never overlap.
