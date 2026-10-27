# Night City Morning Broadcast

An automated overnight pipeline that generates a stylized **Cyberpunk 2077
"Night City"** morning news radio broadcast and plays it through your speakers
at a configured wake time. Two AI-cloned hosts — a smooth corporate-friendly
male anchor and a sharper, sardonic female newscaster — banter over fake city
stats (or reframed Babylon Bee headlines), wrapped in lo-fi radio audio with
jingles and ambience.

It runs unattended via cron: generate at 2 AM, play at 7 AM.

---

## 1. Quick Start

From a fresh clone to your first broadcast in under 10 commands:

```bash
# 1. Clone and enter the project
git clone <your-repo-url> night-city-broadcast
cd night-city-broadcast

# 2. Create a virtualenv and install deps
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Install ffmpeg (required for all audio work)
#    macOS: brew install ffmpeg | Debian/Ubuntu: sudo apt install ffmpeg
#    Windows: winget install Gyan.FFmpeg

# 4. Add voice training clips (see "Voice Setup" below)
#    voice/training_data/male_host/   and   voice/training_data/female_host/

# 5. Set your LLM key
export ANTHROPIC_API_KEY="sk-ant-..."   # or put it in config.toml

# 6. Clone the voices (one time)
python main.py --setup

# 7. Generate tonight's broadcast
python main.py

# 8. Play the latest broadcast
python main.py --play
```

Tip: run `python main.py --dry-run` first to generate and inspect the script
without spending time on TTS and audio rendering.

---

## 2. Voice Setup

The two hosts use voices cloned from real audio you provide.

1. Put training clips here (filenames don't matter — they're discovered with
   `glob`):
   - `voice/training_data/male_host/`   — the male anchor
   - `voice/training_data/female_host/` — the female newscaster
2. Aim for **~2 minutes of clean, single-speaker audio per voice**, split
   across any number of `.mp3` / `.wav` / `.m4a` / `.flac` / `.ogg` files.
3. Run the one-time cloning step:

   ```bash
   python main.py --setup
   ```

What `--setup` does per engine (`[tts].engine`):

| Engine | Behavior |
|---|---|
| `xtts` (default) | Concatenates clips into a reference WAV under `voice/embeddings/`, reused on later runs. Local & free; needs PyTorch (CPU works, GPU is faster). Uses the maintained **`coqui-tts`** fork (the original `TTS` package is archived and won't install on Python 3.12+). On Windows it also needs FFmpeg **shared** libraries — see Troubleshooting. |
| `piper` | **Skips cloning** (Piper uses pre-trained voices); logs a warning. Set `piper_male_model` / `piper_female_model` in config. |
| `elevenlabs` | Uploads clips to the ElevenLabs cloning API and stores the returned voice ids. Needs `elevenlabs_api_key`. |

After setup, `male_voice_id` and `female_voice_id` in `config.toml` are
auto-populated.

---

## 3. Config Reference (`config.toml`)

| Section / Key | Description |
|---|---|
| `[mode].content_source` | `"simulation"` (fake city stats) or `"babylon"` (Babylon Bee headlines reframed for 2077). |
| `[schedule].generate_time` | Documentation of when cron runs generation (e.g. `"02:00"`). |
| `[schedule].playback_time` | Documentation of when cron plays the broadcast (e.g. `"07:00"`). |
| `[llm].provider` | `"anthropic"` or `"ollama"`. |
| `[llm].model` | Model id for the chosen provider (e.g. `claude-opus-4-6`). |
| `[llm].api_key` | Anthropic key. Leave blank to use the `ANTHROPIC_API_KEY` env var. |
| `[llm].ollama_host` | Base URL of the local Ollama server (default `http://localhost:11434`). |
| `[tts].engine` | `"xtts"`, `"piper"`, or `"elevenlabs"`. |
| `[tts].elevenlabs_api_key` | Only used when `engine = "elevenlabs"`. |
| `[tts].male_voice_id` / `female_voice_id` | Auto-populated by `--setup`. |
| `[tts].piper_male_model` / `piper_female_model` | Piper voice model names (engine = piper). |
| `[city].name` / `year` / `factions` | Flavour woven into stats and the script prompt. |
| `[audio].background_volume` | `0.0`–`1.0`; level of the ambience bed under the voices. |
| `[audio].radio_filter` | `true` applies the lo-fi radio EQ + compression chain. |
| `[audio].output_format` | `"mp3"` or `"wav"`. |
| `[playback].method` | `"aplay"`, `"mpg123"`, `"pipewire"`, or `"sonos"`. |
| `[playback].sonos_room` | Sonos room name; required only when `method = "sonos"`. |
| `[playback].volume` | `0`–`100` (best-effort; not all backends honor it). |

---

## 4. Cron Setup

Generate overnight and play in the morning. Edit your crontab with
`crontab -e` and paste (adjust the paths):

```cron
# Generate broadcast at 2 AM
0 2 * * * /path/to/venv/bin/python /path/to/night-city-broadcast/main.py >> /path/to/logs/broadcast.log 2>&1

# Play broadcast at 7 AM
0 7 * * * /path/to/venv/bin/python /path/to/night-city-broadcast/main.py --play >> /path/to/logs/broadcast.log 2>&1
```

The times here should match `[schedule].generate_time` and
`[schedule].playback_time` in `config.toml` (those fields are documentation —
cron is what actually triggers the runs).

---

## 5. Content Modes

Switch with `[mode].content_source`:

- **`simulation`** — A persistent fake Night City economy. Stock market, crime
  blotter, weather/AQI, traffic, eurodollar/yen FX, and civic news. State lives
  in `output/city_state.json` and drifts day-to-day (random walks, occasional
  corpo-war shocks), so the city evolves over time.
- **`babylon`** — Scrapes the latest Babylon Bee headlines and asks the LLM to
  reframe each story as if it happened in Night City. Results are cached for 12
  hours. **If scraping fails, the run automatically falls back to simulation.**

Try a module on its own:

```bash
python content/city_simulation.py   # prints a sample day of stats
python content/babylon_scraper.py    # prints fetched headlines
python content/script_writer.py      # generates a sample script
```

---

## 6. Troubleshooting

**TTS is extremely slow / CUDA errors (xtts).**
XTTS runs on CPU by default, which is slow but works. For speed, install a
CUDA-enabled PyTorch matching your GPU. If you see CUDA out-of-memory errors,
force CPU with `CUDA_VISIBLE_DEVICES="" python main.py`. The CPU build of torch
is installed from the PyTorch CPU index:
`pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu`.

**`No module named 'TTS'` / TTS won't install on Python 3.12+.**
The original Coqui `TTS` package is archived and requires Python `<3.12`. This
project uses the drop-in fork `coqui-tts` instead (`requirements.txt` already
points at it). The import path is unchanged (`from TTS.api import TTS`).

**`Could not load libtorchcodec_*.dll` / `torchcodec ... shared libraries`.**
torch 2.9+ performs audio IO through `torchcodec`, which needs FFmpeg's
**shared** libraries (`avcodec-*.dll`, `avutil-*.dll`, …). A static
`ffmpeg.exe` (e.g. the gyan "essentials" build) does NOT include these. Fix on
Windows:
1. Download a *shared* FFmpeg build, e.g. BtbN's
   `ffmpeg-master-latest-win64-gpl-shared.zip`.
2. Copy the seven `av*.dll`/`sw*.dll` files from its `bin/` either onto your
   `PATH`, or directly into the torchcodec package folder
   (`.venv/Lib/site-packages/torchcodec/`) so they sit next to
   `libtorchcodec_core*.dll`. Windows resolves a DLL's dependencies from its
   own directory first, so the latter works without touching `PATH`.
   torchcodec ships loaders for FFmpeg majors 4–8.

**`ffmpeg not found on PATH`.**
The audio post-processor and `pydub` both need ffmpeg. Install it (see Quick
Start step 3) and reopen your shell so the binary is on `PATH`.

**`Audio device not found` / playback silent.**
Check `[playback].method` matches your system: `aplay`/`pipewire` are Linux,
`mpg123` is cross-platform. Test the player directly with
`python playback/player.py`. For Sonos, ensure the host and speaker are on the
same LAN and `sonos_room` exactly matches the room name.

**API key errors.**
For Anthropic, set `ANTHROPIC_API_KEY` or `[llm].api_key`. For ElevenLabs, set
`[tts].elevenlabs_api_key`. The script writer retries once, then falls back to
`assets/fallback_script.json` so a broadcast still airs.

**Piper voice not found.**
Install the `piper` binary and download the ONNX voice models named in
`piper_male_model` / `piper_female_model`. Cloning is skipped for Piper.

**Nothing played at 7 AM.**
Check `logs/broadcast.log` for the timestamped stage that failed. The pipeline
is designed never to crash the cron job — every stage logs and degrades.

---

## 7. Extending the City Simulation

All simulation logic lives in [content/city_simulation.py](content/city_simulation.py).
To add a new stat category (say, "media ratings"):

1. **Add seed data / baseline.** If your category needs day-to-day memory, add
   a key to `_new_state()` and bump `STATE_VERSION` so old state files reseed
   cleanly. If it's freshly random each day, you can skip persistence.
2. **Write a generator.** Add a `_generate_media()` (or `_drift_media(state)`
   for persistent categories) following the existing functions. Use `_clamp()`
   to keep values in range and the drift conventions already in the file.
3. **Wire it into `generate_city_stats()`.** Call your generator and add its
   result to the returned dict under a new key.
4. **Surface it to the script.** The script writer passes the whole stats dict
   to the LLM, so your new key is automatically available — optionally mention
   it explicitly in `_build_simulation_user_prompt()` in
   [content/script_writer.py](content/script_writer.py) so the hosts reliably
   cover it.
5. **Test it.** `python content/city_simulation.py` prints a full sample day,
   including your new category.

State is written atomically (`.tmp` then rename) and validated on load — if you
change the schema, bump `STATE_VERSION` and old files reseed automatically.

---

## Project Layout

```
night-city-broadcast/
├── main.py                  # Orchestrator (full pipeline + --play/--setup/--dry-run)
├── config.toml              # All configuration
├── paths.py                 # Central path resolution (no hardcoded paths)
├── config_loader.py         # Config loading + shared logging
├── content/                 # city_simulation / babylon_scraper / script_writer
├── voice/                   # cloner / synthesizer + training_data/ + embeddings/
├── audio/                   # postprocessor / mixer + assets/ (background, jingles)
├── playback/                # player
├── assets/fallback_script.json
├── output/                  # scripts / segments / final (gitignored)
└── logs/broadcast.log
```
