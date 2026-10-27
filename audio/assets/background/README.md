# Intro music beds

Place short music beds here (`.mp3`, `.wav`, `.ogg`, `.flac`). The mixer picks
one at random and plays it **under the intro only** — it is *not* looped. The bed
runs beneath the opening (intro/hype) lines, then fades out over the next few
seconds once the intro sequence ends.

Tuning lives in `config.toml` under `[audio]`:
- `intro_bed_volume` — bed level relative to the voices (0–1).
- `intro_bed_fade_ms` — how long the fade-out runs after the intro ends.

A bed shorter than the intro simply ends early (no loop). If this folder is
empty the broadcast still renders — just without a bed.
