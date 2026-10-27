# Topic-switch swooshes

Short whoosh/swoosh stings (`.mp3`, `.wav`, `.ogg`, `.flac`) layered in when the
broadcast switches topics — e.g. from the Babylon Bee stories into the weather.

The mixer fires a swoosh only at MAJOR section changes — when the broadcast
leaves the news block and enters the weather/commute segment or a sponsor read
(detected from the script's content) — not at every story-to-story hand-off
inside the news. One swoosh per section entry; the intro is never swooshed and
the outro jingle covers the close.

Tuning lives in `config.toml` under `[audio]`:
- `swoosh_gain_db` — swoosh level relative to the voices.
- `max_swooshes` — safety cap per broadcast.
- `swoosh_min_gap_ms` — minimum spacing so one pivot can't fire two swooshes.

If this folder is empty the broadcast renders without swooshes.
