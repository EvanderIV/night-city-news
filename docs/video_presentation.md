# Video presentation (TV version)

A high-fidelity MP4 version of the broadcast, alongside the audio one.

- **Audio**: a *clean* mix — no radio band-limiting or hiss — with a slight
  low-shelf bass boost (`[video].bass_boost_db`). Built from the raw voice
  segments, so it's full-range, the opposite of the lo-fi radio broadcast.
- **Video**: a **news-desk composite** — host footage filling the frame, an
  over-the-shoulder story thumbnail (real Babylon Bee images, fetched at render
  time), a headline caption, and a lower-third name bar. Sponsor and sign-off are
  full-screen cards. Picture is synced to the audio via a cue timeline the mixer
  emits, so swooshes/jingles line up exactly.

## Running it

```
python main.py --video        # render audio broadcast + the MP4
```

Or set `[video].enabled = true` in `config.toml` to produce it on every run.
Output: `output/final/broadcast_YYYY-MM-DD.mp4` (the `.mp3` audio is still made).

## Dropping in real assets (placeholders are auto-generated until then)

| Element            | Where                          | Notes |
|--------------------|--------------------------------|-------|
| Male host footage  | `video/assets/hosts/male_host.mp4`   | `.mp4/.mov/.webm/.m4v`, or a still `.png/.jpg`. Looped/cropped to fill. |
| Female host footage| `video/assets/hosts/female_host.mp4` | same |
| Sponsor cards      | `video/assets/sponsors/*.png`  | shown full-screen during the sponsor read |
| Story thumbnails   | (automatic)                    | pulled from the Bee feed/article; titled placeholder card on fetch failure |

Lower-third names and the show name are `[video].male_host_name` /
`female_host_name` / `show_name`. Resolution/fps/quality: `width`/`height`/`fps`/
`crf`/`preset`.

## Requirements

- **ffmpeg** on PATH with `libx264` + `aac` (already required by the audio path).
- **Pillow** (`pip install -r requirements.txt`) for the text/cards.
- On Debian, install a TrueType font for crisp text: `apt install fonts-dejavu-core`
  (the card renderer looks for DejaVu/Liberation/Arial, falling back to Pillow's
  built-in font otherwise).
