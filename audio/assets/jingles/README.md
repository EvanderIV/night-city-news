# Outro jingles (by FM frequency) + prejingle

Outro jingles are grouped into one subfolder per FM frequency, e.g.:

```
jingles/
  88.9/   *.ogg ...
  91.9/   *.ogg ...
  98.7/   *.ogg ...
  prejingle/   scifi_00.mp3 ...   <- smoothing stings (see below)
```

At the end of the show the mixer:
1. picks a random FM frequency **folder** (chosen fairly — uniformly over the
   folders, so a station with more jingles isn't over-represented),
2. picks a random jingle from inside it,
3. plays a random **prejingle** sting first to ease into it; the jingle then
   starts `[audio].prejingle_gap_ms` (default 1000 ms) after the prejingle
   *begins*, so the two overlap (the prejingle is longer than that gap).

Supported: `.mp3`, `.wav`, `.ogg`/`.Ogg`, `.flac` (extension matching is
case-insensitive). If a folder is empty it's skipped; if everything is empty the
broadcast renders without an outro.

Topic-switch swooshes live separately in `../transitions/`.
