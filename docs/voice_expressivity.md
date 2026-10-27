# Voice Expressivity & Delivery Presets — Report

**Date:** 2026-06-13
**Goal:** Give the hosts controllable expressive delivery — specifically the
male host's signature over-the-top **"Good morning, Night City!"** open — and
make it reproducible in the broadcast pipeline.
**Outcome:** Solved. The intro is produced reliably via a **dramatic reference
clip + plain text + a vetted seed pool**, and a general **per-line preset
system** is now wired into the pipeline.

---

## 1. The controls Chatterbox gives us

`generate()` exposes three expressivity knobs (all per-utterance):

| Knob | Default | Effect |
|---|---|---|
| `exaggeration` | 0.5 | Emotional intensity. ~0.7–1.2 = dramatic; >1.2 destabilizes. |
| `cfg_weight` | 0.5 | Lower (~0.3) slows pacing and lets drama land. |
| `temperature` | 0.8 | Higher = wilder intonation, but more mispronunciation. |

Plus two things we control outside the model:
- **Which reference clip** we clone from — this turned out to be the single
  biggest lever for *character* (see §3).
- **Seed** — makes a given take reproducible (added to the helper).

---

## 2. What did and didn't work (the journey)

1. **Cranking knobs alone** got energy but the wrong *character* — high
   `exaggeration` made him sound like an "excited kid," not a booming DJ.
2. **Spelling the vowels out** (`moooooorning`, `Cityyyy`) to force a drawl
   **backfired**: Chatterbox re-articulates repeated letters, producing
   "moo-uh-orning" / "city-ee-ee" breakup. There was no spelling that was both
   a long hold *and* clean — few repeats get normalized away, many repeats
   stutter.
3. **Post-processing pitch-shift down** (to add bass) was **rejected** — the
   simple shifter moved formants and sounded synthetic/awful. (A
   formant-preserving shifter, e.g. rubberband, remains an untested option, but
   we didn't need it.)
4. **The reference clip was the unlock.** The clean `radio_ad...test` stems are
   Stanley's *stoic / narrative* register, so the clone sounded stoic and
   "lacked charm." His *dramatic* "good morning night city" energy lives in the
   **q0/q1 clips** (which carry music grain). Building a reference from the
   q0/q1 clips transferred the charm and matched the original voice actor's
   vocal nature — grain bleed was tolerable.
5. **Plain text + dramatic reference = the answer.** With the dramatic
   reference supplying the delivery, plain **"Good morning, Night City!"** (no
   repeated letters → *no possible mispronunciation*) sounded dramatic *and*
   clean. Removing the elongation hack removed the only remaining defect.
6. **Best-of-N by seed.** At fixed settings the result is stochastic; rendering
   several seeds and auditioning gave multiple "just about perfect" takes
   (seeds 2, 4, 5, 7, 8 — **8** the favourite). Seeds are reproducible, so we
   keep the whole good set and pick one at random per run for variety.

**Validated intro recipe:**
> dramatic reference (`voice/embeddings/male_dramatic_reference.wav`, built from
> q0/q1 clips) + text **"Good morning, Night City!"** +
> `exaggeration 1.2 / cfg_weight 0.3 / temperature 0.6` + seed ∈ {8, 2, 4, 5, 7}.

---

## 3. The preset system (now in the pipeline)

A **delivery preset** is a named bundle of per-line overrides. Lines without a
preset use each host's normal voice.

**Config** — `config.toml` `[chatterbox]`:
- Default references per speaker (`male_reference` = stoic, `female_reference`).
- Default generation params (natural banter).
- `[chatterbox.presets.<name>]` tables, each optionally overriding
  `reference`, `text`, `seeds` (pool — one chosen at random per run), and any of
  the generation params. Shipped preset: **`intro`** (the recipe above).

**Script writer** — emits an optional `"preset"` per turn. It's instructed to
tag the first line `"preset": "intro"` with the exact text "Good morning, Night
City!"; all other lines omit it.

**Synthesizer** — for `engine = "chatterbox"`, `_resolve_preset()` turns each
turn into a render job (reference / text / seed / gen params), and the whole
script is rendered in **one** out-of-process helper call (model loads once).
`_synthesize_chatterbox()` then adds the inter-segment pause and substitutes ~1s
silence for any failed segment.

**Adding a new preset** (e.g. a `deadpan` for the female host's dry asides):
add a `[chatterbox.presets.deadpan]` table with its reference/params, teach the
script writer when to tag it, done — no code changes.

---

## 4. Reusable techniques established

- **Reference clip = character.** Curate references by the *register* you want,
  not just cleanliness. Different references per delivery mode is legitimate and
  powerful.
- **Don't fight the tokenizer.** For sustained/stylized delivery, prefer
  reference + params over trick spelling. (If a *guaranteed* smooth vowel hold
  is ever needed, a pitch-preserving **time-stretch** in post is the right tool
  — untested here, but noted.)
- **Seeds for reproducibility + curated variety.** Lock the good seeds; rotate
  within the vetted set for day-to-day variation without quality risk.
- **Per-line presets** keep the LLM in creative control while we own the actual
  voice settings.

---

## 5. Open items / future ideas

- **Grain cleanup:** the dramatic reference carries music-bed grain from the
  q0/q1 clips. Source separation (e.g. Demucs vocal isolation) could yield a
  *clean* dramatic reference for an even better intro. Not required — current
  result is good.
- **More presets:** `deadpan` (female asides), `hard_sell` (sponsor reads),
  `breaking_news` stinger, etc.
- **Time-stretch drawl** as an optional, deterministic alternative to seed
  roulette for the vowel hold.
- The intro preset **forces** its text; the saved script JSON shows that line as
  written by the LLM ("Good morning, Night City!") — kept in sync by instructing
  the writer to use the exact phrase.
