"""Broadcast script generator.

Takes either simulated city stats or Babylon Bee articles and asks the
configured LLM (Anthropic Claude or a local Ollama model) to produce a
two-host Night City radio script as a JSON list of dialogue turns::

    [ { "speaker": "male_host", "line": "..." }, ... ]

Resilience (per spec): retry once with exponential backoff, then fall back to
the pre-written emergency script in ``assets/fallback_script.json``.

Run standalone to test against simulated data::

    python content/script_writer.py
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_loader import get_logger  # noqa: E402
from paths import FALLBACK_SCRIPT  # noqa: E402

log = get_logger("script")

VALID_SPEAKERS = {"male_host", "female_host"}
MAX_RETRIES = 1  # one retry after the initial attempt (spec: "retry once")

# Structured-output schema for the dialogue array (used to grammar-constrain
# local Ollama models). A draft and a refined draft share this shape.
TURNS_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "speaker": {"type": "string", "enum": list(VALID_SPEAKERS)},
            "line": {"type": "string"},
            "preset": {"type": "string"},
        },
        "required": ["speaker", "line"],
    },
}

# Schema for the judge's selection (best-of-N).
JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"best": {"type": "integer"}},
    "required": ["best"],
}

# Opening greeting pools per mode. We force the first line to one of these
# (post-generation) so model drift can't leak the wrong setting and the dramatic
# intro voice opens on a vetted phrase - but we pick at random for day-to-day
# variety. Kept short and same-cadence so the loud "intro" delivery still lands.
GREETINGS = {
    "simulation": [
        "Good morning, Night City!",
        "Goood morning, Night City!",
        "Rise and shine, Night City!",
        "Wake up, Night City!",
    ],
    "babylon": [
        "Good morning, America!",
        "Goood morning, America!",
        "Rise and shine, America!",
        "Top of the morning, America!",
    ],
}


# --- Prompt construction ---------------------------------------------------

SIMULATION_SYSTEM = """\
You are the head writer for "Night City Morning Broadcast", a satirical \
dystopian radio show set in Night City in the year {year}. The world is \
corporate-controlled, neon-soaked, and dangerous, in the style of Cyberpunk \
2077.

There are exactly TWO hosts:
- male_host: smooth, slightly smug, relentlessly corporate-friendly. His vibe \
is "everything is fine, citizen." He sells the dystopia with a warm voice.
- female_host: sharper, more sardonic. She plays along with the corporate line \
but occasionally lets the cynicism break through.

STYLE RULES:
- Hosts banter naturally and hand off to each other. No monologue longer than \
3-4 sentences.
- Use Cyberpunk 2077 slang naturally (not in every line): choom, chrome, \
flatline, gonk, nova, preem, solo, edgerunner, eddies, ripperdoc, braindance.
- Weave in brand mentions that sound like paid corporate sponsorships, e.g. \
"This traffic report is brought to you by Trauma Team International."
- Total spoken content should run roughly 4-6 minutes (about 700-950 words).
- ALWAYS open with a jingle cue line (e.g. a punchy station ident) and ALWAYS \
close with a sign-off.
- Keep it irreverent and satirical, never sincere about the dystopia.

CONTENT STANDARDS (strict - non-negotiable):
- Broadcast-clean. NO profanity or crude language (no "shit", "fuck", "ass", \
"bitch", "bastard", "damn", "piss", or "hell" as an expletive). Use mild, witty \
phrasing instead.
- NO slurs of any kind, ever.
- Do NOT use God's or Jesus's name in vain, and never use religious \
exclamations as expletives (no "oh my god", "jesus", "jesus christ", "god \
damn", "for god's sake"). Use clean exclamations instead: "oh my goodness", \
"good grief", "wow", "unbelievable", "for crying out loud".
- Humor comes from wit, irony, and absurdity - never shock words. Clever over crude.

BROADCAST STRUCTURE (this running order is mandatory — it is the whole joke):
The show is front-loaded with sensational mayhem and deflates into tedium.
1. COLD OPEN: the male host's signature greeting, then a FRESH back-and-forth \
that immediately teases the biggest, most lurid story of the morning to hook the \
listener. Write this open originally every time - both the co-host's first line \
and the male host's high-energy lead-in should hook on TODAY'S actual top story, \
never generic "big show" filler.
2. SENSATIONAL BLOCK (give this the most airtime, ~50%): lead with the most \
shocking, violent, dramatic material - cyberpsycho rampages, gang and faction \
wars, Maelstrom activity, ripperdoc raids, riots, corpo sabotage. The hosts \
treat horrifying events as thrilling entertainment, breathless and gleeful, \
one-upping each other. High energy.
3. THE WIND-DOWN: segue through mid-tier filler - weather (acid rain, AQI) and \
traffic. Energy starts visibly draining here.
4. THE BORING TAIL: now grind through the genuinely dull but "important" news - \
the stock market (tickers, prices, daily % moves, analyst notes) and the \
economy (eurodollar/yen rate, unemployment, black market index). The hosts get \
audibly bored, perfunctory, checking the clock, reading numbers like a chore. \
This deadpan tedium - right after the gleeful violence - is the satire.
5. SPONSOR READS + SIGN-OFF: limp into one or two corporate sponsor reads \
("brought to you by..."), then the female host's dry sign-off.
The emotional arc - manic excitement over carnage, collapsing into bored \
monotone over stock prices and ads - must be audible in the writing.

OUTPUT FORMAT:
Return ONLY a JSON array (no markdown fences, no commentary) of objects with \
"speaker" (either "male_host" or "female_host") and "line" (the spoken text). \
An object MAY also include an optional "preset" naming a delivery style.

PRESET RULES (these control the male host's delivery voice - follow exactly):
- The VERY FIRST line must be the male host's signature greeting, written \
EXACTLY as "Good morning, Night City!" and tagged "preset": "intro". Keep it \
exactly those four words.
- The next ONE OR TWO male_host lines - only the high-energy "welcome to the \
show" lead-in, BEFORE the actual news starts - may be tagged "preset": "hype" \
to keep them in the big, loud voice.
- Tag NOTHING else. The moment the real news begins (crime stats, weather, \
traffic, stocks, economy, sponsors, sign-off), OMIT "preset" so those lines use \
the hosts' normal voices. NEVER tag stock, economy, weather, traffic, or \
sponsor lines - the loud voice is wrong for them.

The example below shows ONLY the format and tone - DO NOT reuse its wording; \
write original lines tied to the day's actual stories:

[
  {{"speaker": "male_host", "preset": "intro", "line": "Good morning, Night City!"}},
  {{"speaker": "female_host", "line": "Emphasis on the 'good', choom. You're locked into NC-One, the only frequency Arasaka hasn't bought... yet."}},
  {{"speaker": "male_host", "preset": "hype", "line": "We have got a NOVA show for you today, choom - chrome, crime, and corporate carnage, so let's RIP into it!"}},
  {{"speaker": "female_host", "line": "Easy, tiger. First up, the crime blotter..."}},
  {{"speaker": "male_host", "line": "Arasaka slipped two percent on the exchange overnight."}},
  {{"speaker": "female_host", "line": "And remember, today's broadcast is brought to you by Zetatech. Zetatech: we already know what you were going to say."}}
]
"""


BABYLON_SYSTEM = """\
You are the head writer for a satirical morning radio news show. Two hosts read
and riff on the day's real headlines from The Babylon Bee (a satire
publication). The comedy comes from the hosts presenting these absurd satirical
stories as if they were straight, serious breaking news - deadpan delivery,
mock-serious analysis, incredulous reactions, and dry banter. This is NOT
science fiction - it's a normal, present-day, real-world morning-news desk
broadcasting to America, reacting to the day's ridiculous headlines.

HARD RULE: never say "Night City", and never invent any cyberpunk, sci-fi, or
futuristic setting, sponsor, slang, or characters (no "choom", "chrome",
megacorps, etc.). The setting is the real present-day world.

There are exactly TWO hosts:
- male_host: the smooth, upbeat lead anchor. Warm, confident, "and now this"
energy. Sells every story like it's the scoop of the century.
- female_host: the sharper, drier co-anchor. Skeptical, quick with a one-liner,
occasionally lets real exasperation slip through.

STYLE RULES:
- Hosts banter naturally and hand off to each other. No monologue longer than
3-4 sentences.
- Cover the ACTUAL Babylon Bee stories provided: read the headline, give a
sentence or two of mock "coverage," and react. Keep each story's real premise
intact. Do NOT invent unrelated stories, and do NOT relocate them to a fictional
city or the future - report them as today's news.
- Lean into the satire: treat the ridiculous as routine, add straight-faced
analysis, fake expert/eyewitness quotes, and absurd follow-up details.
- Keep it punchy and broadcast-paced. Sharp but PG; clever over crude.
- Brief mock sponsor reads are welcome for flavor ("This segment brought to you
by...").
- If a real commute-weather report is provided, include a short, GENUINE
weather/commute segment - the morning drive and the evening drive home. Be wry
about the weather if you like, but report the ACTUAL numbers and conditions
accurately; never fabricate or satirize the forecast itself.
- Total spoken content should run roughly 4-6 minutes (about 700-950 words).
- ALWAYS open with the signature greeting and ALWAYS close with a sign-off.

CONTENT STANDARDS (strict - non-negotiable):
- Broadcast-clean. NO profanity or crude language (no "shit", "fuck", "ass",
"bitch", "bastard", "damn", "piss", or "hell" as an expletive). Use mild, witty
phrasing instead.
- NO slurs of any kind, ever.
- Do NOT use God's or Jesus's name in vain, and never use religious exclamations
as expletives (no "oh my god", "jesus", "jesus christ", "god damn", "for god's
sake"). Use clean exclamations instead: "oh my goodness", "good grief", "wow",
"unbelievable", "for crying out loud".
- Humor comes from wit, irony, and absurdity - never shock words. Clever over
crude.

BROADCAST STRUCTURE:
1. COLD OPEN: the male host's signature greeting, then a FRESH, punchy
back-and-forth that teases the single most outrageous story of the day BY NAME.
Write this open originally every time - the co-host's first line and the male
host's high-energy lead-in should both hook on TODAY'S actual headlines, not
generic "it's early and the news is dumb" / "we've got a big show" filler.
2. TOP STORY (the most airtime): cover the funniest / most absurd headline in
depth - real back-and-forth riffing and straight-faced "analysis."
3. THE REST: move through the other stories, roughly 1-3 turns each, brisk
hand-offs.
4. WIND-DOWN: a quick "and finally..." lighter item and/or a mock sponsor read.
5. SIGN-OFF: the female host's dry closing line.

PRESET RULES (these control the male host's delivery voice - follow exactly):
- The VERY FIRST line must be the male host's signature greeting, written
EXACTLY as "Good morning, America!" and tagged "preset": "intro".
- The next ONE OR TWO male_host lines - the high-energy "here's what we've got"
lead-in, before the first real story - may be tagged "preset": "hype".
- Tag NOTHING else. Once the stories start, OMIT "preset" so those lines use the
hosts' normal voices. Never tag a sponsor or closing line.

OUTPUT FORMAT:
Return ONLY a JSON array (no markdown fences, no commentary) of objects with
"speaker" (either "male_host" or "female_host") and "line" (the spoken text),
plus an optional "preset". The example below shows ONLY the format and tone -
DO NOT reuse its wording; write original lines tied to the real stories:

[
  {"speaker": "male_host", "preset": "intro", "line": "Good morning, America!"},
  {"speaker": "female_host", "line": "The coffee's hot and the news has somehow already gotten dumber. Thrilled to be here."},
  {"speaker": "male_host", "preset": "hype", "line": "We have got a PACKED show for you this morning, so grab your coffee and let's get into it!"},
  {"speaker": "female_host", "line": "Our top story: the United States has reportedly voted to legally adopt a World Cup tourist."},
  {"speaker": "male_host", "line": "That's right - 'Freddy the German' is now, legally, everyone's son. Sources say the adoption papers were signed in beer."},
  {"speaker": "female_host", "line": "A nation of three hundred million suddenly has one very confused dad. Beautiful."}
]
"""


def _air_time_phrase(config: dict[str, Any]) -> str:
    """Human air time (e.g. '8:40 AM') from [schedule].playback_time, or ''."""
    raw = str(config.get("schedule", {}).get("playback_time", "")).strip()
    if not raw:
        return ""
    try:
        hh, mm = raw.split(":")[:2]
        h, m = int(hh), int(mm)
    except (ValueError, TypeError):
        return ""
    return f"{h % 12 or 12}:{m:02d} {'AM' if h < 12 else 'PM'}"


def _time_block(phrase: str) -> str:
    """Instruction so any on-air time reference matches when the show actually airs."""
    if not phrase:
        return ""
    return (
        f"AIR TIME: this broadcast airs in the morning at approximately {phrase}. "
        "If a host refers to the time of day, use this time; otherwise just greet "
        "the morning naturally. Do NOT state any other clock time (for example, "
        "never say 'six AM').\n\n"
    )


def _weather_block(weather: dict[str, Any] | None) -> str:
    """Instruction + real data for an accurate commute-weather segment."""
    if not weather:
        return ""
    payload = {"morning": weather.get("morning"), "evening": weather.get("evening")}
    return (
        f"REAL COMMUTE WEATHER for {weather.get('location', 'the area')} - include "
        "a short, ACCURATE weather/commute segment using EXACTLY these numbers "
        "(the morning drive and the evening drive home). Be wry if you like, but "
        "do NOT change the temperatures, conditions, or chances of rain:\n"
        f"{json.dumps(payload, indent=2)}\n\n"
    )


def _build_simulation_user_prompt(
    stats: dict[str, Any],
    city_cfg: dict[str, Any],
    weather: dict[str, Any] | None = None,
    air_phrase: str = "",
) -> str:
    factions = ", ".join(city_cfg.get("factions", []))
    # Split the stats by newsworthiness so the model knows what to lead with
    # (sensational) and what to bury in the bored tail (mundane).
    sensational = {
        "crime": stats.get("crime"),
        "civic": stats.get("civic"),
        "corpo_war_shock": stats.get("economy", {}).get("corpo_war_shock"),
    }
    mundane = {
        "stocks": stats.get("stocks"),
        "economy": stats.get("economy"),
    }
    # Real commute weather (if provided) replaces the simulated weather so the
    # forecast the host reads is genuinely useful.
    midblock = {"traffic": stats.get("traffic")}
    if not weather:
        midblock["weather"] = stats.get("weather")
    return (
        f"{_time_block(air_phrase)}"
        "Today's content mode: SIMULATION.\n"
        f"City: {city_cfg.get('name', 'Night City')} | "
        f"Date: {stats.get('date')} | Active factions: {factions}\n\n"
        "Build the broadcast on the mandatory running order from the system "
        "prompt. The data below is grouped to match that arc - keep all numbers "
        "accurate, but invent colorful detail freely.\n\n"
        "=== LEAD WITH THIS (sensational block - the most airtime, breathless "
        "and gleeful) ===\n"
        f"{json.dumps(sensational, indent=2)}\n\n"
        "=== WIND-DOWN FILLER (mid-broadcast, energy draining) ===\n"
        f"{json.dumps(midblock, indent=2)}\n\n"
        "=== THE BORING TAIL (read these last, bored and perfunctory - the "
        "stocks and economy are the tedious payoff after the carnage) ===\n"
        f"{json.dumps(mundane, indent=2)}\n\n"
        f"{_weather_block(weather)}"
        "Weave it all into natural host banter - never a flat list. Then write "
        "the full broadcast script as the JSON array described in the system "
        "prompt (remember: first line is the male host's 'preset': 'intro')."
    )


def _sponsored_block(sponsored: dict[str, Any] | None) -> str:
    """Instruction + data to turn a real Bee 'Sponsored' story into the sponsor read."""
    if not sponsored:
        return ""
    payload = {"title": sponsored.get("title", ""), "summary": sponsored.get("summary", "")}
    return (
        "REAL SPONSORED FEATURE (from The Babylon Bee's Sponsored section, posted "
        "yesterday) - use THIS as the show's sponsor segment INSTEAD of inventing a "
        "mock sponsor. In the wind-down, introduce it with a clear sponsor cue (for "
        "example 'And now, a word from our sponsor...' or 'Today's broadcast is "
        "brought to you by...'), then read the real title and give a straight, "
        "faithful summary of what the piece is, and end on one dry line of host "
        "reaction. Keep the actual title and premise intact; do not satirize it "
        "into something else:\n"
        f"{json.dumps(payload, indent=2)}\n\n"
    )


def _sponsor_products(n: int = 6) -> list[dict[str, Any]]:
    """A random sample of real brands for the fallback sponsor read ([] on failure)."""
    try:
        from content.product_catalog import sample_products

        return sample_products(n)
    except Exception as exc:
        log.warning("Could not load product catalog (%s); sponsor will improvise.", exc)
        return []


def _default_sponsor_block(products: list[dict[str, Any]] | None = None) -> str:
    """YouTuber-style mock sponsor read, used when no real Bee sponsor ran.

    If ``products`` (real brand facts from the catalog) are supplied, the model
    must pick ONE and use its REAL slogan and founding year, so the read stays
    factually accurate; otherwise it improvises a well-known brand.
    """
    intro = (
        "SPONSOR READ (no real sponsor today): near the end, do a short, "
        "YouTuber-style sponsor read - a tight TWO-LINER. First line: a quick "
        "sponsor cue plus the brand and its REAL slogan, '...since <founding "
        "year>'. Second line: a deadpan, satirical jab grounded in what the "
        "product ACTUALLY does (the co-host can deliver this second line). "
        "Example of the SHAPE (do not reuse it):\n"
        "  male_host: \"And today's broadcast is brought to you by Red Bull - "
        "giving you wings since 1987.\"\n"
        "  female_host: \"Because a lower risk of heart attack just isn't as "
        "thrilling.\"\n"
        "Keep it to two lines, PG and clever, then go to the sign-off.\n"
    )
    if products:
        return (
            intro
            + "Pick exactly ONE of these real brands and use its real slogan and "
            "founding year EXACTLY as given (do not invent or change these facts); "
            "the satirical line should riff on its real function ('does'):\n"
            f"{json.dumps(products, ensure_ascii=False, indent=2)}\n\n"
        )
    return intro + "Use a well-known real brand and its real slogan/founding year.\n\n"


def _build_babylon_user_prompt(
    articles: list[dict[str, str]],
    city_cfg: dict[str, Any],
    weather: dict[str, Any] | None = None,
    sponsored: dict[str, Any] | None = None,
    air_phrase: str = "",
) -> str:
    trimmed = [
        {"title": a.get("title", ""), "summary": a.get("summary", "")}
        for a in articles
    ]
    return (
        f"{_time_block(air_phrase)}"
        "Here are today's real Babylon Bee stories (headline + summary). Build "
        "the broadcast by READING and riffing on these ACTUAL stories - present "
        "them as today's news, react to the absurdity, and add straight-faced "
        "mock analysis, fake quotes, and absurd follow-ups. Keep each story's "
        "real premise and key details intact. Do NOT invent unrelated stories, "
        "and do NOT move them to a fictional city or the future.\n\n"
        "Lead with the single funniest / most outrageous headline and give it "
        "the most airtime, then move briskly through the rest. Cover about 6-9 "
        "of them; you don't have to use all.\n\n"
        f"{json.dumps(trimmed, indent=2)}\n\n"
        f"{_weather_block(weather)}"
        f"{_sponsored_block(sponsored) if sponsored else _default_sponsor_block(_sponsor_products())}"
        "Now write the full broadcast script as the JSON array described in the "
        "system prompt."
    )


# --- Content scrubber (broadcast-clean safety net) -------------------------
# A deterministic last line of defense: even when the local model ignores the
# "keep it clean" instruction, every line passes through here before it is saved
# or synthesized. Profanity/blasphemy become clean substitutes; slurs are
# removed outright. Word-boundaried and case-insensitive; the replacement's
# capitalization follows the original match.
#
# NOTE: bare "God"/"Christ" in a SINCERE reference is left alone (the Babylon
# Bee is a Christian publication) - only vain/expletive PHRASES are rewritten.
_CLEAN_SUBS: list[tuple[re.Pattern[str], str]] = [
    # Vain / blasphemous exclamations -> clean exclamations.
    (re.compile(r"\bjesus(?:\s+h\.?)?\s+christ\b", re.I), "good grief"),
    (re.compile(r"\bfor\s+(?:god|christ)['’]?s\s+sake\b", re.I), "for goodness' sake"),
    (re.compile(r"\bswear\s+to\s+god\b", re.I), "swear to goodness"),
    (re.compile(r"\b(?:oh\s+my\s+god|omg)\b", re.I), "oh my goodness"),
    (re.compile(r"\b(?:oh|good|dear|my)\s+god\b", re.I), "good grief"),
    (re.compile(r"\bgod\s*-?\s*damn(?:it|ed)?\b", re.I), "darn"),
    (re.compile(r"\bjesus\b", re.I), "geez"),
    # Profanity -> minced oaths.
    (re.compile(r"\bmother\s*-?\s*fuck\w*\b", re.I), "son of a gun"),
    (re.compile(r"\bfucking\b", re.I), "freaking"),
    (re.compile(r"\bfucked\b", re.I), "screwed"),
    (re.compile(r"\bfuck(?:er|ers|wit|wits)\b", re.I), "jerk"),
    (re.compile(r"\bfuck(?:s|ed)?\b", re.I), "heck"),
    (re.compile(r"\bbull\s*shit\b", re.I), "nonsense"),
    (re.compile(r"\bshitty\b", re.I), "lousy"),
    (re.compile(r"\bshit(?:s|ting|ted|head|heads)?\b", re.I), "crud"),
    (re.compile(r"\bdammit\b", re.I), "darn it"),
    (re.compile(r"\bdamn(?:ed|ing|it)?\b", re.I), "darn"),
    (re.compile(r"\basshole(?:s)?\b", re.I), "idiot"),  # vowel-initial keeps "an" correct
    (re.compile(r"\b(?:dumb|jack|smart)\s*ass(?:es)?\b", re.I), "jerk"),
    (re.compile(r"\bbitch(?:es|ing|y)?\b", re.I), "jerk"),
    (re.compile(r"\bbastard(?:s)?\b", re.I), "scoundrel"),
    (re.compile(r"\bpissed\b", re.I), "ticked"),
    (re.compile(r"\bwhat\s+the\s+hell\b", re.I), "what on earth"),
    (re.compile(r"\bthe\s+hell\b", re.I), "the heck"),
]

# Slurs: removed entirely. Kept tight and unambiguous to avoid false positives
# (the "nigg(a|er)" branch deliberately excludes "niggardly"; "coon" excludes
# "raccoon"/"tycoon" via the word boundary).
_SLUR_RE = re.compile(
    r"\b(?:n[i1]gg(?:er|ers|a|as|ah)|faggot|kike|spic|chink|gook|wetback)\b",
    re.I,
)


def _match_case(replacement: str, original: str) -> str:
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def sanitize_text(text: str) -> str:
    """Rewrite profanity/blasphemy and strip slurs so the line is broadcast-clean."""
    if not text:
        return text

    def _repl(replacement: str):
        return lambda m: _match_case(replacement, m.group(0))

    for rgx, replacement in _CLEAN_SUBS:
        text = rgx.sub(_repl(replacement), text)
    text = _SLUR_RE.sub("", text)
    # Tidy whitespace/punctuation left behind by removals.
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


# --- Response parsing ------------------------------------------------------

def _extract_json_array(text: str) -> list[dict[str, str]]:
    """Pull a JSON array of dialogue turns out of an LLM response.

    Tolerates code fences and leading/trailing prose. Raises ValueError if no
    valid, schema-conformant array is found.
    """
    cleaned = text.strip()
    # Strip markdown fences if present.
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)
        # ["", "json\n[...]", ""] or similar
        cleaned = max(cleaned, key=len)
        if cleaned.lstrip().lower().startswith("json"):
            cleaned = cleaned.lstrip()[4:]

    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON array found in LLM response.")

    candidate = cleaned[start : end + 1]
    data = json.loads(candidate)
    return _validate_turns(data)


def _validate_turns(data: Any) -> list[dict[str, str]]:
    """Validate and normalize a parsed dialogue array."""
    if not isinstance(data, list) or not data:
        raise ValueError("LLM response was not a non-empty JSON array.")
    turns: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        speaker = str(item.get("speaker", "")).strip()
        line = sanitize_text(str(item.get("line", "")).strip())
        if speaker not in VALID_SPEAKERS or not line:
            continue
        turn: dict[str, str] = {"speaker": speaker, "line": line}
        # Carry an optional delivery preset through (e.g. the male "intro").
        preset = str(item.get("preset", "")).strip()
        if preset:
            turn["preset"] = preset
        turns.append(turn)
    if not turns:
        raise ValueError("No valid dialogue turns after validation.")
    return turns


# --- LLM provider calls ----------------------------------------------------

def _call_anthropic(system: str, user: str, llm_cfg: dict[str, Any], fmt: Any = None) -> str:
    # ``fmt`` (a JSON schema) is honored by the Ollama path; Anthropic relies on
    # the prompt + tolerant parsing, so it's accepted and ignored here.
    import anthropic

    api_key = llm_cfg.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No Anthropic API key. Set [llm].api_key or the ANTHROPIC_API_KEY "
            "environment variable."
        )
    client = anthropic.Anthropic(api_key=api_key)
    model = llm_cfg.get("model", "claude-opus-4-8")
    kwargs: dict[str, Any] = {
        "model": model,
        # Generous ceiling: the script plus adaptive-thinking tokens. Stays well
        # under the SDK's non-streaming timeout guard.
        "max_tokens": 8000,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    # Adaptive thinking improves the structured, tone-modulated writing on the
    # Opus 4.x family. Harmless to request; only send it for those models so an
    # older configured model doesn't 400.
    if model.startswith(("claude-opus-4", "claude-fable", "claude-sonnet-4-6")):
        kwargs["thinking"] = {"type": "adaptive"}
    resp = client.messages.create(**kwargs)
    # Concatenate text blocks only (skips any thinking blocks).
    return "".join(
        block.text for block in resp.content if getattr(block, "type", None) == "text"
    )


def _call_ollama(system: str, user: str, llm_cfg: dict[str, Any], fmt: Any = None) -> str:
    import httpx

    host = llm_cfg.get("ollama_host", "http://localhost:11434").rstrip("/")
    model = llm_cfg.get("model", "qwen2.5:7b-instruct")
    payload = {
        "model": model,
        "system": system,
        "prompt": user,
        "stream": False,
        # Unload the model from RAM right after generating (frees its RAM before
        # the Chatterbox synthesis stage runs). 0 = unload immediately; set a
        # duration like "5m" via [llm].ollama_keep_alive to keep it warm instead.
        "keep_alive": llm_cfg.get("ollama_keep_alive", 0),
        # Structured output: constrain to an exact SHAPE via a JSON schema (not
        # just "valid JSON"). Defaults to the dialogue-array schema; callers pass
        # JUDGE_SCHEMA etc. for other steps. The grammar forces correct structure.
        "format": fmt if fmt is not None else TURNS_SCHEMA,
        "options": {
            "temperature": float(llm_cfg.get("ollama_temperature", 0.9)),
            "num_ctx": int(llm_cfg.get("ollama_num_ctx", 8192)),
            "num_predict": int(llm_cfg.get("ollama_num_predict", 3500)),
        },
    }
    timeout = float(llm_cfg.get("ollama_timeout", 1200.0))
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(f"{host}/api/generate", json=payload)
        resp.raise_for_status()
        return resp.json().get("response", "")


def _call_llm(system: str, user: str, llm_cfg: dict[str, Any], fmt: Any = None) -> str:
    provider = llm_cfg.get("provider", "anthropic").lower()
    if provider == "anthropic":
        return _call_anthropic(system, user, llm_cfg, fmt)
    if provider == "ollama":
        return _call_ollama(system, user, llm_cfg, fmt)
    raise ValueError(f"Unknown LLM provider: {provider!r}")


# --- Fallback --------------------------------------------------------------

def load_fallback_script() -> list[dict[str, str]]:
    """Load the pre-written emergency broadcast script from disk."""
    try:
        with FALLBACK_SCRIPT.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return _validate_turns(data)
    except Exception as exc:  # last-ditch; never let this crash the pipeline
        log.error("Fallback script unavailable (%s); using minimal stub.", exc)
        return [
            {
                "speaker": "male_host",
                "line": "Good morning, Night City. Technical difficulties in the "
                "booth this morning, choom - the chrome's acting up.",
            },
            {
                "speaker": "female_host",
                "line": "Translation: everything's broken and nobody's getting "
                "paid to fix it. Stay preem out there. We'll be back tomorrow.",
            },
        ]


# --- Quality pipeline: best-of-N + iterative refinement -------------------

def _build_refine_prompt(draft: list[dict[str, str]]) -> str:
    # Mode-NEUTRAL: the mode-specific theme/greeting/running-order rules live in
    # the system prompt (already in context). Do NOT restate them here, or this
    # prompt would override the active mode (e.g. re-injecting Cyberpunk into a
    # Babylon broadcast).
    return (
        "Below is a DRAFT of the broadcast, as a JSON array of dialogue turns. "
        "Revise it into a sharper, funnier version while STRICTLY following the "
        "rules in the system prompt above - keep the same hosts, setting, theme, "
        "tone, running order, opening greeting, preset tags, and output schema. "
        "Do NOT change the setting or theme. Keep every real fact and number "
        "that already appears accurate (headlines, the weather forecast). Improve "
        "weak or generic lines, cut filler, tighten the hand-offs, and sharpen "
        "the comedy. Return ONLY the improved JSON array in the same schema.\n\n"
        "DRAFT (Speaker: line):\n"
        f"{_render_compact(draft)}"
    )


def _render_compact(turns: list[dict[str, str]]) -> str:
    """Plain 'Speaker: line' rendering - compact input for the judge."""
    return "\n".join(
        f"{t['speaker'].replace('_', ' ').title()}: {t['line']}" for t in turns
    )


def _build_judge_prompt(candidates: list[list[dict[str, str]]]) -> str:
    blocks = []
    for i, cand in enumerate(candidates):
        blocks.append(f"--- CANDIDATE {i} ---\n{_render_compact(cand)}")
    body = "\n\n".join(blocks)
    return (
        "You are the executive producer choosing which take of the morning "
        "broadcast to air. Judge on: faithfulness to the system prompt's format, "
        "theme, and running order; sharpest banter and comedic timing; best flow "
        "and hand-offs; accurate use of any real facts; COMPLETE coverage that "
        "ends with a proper sign-off (and includes the weather/commute segment if "
        "one was provided); and no broken or empty lines. Reject takes that trail "
        "off or feel cut short.\n\n"
        f"{body}\n\n"
        "Return the index (0-based) of the single best candidate as JSON: "
        '{"best": <index>}.'
    )


def _generate_turns(system: str, user: str, llm_cfg: dict[str, Any]) -> list[dict[str, str]]:
    """One validated generation with retry. Returns turns or raises."""
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return _extract_json_array(_call_llm(system, user, llm_cfg, TURNS_SCHEMA))
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
    assert last_error is not None
    raise last_error


def _refine(system: str, turns: list[dict[str, str]], llm_cfg: dict[str, Any], passes: int) -> list[dict[str, str]]:
    """Run critique-and-rewrite passes; keep the last good draft on failure."""
    for i in range(passes):
        try:
            improved = _generate_turns(system, _build_refine_prompt(turns), llm_cfg)
            turns = improved
            log.info("Refine pass %d/%d complete (%d turns).", i + 1, passes, len(turns))
        except Exception as exc:
            log.warning("Refine pass %d failed (%s); keeping previous draft.", i + 1, exc)
            break
    return turns


def _judge(system: str, candidates: list[list[dict[str, str]]], llm_cfg: dict[str, Any]) -> list[dict[str, str]]:
    """Pick the best candidate via the LLM; fall back to the first on any error."""
    if len(candidates) == 1:
        return candidates[0]
    try:
        raw = _call_llm(system, _build_judge_prompt(candidates), llm_cfg, JUDGE_SCHEMA)
        # Parse {"best": N}; tolerate stray text by grabbing the first integer.
        try:
            idx = int(json.loads(raw[raw.find("{"): raw.rfind("}") + 1])["best"])
        except Exception:
            import re
            m = re.search(r"-?\d+", raw)
            idx = int(m.group()) if m else 0
        if not 0 <= idx < len(candidates):
            idx = 0
        log.info("Judge selected candidate %d of %d.", idx, len(candidates))
        return candidates[idx]
    except Exception as exc:
        log.warning("Judge step failed (%s); using the first candidate.", exc)
        return candidates[0]


# --- Public entry point ----------------------------------------------------

def generate_script(
    content: dict[str, Any] | list[dict[str, str]],
    mode: str,
    config: dict[str, Any],
    weather: dict[str, Any] | None = None,
    sponsored: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Generate the broadcast script.

    ``content`` is the simulated stats dict (mode="simulation") or the list of
    article dicts (mode="babylon"). Returns a validated list of dialogue turns,
    falling back to the emergency script on repeated failure.

    Quality scales with two knobs (trading time for quality, ideal for an
    overnight job): ``[llm].candidates`` generates N independent drafts and an
    LLM judge picks the best; ``[llm].refine_passes`` runs critique-and-rewrite
    passes on each draft before judging.
    """
    llm_cfg = config.get("llm", {})
    city_cfg = config.get("city", {})

    air_phrase = _air_time_phrase(config)
    if mode == "babylon":
        # Babylon mode reads the real Bee stories - no Cyberpunk theming.
        system = BABYLON_SYSTEM
        user = _build_babylon_user_prompt(content, city_cfg, weather, sponsored, air_phrase)  # type: ignore[arg-type]
    else:
        system = SIMULATION_SYSTEM.format(year=city_cfg.get("year", 2077))
        user = _build_simulation_user_prompt(content, city_cfg, weather, air_phrase)  # type: ignore[arg-type]

    candidates_n = max(1, int(llm_cfg.get("candidates", 1)))
    refine_passes = max(0, int(llm_cfg.get("refine_passes", 0)))

    candidates: list[list[dict[str, str]]] = []
    for c in range(candidates_n):
        try:
            draft = _generate_turns(system, user, llm_cfg)
            draft = _refine(system, draft, llm_cfg, refine_passes)
            candidates.append(draft)
            log.info("Candidate %d/%d ready (%d turns).", c + 1, candidates_n, len(draft))
        except Exception as exc:
            log.error("Candidate %d/%d failed (%s).", c + 1, candidates_n, exc)

    if not candidates:
        log.error("All %d candidate(s) failed. Using fallback script.", candidates_n)
        return load_fallback_script()

    best = _judge(system, candidates, llm_cfg)

    # Force the opening greeting to a vetted phrase (random pick per run for
    # variety) so model drift can't change the setting (e.g. a Babylon take
    # wandering back to "Good morning, Night City!").
    greeting = random.choice(GREETINGS.get(mode, GREETINGS["simulation"]))
    if best and best[0].get("speaker") == "male_host":
        best[0]["line"] = greeting
        best[0]["preset"] = "intro"
        log.info("Opening greeting set -> %r (%s mode).", greeting, mode)

    _ensure_hype_leadin(best)
    _ensure_signoff(best, mode)

    log.info("Final script: %d dialogue turns.", len(best))
    return best


# Lead-in lines the high-energy "hype" voice belongs on (the welcome-to-the-show
# beat, before the first real story). Used to recover a dropped "hype" tag.
_HYPE_CUES = (
    "packed", "big show", "huge show", "stacked", "loaded show", "great show",
    "big morning", "wild morning", "wild one", "quite a", "what a",
    "let's get into", "let's dive", "let's jump", "let's go", "grab your",
    "buckle up", "strap in", "hold on", "get ready", "here we go",
    "we've got", "we have got", "welcome", "coming up", "here's what we",
    "kicking off", "kick things off", "wait until you hear", "wait till you hear",
    "you won't believe", "you are not going to believe",
)


def _ensure_hype_leadin(script: list[dict[str, str]]) -> None:
    """Tag the opening lead-in male line 'hype' if the model dropped the tag.

    Mirrors the forced intro greeting: weak local models often omit the preset
    the prompt asks for. We only touch the FIRST male_host line after the opener,
    and only if it reads like a welcome lead-in (not already a news story), so the
    loud voice never lands on actual reporting.
    """
    for turn in script[1:]:
        if turn.get("speaker") != "male_host":
            continue
        if turn.get("preset"):  # model already tagged it (intro/hype) - respect it
            return
        if any(cue in turn.get("line", "").lower() for cue in _HYPE_CUES):
            turn["preset"] = "hype"
            log.info("Applied 'hype' preset to the opening lead-in line.")
        return  # only the first male line after the opener is eligible


# Phrases that signal the broadcast has actually closed out (used to detect a
# missing sign-off so we can append one).
_SIGNOFF_CUES = (
    "see you tomorrow", "back tomorrow", "catch you tomorrow", "until tomorrow",
    "until next time", "that's all for", "that's it for", "that does it for",
    "that wraps", "that's a wrap", "thanks for joining", "thank you for joining",
    "thanks for listening", "thank you for listening", "thanks for waking up",
    "thank you for sharing", "stay tuned", "have a great day", "good day to you",
    "stay safe out there", "stay preem",
)
# Clean fallback sign-offs, by mode, appended (female host) when the model's
# script just trails off without closing - so the broadcast never ends mid-thought.
_SIGNOFFS = {
    "babylon": [
        "And that's the morning. Thanks for waking up with us - stay sane out "
        "there, and we'll see you bright and early tomorrow.",
        "That's all for today's broadcast. Thanks for listening, take care of "
        "each other, and we'll be right back here tomorrow morning.",
        "And that wraps up the news. Thanks for starting your day with us - "
        "we'll catch you tomorrow.",
    ],
    "simulation": [
        "And that's the morning feed, choom. Stay preem out there, watch your "
        "back, and we'll see you tomorrow.",
        "That's all Night City's cleared for broadcast today. Stay frosty, and "
        "we'll catch you tomorrow morning.",
    ],
}


def _ensure_signoff(script: list[dict[str, str]], mode: str) -> None:
    """Append a clean sign-off if the script trails off without one.

    Weak local models (and the judge picking a short candidate) sometimes drop
    the closing line, leaving the broadcast feeling cut short. If the final turn
    doesn't read like a sign-off, we add a mode-appropriate one (female host,
    normal voice).
    """
    if not script:
        return
    if any(cue in script[-1].get("line", "").lower() for cue in _SIGNOFF_CUES):
        return  # already closes cleanly
    pool = _SIGNOFFS.get(mode, _SIGNOFFS["babylon"])
    script.append({"speaker": "female_host", "line": random.choice(pool)})
    log.info("Appended a sign-off (script ended without one).")


if __name__ == "__main__":
    # Quick smoke test using simulated data and whatever provider is configured.
    from config_loader import load_config
    from city_simulation import generate_city_stats

    cfg = load_config()
    sample_stats = generate_city_stats()
    script = generate_script(sample_stats, mode="simulation", config=cfg)
    print(json.dumps(script, indent=2))
