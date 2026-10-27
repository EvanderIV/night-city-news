"""Quick standalone test of the cloned voices (no LLM / no API cost).

Synthesizes a couple of Night City lines through the XTTS-cloned hosts and
reports the output segment paths + durations.
"""
from config_loader import load_config
from voice.synthesizer import synthesize_script

DEMO = [
    {"speaker": "male_host", "line": "Good morning, Night City. You're locked into the Morning Broadcast, brought to you by Arasaka. Everything is fine, citizen."},
    {"speaker": "female_host", "line": "Fine is doing a lot of heavy lifting this morning, choom. Let's get into the chrome, the crime, and the corporate weather."},
]

if __name__ == "__main__":
    cfg = load_config()
    print("engine:", cfg["tts"]["engine"])
    paths = synthesize_script(DEMO, cfg)
    from pydub import AudioSegment
    for p in paths:
        seg = AudioSegment.from_file(p)
        print(f"  {p.name}: {len(seg)/1000:.1f}s")
    print("DONE")
