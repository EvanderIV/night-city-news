"""Out-of-process Chatterbox synthesis helper.

Executed by the ISOLATED ``.venv-chatterbox`` interpreter (whose torch /
transformers pins conflict with the main XTTS venv), NOT the main project venv.
The main synthesizer shells out to it.

Reads a JSON job spec from stdin and renders every line. Two execution modes,
selected by ``"mode"``:

  "full"   - load the whole model once (ChatterboxTTS.from_local) and call
             generate() per line. Simplest; both stages resident (~3.2 GB).

  "staged" - run as three phases, loading only what each needs and freeing it
             before the next, so peak weight RAM is the LARGEST single stage
             instead of the sum:
               A. conditioning  -> needs ve + s3gen        (~1.06 GB)
               B. t3 inference   -> needs t3                (~2.1 GB, or less
                                                             with quantize_t3)
               C. s3gen vocoding -> needs s3gen             (~1.06 GB)
             Optional ``"quantize_t3": true`` applies int8 dynamic quantization
             to T3's Linear layers (smaller + faster on CPU, small quality risk).

Spec::

    {
      "model_dir": "models/chatterbox",
      "device": "cpu",
      "mode": "staged",
      "quantize_t3": false,
      "gen_kwargs": {"exaggeration":0.5,"cfg_weight":0.5,"temperature":0.8,
                     "repetition_penalty":1.2,"min_p":0.05,"top_p":1.0},
      "jobs": [{"text":"...","ref":"abs/ref.wav","out":"abs/out.wav"}, ...]
    }

Per-job result line: ``OK <out>`` or ``ERR <out> <msg>``. Final line:
``PEAK_RSS_MB <float>`` (process peak working set, Windows).
"""
from __future__ import annotations

import gc
import json
import sys
import traceback
from pathlib import Path

import psutil

_PROC = psutil.Process()


def _mem() -> tuple[float, float]:
    """Return (current_working_set_MB, peak_working_set_MB)."""
    mi = _PROC.memory_info()
    peak = getattr(mi, "peak_wset", mi.rss)
    return mi.rss / 1e6, peak / 1e6


def _mark(label: str) -> None:
    cur, peak = _mem()
    print(f"MEM {label}: now={cur:.0f}MB peak={peak:.0f}MB", flush=True)


# --- Shared setup ----------------------------------------------------------

def _setup_threads() -> None:
    import os
    import torch
    try:
        torch.set_num_threads(os.cpu_count() or 4)
    except Exception:
        pass


def _resolve_device(requested: str | None) -> str:
    """Resolve a requested device ('auto'/'cuda'/'mps'/'cpu') to a usable one.

    'auto' prefers CUDA, then Apple MPS, else CPU. An explicit 'cuda'/'mps' that
    isn't actually available also falls back to CPU (so the same config works on
    the GPU dev box and the CPU-only server).
    """
    import torch

    req = (requested or "auto").lower()
    cuda = torch.cuda.is_available()
    mps = bool(getattr(getattr(torch, "backends", None), "mps", None)) and torch.backends.mps.is_available()
    if req in ("cuda", "gpu"):
        return "cuda" if cuda else "cpu"
    if req == "mps":
        return "mps" if mps else "cpu"
    if req == "auto":
        return "cuda" if cuda else ("mps" if mps else "cpu")
    return req  # explicit "cpu" (or anything else, passed through)


def _free(*objs) -> None:
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()


# --- Component loaders (staged mode) ---------------------------------------

def _load_ve(model_dir: Path, device: str):
    from chatterbox.tts import VoiceEncoder
    from safetensors.torch import load_file

    ve = VoiceEncoder()
    ve.load_state_dict(load_file(model_dir / "ve.safetensors"))
    ve.to(device).eval()
    return ve


def _load_s3gen(model_dir: Path, device: str):
    from chatterbox.tts import S3Gen
    from safetensors.torch import load_file

    s3gen = S3Gen()
    s3gen.load_state_dict(load_file(model_dir / "s3gen.safetensors"), strict=False)
    s3gen.to(device).eval()
    return s3gen


def _load_t3(model_dir: Path, device: str, quantize: bool):
    """Load T3 with a near-zero memory spike.

    Constructing on the ``meta`` device allocates no real storage, then
    ``assign=True`` lets safetensors tensors back the params directly instead of
    being copied into pre-allocated weights -- avoiding the ~3x transient spike
    of the naive ``T3(); load_state_dict()`` path. The only catch: meta leaves
    the two non-persistent rotary-embedding buffers uninitialized, so we
    recompute them.
    """
    import torch
    from chatterbox.tts import T3
    from safetensors.torch import load_file
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    with torch.device("meta"):
        t3 = T3()
    state = load_file(model_dir / "t3_cfg.safetensors")
    if "model" in state.keys():
        state = state["model"][0]
    t3.load_state_dict(state, assign=True)
    del state

    # Recompute the rotary buffers meta left empty (non-persistent, not in the
    # checkpoint). rope_type is "llama3" for this model.
    rot = t3.tfmr.rotary_emb
    inv_freq, scaling = ROPE_INIT_FUNCTIONS[rot.rope_type](rot.config, device=torch.device(device))
    rot.register_buffer("inv_freq", inv_freq, persistent=False)
    rot.original_inv_freq = inv_freq
    rot.attention_scaling = scaling

    if device != "cpu":
        t3.to(device)
    t3.eval()

    if quantize:
        from torch.ao.quantization import quantize_dynamic
        # Backbone only (~95% of weights); heads/embeddings stay fp32 to protect
        # quality and keep T3.device (reads speech_head.weight.device) working.
        # inplace=True avoids quantize_dynamic's default deep-copy (another spike).
        t3.tfmr = quantize_dynamic(t3.tfmr, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
        t3.cond_enc = quantize_dynamic(t3.cond_enc, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
        print("QUANTIZED t3.tfmr + t3.cond_enc (int8 dynamic, inplace)", flush=True)
    return t3


# --- Staged pipeline -------------------------------------------------------

def run_staged(spec: dict) -> int:
    import torch
    import torch.nn.functional as F
    import librosa
    import numpy as np
    import soundfile as sf
    import perth
    from chatterbox.tts import (
        EnTokenizer, Conditionals, punc_norm, drop_invalid_tokens,
        S3GEN_SR, S3_SR,
    )
    from chatterbox.models.t3.modules.cond_enc import T3Cond

    DEC_COND_LEN = 240000   # ChatterboxTTS.DEC_COND_LEN
    ENC_COND_LEN = 96000    # ChatterboxTTS.ENC_COND_LEN
    PLEN = 150              # T3Config().speech_cond_prompt_len
    SOT, EOT = 255, 0       # start/stop text tokens

    model_dir = Path(spec["model_dir"])
    device = _resolve_device(spec.get("device"))
    quantize_t3 = bool(spec.get("quantize_t3", False)) and device == "cpu"
    gk = spec.get("gen_kwargs", {})
    exaggeration = gk.get("exaggeration", 0.5)
    cfg_weight = gk.get("cfg_weight", 0.5)
    temperature = gk.get("temperature", 0.8)
    repetition_penalty = gk.get("repetition_penalty", 1.2)
    min_p = gk.get("min_p", 0.05)
    top_p = gk.get("top_p", 1.0)
    jobs = spec["jobs"]

    _setup_threads()
    _mark("start")

    # ---- Phase A: conditioning (ve + s3gen) -> small cached conds per ref ----
    refs = {}
    for j in jobs:
        refs.setdefault(j["ref"], None)

    ve = _load_ve(model_dir, device)
    s3gen = _load_s3gen(model_dir, device)
    _mark("A loaded ve+s3gen")
    for ref_path in list(refs):
        s3gen_ref_wav, _ = librosa.load(ref_path, sr=S3GEN_SR)
        ref_16k = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)
        ref_dict = s3gen.embed_ref(s3gen_ref_wav[:DEC_COND_LEN], S3GEN_SR, device=device)
        prompt_tokens, _ = s3gen.tokenizer.forward([ref_16k[:ENC_COND_LEN]], max_len=PLEN)
        prompt_tokens = torch.atleast_2d(prompt_tokens).to(device)
        ve_embed = torch.from_numpy(ve.embeds_from_wavs([ref_16k], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(device)
        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=device)
        refs[ref_path] = Conditionals(t3_cond, ref_dict)
    _free(ve, s3gen)
    _mark("A freed")

    # ---- Phase B: t3 inference (t3 only) -> speech tokens per job ----
    t3 = _load_t3(model_dir, device, quantize_t3)
    tokenizer = EnTokenizer(str(model_dir / "tokenizer.json"))
    _mark("B loaded t3")
    token_results: list = []
    for j in jobs:
        try:
            text = punc_norm(j["text"])
            text_tokens = tokenizer.text_to_tokens(text).to(device)
            if cfg_weight > 0.0:
                text_tokens = torch.cat([text_tokens, text_tokens], dim=0)
            text_tokens = F.pad(text_tokens, (1, 0), value=SOT)
            text_tokens = F.pad(text_tokens, (0, 1), value=EOT)
            with torch.inference_mode():
                speech = t3.inference(
                    t3_cond=refs[j["ref"]].t3,
                    text_tokens=text_tokens,
                    max_new_tokens=1000,
                    temperature=temperature,
                    cfg_weight=cfg_weight,
                    repetition_penalty=repetition_penalty,
                    min_p=min_p,
                    top_p=top_p,
                )
            speech = speech[0]
            speech = drop_invalid_tokens(speech)
            speech = speech[speech < 6561].to(device)
            token_results.append(speech)
        except Exception as exc:
            token_results.append(exc)
            traceback.print_exc()
    _free(t3, tokenizer)
    _mark("B freed")

    # ---- Phase C: s3gen vocoding (s3gen only) -> waveform per job ----
    s3gen = _load_s3gen(model_dir, device)
    watermarker = perth.PerthImplicitWatermarker()
    _mark("C loaded s3gen")
    for j, tokens in zip(jobs, token_results):
        out = j["out"]
        if isinstance(tokens, Exception):
            print(f"ERR {out} t3:{tokens!r}", flush=True)
            continue
        try:
            with torch.inference_mode():
                wav, _ = s3gen.inference(speech_tokens=tokens, ref_dict=refs[j["ref"]].gen)
            wav = wav.squeeze(0).detach().cpu().numpy()
            wav = watermarker.apply_watermark(wav, sample_rate=S3GEN_SR)
            sf.write(out, wav, S3GEN_SR, subtype="PCM_16")
            print(f"OK {out}", flush=True)
        except Exception as exc:
            print(f"ERR {out} s3gen:{exc!r}", flush=True)
            traceback.print_exc()
    _free(s3gen, watermarker)
    _mark("C freed")

    print(f"PEAK_RSS_MB {_mem()[1]:.1f}", flush=True)
    return 0


# --- Full pipeline (baseline) ----------------------------------------------

def run_full(spec: dict) -> int:
    import torch
    import torchaudio
    from chatterbox.tts import ChatterboxTTS, Conditionals, EnTokenizer

    model_dir = Path(spec["model_dir"])
    device = _resolve_device(spec.get("device"))
    quantize_t3 = bool(spec.get("quantize_t3", False)) and device == "cpu"
    gen_kwargs = spec.get("gen_kwargs", {})
    jobs = spec["jobs"]

    _setup_threads()
    _mark("start")
    print(f"LOADING (full, low-spike) from {model_dir} on {device}", flush=True)
    # Build the model manually so T3 uses the low-spike meta+assign loader
    # instead of from_local's naive (spiky) path.
    ve = _load_ve(model_dir, device)
    s3gen = _load_s3gen(model_dir, device)
    t3 = _load_t3(model_dir, device, quantize_t3)
    tokenizer = EnTokenizer(str(model_dir / "tokenizer.json"))
    conds = None
    if (cp := model_dir / "conds.pt").exists():
        conds = Conditionals.load(cp, map_location=torch.device("cpu")).to(device)
    model = ChatterboxTTS(t3, s3gen, ve, tokenizer, device, conds=conds)
    _mark("loaded")

    for j in jobs:
        out = j["out"]
        # Per-job gen_kwargs override the global ones (enables per-line
        # expressivity, e.g. a cranked intro vs. natural banter).
        gk = {**gen_kwargs, **j.get("gen_kwargs", {})}
        # Optional per-job seed makes a take reproducible -> pick the best
        # "best-of-N" roll, then lock its seed to regenerate it forever.
        if j.get("seed") is not None:
            torch.manual_seed(int(j["seed"]))
        try:
            wav = model.generate(j["text"], audio_prompt_path=j["ref"], **gk)
            torchaudio.save(out, wav.cpu(), model.sr)
            print(f"OK {out}", flush=True)
        except Exception as exc:
            print(f"ERR {out} {exc!r}", flush=True)
            traceback.print_exc()

    print(f"PEAK_RSS_MB {_mem()[1]:.1f}", flush=True)
    return 0


def main() -> int:
    spec = json.load(sys.stdin)
    mode = spec.get("mode", "full")
    if mode == "staged":
        return run_staged(spec)
    return run_full(spec)


if __name__ == "__main__":
    sys.exit(main())
