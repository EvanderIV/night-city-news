# Chatterbox RAM Optimization — Report

**Date:** 2026-06-13
**Scope:** Reduce the peak RAM needed to synthesize audio with the Chatterbox
TTS engine on CPU, without sacrificing the (near-perfect) output quality.
**Outcome:** Peak RAM cut from **~7.0 GB → ~4.66 GB** (≈33%) with **zero quality
change**, by fixing how the model is *loaded*. Two intuitively-appealing
techniques (in-process stage unloading, int8 quantization) were measured to
**increase** peak RAM here and were rejected.

---

## 1. TL;DR / Recommended configuration

Use the helper in **`full` mode with `quantize_t3: false`** (the current
default). This loads the model once, keeps it resident, and renders the whole
script. Measured peak: **~4.66 GB**, comfortably inside an 8 GB budget.

| Configuration | Peak RAM | Verdict |
|---|---:|---|
| `full`, naive load (original) | 7001 MB | baseline |
| **`full`, low-spike load, no quant** | **4657 MB** | ✅ **recommended** |
| `full`, low-spike load + int8 | 5157 MB | worse |
| `staged` (in-process unload) | 6668 MB | worse |
| `staged` + int8 | 7178 MB | worst |

> Measurement: peak Windows working set (`psutil` `peak_wset`) of the
> `.venv-chatterbox` process while rendering two ~8 s lines on CPU.

---

## 2. What changed

All changes are confined to the out-of-process helper
**`voice/chatterbox_synth.py`** (run by the isolated `.venv-chatterbox`
interpreter). **The main broadcast pipeline was not modified** — Chatterbox is
still *not* wired in as a TTS `engine` yet; that integration is a separate,
pending step.

### 2.1 Low-spike model loading (the actual win)

The original load path was effectively:

```python
t3 = T3()                       # allocates ~2 GB of random weights
state = load_file("t3_cfg.safetensors")   # ~2 GB more
t3.load_state_dict(state)       # copies state into the random weights
```

That transiently holds **~3× the model size** (random weights + state dict +
copy), which is what spiked the process to 7 GB. The fix:

```python
with torch.device("meta"):
    t3 = T3()                   # allocates NO real storage
state = load_file("t3_cfg.safetensors")
t3.load_state_dict(state, assign=True)   # params *reference* the mmap'd
                                         # safetensors tensors — no copy
```

Because safetensors memory-maps the file and `assign=True` makes the parameters
point straight at those mmapped tensors, **resident RAM right after load was
only ~1.76 GB** — the weights page in lazily as inference touches them.

**Gotcha handled:** building on `meta` skips initialization of the two
*non-persistent* rotary-embedding buffers (`tfmr.rotary_emb.inv_freq` /
`original_inv_freq`), which aren't in the checkpoint. They're recomputed:

```python
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
rot = t3.tfmr.rotary_emb
inv_freq, scaling = ROPE_INIT_FUNCTIONS[rot.rope_type](rot.config, device)  # rope_type == "llama3"
rot.register_buffer("inv_freq", inv_freq, persistent=False)
rot.original_inv_freq = inv_freq
rot.attention_scaling = scaling
```

`run_full()` now builds the `ChatterboxTTS` object manually (normal `ve`/`s3gen`
loaders + the low-spike `t3` loader + `conds.pt`) instead of calling
`ChatterboxTTS.from_local()`, so it benefits from the low-spike `t3` path.

### 2.2 Optional int8 quantization (implemented, but not recommended)

`_load_t3(..., quantize=True)` applies `quantize_dynamic` to **only the
transformer backbone** (`t3.tfmr` + `t3.cond_enc` — ~95% of the weights),
leaving the small heads/embeddings in fp32. This was necessary because:
- quantizing `speech_head` breaks `T3.device` (a property reading
  `speech_head.weight.device`, which a quantized Linear no longer exposes), and
- `inplace=True` avoids `quantize_dynamic`'s default deep-copy (a separate
  +2 GB spike seen in early runs).

It works, but **measured worse** (see §4) — kept behind the `quantize_t3` flag
for completeness only.

### 2.3 Staged (in-process) mode (implemented, but not recommended)

`run_staged()` splits work into three phases — (A) conditioning (`ve`+`s3gen`),
(B) `t3` inference, (C) `s3gen` vocoding — `del`-ing each stage's modules before
the next. **Measured worse** (see §4).

### 2.4 Instrumentation

Added `psutil`-based `_mem()` / `_mark()` to report current + peak working set
at each phase, and a final `PEAK_RSS_MB` line. (`psutil` was added to
`.venv-chatterbox`.)

### 2.5 Supporting / scratch files created

- `download_chatterbox.sh` — resilient curl downloader for the model weights
  (keep; useful given the flaky HF connection on this network).
- `gen_cb_specs.py`, `spec_full.json`, `spec_staged.json`, `spec_staged_q.json`,
  `spec_full_q.json`, `cb_job.json` — benchmark scratch (safe to delete).

---

## 3. Why the obvious ideas failed

- **In-process staging:** On CPU, the allocator does **not** return freed memory
  to the OS. `del t3; gc.collect()` shrinks Python's view but not the process
  RSS, so reloading `s3gen` for phase C stacked *on top of* the unreturned
  memory and pushed the **peak higher**, not lower.
- **int8 quantization:** Converting weights to int8 **touches every weight**,
  faulting the entire model into RAM and defeating the lazy memory-mapping that
  made the fp32 path cheap. You pay the full fp32 read to produce a "smaller"
  model — net peak went *up* for this short-job workload.
- **The real bottleneck was the load transient**, not steady-state weights or
  activations. Fixing the load (meta + `assign`) is what moved the number.

---

## 4. How to push RAM lower (future work)

The hard floor for the single-process design is ~4.6 GB because, during
inference, all weights are resident and the allocator won't release between
stages. Getting under that requires **true process isolation** — the time-for-RAM
trade, which suits the overnight batch job.

### 4.1 Process-isolated stages (primary recommendation, est. ~3 GB)

Run each stage as its **own subprocess that exits**, so the OS reclaims its
memory before the next stage starts. Peak then ≈ the **largest single stage**.

```
orchestrator (main venv or helper):
  1. spawn  phase_cond.py   ->  load ve + s3gen, compute conds for every
                                distinct reference, write conds to a temp
                                file (small), EXIT.            (~2.7 GB)
  2. spawn  phase_t3.py     ->  low-spike load t3, read conds, run t3.inference
                                for all lines, write speech-token tensors to a
                                temp file, EXIT.               (~2.5–3 GB)
  3. spawn  phase_s3gen.py  ->  load s3gen, read tokens + conds, vocode +
                                watermark, write WAVs, EXIT.   (~2.7 GB)
```

The phase logic already exists inside `run_staged()` — the work is to split
those three blocks into standalone entry points and serialize the small
hand-off artifacts (conds, speech tokens) to disk between them. Intermediate
tensors are tiny; the cost is reloading each stage (extra minutes — acceptable
overnight).

**Estimated peak:** ~3 GB, dominated by the s3gen-bearing phases. If those need
to drop further, apply the same meta+`assign` low-spike trick to `s3gen`
(verify its non-persistent buffers, analogous to the rotary fix).

### 4.2 Secondary levers (smaller / situational)

- **Cap concurrent worker threads' memory:** none here (single process), but if
  parallelizing lines across processes, do the opposite of what we want — avoid.
- **`s3gen` low-spike load:** mirror §2.1 for `s3gen` to shave its ~2.7 GB load
  peak; needs a check for meta-uninitialized buffers (mel filterbanks, etc.).
- **Allocator tuning:** on Linux, `MALLOC_TRIM_THRESHOLD_`/jemalloc/tcmalloc can
  coax the allocator into returning freed pages, which would make in-process
  staging viable. On Windows this lever is largely unavailable — another reason
  process isolation is the portable answer.
- **Quantization *only* makes sense with process isolation**, where the int8
  model is the *only* thing resident in its phase and is loaded directly as int8
  (not converted from a resident fp32 copy). Even then, A/B the quality first.
- **Shorter references / chunked long lines** reduce activation memory, but
  activations were not the bottleneck here.

### 4.3 What NOT to do (confirmed dead ends)
- In-process stage unloading (`del` + `gc`) — no OS reclaim on Windows CPU.
- Quantizing in the single-process/full design — defeats lazy mmap.

---

## 5. Caveats / things to watch

- The rotary recompute keys on `rope_type == "llama3"` and the
  `transformers.modeling_rope_utils.ROPE_INIT_FUNCTIONS` API. Both are stable
  for the **pinned** `transformers==5.2.0` in `.venv-chatterbox`; revisit if that
  pin ever changes.
- `assign=True` requires PyTorch ≥ 2.1 (we run 2.6) and assumes all non-buffer
  params are present in the checkpoint (verified: only the 2 rotary buffers were
  meta-leftover).
- Chatterbox embeds an inaudible Resemble "Perth" watermark by design.
- Chatterbox lives in an **isolated venv** (`torch==2.6` / `transformers==5.2`)
  because those pins conflict with the XTTS stack in the main venv. Any
  integration must call it out-of-process (as the helper does).

---

## 6. Current status

- ✅ Low-spike loading implemented and is the helper default (`full` mode).
- ✅ Peak RAM ~4.66 GB, quality unchanged.
- ⏸️ Process-isolated ~3 GB path: **designed, not built** (this report, §4.1).
- ⏸️ Chatterbox as a pipeline `engine = "chatterbox"`: **not wired into
  `main.py` / `voice/synthesizer.py` yet.**
