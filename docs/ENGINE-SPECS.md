> **STANDARDS:** `docs/ENGINE-STANDARDS.md` is authoritative. **Maya v1.1.78** and **Strata v0.2.21.1** are the
> standard; **atomic-engine (atomicchat)** is the workflow replacement; **llama.cpp is removed** from this
> workspace and from every workflow. Where this document cites it, the citation is history.

# ENGINE-SPECS — the reference class, pulled and speced

**Scope.** All 43 inference engines cloned under `C:/Users/eman5/WuBu/research/engine-sources/`, pulled to
current on 2026-11-25, and speced here for what they do, what they prove, and what wubuwizard takes or
refuses. Read-only clones; nothing vendored.

**Rule for this document.** Every claim is either marked *measured* (a number produced on this box or a grep
count in the pinned tree) or *claimed* (the project's own README). A project's marketing is not evidence about
its code. Where the two disagree, the grep wins and the disagreement is recorded.

**Why this class matters more than llama.cpp.** Inkling 2 Mini is a looped transformer: 24 physical layers,
core layers 6..17 executed 3× per token. A single-pass engine can load it and throw two of every three passes
away. An engine that models the loop can keep them — contrast them (LoopCD), draft from them (LoopSpec), or
profit from them (MTP heads). That difference is capability, not throughput, and it is where wubuwizard can
win something a llama.cpp-shaped engine cannot copy.

---

## 1. Pull state (all `behind=0`, measured 2026-11-25)

| engine | version / branch | HEAD | LOC (lang) |
|---|---|---|---|
| Strata | **v0.2.21.1** | 2af30cc | 1,490,171 (py 1607, cpp 1219, cu 322) |
| project-maya | **v1.1.78** | 1530120 | 248,923 (py 338, cpp 313, cu 174) |
| atomic-engine | dev-latest | fa190d7b5 | 134,819 (cpp 679, h 297, py 263) |
| basalt | v0.5.7 | 31174039 | 301,716 (cpp 529, py 207, cu 137) |
| lucesbox | — | a70ab3c | 711,673 (cpp 638, h 460, cu 273) |
| Splash | 1.7.3 | d40c250 | 322,632 (py 203, cpp 167, h 69) |
| MTPLX | v2.13.3 | b16e30e | 784,216 (py 1353, swift 338) |
| TensorFold | main | 4c79cfe3 | 1,456,118 (zig 7046, py 617, cu 193) |
| TurboFieldfare | 0.10.1 | f24e598 | 114,126 (swift 559) |
| loopspec | main | 41aa710 | 9,885 (py 49) |
| ik_llama.cpp | main | ad217a2 | 623,302 (cu 362, cpp 283, h 140) |
| llama.cpp | **b12723** | abca692 | 1,045,176 (cpp 801, h 356, py 269) |
| vllm.cpp | main | 28007b4 | 3,500 files, cpp 1543 / h 1002 |
| vllm | v0.34.1rc0 | d51a514 | 2,915,436 (py 6747, rs 441) |
| sglang | main | 5b4ddbb6 | 3,734,365 (py 9497, rs 448) |
| flashinfer | main | 55643877 | 15,123,728 (cu 7745) |
| LMCache | nightly | bd830fc | 449,088 (py 1427, go 76) |
| vattention | main | 71a0e91 | 2,273,610 (h 2650, c 1235, cu 724) |
| ktransformers | main | a165075 | 272,237 (py 675, cpp 177) |
| PowerInfer | main | 8bd56d6 | 964,717 (cpp 332, h 210, cu 167) |
| flyweight | main | fc323b3 | 144,711 (py 226, cpp 50) |
| BigMoeOnEdge | v0.49.0 | 8e415e4 | 143,614 (swift 193, cpp 170) |
| mistral.rs | master | 21da642 | 718,990 (rs 1000, py 180) |
| candle | main | 1a49dc2 | 262,997 (rs 635, cu 148) |
| frink | main | 55f122a | 451,847 (rs 896) |
| mlx-lm | main | 6fe0ce4 | 89,776 (py 736) |
| Rapid-MLX | rapid-mac-v1.1.2 | 43d166a | 1,709,622 (py 2743, swift 801) |
| MNN | master | 02e1062 | 4,092 files (cpp 2008, h 797) |
| MemVanta | main | 66701ca | 8,514 (cpp 37, py 15) |
| llm-inference-cpp | main | ece521b | 22,759 (cpp 55) |
| h3.c | main | 8974cc0 | 30,692 (c 38, h 20, m 3) |
| rec | v0.1.2 | 0a4aea6 | 15,167 (py 64, go 32) |
| MACaQ | main | a8c6c47 | 839,057 (cpp 642, h 290) |
| EAGLE | main | cb7e084 | 22,072 (py 51) |
| AngelSpec | main | 0505773 | 91,961 (py 242) |
| AdaFlash | main | 478bf0f | 9,533 (py 48) |
| SpecPilot | main | 1058dc2 | 1,203,593 (py 2111, cuh 200) |
| HyperQwen | main | 3d6c954 | 40,217 (py 160) |
| gpt4all | main | b666d16 | 26,920 |
| KuiperInfer | main | 64e9561 | 21,079 (cpp 82) |
| inference-optimisation-skill | main | fda968e | 313 (py 2) |
| **FlexGen** | master | 951c23b | 13,334 (java 66) — **not an LLM engine, miscloned** |

---

## 2. Who actually runs Inkling 2 Mini (measured: files whose source mentions it)

This is the competitive scan, and the numbers are grep counts, not claims:

| engine | files w/ "inkling" | w/ "inkling2" | reading |
|---|---|---|---|
| Strata | 291 | 247 | deepest integration of any engine |
| TensorFold | 219 | 184 | Zig, Metal+CUDA |
| splash | 124 | 92 | DFlash 2 spec decoding |
| **basalt** | 108 | 96 | has a bench corpus for it (see §3) |
| vllm | 171 | 78 | upstream transformers path |
| sglang | 282 | 83 | LoopSpec integration target |
| MTPLX | 210 | 61 | Metal + MTP heads |
| Rapid-MLX | 54 | 30 | Apple |
| lucesbox | 35 | 31 | AMD |
| ik_llama.cpp | 32 | 26 | fork, MTP/DFlash landed here first |
| llama.cpp | 53 | 39 | **b12723 does reference inkling2** |
| project-maya | 18 | 18 | 26B, one-to-16 GPUs |
| flyweight | 12 | 12 | experts on CPU |
| BigMoeOnEdge | 6 | 6 | flash-streamed experts |
| atomic-engine | 18 | 1 | mostly KV-streaming work |
| PowerInfer | 5 | 0 | neighbouring MoE support only |

**Correction to an earlier claim in `INKLING2-PORT.md`.** That doc states llama.cpp's `inkling2` is
single-pass only. At b12723, 39 files mention `inkling2`. That claim is **stale or was never verified for this
tag** and must not be repeated as fact. It does not change the ruling that llama.cpp is not the comparative
spec — but it does mean the load failure I hit earlier may have been a tensor-name/format mismatch rather than
an absent architecture, and that distinction was assumed rather than tested.

---

## 3. The named three

### Strata — v0.2.21.1, 1.49M LOC (py/cpp/cu)
Consumer-GPU chat/code/vision product. Largest Inkling 2 Mini integration by file count (247). Ships a **CI
gate suite specifically for this model** — measured file list:

```
.github/ci/inkling/b70_gate.sh        .github/ci/inkling/c32_chat.py
.github/ci/inkling/chat_check.py      .github/ci/inkling/longctx_chat.py
.github/ci/inkling/serve_chat.py      app/host/tests/inkling-family.test.mjs
```

**What we take:** the gate *shape* — separate short-chat, long-context-chat, and serve-level checks per model
family, run in CI, rather than one grand suite. Our suite is a single binary; this is the better decomposition
because a long-context regression and a chat regression are different bugs.

### project-maya — v1.1.78, 249k LOC
Multi-model local server: GLM-5.3, DeepSeek-V4.1-Flash, MiniMax-M3.1-Flash, Hy4-Flash, **Inkling 2 Mini 26B**;
one to sixteen GPUs; OpenAI + Anthropic APIs; concurrent requests.

**What we take:** nothing structural — it is a serving product, and serving is out of scope until the graph is
correct. It is evidence of the *target*: a 26B looped MoE served locally is a solved problem for four engines
already, so correctness is not gated on hardware.

### atomic-engine — dev fa190d7b5, 135k LOC, built *on top of llama.cpp*
Self-described: "inference engine under development on top of llama.cpp, targeting **MoE expert streaming from
SSD** and adjustable managed memory budgets. The experimental engine admits model, state, compute and expert
buffers under a **shared budget with an explicit runtime reserve**."

**This is the closest published analogue of the staging hierarchy wubuwizard is aiming at** (SSD → RAM
transient → VRAM workspace) and of the "declare scratch, recycle it" idea. Measured: its benchmark tree
contains `benchmark_kv_stream.py` plus `benchmarks/results/kv-stream-pr-evidence/gemma4-it-memory|throughput/`
with per-variant KL-divergence logs (`gemma4-baseline`, `-streaming`, `-f16-check`, `-nofa-check`,
`-coherence-check`).

**What we take, and this is the single most useful thing in this whole pull:** the evaluation *pattern* —
streaming vs baseline compared by **KL divergence on logits**, with a separate coherence check, rather than
by eyeballing output. That is exactly the measurement my own staging work is missing, and it is cheap: I
already have logit dumps and a detokeniser. A staging change that alters logits beyond a KL threshold fails,
regardless of how fast it is.

---

## 4. MoE weight-streaming class (our actual target)

The premise — experts on flash, hot path in VRAM — is independently implemented at least five times:

- **atomic-engine** — SSD expert streaming, shared budget + explicit runtime reserve (above).
- **BigMoeOnEdge** (v0.49.0) — "experts are read from flash as the model needs them", llama.cpp-based, claims
  a 12 GB Android phone running a 35B-A3B at 5–6 tok/s. Storage→VRAM streaming on a *phone*.
- **flyweight** — GGUF/safetensors native C++/CUDA, "built for one NVIDIA card and a lot of host RAM: MoE
  checkpoints larger than VRAM run with their routed experts on the CPU."
- **ktransformers** — CPU-GPU heterogeneous experts; `kt-kernel`; MXFP4 routed experts, LoRA fine-tuning of
  them (Oct 2026).
- **PowerInfer** — activation locality: hot experts on GPU, cold on CPU, learned at load time.
- **basalt** (v0.5.7) — "what does not fit in VRAM (the experts of a mixture-of-experts model) in system RAM",
  optionally more GPUs; sm_120 / GB10.

**What this means for wubuwizard, stated plainly:** expert streaming is *not* the differentiator, because it
is already table stakes across five engines including two that support our exact model. The differentiator
stays the loop. The staging hierarchy remains necessary — but as an *enabler* for holding three passes of
recurrent state, not as the headline.

**Direct reuse from basalt (measured file list):** a ready-made bench corpus for our model —
`data/bench/inkling2/{prefill,prose,struct,conv}.ids` and `data/profiles/inkling2-24x128.prof`. Those are
token-id files: prefill, prose, structured, conversation. They give us four canonical prompt classes with
known token ids, which is exactly what a gate suite needs and what I have been improvising with single ids.

---

## 5. Looped decoding class (our differentiator)

- **loopspec** (9,885 LOC, py) — "Pipelined Self-Speculative Decoding for Looped Transformers… drafts upcoming
  tokens while verifying earlier ones, **reusing work across recurrent steps**". Lossless greedy + sampling;
  runs through SGLang; targets Ouro and Raven. Verified layout: `sglang_recurrent/`, `benchmark.py`
  `download.py`, `demo/`, `experiments/`.
- **MTPLX** (v2.13.3) — runs Inkling 2 Mini on Metal using "the model's own multi-token prediction (MTP)
  heads", rejection sampling with **residual correction at the model's native temperature**, and validates by
  comparing a thousand fast-path samples against a thousand plain-path samples *by token id*. Measured:
  `mtplx/models/inkling/{attention,attention_metal,cache,activations}.py` + `forge_inkling*.py`.
- **ik_llama.cpp** — README states MTP and DFlash landed there before upstream: "Many inference features
  landed here first: MLA, quant repacking, fused delta-net, tensor parallel, MTP, DFlash."

**Where wubuwizard sits.** LoopCD (implemented, committed) and LoopSpec are the same insight from two
directions: intermediate recurrent states are *predictions*, not waste. LoopCD contrasts pass 3 against pass 1
at decode; LoopSpec drafts from earlier passes and verifies. MTPLX gets the same benefit from dedicated MTP
heads trained for it. **Only the first two work on a stock checkpoint with no MTP head**, and LoopSpec's
greedy mode is lossless — which makes it the natural next step after LoopCD, and it composes: draft with
earlier passes, then guide with LoopCD.

**Validation discipline worth copying:** MTPLX proves equivalence by sampling a thousand outputs from each path
and comparing distributions by token id, not by asserting correctness. That is the same standard as the
atomic-engine KL check and it is the right one for anything that changes the decode path.

---

## 6. KV / memory management

- **vattention** (OSDI'24) — decouples virtual from physical KV allocation via CUDA virtual memory APIs; physical
  on demand, virtual stays contiguous. Removes paged-attention fragmentation without paging cost.
- **LMCache** — KV cache as a separate managed layer (offload/reuse across requests).
- **MemVanta** (8.5k LOC) — mmap model access, paged KV, Q4/Q8 kernels, **bounded adaptive prefetching**, with
  memory and throughput benchmarks against pinned llama.cpp. Its whole thesis is "how much resident memory can
  we avoid while still executing correctly" — a direct test of our staging premise.
- **vllm.cpp** — continuous batching, paged KV, CUDA/CPU/Metal/Vulkan, no Python. Closest in *philosophy* to
  wubuwizard (C++-only, multi-backend) and worth reading for batching structure.
- **mistral.rs** — paged attention default on Metal, automatic memory planning.

**What we take:** MemVanta's framing (measure resident-memory avoided *at fixed correctness*) and
vattention's virtual/physical split, which is the correct model for a recyclable declared scratch pool: the
*addresses* are stable and declared at init, the *physical* pages come and go. That is the C-side version of
what the "declare it in a hive scratch, JIT-init, boom" idea was reaching for.

---

## 7. Quantisation (format authority)

- **llama.cpp b12723** — canonical ggml quant layouts remain the format authority for dequantisers.
- **ik_llama.cpp** — SOTA quant types, **quant repacking** (row-interleaved for CPU/GPU split). Note its own
  warning: `-rtr` repacks everything left in RAM to row-interleaved, and not all quant types have a CUDA
  implementation — a footgun in exactly the hybrid MoE case we care about.
- **MACaQ** — model-agnostic quantisation, claims a 122B model quantised in 64 min on 32.5 GB, and
  llama.cpp-compatible GGUFs.

**Measured correction carried from the port:** the Inkling 2 Mini UD file's type-17 expert tensor is **not**
canonical ggml `IQ2_XS`. It uses a 9-bit grid index plus a **7-bit** sign index (`q>>9` reaches 127 against a
128-entry sign table), not 9+3. Any check that assumes the canonical 12-bit packing fails on this file. This is
a property of the checkpoint, and it is invisible in all 43 engines' documentation.

---

## 8. Serving and research engines, flat

`vllm` (v0.34.1rc0), `sglang` (3.7M LOC, RadixAttention), `flashinfer` (15.1M LOC of kernels — attention/GEMM/
MoE backends), `EAGLE` (spec-dec research), `AdaFlash` (on-policy distilled diffusion drafters, adaptive
candidate length), `AngelSpec` (Tencent Hunyuan; trains MTP/DFlash drafters; DFly2 reuses target's sparse KV and
top-k so drafting cost is **constant in context length**), `SpecPilot` (agentic search that found per-deployment
spec-dec configs, "more than doubles stock's speedup, without training"), `HyperQwen` (Qwen on consumer GPUs,
150k context on 24 GB), `candle`/`frink`/`mistral.rs` (Rust), `mlx-lm`/`Rapid-MLX`/`TensorFold`(Zig)/`Splash`/
`MTPLX`/`TurboFieldfare`(Gemma 4 26B-A4B in ~2 GB RAM)/`h3.c` (MiniMax-H3 on Metal) — Apple-silicon side,
`MNN` (mobile; Hexagon DSP backend), `gpt4all`, `llm-inference-cpp` (C++20 from scratch, every stage checked
against HF reference — same discipline as this port), `rec` (Recursive's own engine), `KuiperInfer` (course
framework), `inference-optimisation-skill` (an agent skill for hill-climbing server speed).

**Miscloned, should be deleted:** `FlexGen` at 951c23b is *"a flexible random map generation library for
games"* — Java, 66 files, last commit 2014. It is not FlexGen-the-offloading-engine. Anything that referenced
it as an offloading reference was referencing a map generator.

---

## 9. Matrix — capability vs wubuwizard's needs

| need | who has it | status for us |
|---|---|---|
| Runs Inkling 2 Mini | Strata, TensorFold, splash, basalt, MTPLX, lucesbox, maya, sglang, vllm, llama.cpp b12723 | **proven possible**; correctness is not hardware-gated |
| Looped 3-pass decode | wubuwizard (LoopCD), loopspec, MTPLX (MTP) | **our edge**; only two others, neither is a C engine |
| Expert streaming SSD→VRAM | atomic-engine, BigMoeOnEdge, flyweight, ktransformers, PowerInfer, basalt | table stakes, not a differentiator |
| Shared memory budget + reserve | atomic-engine | worth copying literally |
| KL-divergence gate for streaming | atomic-engine | **adopt** |
| Reuse across recurrent steps | loopspec | next step after LoopCD |
| Lossless spec-dec | loopspec (greedy), MTPLX (residual-corrected) | adopt as the quality bar |
| Virtual/physical KV split | vattention | model for the declared-scratch pool |
| Resident-memory-avoided metric | MemVanta | adopt as the staging metric |
| 7-bit-sign IQ2 variant | **nobody documents it** | ours to record |

---

## 10. What this changes

1. **Expert streaming is not the headline.** Five engines do it, two on our exact model. Claiming it as the
   differentiator would be claiming parity as innovation. The loop is the edge.
2. **Measurement gets a standard.** KL divergence against a baseline, coherence checks, and thousand-sample
   token-id comparisons (atomic-engine, MTPLX) — not "results unchanged, therefore no effect", which is the
   mistake a stale binary already tricked me into twice.
3. **Gates get a shape.** Strata's per-family CI split (short chat, long context, serve) plus basalt's four
   prompt classes (`prefill/prose/struct/conv` token ids) replace my single improvised prompt id.
4. **A stale claim is retracted.** llama.cpp at b12723 does reference `inkling2` in 39 files. The port doc's
   "single-pass only" line is not to be repeated without re-testing against this tag.
5. **One repo is wrong.** `FlexGen` is a map generator; delete it before it pollutes a citation.

---

## 11. Addendum: what the pull actually settled (hours after writing §1-10)

The spec above flagged that llama.cpp at b12723 has `src/models/inkling2-loop.cpp` and retracted a stale
"single-pass only" claim. Following that file further settled something the port had been stuck on for
several turns.

**`src/models/inkling2.cpp:251`:**

```c
return ggml_scale(ctx0, ggml_add(ctx0, cur, out), 1.0f/sqrtf((dynamic ? 2 : 1)*n_conv + 1));
```

The dynamic (full-layer) short convolution normalises by `1/sqrt(2k+1)`; the static convs use `1/sqrt(k+1)`.
The port's own code already did this correctly - `in2_shortconv_dynamic` divides by `sqrtf(2*k+1)` and carries
a comment warning that the alternatives are a silent 1.34x error - but the independent check used `1/sqrt(5)`
and therefore disagreed with a correct implementation. Measured: `1/sqrt(5)` gives max|diff| 3.72; `1/3` gives
**9.5e-07**, and the ratio implied by the data is `0.333333` against `1/3 = 0.333333`.

Two things follow, and they are the reason this addendum exists:

1. **This is the correct use of llama.cpp in this project.** Not as a comparative spec (barred), not as a
   throughput target (barred), but as a *semantics* source - the same role `ggml-quants.c` plays for
   dequantiser layouts. One line of its model code resolved a question that eight engines' READMEs and my own
   reasoning had not.
2. **The `inkling2-loop.cpp` file (123 lines) plus `inkling2.cpp` (483 lines) are a compact, readable statement
   of this architecture's tensor semantics** - dynamic conv conditioning, `attn_res_block` boundaries, shortconv
   cache semantics, per-head tap layout `(n_conv, n_heads)`. Against 1,045,176 lines of llama.cpp, those 606 are
   the part that matters for this port, and they are worth reading line by line before the next checkpoint.

Also actioned from §9: `FlexGen` deleted from the clone set - it is a Java map generator from 2014, not the
offloading engine, and citing it as an offloading reference would have been an error traceable to this pull.

### Why llama.cpp cannot load this checkpoint - diagnosed, not guessed

Built b12723 (`llama-cli.exe`, MinGW-UCRT 16.1, needs `libgcc_s_seh-1`/`libstdc++-6`/`libwinpthread-1`
copied beside it or the loader returns 127 with no message) and attempted the load three ways:

```
arch as shipped      -> failed to load model (no detail at default verbosity)
arch=inkling2_loop   -> key not found in model: inkling2_loop.context_length
block_count=24       -> key inkling2.attention.head_count has wrong array length; expected 24, got 48
```

The third line is the diagnosis. `inkling2.attention.head_count` is a **per-layer array of length 48** in the
file, matching its declared `block_count = 48` (the *logical* count: 24 physical layers with the core looped
three times). `llama.cpp`'s `inkling2_loop` instead takes `n_phys = hparams.n_layer()` and expects both
`block_count` and every per-layer array to be **physical** (24), then expands to cache slots itself:

```cpp
n_phys = hparams.n_layer();
const size_t n_slot = n_phys + (n_phys ... )*(core_end - core_start);
hparams.n_layer_all = n_slot;
```

So the two disagree on a **metadata convention**, not on capability. llama.cpp ships a real looped
implementation for this architecture (`src/models/inkling2-loop.cpp`, 123 lines, with `loop_q/k/v/ffn_ctrl_{down,gain,forget}`
and a `bank` that starts at `prelude_blocks.size() + 1`); it simply cannot read a file authored to the
logical-count convention. Loading it there would require rewriting the KV section - every per-layer array
truncated to 24 and the arch string re-prefixed - which shifts the 8.4 GB data section.

**That closes the earlier load-failure question** ("tensor blk.24.attn_q.weight not found"): the file says
`block_count 48`, so the non-loop path wants blk.24..47, which do not exist. It was never a missing
architecture and never a corrupt file.

**Decision, per the standing ruling:** making llama.cpp ingest this file is the nerd snipe. The implementation
was read for *semantics* (`mix`, `control`, `convolve`, `tau`, `clip` - all extracted above and all now
reflected in the port), and the loading attempt stops here. The file's own convention is the one the port
already uses.
