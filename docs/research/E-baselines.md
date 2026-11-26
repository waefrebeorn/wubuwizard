# E — Empirical prefill baselines: what real engines actually do on a 6 GB laptop GPU

**Status:** measured-baseline survey, 2026-11-26. Citation ids 161-200 (see `LEDGER.md`).
**Scope:** PREFILL only (prompt processing / prompt-eval, tokens/s). Decode is quoted only where
the same run reports it, because a prefill claim without its decode number is usually a
mis-labelled decode number.
**Purpose:** replace the guess "llama.cpp is faster" with numbers, so the engine's own prefill
(0.11 s/token f32 = **9.09 t/s**, 8.0 h for 262k) can be placed on a real scale.

Every number below is tagged:

- **[M]** measured on hardware by a named person/repo, with the run's own settings.
- **[D]** derived here by arithmetic from [M] numbers — the arithmetic is shown so it can be checked.
- **[V]** vendor/author claim, not independently reproduced. There are only three of these and they
  are labelled inline.

---

## 0. Answers to the four questions, up front

1. **llama.cpp CUDA on a 6-8 GB laptop GPU, 12B-class Q4:** a 12B dense model at Q4/Q6 that *fits*
   prefills at **700-900 t/s at 4k** on an RTX 5060 Laptop 8 GB [171]; a 9B Q4_K_M that fits
   comfortably reaches **1,828.8 t/s at pp512** on an RTX 4060 Laptop 8 GB [168]. The exact model in
   question (Inkling-2-Mini, 12.8 GB file, most experts on CPU) prefills at **101 t/s at 10k
   context** on an RTX 4060 8 GB [170], and **545 t/s at 4k** after requantisation + kernel fusion +
   `-ub 512 -b 2048` [162]. At 32k on a 24 GB card the same model falls to **400 t/s** from 965
   [164]. No 128k llama.cpp number exists for a 6 GB card, because the KV cache does not fit — that
   absence is itself the finding.
2. **Fall-off:** measured, and it is not one curve. With flash attention working and everything in
   VRAM the fall is **13-20% from 512 to 16k** [173]; with FA broken or absent it is **49% to 783%**
   over the same span [173]; on unified memory it is **about 2x per position-doubling** (317 → 168
   t/s from 6k to 35k) [185]. The measured cause is a phase change: at pp512 matrix multiply is
   93.3% of prefill time and softmax 0.84%; at 32k softmax is **17.8%** and the two attention
   matmuls are **59% of all matmul time** [175].
3. **262k:** **8 minutes** on a 24 GB RX 7900 XTX with a dedicated engine (1,193 t/s measured at
   262k) [165]; **6 minutes** on an A100 (715 t/s at 250k) [166]; **≈45 minutes** on an 8 GB laptop
   card at the measured 101 t/s [170]; **≈3.5 hours** on a 96 GB unified-memory iGPU once the
   measured halving-per-doubling is applied [185]. Hours, not days — and the engine's 8.0 h is
   already in the hours band, which is the uncomfortable part.
4. **Kernels:** quantised tensor-core GEMM (MMQ-class) is worth **5.7-6.3x over the same card's
   cuBLAS path** [182] and **8.5x over the same machine's CPU** [172]; tensor cores themselves are
   worth **5x** versus a coreless GPU [181]; flash attention is worth **3.1x at 32k** [175] and,
   when it silently fails to bind, the measured penalty is **20-37x** [183]. A scalar C
   implementation is **~200x** off the pp512 baseline [168] and **3 orders of magnitude** off the
   card's tensor peak (§7).

---

## 1. What "6 GB laptop GPU" actually constrains

The RTX 4050 Laptop is 2,560 shaders, 6 GB GDDR6, **192 GB/s**, 10.91 TFLOPS FP32/FP16 vector at
80 W (9.73 at 60 W) [193]. The RTX 4060 Laptop is the same generation with 8 GB and the same 128-bit GDDR6 class,
272 GB/s in the same table [181]. Two consequences drive every number in this report:

- **Weights stream at 192 GB/s.** A 7.9 GB model read once costs 41 ms; at one token per read that
  is a 24 t/s decode ceiling before any compute [D]. Prefill escapes this because one weight read
  serves M prompt rows — which is exactly why prefill tok/s is 5-20x decode tok/s in every table
  below, and why *batching is the prefill optimisation*.
- **6 GB cannot hold a 12B Q4.** A 12B at 4.5 bpw is 6.8 GB of weights plus KV plus CUDA context,
  so the model either spills to host RAM over PCIe or keeps some layers on the CPU. Every "6 GB,
  12B" number in the wild is therefore a *hybrid* number, and hybrid prefill is bounded by the
  slowest device in the loop, not by the GPU.

That distinction explains an apparent contradiction in the data: a 9B that fits gets 1,828 t/s
[168], while a 12B that spills gets 700-900 t/s [171] and the 12.8 GB MoE gets 101-545 t/s
[162][170]. Same class of card, 2-18x apart, entirely from residency.

---

## 2. The laptop-GPU table (the core evidence)

| Hardware | Model / quant | Engine, settings | Context | Prefill | Decode | Date | Src |
|---|---|---|---|---|---|---|---|
| RTX 4060 Laptop 8 GB, i7, 24 GB RAM | Qwen3.5 9B Q4_K_M (fits fully) | llama.cpp CUDA, `-ngl 99`, llama-bench pp512/tg128 | 512 | **1,828.8 t/s** | 37.7 | 2026 | [168] |
| RTX 5060 Laptop 8 GB | Gemma 4 12B dense, 6.18 GB per-tensor quant | llama.cpp, 48 layers, `-ngl -1 -c 4096 -fa on`, mean of 3 | 4,096 | **700-900 t/s** | 27.4 (40.0 w/ MTP) | 2026-07-06 | [171] |
| RTX 4060 8 GB + 16 GB DDR4 | **Inkling-2-Mini** (7.9 GB IQ2_XXS class) | llama.cpp, stock flags | 10,000 | **101 t/s** | 17.11 | 2026 | [170] |
| RTX 4060 8 GB | **Inkling-2-Mini** requant 12.8 GB Q3_K experts | llama.cpp + fusions, `-ngl 99 -fa 1 -ub 512 -b 2048`, blk 6-15 experts on GPU, 7.3 GB peak | 4k (8k ctx) | **545 t/s** (308 stock) | 31.1 (17.4 stock) | 2026 | [162] |
| RTX 4050 Laptop 6 GB, i5-13420H, 64 GB DDR5 | Qwen3.6-35B-A3B Q5_K_M, RAM offload | LM Studio, 34-36k windows | 34-36k | **≈292 t/s [D]** from a reported 2-minute TTFT | 25-35 | 2026 | [169] |
| RTX 4050 Laptop 6 GB | LFM2.5 1.2B | LM Studio | not stated | **8,500 t/s** | 129 | 2026 | [169] |
| RTX 4070 Ti 12 GB | **Inkling-2-Mini** IQ3_XXS 10.81 GiB, blk 20-23 experts on CPU | llama.cpp, 64k ctx | short | **1,239-1,399 t/s** | 69-75 | 2026 | [163] |

The nearest fully-offloaded control is the RTX 4060 8 GB desktop at 272 GB/s [194]; the laptop
parts below differ from it by clock and power, not by architecture.

Read the table by residency, not by card:

- **Fits with room** (9B/5.4 GB in 8 GB): 1,828 t/s. This is the number to beat for short prompts.
- **Fits exactly** (12B/6.18 GB in 8 GB): 700-900 t/s. The 2x loss against the 9B is the extra
  parameters plus the tighter KV budget, not a different kernel.
- **Spills to host RAM** (12.8 GB in 8 GB; 7.9 GB IQ2 in 6-8 GB): 101-545 t/s, and the spread is
  almost entirely kernel quality + how many experts stay on the GPU. Same card, same model,
  **5.4x** between the stock build and the tuned one [162][170].
- **Tiny model** (1.2B in 6 GB): 8,500 t/s — proof that the card's MMA throughput is fine and that
  everything above is a residency/bandwidth story.

The 4050's 2-minute TTFT at 34-36k [169] is the only direct long-context number I found on a 6 GB
card. It is worth stating plainly: **that is 292 t/s, and the user called it "outstanding"** — for a
35B MoE that mostly lives in system RAM, on a 6 GB laptop.

---

## 3. 12B-class Q4/Q2 specifically

Two hard data points and one useful control:

| Run | Config | pp512 / short | Long context | Src |
|---|---|---|---|---|
| Gemma 4 12B Q4_K_M, **RTX 5080 16 GB** | llama.cpp, 41 configs, CUDA 12.4 | **4,983 t/s** GPU vs **589 t/s** CPU | — | [172] |
| Gemma 4 12B Q4_K_M, same card | FA on, per the repo's own table | 4,618 t/s; repo claims "+11% PP" but this is **below** its 4,983 baseline row, so the +11% is against a different config and cannot be read as an FA delta | — | [172] |
| Gemma 4 12B Q4_K_M, same card | `bs=4096` | **5,120+ t/s** | — | [172] |
| Gemma 4 12B, **RTX 5060 Laptop 8 GB** | tight per-tensor quant | 700-900 t/s | 33-37 t/s decode at 16k | [171] |
| 12B QAT Q4_0, **RTX 5080** desktop | llama.cpp CUDA, 8k prompts | **3,100 t/s** | — | [161] |

[D] Scaling the 5060 Laptop's 700-900 t/s against the 5080's 4,983 t/s gives a 6-7x ratio against
~3.5x the memory bandwidth (960 vs 272 GB/s) and ~3.6x the vector FLOPS. The laptop card is *not*
underperforming — it is bandwidth-limited, and pays a further factor for having almost no VRAM
headroom at 12B.

**The Q2/IQ2 case is worse than the Q4 case, not better.** IQ2_XXS is not a faster format: on CPU it
runs at 43.54 t/s against Q4_0's 146.20 in mainline llama.cpp, because the lookup-table decode costs
more than the bytes saved [174]. Only after runtime repacking (ik_llama.cpp) does it reach 154 t/s
[174]. On GPU the same shape holds — low-bit formats buy memory, and pay in kernel complexity
unless the kernel is written for them. **For a 7.9 GB IQ2_XXS on a 6 GB card, the quant is buying
you the ability to run at all, and charging you ~2-3x in prefill versus a 4-bit format of the same
size class.**

---

## 4. Fall-off with context: four measured curves and what causes it

### 4.1 The clean case — FA working, all in VRAM

Arc Pro B65, llama.cpp b11574 Vulkan, mean of 5 runs, 2026-10-12 [173]:

| Model | Test | fa=1 | fa=0 | fa=1 fall 512→16k |
|---|---|---|---|---|
| Qwen 3.8 27B | pp512 / pp4096 / pp16384 | 434 / 358 / 221 | 446 / 432 / 388 | **−49%** |
| Gemma 4 31B QAT | pp512 / pp4096 / pp16384 | 294 / 111 / 38 | 413 / 379 / 335 | **−87%** |
| Muse Glimmer 28B | pp512 / pp4096 / pp16384 | 523 / 375 / 275 | 540 / 483 / 430 | **−47%** |
| Gemma 4 31B, fa=0 only | pp512→16k | — | 413 → 335 | **−19%** |
| Qwen 3.8 27B, fa=0 only | pp512→16k | — | 446 → 388 | **−13%** |

Corroboration on a CUDA card, though not a clean A/B: the same RTX 5080 measured **4,983 t/s at
pp512** with Q4_K_M [172] and **3,100 t/s at 8k-token prompts** with QAT Q4_0 [161] — **−38%**.
Q4_0 is normally the *faster* format, so the true context penalty is at least that large.

Two independent lessons:

- **With FA disabled, the curve is nearly flat** (−13% to −20% to 16k). That is the signature of a
  prefill that is still weight-bound: more tokens per launch, same bytes read.
- **With FA enabled on this driver, prefill collapses** — and on Gemma it collapses to 38 t/s, a
  10x loss. The author's own conclusion: "Gemma's prefill collapse was the flash-attention
  setting, not the card" [173]. This is the single most important warning in the file: *FA is not
  monotonically good; it is good when its kernel binds and catastrophic when it half-binds.*

### 4.2 The long case — 100k of KV already resident

V100 32 GB, Qwen3.8-27B UD-Q5_K_XL, 100,000 restored tokens, q8_0 K/V, appending a suffix [186]:

| Appended | Prefill | vs pp512 (784 t/s) |
|---|---|---|
| +128 | 155.6 t/s | −80% |
| +256 | 211.1 t/s | −73% |
| +1,000 | 300.9 t/s | −62% |
| +1,000 (optimised fork) | 433.2 t/s | −45% |

This is the cleanest isolation of the effect: **same model, same card, same build — only the KV
length changed, and prefill fell 2.6-5x.** Note the shape: the penalty is largest for the shortest
append, because fixed per-request overhead amortises over fewer tokens. Any "prefill tok/s" number
quoted without its N_KV is therefore not comparable to another one.

### 4.3 The unified-memory case — the halving rule

Ryzen AI Max+ 395 (gfx1151, 256-bit LPDDR5X, 96 GB), Qwen3.8-27B UD-Q4_K_XL, ROCm, 2026-08-17 [185]:

- **~317 t/s at a 6k prompt → ~168 t/s at 35k**, author's summary: **"prefill halves per
  position-doubling"**.
- Full-262k fresh prefill is described as an **"hours-scale operation"**; practical fresh-prompt
  ceiling before TTFT hurts: **60-100k**.
- Warm-prefix TTFT 0.2-0.6 s — the KV cache holds 262k cheaply, it is the *fresh* prefill that
  costs.

This is the closest match to a 6 GB laptop's economics: weights do not fit, so the loop is bound by
a shared memory system rather than by the GPU's MMA units.

### 4.4 The mechanism, measured

The only phase-attribution I found that breaks prefill into ops at both ends of the range is
ikawrakow's LLaMA-3.1-8B Q4_K_S profile [175], CPU, 32k prompt vs 512 prompt:

| Op | Share of PP-32k | Share of PP-512 |
|---|---|---|
| MUL_MAT | 80.22% | 93.34% |
| **SOFT_MAX** | **17.81%** | **0.84%** |
| quantize / MUL / RMS_NORM / ADD / SILU / ROPE | 0.2-0.5% each | ~1% each |

And within the matmuls at 32k: `kq` 34.05%, `kqv` 25.32% — **59.4% of all matmul time is the two
attention products** — versus `ffn_out` 11.4%, `ffn_up` 11.0%, `ffn_gate` 10.3%.

[D] That is the whole fall-off in one table. Prefill FLOPs per token are `2P + 4·L·d·N`. For a 12B
dense (L=48, d=2304) that is 24.0 GFLOP of weights and 1.8 / 14.5 / 58.0 / 116.0 GFLOP of attention
at 4k / 32k / 128k / 262k — i.e. attention is **7% → 38% → 71% → 83%** of prefill math. The curve is
not a bug; it is the quadratic term arriving.

The softmax number is separate and equally important: it is a *memory* cost, not a FLOP cost. At
32k the profile moved 2,080 GiB through softmax at ~25 GiB/s, near the 30 GiB/s copy ceiling of that
CPU [175]. On a GPU the same op is cheap only because FA keeps the score tile in shared memory and
never writes the N×N matrix. **A scalar implementation that materialises scores is paying the
2,080 GiB version of this.**

### 4.5 Other things that look like fall-off

- **KV-cache type fallback.** RTX 3090, Qwen3.8-27B IQ4_NL: switching K from q8_0 to q4_1 dropped
  prefill **1,276 → 34 t/s** with decode unchanged, because `ggml_cuda_get_best_fattn_kernel`
  returns `BEST_FATTN_KERNEL_NONE` for unsupported KV types in default builds and the scheduler
  silently runs `FLASH_ATTN_EXT` **on the CPU**. A 90k prompt took 44 minutes [183]. The same build
  with q8_0/q8_0 logged 1,174-1,276 t/s. V=q4_0 showed the same: 1,174 → 195 t/s [183].
  **Directly relevant to any 6-bit KV plan: the failure is silent, and it costs 20-37x.**
  Compounding it, the CUDA MMA FA path only takes f16 KV tiles natively and routes quantized KV
  through a scratch buffer whose cost grows with context [184].
- **Slow degradation with context even without a type change.** Two independent user reports bracket
  the effect across years: a 2026 Vulkan sweep shows pp degrading from a small context increase on a
  27B Q4_K_M [199], and the 2023 report that GPU prefill collapses as past context grows while the
  CPU degrades far less [200] is the symptom flash attention was later built to fix.
- **Silent VRAM spill.** "Past your VRAM budget the Windows driver doesn't OOM — it silently spills
  to shared memory and you lose 30% with zero error messages" [171]. On a 6 GB card this is the
  default state, not an edge case.

---

## 5. The other engines, measured

### 5.1 ik_llama.cpp — a kernel-quality experiment, not a different algorithm

Same model, same quant, same CPU, mainline vs fork; the fork's advantage is entirely in the quantised
matmul [174] (LLaMA-3.1-8B, pp512, no FA, 2025-01-21):

| Quant | llama.cpp | ik_llama.cpp | Speedup |
|---|---|---|---|
| Q4_0 | 146.20 | 260.71 | 1.78x |
| Q4_K_S | 102.85 | 258.78 | 2.52x |
| **IQ2_XXS** | **43.54** | **154.17** | **3.54x** |
| IQ4_XS | 70.67 | 256.51 | 3.63x |
| IQ3_S | 28.96 | 149.98 | 5.18x |
| geometric mean, all 16 types | — | — | **2.88x** |

Context scaling, Qwen3-30B-A3B, CPU only, PP as a function of N_KV [176]: llama.cpp+FA **139.25 →
13.28 t/s** (0 → 15,872); llama.cpp without FA 141.84 → 38.13; ik 474.34 → 148.68. So at 16k the
fork is **11.2x** mainline-with-FA and 3.9x mainline-without.

Two honest caveats: (a) the fork is *not* universally faster — 70B UD-Q4_K_XL on a 3090 measured
**14.57 t/s vs llama.cpp's 123.28** [177], an 8.5x loss from a quant path the fork does not
specialise; (b) ik's own maintainer notes FA is *slower* than no-FA on CPU in both engines
(174.09 → 137.87 t/s, −26%) because fused attention bypasses the optimised `iqk_mul_mat` [175].

**Lesson for a from-scratch engine: 2.9-5.2x is available from the quantised matmul alone, at
identical FLOPs and identical memory traffic.** That is the cheapest large multiplier on the list.

### 5.2 KTransformers — CPU+GPU hybrid MoE, the 671B case

DeepSeek-V3/R1 671B Q4_K_M, 14 GB VRAM + 382 GB DRAM, dual Xeon 6454S [178]:

| Config | Prefill | Decode |
|---|---|---|
| llama.cpp, 2x32 cores | 10.31 t/s | 4.51 t/s |
| KTransformers, 32 cores | 54.21 | 8.73 |
| KTransformers, 2x32 cores | 74.36 | 11.26 |
| + AMX MoE kernel (V0.3) | 255.26 | 13.69 |
| + 6-expert selection | **286.55** | 13.69 |

That is **27.8x over llama.cpp at prefill, 3.0x at decode** — the single largest engine-to-engine gap
in this survey, on the same silicon. Its short-prompt behaviour is also instructive: V0.2.1 gives
111 / 112.5 / 102 / 101 t/s at 1k / 2k / 4k / 8k [178] — **flat to 8k**, because the CPU-side expert
matmul dominates and the attention term is a rounding error at those lengths.

Mechanisms, with the numbers attached [179][180]: AMX-specialised CPU kernels at **21.3 TFLOPS
sustained, 3.9x PyTorch native**; CUDA-graph capture cuts GPU kernel-launch overhead from **>20% to
~0**; expert deferral overlaps CPU expert work with GPU attention for **1.45x decode** at <0.5%
accuracy change. Vendor-authored (MadSys/Tsinghua, SOSP'25) but the prefill ladder is a measured
table with hardware specified, and the llama.cpp baseline is the same box.

**The >20% launch-overhead figure is the answer to the -5.5% offload mystery** (§7.4).

### 5.3 vLLM and SGLang on consumer GPUs

- **vLLM, RTX 4060 Laptop 8 GB, vLLM 0.24, Qwen2.5-1.5B** [189]: 39 tok/s (HF baseline) → 57
  (vLLM, `max_num_seqs=1`) → **782** (continuous batching, 32 seqs) → 1,188 (AWQ INT4). CUDA graphs
  **−10%**, speculative decoding **−63%**, chunked prefill a **no-op at 540-token prompts** (the flag
  exists to interleave prefill with in-flight decode; `vllm bench` exposes the prefill/decode cases
  separately [195]). Two
  things follow: vLLM's whole win is batching, and its optimisations are tuned for shapes a
  single-user 6 GB laptop does not have.
- **vLLM, RTX 4090, Llama-3.1-8B BF16 vs AWQ** [188]: prefill "barely improves" with 4-bit because
  it is compute-bound — AWQ helps decode 1.92x and prefill almost not at all. Kernel profile at
  batch 64: **linear GEMMs 86-89% of kernel time, attention ~9%**. This is the short-context regime
  stated numerically from the other direction: at 64×128 tokens, attention is not the problem.
- **SGLang + KTransformers** [179]: prefill up to **20x** over baselines across all prompt lengths,
  decode up to 4x; measured on dual Xeon 8452Y with A100-40GB and RTX 4080-16GB. Only a chart is
  published for the per-length prefill, so treat the 20x as [V] and the KTransformers README ladder
  [178] as the [M].
- **Consumer-GPU vLLM prefill tok/s is genuinely hard to source.** The measured-benchmark sites
  either report decode only (quantized.uk states outright: "Not covered: Prompt processing (prefill)
  speed" [190]) or report aggregate throughput with batching confounded in. That is a gap, not a
  finding — but it does mean any claim of the form "vLLM prefills at X on a 4060" should be treated
  as unsourced unless the prompt length and concurrency are given.

### 5.4 ExLlamaV2 / ExLlamaV3

- **EXL3 is the fastest prefill on consumer cards in the community dataset.** Qwen3.8-125B (6B
  active) EXL3 4.05 bpw, RTX 4090, 32,768 ctx, part in system RAM: **1,944 t/s prefill**, 33.6 t/s
  decode [192]. GLM-5.3 320B (18B active) on an RTX 3090: **844 t/s** prompt with part in RAM [191].
- EXL2 measured 235 tok/s decode for Llama-3.1-8B 4.65 bpw on a 4090 vs 148 for Q4_K_M [190] — but
  that dataset excludes prefill, so it cannot be used for this question.
- History worth keeping: a 2024 measurement set found "Exllamav2 processes prompts significantly
  faster than llama.cpp, even at full GPU offload" and that with FA, ExLlamaV2's prefill-vs-length
  curve "almost loses the quadratic component" — while llama.cpp's FA *slowed decode* on the same
  box. Both engines have since changed; the durable point is that the prefill curve is an
  implementation property, not a law.

### 5.5 What a well-provisioned machine looks like (the ceiling, for calibration)

| System | Model | Prompt | Prefill | Time | Src |
|---|---|---|---|---|---|
| RX 7900 XTX 24 GB, LexiEngine NVFP4 | Inkling-2-Mini | 4k / 16k / 262k / 524k | 2,430 / 2,332 / **1,193** / 764 t/s | 3.7 min @262k | [165] |
| Same card, llama.cpp | Inkling-2-Mini | 4k | 1,233 t/s | — | [165] |
| A100-40GB, MXFP8 | Inkling-2-Mini | 8k / 131k / 250k / 980k | 1,900 / 1,030 / **715** / 245 t/s | 67 min @980k | [166] |
| 4x V100-32GB | Qwen3.8-27B class | 262,144 ctx, 209,715-token input | **1,772 t/s** | ~2 min | [187] |
| M2 Max 64 GB, MLX W4W8 | Inkling-2-Mini | 512,444 (NIAH) | — | **90.8 min cold prefill** | [167] |

Note the 7900 XTX numbers: 2,430 t/s at 4k → 1,193 at 262k is a **51% fall over a 64x context
increase** — i.e. a well-written attention path plus 24 GB of KV residency keeps the curve almost
flat where the 6 GB card cannot. **The difference between "prefill degrades 2x at 262k" and "prefill
degrades 20x" is residency, not math.**

---

## 6. What 262,144 tokens actually costs

Anchors are [M]; the last two columns are [D] extrapolation with the stated assumption.

| Baseline | Measured anchor | Assumption | Implied 262k | Class |
|---|---|---|---|---|
| **This engine, 4050 6 GB, f32 scalar** | **9.09 t/s** (0.11 s/tok) | linear | **8.0 h** | measured |
| llama.cpp, 4060 8 GB, Inkling stock | 101 t/s @10k [170] | flat (optimistic) | 43 min | D |
| llama.cpp, 4060 8 GB, Inkling stock | 101 t/s @10k [170] | 2x per doubling past 32k | **~1.4 h** | D |
| llama.cpp, 4060 8 GB, Inkling tuned | 545 t/s @4k [162] | flat — **unattainable**, 3.1 GB KV + 12.8 GB weights cannot co-reside in 8 GB | 8 min | D, upper bound |
| llama.cpp, 4050 6 GB, 35B-A3B offload | ~292 t/s @35k [169] | flat | 15 min | D |
| llama.cpp, 4050 6 GB, 35B-A3B offload | ~292 t/s @35k [169] | halving ×3 past 35k | **~2 h** | D |
| llama.cpp, gfx1151 96 GB unified, 27B Q4 | 168 t/s @35k, halving/doubling [185] | 3 doublings to 262k | **~3.5 h** | D (author: "hours-scale") |
| llama.cpp, 3090 24 GB, Inkling Q4 | 400 t/s @32k [164] | further halving | 45-90 min | D |
| A100-40GB | 715 t/s @250k [166] | measured at 250k | **6.1 min** | M |
| RX 7900 XTX, LexiEngine | **1,193 t/s @262k** [165] | measured at 262k | **3.7 min** | M |

**The honest answer to "minutes, hours, or days": on a 6 GB laptop GPU, 15 minutes to 2 hours, with
~45 minutes the most defensible single number for a 12B-class model that mostly lives in host RAM.
Days is not implied by any baseline. And the engine's 8.0 h sits inside the plausible band — at the
slow end of it, but not in a different regime.**

That reframing matters for the project: the gap to close is **4-30x**, not 200x, and most of it is
in two kernels plus residency discipline. The 200x figure is real but it is the gap at *pp512*,
where the model fits and the comparison is pure arithmetic throughput.

---

## 7. Which kernels produce these numbers, and in what order of magnitude

### 7.1 Quantised tensor-core GEMM (MMQ) — 5.7-6.3x, and 5x for tensor cores at all

The cleanest single measurement in the survey, because *nothing* changes except the GEMM path
(RTX 5090, Qwen3.5-35B-A3B, llama.cpp b8240) [182]:

| Build | Q4_K_M pp512 | Q4_K_M tg128 | MXFP4 pp512 |
|---|---|---|---|
| CUDA 12.8 + **MMQ** | **5,611 ± 26** | 211 ± 8 | **6,566 ± 70** |
| CUDA 12.8 + cuBLAS | 989 ± 35 | 152 ± 8 | 1,050 ± 9 |
| CUDA 13.1 + cuBLAS | 772 ± 6 | 121 ± 2 | 721 ± 25 |

**5.7x on Q4_K_M, 6.3x on MXFP4, for prompt processing alone**; decode moves only 1.4x. The reason
is stated in the article and confirmed by the PR history: the cuBLAS path dequantises quantized
weights to f16 *before* the GEMM [196][198], so it pays the unpacking it was meant to avoid and then runs
a dense f16 matmul; MMQ performs the dot product with the weights still quantised, using integer
tensor-core MMA with per-block scales applied in registers [182][197].

The tensor-core contribution by itself is measured by Puget across the GeForce stack [181]: the
GTX 1080 Ti, with **zero tensor cores**, scored **5x slower** than the RTX 4060 at prompt
processing, despite having 484 GB/s of bandwidth and 11 GB of VRAM. Their conclusion — PP rank order
tracks FP16 TFLOPS/tensor-core count, TG tracks bandwidth — is the correct mental model and it is
why *prefill and decode need different kernels*.

[D] Sanity check on the ceiling: 1,828.8 t/s × 2 × 9e9 params = **32.9 TFLOP/s of Q4 MACs** on an
RTX 4060 Laptop whose vector FP16 peak is 15.1 TFLOPS [181]. Ada's dense FP16 tensor rate is 2x its
vector rate and INT8 is 2x that again on consumer parts, so the card's INT8 tensor ceiling is
~60 TOPS. The baseline is therefore running at **~54% of INT8 tensor peak** (and above its FP16
tensor peak) — which is only possible because MMQ issues integer MMA against 4-bit blocks rather
than dequantising to f16. That is what a good quantised kernel looks like. A scalar implementation
at 3.32 GFLOP/s is **0.03%** of vector peak — 1,800x below the card's arithmetic, and ~200x below
the baseline's achieved rate.

### 7.2 Flash-attention-style fused KV attention — 3.1x at 32k, 20-37x when it is absent

Measured on the same card and model [175]: GPU with FA is **3.1x faster than GPU without FA at 32k**
(and **53x** faster than the CPU at the same length). The mechanism, from the same profile: without
FA, `kq` and `kqv` are separate materialised matmuls accounting for 59% of matmul time at 32k, and
softmax moves 2,080 GiB through memory [175].

The counter-evidence is as important: on the Arc B65 Vulkan driver FA *lost* 49-87% [173], and on
CPU it loses 26% because it bypasses the hand-tuned quantised matmul [175]. **FA is a
two-orders-of-magnitude lever when it binds and a 2-10x regression when it half-binds** — the failure
mode is a kernel-selection problem, not an algorithm problem [183].

### 7.3 Fused MoE + expert placement — up to 27.8x when experts live in the wrong memory

KTransformers' 10.31 → 286.55 t/s [178] is not one kernel; it is AMX CPU GEMM (3.9x PyTorch)
[179] + fused gate/up/SiLU + expert deferral for CPU/GPU overlap (1.45x decode) [180]. On the
llama.cpp side the same lever appears smaller but real: Inkling-2-Mini on a 4060 went 308 → 545 t/s
prefill (1.77x) from requantisation plus kernel fusions plus `-ub 512 -b 2048` [162].

### 7.4 Batching and launch overhead — the −5.5% mystery, explained

KTransformers measured GPU kernel-launch overhead at **>20% of total time**, and removed it with
CUDA-graph capture [179]. vLLM measured CUDA graphs at −10% on a 1.5B model on a 4060 Laptop,
because "graph replay overhead exceeds savings" when each kernel is tiny [189].

[D] That is exactly the signature of the observed **−5.5% dense GPU offload on decode**: a decode
matvec on a 7.9 GB model at 192 GB/s is ~41 ms of memory time spread over thousands of launches of
1-5 µs each, so per-launch overhead and PCIe round-trips eat the arithmetic win. The fix is not a
faster matvec — it is **fewer, larger launches** (CUDA graphs, persistent kernels, fused
multi-matvec, or batching several matvecs per launch). Prefill does not have this problem because
one launch already covers M rows, which is why every prefill number in this file is 5-20x the decode
number from the same run.

### 7.5 What does *not* matter much

- Quantisation level, for prefill, once residency is fixed: vLLM measured AWQ as a 1.92x decode win
  and "prefill barely improves" [188]; llama.cpp's FA-on pp512 moved only +11% from a full kernel
  change [172].
- Softmax micro-optimisation (vectorised `expf`) when the score matrix is materialised — the cost is
  the 2,080 GiB of traffic, not the transcendental [175].
- Speculative decoding for prefill: it does not touch prefill, and on a small model with high
  concurrency it cost 63% of throughput [189]. It is a decode-latency tool.

---

## 8. RETROFIT VERDICT — the top 3, for a from-scratch C implementation

Ordered by (multiplier available) ÷ (implementation cost). All three are *retrofittable*: none
requires changing the checkpoint or the tokenizer.

### #1 — Quantised tensor-core GEMM with in-register dequant (MMQ-class)

**What the baselines get:** 5.7-6.3x over a dequant-then-GEMM path on the same card [182]; 8.5x over
the same machine's CPU [172]; 2.9-5.2x over a *mature scalar/SIMD* implementation from repacking
alone [174].
**What a from-scratch C engine must build:**
- One `mma.sync`-based tile kernel (m16n8k32 int8 or m16n8k16 f16) with the A-operand as the
  activation tile and B as the quantised weight block; block scale applied **after** the MMA
  accumulate, in registers, never by expanding the block to f16 first.
- A weight layout that matches the tile (super-block of 32 or 256 with contiguous scales), decided
  **at load time**, not at multiply time. This is the "runtime repacking" that produced ik's 3.5x
  [174] and it is worth more than the intrinsics.
- Three shapes only to start: fused gate+up, down, and QKV. Everything else can fall back.
- A scalar reference of each kernel plus a bit-exactness gate, because a wrong scale placement in an
  MMA kernel produces plausible-looking text and a silent quality loss.
**Realistic ceiling:** 1,500-1,900 t/s at 4k for a model that fits (the measured band [168][171]),
i.e. **~165-210x** over 9.09 t/s. **Minimum viable first step:** get *one* projection from 9 t/s to
150+ t/s and confirm the numerics — that alone is 16x and proves the layout.

### #2 — Fused attention with quantised KV that never materialises scores, plus a hard kernel-selection gate

**What the baselines get:** 3.1x at 32k on GPU versus unfused [175]; 53x versus keeping attention on
the CPU [175]; and the *negative* case, 20-37x lost, when the FA kernel silently refuses the KV type
and the op falls back to the CPU [183].
**What a from-scratch C engine must build:**
- Online-softmax flash attention over a K/V tile (e.g. 128 keys × 128 dims) with the running max and
  normaliser in registers; causal masking by block, never per-element.
- Dequantisation of the KV block **inside the load path**, into the same shared-memory tile the MMA
  reads — no scratch buffer sized by allocated context (the failure in [184]).
- Explicit support for **the exact KV type in use**. The measured trap: default CUDA builds support
  f16/q8_0/q4_0/bf16 and reject others, including mixed K/V types, and the fallback is silent [183].
  For a 6-bit KV at 12.06 KB/token this is the first thing to test, not the last — add an assertion
  that the attention op ran on the intended device, and fail loudly.
- A no-FA path that is still tile-blocked, because on some drivers FA loses [173].
**Realistic ceiling:** flat-to-2x prefill degradation to 32k instead of 5x [173][164]; at 262k, the
difference between ~45 min and ~2 h. **Minimum viable first step:** run the existing 262k prefill
with the KV read path instrumented, and confirm where the 0.11 s/token actually goes — attention vs
weight streaming. Two prior guesses at this were wrong (`ENGINE-THROUGHPUT-BASELINE.md`), so measure
before building.

### #3 — Expert-residency scheduling with overlapped streaming (the MoE-specific lever)

**What the baselines get:** 27.8x prefill over llama.cpp on the same box for a 671B MoE [178];
1.77x on a 4060 8 GB for exactly this model from requant + fusion + ubatch/batch sizing [162]; and
>20% → ~0 from removing launch overhead [179].
**What a from-scratch C engine must build:**
- A residency plan computed from free VRAM at load: which expert blocks stay resident, which stream
  per token, expressed as an explicit table rather than an `-ngl` heuristic. The measured spread
  between "stock placement" and "blk 6-15 experts on GPU" is 308 → 545 t/s on the same card [162].
- Double-buffered expert streaming: prefetch block N+1 while computing block N, on a separate
  stream, so PCIe latency overlaps matmul instead of serialising with it [180].
- A fused gate/up/SiLU/down expert kernel so an expert is one launch, not four — the launch-overhead
  argument of §7.4 applied to the part of the model that has the most launches.
**Realistic ceiling:** 1.7-3x on top of #1 and #2 for a spilling MoE; the 27.8x figure is not
transferable to a 6 GB card because it depends on AMX-class CPU throughput on the expert side.
**Minimum viable first step:** measure prefill with 0 vs 8 vs 16 expert blocks resident and fit the
curve — if it is flat, residency is not the constraint and #3 should be dropped.

### Not in the top 3, deliberately

- **Speculative decoding / MTP** — decode-only. Measured +46% decode on an 8 GB laptop [171],
  −63% aggregate when concurrency is high [189]. Irrelevant to prefill.
- **KV quantisation below 8-bit** — saves memory, costs a kernel path, and on the measured evidence
  risks a 20-37x silent fallback [183]. Do #2 before touching KV width again.
- **Chunked prefill** — a scheduler feature for interleaving with decode. Measured as a no-op at
  540-token prompts [189]. It does not raise single-stream prefill throughput.

---

## 9. Gaps and things I could not measure

- **No llama.cpp 128k-context prefill number on a 6 GB card exists**, because the KV cache does not
  fit at any usable quantisation. Every "128k on 8 GB" report in the wild is either prefix-cached
  (warm) or has the model partly on the CPU. Treat any such figure as a hybrid number.
- **The 4050-specific corpus is four reports** [169]. The 6 GB band is thin; the 4060 Laptop 8 GB
  band is the nearest well-populated proxy and is what most of §2 relies on.
- **vLLM/SGLang prefill tok/s at a stated prompt length on a consumer card**: not found. Both the
  measured-benchmark indexes either exclude prefill explicitly [190] or report only aggregate
  throughput with batching confounded [189].
- **KTransformers' per-length prefill chart** is published as an image only [179]; the numeric ladder
  in [178] is the citable version.
- **All RTX 4050/4060 Laptop numbers are single runs from community reporters**, not controlled
  repetitions. The two that report variance (llama-bench `±`) are [168] and [182]; the rest are
  point estimates, and [171] states a spread ≤0.4 t/s over 3 runs.

## 10. Ledger note

Ids 161-200 were appended to `docs/research/LEDGER.md` at retrieval time; no reset was run and no
existing range was renumbered. `bench-ledger.json` / `sources.py` are not present on this checkout,
so `LEDGER.md` is the ledger of record.
