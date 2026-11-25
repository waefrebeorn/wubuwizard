# Inkling 2 Mini — reality check against the reference class

Written 2026-11-25 after the user asked for a devil's advocate and online research. Every number below is
either sourced or measured on this box; none is estimated except where it says so.

## 1. What the reference class actually does

| engine | hardware | decode | notes |
|---|---|---|---|
| **project-maya 1.1.78** | RTX 3090 24 GB | **124.6 tok/s** decode, **2,542 tok/s** prefill | v1.1.72 measurement; v1.1.75 adds drafting, 140 -> ~200 tok/s on new code |
| **maya, 8 GB card** | 8 GB NVIDIA + 16 GB RAM | not stated | 15% of experts resident on GPU, rest pinned in ~**13.0 GiB system RAM**, **no SSD tier** |
| **atomic-engine (atomic.chat)** | Apple M5 Max | **43 tok/s at 8 GB RAM**, **84** on AD-IQ2_M, **122-143** with `dspark` | 26.4B model in 8 GB via **SSD expert streaming**, greedy output byte-identical to stock |
| OpenRouter providers | cloud | 68.8 - 197 tok/s | latency 0.56 s |
| Mac Studio 128 GB | M-series | **29.1 -> 16.3 tok/s** as context grows | 290 s to first token at 144k |
| **wubuwizard (this repo)** | **RTX 4050 6 GB, 16 GB RAM** | **0.013 tok/s** (77 s/token) | 42.8 s per 5-token forward after the fixes below |

Model: 24 physical layers, 26.4B total / **3.4B active**, 8 of 128 experts per layer, 1M context, looped 3x.

**The gap is ~9,600x against maya and ~2,000x against a laptop running the same model in 8 GB of RAM.**

## 2. The honest arithmetic

Per token, on this model: **6.52 GMAC = 13.0 GFLOP** (26.8M dense + 63.7M MoE MACs per layer, x24 layers x3
loop passes). Measured: 77 s/token.

- **0.169 GFLOP/s delivered** — **0.2% of one Zen4 core's AVX2 FMA peak** (~75 GFLOP/s), let alone the eight cores.
- Effective weight bandwidth **14.2 MB/s** against 40-60 GB/s of DDR5 — **~2,800x below memory bandwidth**.

An engine that is 0.2% of one core and 0.03% of memory bandwidth is not blocked on either. It is blocked on
**instructions per useful flop** — the inner loop is scalar, per-element, and switches on type per block.

## 3. Three hypotheses I had, all measured, all wrong

This is the part worth keeping.

| hypothesis | prediction | measurement | verdict |
|---|---|---|---|
| MoE materialises 32 MB of floats per expert (14x the ~1.5 MB of quant bytes) | large win from fusing dequant into the dot | fused vs staged: **identical output, no speed change**; 42,774 vs 47,495 ms | traffic was never the constraint |
| Dense weights re-dequantised 72x a token | cache them, big win | **61,346 -> 60,750 ms = 0.8%**, inside noise | dead |
| Reads are per row (2.2M seek+read pairs + mallocs a token) | syscall-bound, so mmap gives a huge win | **61,346 -> 42,774 ms = 1.43x** | real but only 30% |

All three were plausible, all three were stated with confidence, and together they explain **1.4x of a 9,600x
gap**. The 2.2M-reads-per-token finding is real and worth having — `in2_dequant_slice` genuinely did
`malloc` + `_fseeki64` + `fread` + `free` per row — but fixing it moved the needle by 30%, not 50x, because
stdio was already page-caching most of it.

**Conclusion: no single hotspot dominates. The whole inner loop is uniformly ~500x off peak.** That is a
rewrite, not a tuning exercise, and pretending otherwise would waste another stretch.

## 4. Assumptions that are at stake

1. **"The speedup is in the SSD -> RAM -> VRAM staging hierarchy."** At the current implementation stage this is
   false by measurement: staging changes were worth 0-30%, and the engine is 500x off compute peak. Staging
   becomes the binding constraint only *after* the kernels are vectorised. Reversing the order — kernels first,
   staging second — is the only ordering that produces a usable engine.

2. **"Verify every seam, then optimise."** The seam verification was necessary and it did find five real bugs
   (IQ2_XS packing was mine, the `.scale` tensors were load-bearing, the message gate, the dead-code attention
   path, the prompt being part of the instrument). But 300 verified seams produced an engine at 0.013 tok/s,
   which has no user. Verification needs a **performance floor gate** alongside it — e.g. ">= 1 tok/s at a
   20-token prompt" — or correctness is measured against nothing.

3. **"wubuwizard competes with maya on tok/s on this box."** It cannot, and the hardware says so: maya's Inkling
   path states a **floor of one 8 GB NVIDIA GPU**, and this box has **6 GB**. The engine built for this shape of
   machine is **atomic-engine**: a hard RAM budget (`--engine-budget-mib`), experts streamed from SSD, and a
   ~0.5 GB `q8_0` drafter for speculative decoding — 43 tok/s in 8 GB of RAM. That, not maya's full-GPU path, is
   the correct reference for a 6 GB / 16 GB machine, and it is exactly the hierarchy the standing goal describes.

4. **"Hand-writing the kernels is the way to close 500x."** Maya achieves 124.6 tok/s with purpose-built
   `ik2` CUDA kernels; this engine's scalar C is 500x off one core. Closing that by tuning is not credible.
   The options are (a) absorb maya's kernel design, which is the sanctioned role of the
   `inference-engine-absorption` skill, or (b) write AVX2 integer-dot kernels from scratch. Both are real work;
   only (a) has a demonstrated endpoint.

5. **"Fusing dequantisation into the dot is strictly better."** I believed it enough to make it conditional on
   `n_tok == 1`, predicting staged would win for prefill because it dequantises each expert once for all
   positions. Measured: **staged 47,495 ms vs fused 42,774 ms** — the prediction was wrong in direction. Rule
   going forward: A/B every kernel change at **both** `n_tok = 1` and `n_tok > 1`, because prefill and decode
   have different cost structures and I cannot predict which way it falls.

## 5. What to do, in order

1. **Quantised dot kernels** — never materialise floats for a weight. Integer activations x LUT, AVX2, gated on
   logit equivalence. This is the only route past ~1 tok/s on CPU, and it is what "make it in the quant kernels"
   actually requires. Target: 50-100x on the expert matvec.
2. **KV cache** (`docs/INKLING2-KV-CACHE.md`) — generation is O(n^2) today; every reference engine caches. Gate:
   cached step must match full recompute, ids identical, `max|dlogit| < 1e-4`.
3. **GPU offload of the dense weights and hot experts** into the 6 GB — the user's rule that CPU-bound work
   belongs on the GPU is empirically right: 0.2% of one core is the whole story.
4. **Expert streaming with an explicit budget**, atomic-engine style — only after (1)-(3), because streaming a
   fast kernel helps and streaming a slow one does not.

## 6. What is genuinely proven and should not be re-litigated

- The port is **arithmetically correct**: templated prompt -> `<|content_thinking|> The user`, logits 21.29 /
  20.50 / 20.59, and every seam checked is bit-exact or within 1e-6.
- IQ2_XS decoding is canonical in this file (`ls = 13`, grids 8/25, `db = 3.375d`).
- The `.scale` companions are load-bearing (corr 0.48 -> 1.0000000) and the loop gate uses **separate** norms.
- The 8.4 GB file maps fully into this box's 16 GB, so RAM-resident weights are available without a streaming
  tier — SSD streaming on *this* machine is for headroom, not necessity.

The engine is correct and 500x slow. Those are independent facts and only the second one blocks use.

## 7. Kernel results (measured, C11 + AVX2 intrinsics)

All of these are C11 with `immintrin.h`, `__attribute__((target("avx2,fma")))` and a
`__builtin_cpu_supports` dispatch, so the rest of the binary stays baseline. No C++ - the project rule is C11
only, so maya's `iq_avx2.cpp` is a source of *algorithms*, never a translation unit to link.

| change | gate | result |
|---|---|---|
| `dot_row_q2_0` AVX2 (broadcast quant word, `srlv` by 2j, mask, `fmadd`) | 64 random block sets | **12.64x** on a 2304-element row (0.028 s vs 0.354 s / 120k), worst rel err **5.934e-08** |
| same, end to end | logits | 42,774 -> **38,475 ms**, `151451 / 29.6508` identical |
| fused **scalar** IQ2_XS dot | A/B twice | **SLOWER**: 48,014 / 48,121 ms vs 42,230 / 42,227 ms. Kept behind `IN2_FAST`, default off |
| IQ2_XS **int8 AVX2** (`maddubs` of grid magnitudes against `sign(act_i8)`) | A/B twice | 38,460 / 38,399 -> **35,800 / 35,738 ms**; top1 identical, logit 29.6508 -> **29.6436** (int8 activation) |
| both kernels, default | logits + wall clock | **35,655 ms** from 42,774 ms = **1.20x** |

Two lessons that cost real time here:

1. **Skipping the float buffer is not a speedup.** The scalar fused IQ2_XS dot removed the 2304-float row and
   was 14% slower, because it serialised eight dependent `double` adds per 32-element group while the
   materialised dot interleaved two accumulators over a contiguous row.
2. **The amortisation is the win, not the kernel.** Quantising the activation to int8 *once per matvec* and
   reusing it across all 2,304 rows of the expert is what made the vector path pay; a per-row quantisation
   would have thrown that away.

Still uncovered by a vector kernel: IQ2_XXS (16, blk.5's gate/up), IQ1_S (19), IQ2_S (22), NVFP4 (40). Those
still materialise floats, which is why 1.20x and not 5x. The 899-line C++ kernel set covers 16/17/18/22/23 and
is being translated 1:1 into C11 (`src/wubu_iq_c11.c`) rather than re-derived.
