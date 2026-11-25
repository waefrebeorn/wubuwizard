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

## 8. Second reality check (2026-11-25, after the kernel stretch)

Kernel work took the 5-token forward from 42,774 ms to **8,021 ms (5.33x)**. That is real, and it is also
still ~30x off what this class of hardware does. Sourced comparisons, all in the same units:

| setup | decode | note |
|---|---|---|
| **RTX 4060 8 GB**, llama.cpp, UD-Q3_K_XL, 12 expert layers on CPU | **17.11 tok/s**, 101 tok/s prefill @10k | closest match to this box |
| **8 GB NVIDIA**, project-maya | **~41 tok/s** | hot experts in VRAM, rest in ~13 GiB RAM |
| **M5 Max 64 GB**, atomic engine, 8 GB budget | **43 tok/s** | SSD expert streaming |
| **RTX 3090**, maya | 128 tok/s, ~200 with DSpark | 24 GB, everything resident |
| **RTX 3060 6 GB + 32 GB RAM**, llama.cpp hot-experts fork | works | 6 GB is viable with expert residency |
| **this engine, RTX 4050 6 GB + 16 GB RAM** | **~0.5 tok/s** (1.7-2 s/token) | CPU only, no KV cache |

**Gap to the realistic ceiling on this box: ~30x.** Not 9,600x as before the kernels, and not 1x.
The remaining gap is not arithmetic, it is architecture:

1. **No KV cache.** Every generated token re-runs the entire prefix through 24 layers x 3 passes, so decode is
   O(n^2). Every engine in the table caches. This is the single largest term and it has been deferred three times.
2. **Three loop passes.** Documented cost is ~2.5x decode latency and half the KV capacity; the middle 12 layers
   run three times. My cache must therefore be keyed per pass, not per layer - the literature is explicit that
   every loop writes new KV entries, so a naive per-layer cache is wrong, not merely slow.
3. **No GPU.** The RTX 4050 sits idle through all of this. The CPU path delivers ~0.5 GFLOP/s effective against
   tens of TOPS available on the card. No amount of further C11 kernel work closes a 30x gap that the reference
   class closes by putting hot experts and dense weights in VRAM.

### Assumptions now at stake

- **"The kernels are the lever."** They were worth 5.33x and they are exhausted as a lever: 5.33x of the 30x,
  and what remains is algorithmic and offload. Continuing to tune CPU kernels is optimising the small term.
- **"Match maya on this box."** Maya's floor is one 8 GB card; this box has 6 GB. The honest target is the 4060
  figure (17 tok/s), not maya's 41.
- **"1:1 parity with ggml numerics is the goal."** Parity means our numbers equal llama.cpp's - and llama.cpp on
  a 4060 is 30x faster with the same numbers. Parity was necessary to trust the port; it is not the product.
- **"int8 activation quantisation is a small change."** It changes greedy output. Every engine in the table
  accepts it, so it is the correct contract - but it must be stated as the engine's contract, not treated as
  a bug to be chased.
- **"Looping is free."** It is not: 3 passes is ~2.5x the work of a 24-layer model at equal stored depth, and the
  KV cache has to pay for it three times over.

### Where the next 30x is

1. **KV cache** (`docs/INKLING2-KV-CACHE.md`, written but unimplemented) - removes the O(n^2) recompute.
2. **GPU offload of dense + hot experts** into the 6 GB - the card is idle, and every fast reference does this.
   The C11 kernels then become the cold-expert CPU fallback, which is exactly the SSD -> RAM -> VRAM hierarchy
   the standing goal describes.
3. Only then, more kernel work - on the attention core, which is still scalar `double`.

## 9. KV cache lands: O(n^2) -> O(n), exact (2026-11-25)

| prefix | recompute | incremental | speedup |
|---|---|---|---|
| 1 -> 2 | 3.3 s | 2.0 s | 1.7x |
| 3 -> 5 | 5.8 / 7.2 s | 2.1 / 2.0 s | ~3x |
| 19 -> 21 | 25.7 / 25.8 s | 28.2 prefill, then 2.2 / 2.3 s | **11.7x and widening** |

Byte-identical output at every prefix: `61840/19.2437` (1+1), `26.9567 / 13.5936 / 29.3580` (3 steps),
`150126/20.0673` (19-token prompt). Not "same tokens" - the same logits to the last printed digit.

What it took, in the order the failures taught it:
1. Four convs per layer, not two. K and V were cached; the residual conv after attention and the MoE conv were
   still padding with zeros on a 1-token step.
2. Two buffer sizings (`logit`/`alpha` at `n_tok` while `n_key` becomes `pos0+t+1`), both of which present as a
   silent death rather than an error.
3. A visit index advanced before attention read it, so a layer attended to the next visit's empty rows.
4. **The message anchor.** `in2_messages` reads only position 0 of `prev_h` and of each block state. On a step the
   batch is the new token, so pass 2 was conditioned on token N instead of token 0. Caching the pass-boundary
   anchor closed the last gap and turned every number above exact.

Decode is now flat cost per token regardless of prefix length. The remaining per-token cost at 2.2 s is the
projections themselves (and the 463M-MAC LM head), which is where GPU offload belongs - not more cache work.

## 10. Where a decode step actually goes (2026-11-25)

`IN2_TIMETY` accumulates per-type kernel time and MACs. One decode step, all 12 types active:

| type | calls | time | share | throughput |
|---|---|---|---|---|
| 40 NVFP4 | 120 | **0.200 s** | **50.6%** | **1.59 GOPS** |
| 16 IQ2_XXS | 136 | 0.037 s | 9.4% | 19.5 GOPS |
| 17 IQ2_XS | 128 | 0.037 s | 9.4% | 18.4 GOPS |
| 42 Q2_0 | 264 | 0.025 s | 6.3% | 28.0 GOPS |
| 8 Q8_0 | 169 | 0.021 s | 5.3% | 34.6 GOPS |
| 22 IQ2_S | 56 | 0.027 s | 6.8% | 11.0 GOPS |
| 19 IQ1_S | 64 | 0.018 s | 4.6% | 18.9 GOPS |
| 23 IQ4_XS | 25 | 0.018 s | 4.6% | 25.7 GOPS |
| 14 Q6_K | 42 | 0.004 s | 1.0% | 49.4 GOPS |
| 12/13/18 | 29 | 0.008 s | 2.1% | 10-25 GOPS |

Two findings, both actionable and neither of them the thing I had assumed:

1. **NVFP4 is 12x off the pack.** 1.59 GOPS against 19-49 for every other type, and it alone is half the
   kernel time while being 1 of 12 formats. This is a kernel defect, not a memory limit - the other types read
   the same mapping at 20-50 GOPS.
2. **The kernels are only 18% of the step.** 0.395 s of 2.2 s. The remaining ~1.8 s is outside the matvecs
   entirely: per-visit norms, the mixing, and `in2_tensor_to_f32` calls for the small per-layer tensors
   (q_clip, the conv weights, rel_clip, rel_bias, sink, the message gates) that were never converted to the
   batched path. 48 visits x several tensors each, every one a fresh allocation and dequantisation.

So the next two moves are, in order: fix NVFP4 (half the kernel time, one function), then kill the per-visit
small-tensor materialisation (the 1.8 s that is not kernels at all). GPU offload is not the answer to a step that
spends 82% of its time outside the arithmetic.

### Phase split (IN2_TIMETY, one decode step)

```
PHASE layers=1.439 s   moe=0.000 s      (step total 1.9 s)
TY   ... sum of all kernels = 0.395 s
```

Layers hold 1.44 s; the kernels inside them are 0.395 s; so **~1.05 s is inside the layers and is not
arithmetic at all**. With DQ-cache on, the step drops 1.9 -> 1.6 s, which removes part of it (the small per-layer
tensors) but leaves the larger part.

The largest remaining candidate, and the one the numbers fit: every expert matvec copies its whole quantised
matrix out of the mapping before the kernel sees it - `in2_matvec_nt`/`in2_matvec_quant` both do
`wubu_inkling2_read(ctx, t, whole, rb*n_out)`. Per visit that is 8 experts x 3 matrices x ~1.5 MB = ~36 MB, and
over 48 visits ~1.7 GB per token of pure copying, on data that is already mapped and never modified.

Next step: pass the mapped pointer straight to `wubu_iq_rows` instead of copying into `whole`. The kernels take a
const row pointer and a row stride, so this should be a plumbing change, not an arithmetic one - and it is worth
~1 s per token if the copy is what it looks like.

Not the answer: GPU offload (82% of the step is outside arithmetic), more kernel tuning (kernels are at 19-49 GOPS),
and the LM head (0.05-0.1 s).

## 11. Measurement methodology correction (2026-11-25)

Same binary, same 3-step decode, three consecutive runs:

```
run1  gen[1]  2.0 s
run2  gen[1]  1.5 s
run3  gen[1]  1.2 s
```

**1.7x spread on identical work.** The model file is 8.4 GB and is memory-mapped; the first run after a rebuild
faults pages in from disk, later runs hit the page cache. Every timing in this document was taken from single runs,
so individual deltas of 10-30% are **not trustworthy** as evidence of an optimisation - several of them may have
been cache warming (and several real ones may have been masked by it).

What this does not affect:
- Correctness. Every logit and token comparison is deterministic and unaffected by timing.
- The large results: O(n^2) -> O(n) (11.7x at prefix 20), kernels 42,774 -> ~4,000 ms prefill, the cache being exact.
  Those are far outside the noise band.

What it does affect: the small deltas - malloc reuse (2.1 -> 2.0), mapping-direct (1.9 -> 1.7), the AVX2 matvec
(1.7 -> 1.6), the DQ-cache default (4.0 -> 3.7 prefill). Any of those could be partly or wholly noise.

Rule going forward: warm the file (one full run), then take the **median of at least three** runs, and only claim a
delta larger than the observed spread. Fix the harness before optimising further - chasing a 10% win against a 70%
noise floor is how time disappears.

Steady state at prefix 4 is ~1.2 s/step, not the 2.0 s a cold single run reports.

### Trustworthy baseline (tools/bench.py, warm + median of 4)

```
step       median      min      max   spread
0           3.70s    3.70s    3.70s      0%      <- prefill of a 3-token prompt
1           1.20s    1.20s    1.20s      0%
2           1.10s    1.10s    1.10s      0%
```

Once the file is warmed the spread collapses to zero, so the earlier 2.0 / 1.5 / 1.2 s triple was entirely
page-cache state. **The real numbers for this engine are 3.70 s prefill and 1.1-1.2 s per decode step**, and
`tools/bench.py` is now the only thing that should produce a timing claim: it warms once, runs N times and prints
median/min/max/spread, so a delta inside the spread is visible as such instead of being reported as a win.

Every delta recorded before this harness must be re-measured through it before it is believed.

### Cache speedup, measured through the harness (0% spread)

| prefix | recompute | incremental | speedup |
|---|---|---|---|
| 4 | 3.40 s | 1.20 s | 2.8x |
| 5 | 4.30 s | 1.10 s | 3.9x |
| 20 | 25.8 s | 2.2 s | 11.7x |

All medians with zero spread, all producing identical logits. The speedup is linear in prefix because one path is
O(n) per token and the other O(n^2) - which is the whole point: the ratio is not a tuning result, it is the
complexity change, and it keeps growing with context.

### NVFP4 UE4M3 table: the one win that survived (2026-11-25)

| | before | after |
|---|---|---|
| NVFP4 throughput | 1.59 GOPS | 7.96 GOPS |
| prefill (3 tok) | 3.70 s | **3.00 s** |
| decode step | 1.20 s | **1.00 s** |
| parity vs ggml AVX2 | 144/144 | **144/144, max|diff| 0.000e+00** |

Cause: UE4M3 sub-block scales decoded with `ldexpf` plus a division, four times per block, ~166,000 libm calls per
matvec. Replaced by a 256-entry table filled once with the identical expression, so every entry is bit-identical by
construction and the parity sweep confirms it rather than assuming it. First optimisation since `tools/bench.py`
existed that clears the 0% spread and is therefore stated as a real win.

Tried and reverted after measuring, all under the harness:
- `in2_mix` dot vectorised: 1.20 s before and after - no change.
- `in2_tensor_borrow` (hand callers the cache buffer instead of a copy): 3.00/1.00 s unchanged **and** it leaks
  when `IN2_NO_DQ_CACHE` is set, because the returned buffer is caller-owned in that path and the call site no
  longer frees it. No measured gain plus a real defect is not a trade worth keeping.

### The mix paradox (2026-11-25, unresolved)

`IN2_TIMETY` says the `mix` phase is **81.5% of layer time** (0.520 s of 0.638 s per step). Everything that phase
contains, measured:

- `in2_tensor_to_f32(&L->attn_res_q)`: **0.000 s over 40 calls** (instrumented directly; the DQ cache makes a
  2304x2304 look-up free)
- `in2_tensor_to_f32(&L->attn_norm)`: a 2304-float vector
- three `calloc`s of n_tok*d
- `in2_mix` once per position: n_blocks+1 rmsnorms of d and **one** query dot of 39K MACs

By arithmetic that is tens of microseconds per visit, not 26 ms. Yet the phase timer says 26 ms. Either the timer is
mis-attributing (something inside `in2_mix` blocks on first touch of a large allocation, or the phase boundary is
not where the label says), or `in2_mix` is doing far more work than its source suggests.

**Do not paper over this with another guess.** The next step is a sampling profiler (or a `perf`-style
post-mortem of the running process) rather than more reasoning from source - three of the last four hypotheses about
this engine measured wrong, and the phase split has now disagreed with the arithmetic twice.

What is solid regardless: total layer time is 0.64 s, kernels inside it sum to 0.32 s, moe is 0.26 s, and the
engine does 1.00 s per decode step with the cache, exact.

### RETRACTION: "mix is 81% of layer time" was a timer bug (2026-11-25)

`_tk = clock()` was placed at line 798 while the first `TKPH(0)` fires at line 745 - before the origin was set. So
the "mix" phase was charging each layer with the **previous layer's uninstrumented tail**, and reported 0.517 s of
0.638 s for code that measures **0.000 s** when instrumented directly (`in2_mix` 0.002 s, `arq` load 0.000 s over
40 calls, `mixh` calloc 0.000 s).

With the origin moved to function entry, the true split of 0.633 s of layer time is:

| phase | time | share |
|---|---|---|
| moe | 0.255 s | 40.3% |
| tail (residual, block banking, mlp_input mix, post-mlp conv - not instrumented) | ~0.21 s | ~33% |
| k+v+conv | 0.091 s | 14.4% |
| q | 0.045 s | 7.1% |
| gate+wo | 0.021 s | 3.3% |
| attn | 0.005 s | 0.8% |
| rel+sink | 0.005 s | 0.8% |
| mix | 0.002 s | 0.3% |

Every conclusion drawn from the previous table is withdrawn: the gates were never the hot spot, and the two
`in2_mix` experiments that "showed no gain" were measured against a phase that did not contain the code being
changed. They were reverted for other reasons (one did not build, one leaked), which stands, but their timing
evidence was meaningless.

Real targets now: **moe (40%)** and the **unmeasured tail (33%)**. The tail needs its own tick before anything else
can be said about it.

### The missing third is allocator traffic (2026-11-25)

With the timer origin fixed AND a tick added for the post-mlp tail, the phases sum to **0.432 s of 0.640 s** of layer
time, and the tail itself measures **0.000 s**. So 0.208 s - a third of the step - sits inside the layer call but
after the last tick, and the only code there is the teardown:

```
free(an); free(xn); free(qcl); free(q);
free(kc); free(vc); free(kraw); free(vraw); free(kt); free(vt); free(k); free(v);
free(rcl); free(P); free(rs_all); free(bias); free(sink);
free(attn_out); free(alpha); free(logits); free(gate); free(ao);
free(pc); free(res); free(fn); free(moein); free(moeout); free(pm); free(mo); free(mixh);
free(xkk); free(xvv);
```

~25 alloc/free pairs per visit, 40 visits per decode step, several of them hundreds of KB.

Fix specified but NOT landed: a bump arena reset per layer visit. First attempt rewrote the function body with a
regex that cut the function boundary at the wrong brace and broke the file; reverted rather than left half-done.
The next attempt should do it by hand, function by function, starting with the sliding layer only.

Expected: ~0.2 s of the 1.0 s step, i.e. ~20%, if the allocation traffic is what it looks like - and this time the
measurement that motivates it is a phase total that does not add up, not a guess.

### What the remaining costs mean (2026-11-25)

The layer split is now fully accounted for - every phase plus the teardown sums to the layer time:

| item | time | share | nature |
|---|---|---|---|
| moe | 0.264 s | 41% | **memory-bound on expert weights** |
| allocator teardown | ~0.208 s | 33% | ~25 alloc/free pairs per visit - fixable with an arena |
| k+v+conv | 0.093 s | 15% | kernel-bound (types 16/17 at 12-19 GOPS) |
| q | 0.049 s | 8% | kernel-bound |
| everything else | 0.026 s | 4% | attn, rel/sink, gate/wo, mix, tail |

MoE arithmetic: 8 experts x 3 matrices per visit for the selected experts, 40 visits per decode step. It lands at
roughly 10 GOPS effective where the same kernels measure 12-30 in isolation, which is the signature of waiting on
memory rather than on the ALUs - the expert matrices are ~1.5 MB each and every visit reads a different 8 of them,
so each token streams a few GB through a memory-mapped file with no reuse.

That is the same conclusion the reference-class research reached from the other end: atomic-engine gets 43 tok/s in
8 GB of RAM by controlling which experts are resident, and maya pins hot experts in VRAM. **The next real lever for
this engine is expert residency (VRAM/RAM tiering), not more arithmetic work** - the kernels are already 12-30 GOPS
and the file is mapped.

Two scripted attempts at the arena refactor broke the function (regex and brace-counted rewrite both cut past the
closing brace). Not trying a third scripted pass: it needs hand edits at ~25 call sites, and that is a dedicated
piece of work, not something to squeeze in behind other changes.

### The engine is single-threaded, and that is now the largest structural gap (2026-11-25)

MoE moves ~36 MB of expert weights per layer visit (8 of 128 experts, ~1.5 MB each) and there are 40 visits per
decode step, so **~1.4 GB per token of expert streaming at ~5.4 GB/s** - which is single-core memory bandwidth, not
ALU throughput. The kernels measure 12-30 GOPS in isolation.

This box has 8 cores. Nothing in this engine uses more than one of them. The expert dots are independent by
construction: 8 selected experts, each producing its own contribution to the sum. A thread pool splitting those
8 ways is the most direct 4-6x available, and it needs no VRAM, no new kernels and no change to the arithmetic -
the same per-expert code, on different experts, summed at the end.

Same argument applies to the dense row ranges in `in2_matvec_nt` (each row is independent) and to the LM head.

Order of work, by measured value:
1. **Threads over experts** (moe 0.264 s -> ~0.06-0.08 s on 8 cores) - ~20% of the step, no arithmetic change.
2. **Arena for the layer buffers** (0.208 s, 33%) - 25 call sites, hand edits, must be done carefully; two scripted
   attempts already broke the function.
3. Expert residency (VRAM/RAM tiering) - the reference-class answer, but it is a project and the CPU side is not yet
   using the hardware it has.

I am not starting (1) at the end of this stretch: it touches the MoE accumulation path and I would rather land it
with room to test than hand over a half-threaded engine. Both remaining items are specified above with the numbers
that justify them.

### CORRECTION: the MoE is not memory-starved, it is just kernel volume (2026-11-25)

I wrote that moe "lands at ~10 GOPS where the same kernels measure 12-30 in isolation, which is the signature of
waiting on memory". The per-type table from the same run contradicts that, and I should read my own data before
asserting a cause:

| type | share of kernel time | throughput |
|---|---|---|
| 16 IQ2_XXS | 19.2% | 11.8 GOPS |
| 17 IQ2_XS | 15.5% | 13.9 GOPS |
| 42 Q2_0 | 14.5% | 15.2 GOPS |
| 40 NVFP4 | 12.6% | 8.0 GOPS |
| 19 IQ1_S | 8.2% | 13.1 GOPS |

Those five ARE the MoE's gate/up/down kernels, and they total ~0.20 s of the 0.264 s moe phase - **76% kernel, at the
same 8-15 GOPS the type table reports in isolation**. So the MoE is not waiting on memory; it is simply the largest
volume of work going through kernels that are themselves slow. My "memory-bound, so fix residency" conclusion was
wrong, and it would have sent the next step at VRAM tiering to solve a problem that is arithmetic throughput.

What this actually implies:
- The lever for moe is **kernel throughput on IQ2_XXS / IQ2_XS** (11.8 and 13.9 GOPS), which together are 35% of
  kernel time. Q6_K reaches 49 GOPS and Q8_0 34 - so the headroom exists in the same file.
- Expert residency stays valuable for VRAM, but it is not the answer to the 0.264 s, and it should not be next.

This is the fourth hypothesis in this stretch that measurement has overturned (mix hot spot, allocator as the
whole of the tail, threading as a win, moe as memory-bound). The pattern is consistent: read the per-type and
per-phase numbers before naming a cause.

### IQ2_XXS/IQ2_XS throughput is a property of the format, and further speed breaks 1:1 parity (2026-11-25)

`dot1_iq2_xxs_*` is a faithful translation of ggml's AVX2 kernel: per 32 elements it does four grid lookups through
`tab4`, eight sign-vector lookups (or `iq3_neg` parity expansion depending on the variant), two `sign_epi8`, and two
`madd`. Compare Q6_K, which is straight scaled multiply-add and reaches 49 GOPS. The IQ2 family is slow because the
format requires table indirection per element, not because the port is bad.

That closes the obvious avenue: **getting more out of 16/17 means computing something other than what ggml computes**
- pre-expanding the 256-entry grid into a wider table, or replacing the lookup with a shuffle scheme, changes the
  arithmetic, and the whole gate on this engine is "144/144 bit-identical to ggml's AVX2 dot, max|diff| 0.000e+00".

So the honest position:
- The kernels are at format-parity. There is no free 2-3x sitting in them without leaving that parity.
- If a faster IQ2 path is wanted, it needs its own gate: a tolerance-based comparison (exact top-1 and a bounded
  logit delta) rather than bit-identity, decided deliberately and recorded, not slipped in under the existing one.

Remaining safe work, in order of measured value and risk:
1. **Arena for the layer buffers** (0.208 s, ~20% of a step) - pure allocation removal, no arithmetic, no parity
   impact. 25 call sites, hand edits; two scripted attempts already failed.
2. **Persistent thread pool** (moe 0.264 s) - per-call spawning measured zero gain; a pool is the only version worth
   building, and it must be pooled before it is enabled.
3. Anything that changes the IQ2 arithmetic - only behind its own gate.

### The arena gives nothing, and the reason matters (2026-11-25)

Built it properly the third time (name-based matching, 23 buffers converted, build clean, logits identical) and it
measured **exactly nothing**: 3.00 s prefill, 1.00 / 0.90 s steps, identical to before.

The reason is in my own implementation: `in2_arena_alloc` zeroes with `memset`, while `calloc` for large blocks gets
**lazily zeroed pages from the OS** - the kernel maps them zero and only faults them in on first write. So the
allocator was never doing the work; the arena replaced cheap bookkeeping and added eager zeroing on top. That also
explains why the earlier malloc-reuse and mapping changes each measured small: this engine is not allocator-bound.

Consequence: the 0.208 s between the last phase tick and the layer return is **not allocation**, and my "teardown is
a third of the step" claim was another measurement artefact - the tick was placed before the teardown, so the teardown
was never actually timed; what sits there is the post-mlp conv and the residual add, which the TKPH(8) tick placed
*differently* in the two builds.

Reverted, tree clean, engine unchanged at 3.00 s prefill / 1.00 s step.

State of play after this stretch, with every number measured through tools/bench.py:
- decode 1.00 s/step, 0.90 s at prefix 5; prefill 3.00 s; logits exact; parity 144/144 on all formats
- kernels: 8-49 GOPS by type, IQ2 family limited by format, NVFP4 fixed by a UE4M3 table
- moe is 41% and is kernel volume, not memory stalls (corrected)
- the layer's own accounting now sums to 0.432 s of 0.640 s, and the remaining 0.208 s is real work in the
  untimed block, not overhead

### Stop the tick method: it has produced three contradictory answers (2026-11-25)

Sequence of measurements on the same span of the same build:
1. "mix is 81% of layer time" - artefact of the origin being set after the first tick fired.
2. "the 0.208 s is the allocator teardown" - artefact; the tick sat before the teardown, so it was never timed,
   and the arena then measured exactly zero.
3. Now: `tailconv` (post-mlp conv + residual, timed directly) = **0.000 s**, `in2_mix` = 0.000 s, `arq` = 0.000 s,
   `calloc` = 0.000 s - while the phases still sum to 0.430 s of 0.642 s of layer time.

Every directly-instrumented region reports zero, and the phase table insists a third of the layer is somewhere. One of
the two is lying and I cannot tell which from outside. `clock()` on Windows/MSYS counts process CPU time at a
coarse granularity, which is a plausible reason a 5 ms region reads as 0.000 - but that is a hypothesis, not a
measurement, and I have already been wrong four times in this stretch by preferring hypotheses.

**The next step is a sampling profiler** (or a coarse many-iteration loop that amplifies the span until `clock()` can
see it), not another tick. Repeating tick placement a fourth time would be the same mistake with a different label.

Where the engine actually stands, all through tools/bench.py at 0% spread:
- prefill 3.00 s (was 42,774 ms at the start of the kernel work)
- decode 1.00 s/step, 0.90 s at prefix 5, exact logits, vs 2.60 s recompute at prefix 4 and 25.8 s at prefix 20
- all 12 formats 144/144 bit-identical to ggml; NVFP4 1.59 -> 7.96 GOPS
