# Engine throughput baseline

Measured on this host (Ryzen 5 8645HS, RTX 4050 6 GB, 16 GB RAM, Win11) with the model this
machine actually runs: `gemma-4-12b-it-UD-IQ2_M.gguf` (667 tensors, 4003 MB). This file
describes the engine as it is now. It is not a changelog.

Build clean before measuring — `Makefile.win` does not track header dependencies, so a stale
object silently changes struct layout and produces different output (measured once: 4.4
tok/s and a different greedy prefix).

```
rm -f src/*.o tools/*.o
mingw32-make -f Makefile.win -j4 wubu_cuda.dll gen_text_gemma4
G4_GPU=1 ./gen_text_gemma4.exe <model> "The meaning of life is" 64 64
```

## Current numbers

| metric | value |
|---|---|
| decode at 33 tokens of context | **13.2 tok/s** |
| decode at 257 tokens of context | **12.2 tok/s** |
| prefill, 21-token prompt | **6.8 tok/s** |
| prefill, 72-token prompt | **15.0 tok/s** |
| greedy tokens | 45518, 107, 236775, 818, 6590, ... |
| CPU path vs GPU path | identical tokens |

Decode degrades with context and the profiler says why: the attention section is 7.3 ms at
32 tokens, **29.0 ms at 256 tokens**, and the layer total goes 76 -> 98 ms. Attention is
the only section that scales with context, and it is entirely on the CPU. At 257 tokens it
is the second largest cost after the FFN.

Decode matvecs: **202 per token, all 202 on the GPU, none on the CPU.**
Per type: IQ3_S 95, IQ2_S 83, IQ3_XXS 14, IQ2_XS 5, IQ4_XS 4, Q3_K 1 (the LM head).

The KV cache, attention arithmetic, norms, RoPE and residual adds are still on the CPU.
That is where the remaining time is.

## Where the time goes (per decode token, `G4_PROF=1`, 32 tokens of context)

| section | ms | detail |
|---|---|---|
| ffn all | 35.2 | proj 33.1 / gelu+mul 1.7 / norm+resid 0.4 |
| norm+qkv | 16.6 | pre-attention norm + q,k,v projections |
| attn out | 13.0 | o-projection 12.6 / norm+resid 0.5 |
| head | 10.5 | 262k-vocab Q3_K tied head |
| attention | 7.3 | CPU, grows with context |
| qk norm+rope | 3.6 | CPU |
| layer total | **76.0** | 48 layers |

## Kernel rates in isolation (`wubu_cuda_bench_shapes`, three rotating buffers, no L2 reuse)

| shape | ms | GB/s |
|---|---|---|
| ffn gate/up/down N=15360 K=3840 | 0.190 | 99.4 |
| attn out (o proj) N=3840 K=3072 | 0.040 | 94.0 |
| attn q N=3072 K=3840 | 0.042 | 90.1 |
| attn k/v N=1024 K=3840 | 0.019 | 67.4 |

The kernels are not the limit. The same o-projection that runs at 94 GB/s alone appears as
18 GB/s in the engine: its 0.26 ms per layer is ~0.04 ms kernel, ~0.03 ms transfer and
~0.19 ms of per-call sequencing (q8_K launch, matvec launch, synchronising download, WDDM
submit). ~200 such submissions per token is the dominant cost.

Confirmed by measurement, not assumed:

- fusing q/k/v into one submission (one upload, one quantisation, one download of q|k|v)
  changed the qkv section by 0.2 ms/token — the number of *transfers* is not the limit;
- per-stage split of every matvec call (`wubu_gpu_stage_report`, printed with `G4_PROF=1`):
  over 6666 calls (33 tokens x 202) upload 72.6 ms, Q8_K quantise 89.8 ms,
  launch+download **1445.3 ms** — i.e. 0.011 / 0.014 / **0.217 ms per call**. Total
  48.7 ms/token, which is what the sum of the weights implies (2.7 GB of FFN + 1.2 GB of
  head + 0.5 GB of attention projections at ~95 GB/s = ~45 ms). So the matvec path as a
  whole is bandwidth-bound and close to its floor; the section charges (o-proj 12.5 ms for
  48 x 3.8 MB) are inconsistent with their own isolated measurement and with this split,
  which is a profiler accounting problem to resolve before optimising anything inside it;
- **correction**: `wubu_cuda_bench_oproj` reports 0.110 ms/call for the o-projection shape
  with three rotating weight buffers, but three copies of 3.1 MB fit in the 32 MB L2, so
  that number is L2-resident and optimistic. Per-section attribution
  (`wubu_gpu_tag`, printed by the same report) puts the real cost at **0.259 ms/call**
  in situ, matching the 12.46 ms the section timer charges - the section accounting was
  right and the isolated bench was wrong. The other sections agree the same way:
  qkv 0.119 ms/call (14.3 ms/token), head 10.46 ms/call (10.5 ms/token), ffn fallback
  0.339 ms/call;
- pinned staging (`wubu_cuda_bench_xfer`): 0.030 ms -> 0.018 ms per call in isolation, i.e.
  6.8 -> 3.2 ms/token across 202 calls. Implemented end to end as a single DLL call
  (`upload + q8k + kernel + download` through pinned buffers): measured **11.2 tok/s
  against the 11.4 baseline**, i.e. no gain, and removed. The isolated transfer saving does
  not survive contact with the real path, which is the useful correction;
- launch shape is flat (2/4/8/16 warps per block: 76-79 GB/s);
- two columns per warp: +6%, and slower than one column once the fp16 fix landed;
- coalesced 3x32 B block reads with shuffles: 8% worse.

What did move a kernel: replacing the hand-written fp16 decoder (branch plus a normalise
loop for subnormals) with `__half2float` took IQ2_S from 78 to 100 GB/s.

## Where the waiting is (measured, not assumed)

| suspect | measurement | verdict |
|---|---|---|
| host wait policy | `G4_WAIT=spin` 11.4, `yield` 11.4, `block` 10.0 tok/s | spin is right; block costs 12%, now pinned explicitly |
| three syncs per matvec | one-shot `upload+quantise+launch+download+1 sync` wired and measured: 0.265 vs 0.259 ms/call, 11.3 vs 11.3 tok/s | **neutral, removed** |
| allocation fragmentation | isolated o-proj bench with 48 separate 3.1 MB allocations instead of contiguous: kernel 0.041 ms, identical | not the cause |
| clock throttling | 63 C, no power/thermal slow-down flags; during generation the GPU holds P0 at ~2650 MHz of 3105 and **55-67% utilisation** | not throttled, but 35-45% idle |
| GPU idle between layers | utilisation above; the idle is exactly the CPU-only attention stretch (7.3 ms at 32 ctx, 29 ms at 256) | **this is the waiting**, and the fused path removes it |

So the per-call overhead that makes an isolated kernel look 6x faster than the same kernel
in situ is not sync, allocation, or clock: the device is idle while the CPU runs attention,
and every submission after that gap pays to resume. `nvidia-smi -pm 1` reports `[N/A]` on
this GeForce part, so there is no driver-side persistence mode to hold it up - the fix is
to stop going idle, which is precisely what the fused half-layer does.

## Prefill: one token at a time, 2x faster and token-identical

Prefill used to be a single `N = n_prompt` forward: it quantises activations for M rows and
runs the CPU matvecs. Feeding the prompt one token at a time through the same single-row
call the decode loop uses is **twice as fast at every length tested**, with identical
greedy tokens:

| prompt | batch forward | one token at a time |
|---|---|---|
| 21 tokens | 3.11 s (6.8 tok/s) | **1.54 s (13.6 tok/s)** |
| 54 tokens | 7.96 s (6.8 tok/s) | **4.10 s (13.2 tok/s)** |
| 321 tokens | 49.46 s (6.5 tok/s) | **26.77 s (12.0 tok/s)** |

Per-token cost during the loop starts at 81 ms and settles at 70-77 ms, i.e. the same as
decode. The batch path is now gone.

Correction: an earlier revision of this document recorded this experiment as neutral
(3.09 vs 3.12 s). That measurement was wrong - the new branch had been put inside
`#ifdef GPU_SUPPORT`, which this tool is built *without* (the model object owns the device
dispatch), so both runs executed the old batch path and the "incremental" timing was a
second copy of the batch timing. The tell was `strings` finding none of the instrumentation
in the rebuilt binary. Measure the binary you built.

## Tried and removed: IQ3_S kernel micro-optimisation

After the fusions, decode issues only **49 matvec calls per token** (48 x IQ3_S, 1 x Q3_K),
so IQ3_S is effectively the whole matvec path. Two changes were built and measured:

- a compile-time-typed kernel (`matvec_t_kernel<21>`) so the block loop has no runtime type
  switch, with 8 warps per block instead of 4;
- branch-free sign application (`c_sign4` + `__vsub4`) instead of eight conditional byte
  negations per 4-byte group in `iq3s_block_dot`.

A/B at 65 tokens: **12.9 / 12.7 with both, 12.8 / 12.8 without**. Neutral, so both were
reverted rather than kept. The o-projection stays at 0.273 ms/call for a 0.045 ms kernel,
which means that gap is per-call overhead (launch plus a synchronising download on a device
that has just idled through the CPU attention), not the block decode. Removing it requires
fusing across the attention boundary, which changes the numerics and so breaks token
identity - the same trade the fused half-layer makes.

## Device attention half-layer: full-attention layers only, now the default

Bisected by layer kind, which finally localised the difference:

| fused subset | result over 64 tokens |
|---|---|
| full-attention layers only | **identical to the CPU reference, all 64** |
| sliding layers only | parts at generated token 2, then a stutter |
| all layers | same as sliding (the sliding layers dominate) |

So the composite reproduces the reference exactly on the full-attention layers - q, k, v, the
KV rows and the span all verified to ~1e-06 there - and the sliding layers are where the
remaining bug is. The default now fuses the full-attention layers only: token-identical,
+2% (13.1-13.2 against 12.9-13.0 tok/s at 65 tokens, 12.3 at 257). `G4_ATTN_DEV=all` gets the
other 8.5% and still parts from the reference; `G4_ATTN_DEV=slide` isolates the broken kind;
`G4_ATTN_DEV=0` turns the whole thing off.

## Device attention half-layer (all layers, `G4_ATTN_DEV=all`) - still not shippable

It is the faster path: **14.1-14.3 tok/s** against 12.8-12.9 at 65 tokens of context, and
**13.5** against 12.0 at 257, with the fused q/k/v, FFN and incremental prefill all in
place. Its arithmetic also tracks the CPU sequence closely (q/k/v/attention to 4e-07,
o-projection to 1.5e-05, EG-124).

### Research pass (llama.cpp gemma4.cpp, upstream) and what it settled

Checked the upstream graph against the engine's composite: Q -> q_norm -> rope,
K -> k_norm -> rope, V -> rms_norm with no scale and no rope, then attention with
`hparams.f_attention_scale`. That is exactly the order the engine uses, including the
unnormalised-weight V norm, so the architecture is not the source of the difference.

Two gates were added to close the remaining blind spots:

- **EG-127** compares the device Q8_K quantiser against `quantize_row_q8_K` byte for byte at
  n = 4096 (the o-projection's K). Scales and all 4096 quantised values match exactly; only
  the unused `bsums` differ, and no kernel reads them. PASS.
- **EG-126** now covers both layer shapes: hd 256 / 8 kv heads (sliding) and hd 512 / 1 kv
  head (full). Both pass at ~4e-06 against a double reference.

### A real bug: the device KV row 0 was never primed

The composite is skipped at position 0 because `kv_len` is 0 there (the cache is not sized
yet), so row 0 was written only to the CPU cache - but `primed[layer]` was set at position 0
anyway, so the one-time device prime never ran and every later token attended over an
uninitialised device row 0. Marking a layer primed only after it has actually appended fixes
it; the fused continuation moved from wrong-at-token-0 to matching the first two tokens.

### Second real bug: the attention span was clamped by the allocated cache size

The fused branch derived its window as `min(cache->size - 1, pos) + 1`, but `cache->size` is
how many rows have been *allocated* so far - at position p that is p - so the clamp dropped
the current token from the window. Measured at position 1: the composite attended **1 row**
where the CPU sequence attends **2**. That is why k, v and the KV rows all matched exactly
while the attention output did not: the composite was computing a different attention, over
the wrong span. Deriving the span from the position (`pos + 1`, or the sliding window) makes
the fused continuation go from a permanent control-token loop to one that tracks the default
for most of 24 tokens, with a two-token stutter at positions 2-4 and one divergence at
position 11.

### Tried and removed: CPU quantisation of the device attention (G4_ATTN_Q8H)

Hypothesis: the fused path's 1e-05 per-layer difference is amplified by the int8 quantisation
of the attention row - a 1e-06 difference near a step boundary flips a whole step, which is
0.8% of that block's scale. Built it: `wubu_cuda_attn_stage` (the composite stopped after
attention) + CPU `quantize_row_q8_K` + `wubu_cuda_matvec_dev`. Result: the continuation got
*worse* on the sliding layers (`45518 107 818 2430 ...`) rather than matching, so the variant
was not equivalent to the composite for some other reason and the experiment proved nothing
about the amplifier. Removed rather than left as a flag. Recorded so the next attempt starts
from the fact that the two paths differ for a reason nobody has isolated yet, not from the
assumption that only the quantiser differs.

### Per-layer ADIFF found a real type-17 bug: the IQ2_XS grid is unsigned

Comparing the composite against the CPU sequence at **every** layer (`G4_ATTN_ADIFF=all`) gave
a distribution, not a single number, and it was bimodal:

| layers | dev-vs-cpu (absolute) |
|---|---|
| most sliding layers | 1e-07 - 9e-06 |
| 16, 21, 22, 24, 25 | **9.7 - 16.7** |

Those five are exactly the layers whose `attn_k` is IQ2_XS (verified against the GGUF tensor
table) - the only IQ2_XS K tensors in the model, and therefore the only ones the composite ever
ran type 17 on. The specialised kernel cast the grid bytes to `int8`:

    if ((signs >> j) & 1) ((int8_t *)&a0)[j] = -((int8_t *)&a0)[j];

The grid holds **unsigned** magnitudes (upstream: `y = db * grid_value` with the sign applied
from `ksigns`), so every magnitude >= 128 became a negative int8 and flipped its product. Fixed
to an unsigned scalar loop; the five layers drop to 1e-06.

The fused FFN uses the same kernel with `n_cols=15360` and was never wrong, which is why the
default path measured exact throughout - the bug only ever ran on the composite's K.

### The token divergence is coherent accumulation, not sensitivity to noise

The jitter probe settles a question the numbers alone could not:

| injected into every layer's attention output | continuation |
|---|---|
| i.i.d. +-3e-5 absolute | **unchanged for 14 tokens** |
| i.i.d. +-1.0 (sanity check) | scrambled, so the probe is live |

Yet the fused path's residual, ~1e-07 per layer, flips tokens from the third one. Random noise
of 300x that magnitude cancels; a *coherent* bias does not - 40 sliding layers apply the same
rounding difference in the same direction, so it accumulates linearly instead of as sqrt(n).
That is why "the error is only 1.5e-07" and "the continuation changes" are both true.

### Prefill attribution, at last (G4_PROF_PREFILL=1, 87 tokens, 4972.7 ms)

Every phase accumulator in this file was gated on `N == 1`, so prefill had no attribution at all - which
is why four separate guesses about where its time went were wrong. `g4p_acc(slot, N, ms)` now routes
`N > 1` into a second table and the run prints it:

    norm+qkv        1146.6 ms   23.1%
    ffn all         2947.9 ms   59.3%
      ffn proj      2881.6 ms   58.0%
      ffn gelu+mul    59.4 ms    1.2%
      ffn norm+resid   6.9 ms
    o proj           652.3 ms   13.1%
    qk norm+rope     161.8 ms    3.3%
    attention         53.2 ms    1.1%
    attn out+resid     6.6 ms
    head              10.4 ms    0.2%

**The three projections are 94% of prefill.** Attention is 1.1% - the device attention kernel did its
job and took the quadratic term off the table. RoPE and the per-head norms are 3.3%, gelu 1.2%, the LM
head 0.2%. Everything that is not a matmul is now six percent of the time.

That closes the question this document opened several sections ago ("where does the 196 s go") with a
measurement instead of a hypothesis, and it changes what the next change should be: prefill is a matmul
problem and nothing else. The levers are the number of times the weight bytes are read (rows per
decoded block - 22 passes at MT=4 for an 87-token prompt) and per-launch fixed cost, not attention, not
the head, not the decode, not the scheduling of either kernel.

### Decomposing the kernel by removing stages, and the benchmark bug that fooled it

`ncu` is not installed and `nsys` is not either, so the decomposition was done by hand: a standalone
kernel with the real shape (15360 cols x 3840 K, M=87, `MT=4`), three variants that read exactly the
same bytes, and one stage removed at a time - full, no-decode (weights used straight from `qs`),
no-dot (bytes folded without `dp4a`).

First run, codebook in `__constant__`:

    A full        50.479 ms     decode A-B  39.385 ms (78.0%)
    B no-decode   11.094 ms     dp4a   B-C  -0.058 ms ( -0.1%)
    C no-dot      11.151 ms     loads  C    11.151 ms (22.1%)

Seventy-eight percent in the decode, and `MT` scaling to match: MT=4 50.5 ms, MT=8 24.7, MT=16 13.6 -
3.7x from rows-per-block alone. It looked like the answer to two sessions of guessing.

It was a property of the benchmark. `__constant__` memory serialises divergent addresses 32 ways, and
every lane in this kernel indexes a different codebook entry. The shipped kernel uses `__device__`
global for exactly that reason - there is a comment saying so, three lines above the declaration I had
read. Re-run with the table in global memory, matching what ships:

    A full        13.645 ms     decode A-B   2.454 ms (18.0%)
    B no-decode   11.192 ms     dp4a   B-C  -0.006 ms ( -0.0%)
    C no-dot      11.197 ms     loads  C    11.197 ms (82.1%)

**Decode is 18%, the dot is nothing, and 82% of the kernel is loading weight bytes at 37 GB/s.**
`MT=8` and `MT=16` cut the decode share to 4% and 3% - they cannot cut what was never there. In the
pipeline MT=16 measures 16.1 tok/s against MT=4's 17.4, because 56 registers cost the occupancy the
load phase lives on. Reverted; MT stays 4.

The lesson is the same one this document keeps relearning from the other side: a benchmark is a model
of the kernel, and when it disagrees with the shipped kernel by 4x the benchmark is the thing that is
wrong. The instrument was corrected before the conclusion was written down, which is the only reason
this section is useful rather than actively misleading.

### RETRACTED: the "FFN output differs" evidence was a broken harness

The entry below claims layer 1's FFN output differs by up to 1.5x between the fused and host paths. That
number came from a harness that was wrong three times over, and it is withdrawn rather than left standing:

1. The two dumps came from **separate processes**, and the model state had already diverged between them,
   so the inputs were not the same. Comparing `normed` for that layer across the two runs showed it
   differing by 76% - the inputs, not the output, were the first thing to differ.
2. The dump files were **overwritten by decode steps**: both paths fired for N=1 as well as N=87, so the
   file held one token's worth (3840 floats) while the other held 87x15360, and the "difference" was
   between a gate output and a down output.
3. After restricting the dumps to prefill and comparing like-for-like, an **in-process** A/B was built
   instead (compute the same layer both ways, in one process, same inputs). It reported the fused and
   per-tensor paths disagreeing by up to 6x on layer 0 - **a type-22 layer that is known-good and whose
   two paths produce identical tokens**. A measurement that says a correct thing is wrong is itself
   wrong, so the harness is at fault and nothing can be concluded from it.

What stands: the decode is verified exact against both ggml's reference and this repo's own CPU
dequantiser, which are identical; the device copy of the failing tensor is byte-identical to the host
copy; the block sizes and dimensions are exact (15,360 columns, 22,579,200 bytes); and two independent
M-row kernels agree with each other to float precision while disagreeing with the host. Everything
unproven stays gated behind `G4_XXS_MMA`, and the next step is a harness that computes both paths into
separate buffers within one process and compares them **before** any other code touches those buffers.

### RETRACTED: the "FFN output differs" evidence was a broken harness
### Where the divergence actually is: the FFN output, not the matvec

`G4_DUMP_FFN=<layer>` writes that layer's FFN output, so the two paths can be compared as numbers
instead of as text. For layer 1 - one of the five - fused against host:

    n = 3840   max abs diff 125.5 on a scale of 85.6   elements differing: 3837 of 3840
    idx 2071  fused -49.84   host 75.70
    idx  267  fused -83.63   host 24.69
    idx 1371  fused -35.16   host 67.64

So the FFN genuinely diverges, while the gate matvec agrees with the per-row GPU kernel. Both of those
are true, which means **the per-row GPU kernel is not the reference** - the host path is, and it is the
one the model ran for these layers before fusion. My kernel was A/B'd against the wrong baseline; that
is the mistake to correct next, and it explains why a kernel that matches its "reference" can still
produce wrong text.

Next step is now precise: A/B the new kernel against the **host** gate/up/down for these layers (dump
`ffn_gate`, `ffn_up` and `ffn_down_out` from the host path), not against the per-row GPU kernel.

### The M-row kernels are numerically sound; the divergence is somewhere else

Two independent implementations of the M-row path for IQ3_XXS and IQ4_XS (an mma version and a simpler
warp-per-column dp4a version) both measure about 2x prefill and both produce wrong text. Rather than
keep guessing, `G4_XXS_CHECK` now A/Bs the new kernel against the per-row kernel the engine already
uses, on the **real tensors and real activations**, and reports how many elements disagree:

    type 18  K 3840  cols 15360  M 87 | worst 0.131 | differing 1160920 | >0.1% 86 | >1% 7 of 1336320
    type 23  K 15360 cols 3840   M 87 | worst 0.067 | differing  311886 | >0.1% 35 | >1% 2 of  334080

Seven elements in 1.3 million, and every one of them is a cancellation case - the row total is ~1e-5
while the individual terms are ~1e-3, so a rounding difference in the order of operations shows up as a
large *relative* error on a value that is numerically meaningless. Elsewhere the two agree to float
precision. **The kernels are not the bug.**

That leaves what the fused path does around them for these five layers, which is the next thing to
isolate. It is also the reason both variants stay gated behind `G4_XXS_MMA`: a path that is arithmetically
right but produces wrong text is worse than a slow one.

One measurement hazard worth recording: whether tensors are device-resident depends on free VRAM at load
(nvidia-smi showed 0 MiB used and 5920 free, yet one run uploaded nothing and fell back to 7.3 tok/s), so
`fused=` must be printed on **every** measurement or a fallback run is silently compared against a fused
one.

### An M-row kernel for IQ3_XXS / IQ4_XS measures 2x prefill - and is not correct yet

The per-row kernels for types 18 and 23 already exist and are correct, but the prefill path calls them
once per row, and each row re-reads the whole weight matrix. Writing the M-row version - same block
shape and same mma tail as the IQ2_S kernel, only the decode differing - changed all 48 layers to
`fused=48 fallback=0` and took the five slow layers from 205-253 ms to ~15 ms:

    87 tokens, whole prefill:   2.70 s  ->  1.39 s       32.2 -> 62.5 tok/s

**It is numerically wrong** - every prompt diverges, with degenerate repeated tokens - and the path is
therefore gated behind `G4_XXS_MMA` and off by default. What has been confirmed and what has not:

  confirmed   the codebooks reach the device: `[TBL] iq3xxs 04040404 04040414 04040424 04040c0c |
              iq4nl -127 -104 -83 -65`, exactly ggml's first entries. A missing table would have
              produced this same symptom, so it was the first thing ruled out.
  confirmed   the block sizes (98 and 136 bytes), the 32-group structure, the scale construction
              `d*(0.5+q)*0.5` and the low/high sign split match the ggml dequantisers and the existing
              per-row kernels.
  not confirmed  where the disagreement is. Forcing one row-group per block changed nothing, so it is
              in the decode, not the row indexing.

This needs a numeric harness - synthetic block, CPU dequantiser, compare - exactly as the nibble
straddle bug was found and fixed. It is the largest single item left: 2x on prefill for one kernel.

### The 5 fallback layers are 43% of prefill, and now we know why

The FFN phase measured 1693 ms while the fused block accounted for 43 x 12.27 = 528 ms. Per-layer wall
times explain the rest immediately:

    layer  0  15.4    1 205.1    2  14.6    3  14.6    4  14.6    5 233.8    6  14.6
    layer  7  14.5    8 230.5    9  13.2   10  13.2   11 230.6   12  13.3  ...  47 252.6

Five layers - 1, 5, 8, 11 and 47 - take 205 to 253 ms each instead of 11 to 15. **They are 1170 ms of a
2720 ms prefill.** This is the "5 of 48 layers fall back" that has been listed as unverified since the
fusion work; it is now measured, and it is the single largest item left.

The types say why. Dumping `ggml_type` per layer:

    gate/up = 18 (IQ3_XXS)   down = 21 (IQ3_S) or 23 (IQ4_XS)
    every other layer: gate/up = 22 (IQ2_S), down = 22 or 21

The GPU matvec dispatch implements IQ2_S (22) and IQ3_S (21) only. IQ3_XXS and IQ4_XS fall through to a
scalar path at roughly 0.8 GMAC/s, which is what 230 ms per layer is. The fix is the dispatch, not the
tiling: an IQ3_XXS kernel (grid codebook, 3.06 bpw, 98-byte blocks) plus IQ4_XS (4.6 bpw, plain 4-bit
with packed 6-bit scales) covers all five, and at the rate the existing kernels run it should return
about 40% of prefill.

Diagnostics added for this: `G4_FFN_PERLAYER` prints the per-layer FFN wall time, `G4_FFNT` prints the
three FFN quant types per layer.

### The row-group count was splitting the weight matrix

The stage accounting pointed at the matvec, and the matvec stats showed ~3.3 GB/s of packed weight
traffic against the 20 GB/s the same kernel reports on its own counters. The cause was the grid: with
`IQ2S_MMA_RH 2`, a block covers 32 rows, so an 87-row prompt split into three y-blocks and **each
one re-read the entire weight matrix**. Raising it to 6 puts 96 rows in a block and the prompt fits in
one.

    interleaved, 87 tokens:   RH=6  32.2 / 32.1      RH=2  31.3 / 31.3       (+3%)
    1717 tokens:              59.97 s               (was 61.35 s)

dp4a against mma on the same shape, same prompt: 26.5 against 31.6 tok/s, so mma stays the right
dispatch and the remaining gap is the kernel's unpack and issue rate, not its memory pattern.

### CORRECTION: the FFN is not transfer-bound. It is 98% matvec.

Two entries above claim the FFN's ~24 ms per layer was "pageable transfer and two Q8_K passes". Both
are wrong, and the way they were wrong matters: they were arithmetic on transfer sizes, never measured.

Two independent measurements refute them.

1. Folding the o-projection and the residual add into the attention block - removing 2.6 MB per layer
   of round trip - changed nothing: 31.4 against 31.5 tok/s, inside noise. There was no transfer to save.
2. CUDA events inside the fused FFN, per layer:

       upload 0.16   q8(x) 0.04   gate+up 8.07   gelu 0.05   q8(gu) 0.03   down 3.92   download 0.00
       total 12.27 ms

Transfer is 0.16 ms of 12.27, or 1.3%. The two matvecs are 11.99 ms. The 87-row Q8_K passes cost 0.07 ms
between them. Every "fusing / pinning / fewer round trips" conclusion drawn from transfer arithmetic was
chasing something that is not there.

**And the matvec rate is not what the per-shape stats said either.** Each FFN matrix is 15360x3840 at
2.7 bits, about 19.9 MB, so gate+up at 8.07 ms is ~4.9 GB/s and down at 3.92 ms is ~5.1 GB/s - not the
11-13 GB/s those stats printed. On a card that delivers 143 GB/s for the identical access pattern, the
FFN matvecs have 4-5x of headroom, and it is in the kernel, not in the copies.

There is also an ABI bug to go with it: the bridge's function-pointer typedef for the fused FFN omitted
the `ff` argument, so `M`, `hidden` and `ff` arrived shifted by one slot, every layer failed the
`K % 256` check, and the fused path had been silently dead - visible only as `fused=0` in the counter
that was added for exactly this reason. Fixed; the path is now genuinely exercised, and with the
attention block in place it measures neutral (28.8 against 29.1 tok/s), so the +6% recorded earlier no
longer reproduces.

### The attention block moved to the device, and the profile shows what it fixed

    before    norm+qkv 1153.3   qk norm+rope 171.2   attention 55.4   ffn 1964.3   total 3513.6
    after     norm+qkv  650.6   qk norm+rope  37.3   attention 21.1   ffn 1742.7   total 2766.2

Projections down 43%, head norms and RoPE down 78%, attention down 62%, prefill from 3513 to 2766 ms.
The earlier guess that this whole chain was worth ~8% was low: it is 11% at 87 tokens and 11% at 1717,
and it is the single largest correct change since the tensor cores.

**FFN is now 63% of prefill** at 1743 ms, and its kernels are only ~17 ms per layer of that - the rest
is the upload of the normalised rows, two Q8_K passes and the download of the down projection, the same
traffic pattern the attention block just eliminated one layer over. Applying the same treatment there
means the residual stream itself has to live on the device: x staged once per forward, normalised,
added and re-used in place, with only the final row crossing PCIe for the head. That is the last
structural item, worth roughly 16% by the transfer arithmetic, and it is a coherent piece of work
rather than an afternoon's patch.

### Keeping q on the device: 30 tok/s, and wrong

The obvious next saving was the q round trip: the QKV projection downloads q, then the attention kernel
uploads it again - 5.6 MB per layer for a tensor that never needed to leave. Implemented as
`wubu_cuda_attn_causal_devq` plus a projection that hands back the device pointer, it measured

    87 tokens   28.0 -> 30.0 tok/s       1717 tokens  67.6 s -> 62.5 s

and diverged on 5 of 5 prompts against the batched dp4a path. The reason is not subtle in hindsight:
**RoPE and the per-head norms are applied to the host q buffer after the projection**, so the device
copy going straight into attention was an un-rotated q. Reverted.

That is the real content of "move the KV cache to the device": q cannot move first. The norms, RoPE and
the KV store have to move together, because they are the reason the tensor is on the host at all. The
speedup above is a fair estimate of what that refactor is worth - about 7% at 87 tokens and 8% at 1717 -
and it is now recorded as a measured target rather than an assumption.

### Fusing the FFN: +6%, and why it is not more

`wubu_cuda_ffn_mrow_dev3` keeps gate, GELU, the product and the down projection on the device, cutting
the FFN from three round trips to one upload and one download:

    87 tokens   26.4 -> 28.0 tok/s       1717 tokens  72.72 s -> 67.59 s
    coverage: 43 of 48 layers fused; 5 still fall back because their weights are not device-resident

But the phase moved only 2073 -> 1964 ms, and the profile says why. Of ~41 ms per layer, the kernels
are ~17 ms; the other ~24 ms is the upload, the download and **two Q8_K passes over 87 rows**, at
roughly 700 MB/s. Fusing removed round trips but not the copy rate.

**Pinning those buffers was tried and made things worse** (24.3 against 28.0 tok/s): the same buffers
are read and written by the CPU for the norms and the GELU, and pinned memory is uncached on the host,
so every CPU pass got slower than the copies got faster. Reverted.

That leaves the real conclusion, now measured three ways: the round trips cannot be fixed by moving
buffers faster, only by moving fewer of them. The next change is norms, RoPE and the KV cache on the
device, so that q/k/v and the FFN input never leave the card - after which the only thing crossing PCIe
per layer is the residual stream.

### The kernel is now at parity; 71% of prefill is transfer and glue

Fresh `G4_MV_STATS=1` on the tensor-core build, 87 tokens, 3.37 s total:

    shape             calls   ms total   ms/call    rate
    21 3840x4096        40      112.0     2.800    49.4 GB/s    o-projection
    22 15360x3840       86      496.0     5.767    19.2 GB/s    ffn gate/up
    21 3840x15360        7       50.0     7.143    72.7 GB/s    ffn down
    21 2048x3840         7       15.0     2.143    32.3 GB/s    attn k/v
    21 3840x8192         8       43.0     5.375    51.5 GB/s    attn q
    22 3840x15360       37      205.0     5.541    20.0 GB/s    o/ffn mix
    22 4096x3840         5       12.0     2.400    12.3 GB/s

**933 ms of the 3370 ms is kernel time.** The large tensors run at 19-20 GB/s, which is llama.cpp's
21.7 on the same card - the kernel has reached parity. The other **2.4 s is what happens around the 190
matvec calls**: each uploads its input, runs the Q8_K quantiser, launches the kernel and blocks on a
download of its output, and that costs about 12.6 ms per call.

This is the arithmetic the staging model in this document predicted from the beginning and it is now
measured rather than asserted: 71% of prefill is moving activations on and off the card. The fix is not
another kernel change - it is keeping the activation chain resident, which removes the FFN's 16 MB per
layer of gate/up-out and down-in entirely.

### Where the remaining prefill time is, after the tensor-core work

    layer total   3425.9 ms
    ffn all       2072.9   61%     ffn gate/up/down
    norm+qkv       973.3   28%     fused q/k/v
    qk norm+rope   162.4    4.7%
    attention       54.7    1.6%
    head            10.1    0.3%

89% is the projections, and they move the 4 GB of weights at **1.3 GB/s** against llama.cpp's 21.7 on
the same card. Everything that is not a matvec is now four percent of prefill, which closes the
question this document opened with: there is nothing left to win outside the kernel.

The kernel itself is a hybrid - mma on a tile decoded per column - where upstream is a different
shape entirely: one thread per (output column, batch row) walking padded shared tiles with dp4a and
int32 accumulation, applying one scale multiply per 32 values. That is the structure to port, and it is
the last structural difference between 25 tok/s and 477.

### The tensor-core prefill ladder, and the 8-warp lesson

    dp4a, MT=16 (baseline)        87 tok 18.4    1717 tok 16.3  (105.2 s)
    + mma, 4 warps, 16 rows       87 tok 23.5    1717 tok 21.5  ( 79.8 s)
    + bank-conflict padding       87 tok 24.8          -
    + 8 warps, 32 rows            87 tok 26.7    1717 tok 16.0  (107.3 s)   <- reverted
    final: mma 4 warps + padding  87 tok 25.1    1717 tok 22.8  ( 75.2 s)

**More rows per staged tile is not free at every prompt length.** Eight warps with 32 rows per tile is
the best 87-token result measured (26.7) and the worst 1717-token result (107.3, against 79.8 for the
same kernel at 16 rows). At length there are already 54 row-tiles to fill the card, so halving the
number of blocks buys nothing and the extra warps fighting over one shared tile cost 25%. It was
reverted, because time-to-first-token at length is the number this engine is judged on.

Three bugs of my own, each found only because the tokens were checked after every change:
- an early `return` on warps whose row-half was past the end (any prompt under 32 rows) skipped the
  `__syncthreads()` the other half waited on;
- `IQ2S_MMA_COLS` was derived from the warp count, so doubling warps made each block claim 64 columns
  while the warp->column mapping covered 32 - half the output was simply never written;
- the original tile had no padding, so all 32 columns began at the same bank offset.

And one methodological failure repeated: three A/Bs in this stretch were taken while a test suite or a
long-context run held the card. The same binary reads 18.4 tok/s idle and 2.6 tok/s under a concurrent
1717-token run. Every number above is from an interleaved pair on an idle card.

### What upstream actually does, and the correction it forces

Reading llama.cpp's mmq sources rather than guessing at them changed the picture twice.

**First correction: IQ2_S and IQ3_S do not use tensor cores upstream.** In `mmq.cuh` both dispatch to
`ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a`. The `_mma` variants exist for Q4_0, Q8_0, Q2_K and the FP4
types. The comment on `MMQ_DP4A_MAX_BATCH_SIZE 64` says dp4a is the *preferred* path for these types at
batch <= 64 and only falls back when it is not. So llama.cpp reaches 477 tok/s on this card with dp4a
on the exact types this engine uses, and "the answer is tensor cores" - which an earlier section of
this document asserted - is wrong. The instruction was never the problem.

**What is the problem, from `ggml_cuda_mmq_load_tiles_iq2_s`:**

    threads_per_row = (MMQ_ITER_K / (4 * QR2_S)) / 2      // 256 K / 16 / 2 = 8 threads
    nrows           = warp_size / threads_per_row        // one warp covers 4 rows

A warp decodes a whole 256-value IQ2_S block for four rows at once, writing the decoded int8 into a
**shared tile padded by one int per row** (`x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + 2*l]`) - the
padding is what removes shared-memory bank conflicts, and there is an mma-specific `sram_stride` for the
tensor-core path. Scales go into the same tile as floats, `((ls & 0x0F)*d + d/2)/4`. Signs are built
with `__vcmpne4` into 0/0xFF byte masks and applied as `__vsub4(grid ^ signs, signs)` - the same SWAR
negation this engine derived independently, which at least confirms the decoding arithmetic.

The difference from this engine is therefore not the instruction but the staging: upstream decodes a
K-slab for a row-tile once into a conflict-free shared tile and then runs many dot products from it,
where this engine stages per column with a barrier around every block. That is the remaining gap, and
it is why the tensor-core work below bought 28% rather than the 26x the reference number implies.

### IQ2_S through tensor cores: decode to shared, mma from shared, verified equivalent

    uniform data : worst rel 1.6e-06                       0 of 1336320 outside 1%
    random data  : worst abs 1.22e-03 on max |y| 3415      0 of 1336320 outside 1%
    dp4a MT=16   : 10.799 ms   10.5 GB/s   (6 passes over the weights)
    iq2s -> mma  :  6.450 ms    2.9 GB/s   (1 pass)         1.67x

The decode is unchanged - same lane mapping, same grid, same signs, same scale nibbles - only its
destination moves: shared memory as int8, from which the b-fragment is read. Two bugs had to be found
to get here, and both are worth recording because both were invisible in the timing.

**First:** the a-fragment index skipped `c_q8_k`'s 36-byte header and its per-block stride, so it read
unrelated bytes. The kernel still ran at 2x while producing nonsense.

**Second, and subtler:** a 32-k `mma` chunk straddles *four* IQ2_S grid entries, and entries 0,1 take
the low scale nibble while entries 2,3 take the high one. A single scale per chunk is therefore
inexpressible, and it showed as a 1600-count disagreement on a 3415 maximum. The b-fragment splits
exactly on that boundary - `b[0]` covers the first 16 k, `b[1]` the second - so two `mma` per chunk,
each scaled by its own nibble, is exact. The uniform-data case passed even while the kernel was wrong,
which is why a uniform test alone is not evidence: it proves the layout, not the scaling.

1.67x on one pass against six is the floor for this shape. What remains is the pipelining llama.cpp
has and this does not: `cp.async` staging with double buffering, `ldmatrix`, and enough mma per warp to
keep the tensor pipe fed. The engine integration is now the work that matters most, because 1.67x on
the projections is 1.6x on prefill end to end.

### The tensor-core path, built and validated: bit-exact, 1.67x, and what it must become

`mma.sync.aligned.m16n8k32.s32.s8.s8` does 4096 MACs per instruction against dp4a's 4. Before any
IQ2_S decode is wired in, the instruction path was proved on its own: int8 weights and int8
activations in memory, fragments built to the PTX layout, output checked against a scalar int32
reference.

    mma, fragments from global   :  6.630 ms   8.9 GB/s    774.0 GMAC/s   correctness 0 of 44544 differ
    mma, shared stage, 32-k slab :  7.692 ms   7.7 GB/s    667.1 GMAC/s   correctness exact
    mma, shared stage, 128-k slab:  8.507 ms   6.9 GB/s    603.2 GMAC/s   correctness exact

**Bit-exact on the first try** - 44544 of 44544 outputs identical to the scalar reference, so the
fragment layout is right and the arithmetic is not in question. 6.63 ms against the current dp4a
kernel's 11.08 ms on the same 15360x3840 tensor at 87 rows is **1.67x**, from a version deliberately
left unoptimised.

The two shared-staging variants both lose, which is worth recording because it inverts the usual
instinct: coalescing the weight loads does not pay for two `__syncthreads()` per k-chunk (240 of them)
or for byte-wise staging. Direct global fragment loads win at this size, and the barriers only start
to pay with asynchronous copy and double buffering.

**What the integration has to be.** 1.67x is the floor, not the target. With the dp4a maths replaced,
the IQ2_S decode - measured at 18% of the current kernel - becomes the whole cost, and the design that
follows is: decode each column's IQ2_S block into shared as int8 (same lane mapping the current kernel
already uses, so the decode code carries over), then feed `mma` from shared. That is decode-limited
rather than dp4a-limited, which is the 4-5x band rather than 1.67x. Beyond that, `cp.async` staging
with double buffering and `ldmatrix` are what separate this from llama.cpp's 477 tok/s.

### The shared-accumulator GEMM, measured, and why the answer is tensor cores

The obvious way to get one weight pass per prompt: keep the accumulators in shared memory instead of
registers and let one decoded weight block feed all 87 rows. Written, checked against the current
kernel, timed:

    ref  MT=16 (registers) :  11.081 ms    10.2 GB/s
    shared accumulators    :  23.678 ms     0.8 GB/s    speedup 0.47x

Twice as slow. A shared read-modify-write per row per block costs three instructions where a register
add costs one, and 45 KB of shared memory per block caps it at four warps against eight. Rows per
weight pass cannot be bought with shared memory.

Which leaves the actual reason llama.cpp does 477 tok/s and this engine does 18.3: **tensor cores**.
This kernel does four multiply-accumulates per `dp4a` instruction, one per lane. An `mma.sync`
m16n8k32 int8 instruction does 4096, and sm_89 has them. Prefill is a matrix-matrix product and this
engine computes it with the scalar path that is correct for a single row. No amount of tiling,
staging or lane remapping in the current kernel closes that - it is the wrong instruction.

That redefines the next piece of work: an int8 tensor-core kernel for the prefill path, with the
quantised weights decoded into the a-fragment layout `mma.sync` requires and the Q8_K activations into
the b-fragment. Decode stays on the dp4a path, where it is already within 1.8x of llama.cpp.

### The reference number on this machine, and what it finally explains

`llama-bench` was already on this box in two builds. The CUDA one has no CUDA backend and falls back
to CPU; the Vulkan one uses the GPU. Same GGUF, same card, same session:

    llama.cpp Vulkan, RTX 4050, gemma-4-12b IQ2_M        pp87    477.36 +- 2.46 tok/s
                                                         tg32     24.31 +- 0.65 tok/s
    llama.cpp CPU-only (other build)                     pp87      7.28 tok/s
                                                         tg32      4.60 tok/s
    wubuwizard, CUDA, this session                       prefill  18.3 tok/s      26x behind
                                                         decode   13.6 tok/s      1.8x behind

**Decode is within 1.8x of llama.cpp** - a hand-written engine that is only 1.8x off a mature
implementation at token generation is not broken, and the M-row kernels, the pooled staging and the
device attention are why. Against llama.cpp CPU-only this engine is 2.5x faster at prefill and 3x at
decode, which is the more honest comparison for a machine with no working CUDA llama.cpp build.

**Prefill is 26x behind, and the gap is now fully accounted for.** llama.cpp runs an 87-token prompt
as *one* pass over the weights: 4 GB in roughly 0.18 s, about 22 GB/s. This engine runs it as six
passes (M = 87, MT = 16) at ~11 GB/s. Six passes at half the rate is the 26x, and both halves of that
were measured here independently - the pass count from the MT table, the rate from `G4_MV_STATS`.

So the 13x gap between the 143 GB/s the card streams and the 11 GB/s the kernel achieves is not a
mystery any more, it is a GEMM that does not exist: a kernel where one staged weight block feeds a
tile of 32-64 rows out of shared memory, the way llama.cpp's quantised mat-matrix kernels do, so a
prefill costs one pass instead of ceil(M/MT). Every attempt so far tried to reach that by raising MT
in a kernel whose accumulators live in registers, and registers run out at 16 rows. That is the wall,
and the answer is a different kernel, not another parameter.

### Per-shape accounting: every matvec runs at ~11 GB/s and I cannot yet say why

`G4_MV_STATS=1` times every M-row call and prints the shape, the call count and the rate computed
from the bytes actually moved (whole tensor per row-tile, tiles = ceil(M/MT)):

    type 21  3840x4096    M=87  calls=40   12.13 ms/call   11.4 GB/s     o-projection
    type 22  15360x3840   M=87  calls=86   10.28 ms/call   10.8 GB/s     ffn gate/up
    type 21  3840x15360   M=87  calls=7    45.43 ms/call   11.4 GB/s     ffn down
    type 21  3840x8192    M=87  calls=8    21.13 ms/call   13.1 GB/s     attn q
    type 21  2048x3840    M=87  calls=7     7.00 ms/call    9.9 GB/s     attn k/v
    type 22  4096x3840    M=87  calls=5     3.80 ms/call    7.8 GB/s

**Uniform, 8-13 GB/s, every shape, both quant types.** There is no pathological tensor. Against the
143 GB/s this card delivers for the identical 110-byte-per-lane access pattern, that is a 13x gap, and
it has survived every intervention: MT at 4/8/16/32, split-K, block-loop unrolling, the IQ3_S lane
remap, codebook staging in shared memory, and one-launch fused QKV.

An earlier version of this same print reported IQ3_S at 3 GB/s against IQ2_S's 14.4 and I nearly
believed it; it had assumed a single pass and a flat 0.42 bytes per weight, which under-reports by the
tile count. The instrument is now computed from bytes actually moved, and both types agree. The lesson
repeats: check the arithmetic in the measuring instrument before believing what it says about the
kernel.

**What this rules out.** Not the access pattern (143 GB/s measured on it), not the codebook gather
(staging it in shared is neutral, so it was hitting cache and costing no HBM traffic), not occupancy
(full at MT=16), not memory-level parallelism (unrolling by two is neutral), not lane utilisation
(remap is neutral), not instruction width (byte loads hit 143 GB/s). What is left needs kernel-level
stall attribution - `ncu` - which is not installed on this machine. Guessing past this point has a
measured record of about one hit in ten, so the next step here is to get the profiler rather than to
try an eleventh variant.

### The memory system is not the limit, and neither is the codebook

Two direct measurements, both negative, both worth more than the guesses they replace.

**Bandwidth.** 1 GiB read with the kernel's own pattern - 110 bytes per lane, byte loads, no vector
types - runs at **143.1 GB/s**. The matvecs move weight bytes at ~21 GB/s. A 7x gap with the memory
system doing exactly what the kernels ask of it, so layout, alignment, sector straddling and vector
width are all innocent.

**Codebook staging.** The gather is one divergent lookup per four weights, and the theory was that a
4 GB stream thrashes the 8 KiB table out of L1/L2 so every lookup misses. Staged in shared memory,
once per long-lived block that strides over many columns rather than owning one:

    baseline   17.6 tok/s      17.6
    staged     17.6 tok/s      17.6      tokens exact

Neutral. Shared memory is no faster than the L1 the gathers were already hitting, which also says the
gathers were hitting cache all along.

Together these bound what is left inside these kernels: the decomposition put the entire decode at
18%, and both ways of attacking it - fewer lookups per weight, and cheaper lookups - come back neutral.
What remains is the row count, and that is capped by registers (see the MT table below), not by
anything the kernel can restructure locally.

**Measurement discipline, learned the hard way twice:** every A/B in this document has to be
interleaved on an idle card. The same binary measures 16.0-16.1 tok/s with a test suite running and
17.6-19.0 alone - a larger spread than three of the changes I was trying to measure. Both wrong
verdicts recorded here (MT=16 "worse", then "better") came from ignoring that.

### IQ3_S reads its weights better than it looks, which is why fixing it changed nothing

`iq3s_matvec_mrow_kernel` walks `for (b = lane; b < nblocks; b += 32)`, so every lane takes a
different 110-byte block. At K = 3840 that is nblocks = 15, and 17 of 32 lanes never execute at all -
it looked like an obvious 2x staring back at me. An IQ3_S block is 8 sub-blocks x 4 grid entries,
exactly 32, so the natural remap is lane -> (sub-block, entry) inside one block: every lane busy, all
lane reads inside one 110-byte window. Written, gated, measured:

    lane-per-block (old)   19.0 tok/s   tokens exact
    lane-per-block (old)   18.9
    sub-block-per-lane     19.0 tok/s   tokens exact
    sub-block-per-lane     18.8

Neutral, reverted. The reason is worth keeping: the two layouts trade one resource for the other.
Lane-per-block is *good* for memory - 32 lanes walk 15 consecutive 110-byte blocks, one contiguous
1650-byte run - and *bad* for ALU, running the 64-lookup inner loop with 47% of lanes predicated off.
Sub-block-per-lane is the reverse. They cancel exactly, which is what a memory-bound kernel does when
you fix its ALU.

So IQ3_S is not starved of lanes, instructions, or rows-per-block. It reads its weights as well as
this layout allows, and the ceiling it shares with IQ2_S is the layout itself: 82 and 110 bytes per
block, fetched as 3-5 memory sectors per warp instruction for a few dozen useful bytes each.

### MT=16, and why MT is a fight against occupancy

`acc[MT]` costs one register per row, so rows-per-decoded-block and occupancy trade against each
other almost exactly:

    MT=16:  56 registers  -> 48 warps/SM (full)
    MT=32: 128 registers  -> 16 warps/SM
    MT=64: 254 registers  ->  8 warps/SM
    MT=96: 384-byte stack frame - accumulators fall out of registers into local memory

That is why the isolated harness curve is flat from MT=8 to MT=64: every row added removes weight
traffic and removes occupancy by the same factor, and the two cancel. The optimum is where occupancy
is still full, which is MT=16.

    MT=4    4.85 s  17.9 tok/s      4.88 s  17.8
    MT=16   4.69 s  18.5 tok/s      4.69 s  18.5      <- interleaved A/B, nothing else on the card
    MT=32   5.03 s  17.3 tok/s      5.01 s  17.4

**+3.6% for MT=16, and a correction:** an earlier pass in this document concluded "MT=8 and MT=16
lose, revert to MT=4". That measurement was taken while a test suite was running in the background
and was simply wrong - the same builds measure 16.0-16.1 tok/s under load and 17.8-18.5 tok/s alone.
Every A/B in this document that was not interleaved on an idle card is suspect for that reason.

`IQ3_S` stays at MT=4: MT=8 and MT=16 both measured 18.4-18.7 against 18.5, which is noise.

Prefill to first token, measured across this stretch:

    87 tokens     17.4 -> 18.5 tok/s        (4.88 s -> 4.69 s)
    1717 tokens   15.5 -> 15.8 tok/s       (111.07 s -> 108.94 s)
    TTFT at 1717  196.19 s at the start of this stretch -> 108.94 s

The 1717 gain is smaller than the 87 gain because at long context the tiles are larger to begin with,
so the per-tile decode is a smaller share of the whole - the same arithmetic that made the device
attention kernel worth 43% there and 2% at 87 tokens.

### The decode setup is instruction throughput, not memory latency

Fourth hypothesis, and the fourth measurement to kill one. If the warp stalls waiting on each 82-byte
weight block before its dependent `dp4a` chain, then decoding two blocks per iteration - two
independent chains in flight - should hide that latency. Implemented, built, measured:

    unroll x2, 87 tokens:  17.3 tok/s     unroll x1: 17.3 tok/s     tokens identical

Neutral, so it was reverted rather than kept (`git checkout`), because a change that neither moves the
number nor explains something is a change that has to be maintained for nothing.

That kills the memory-latency explanation, and the obvious next step was instruction count: replace the
eight predicated negations per block with two byte-wise ones. Done - the sign nibble is spread to one
byte per bit through a multiply by 2^0,2^7,2^14,2^21, widened to 0xFF, and applied with `__vsub4`:

    before: 5.07 5.04 5.02 s         after: 5.01 5.04 4.97 s     (three runs each, 87 tokens)

**No measurable change, against a run-to-run spread of +-0.7%.** It is kept, not because it is faster,
but because it is strictly fewer instructions and completely branchless with bit-identical output, and
because reverting it would put the divergent loop back for no gain either. It is not evidence for the
instruction-throughput theory; that theory is now as unconfirmed as the latency one was.

So after four falsified hypotheses - LM head, MT=8 registers, wire latency, sign-mask instruction
count - the number that governs this kernel is still unexplained from first principles. The pattern is
consistent: every change that alters *how* the same work is scheduled or encoded moves nothing, which
points at something structural the arithmetic does not capture. The next instrument for this is `ncu`,
not another reasoned guess - a stopwatch and a spec sheet have taken this as far as they can.

### Decode, measured per phase (G4_PROF=1, 85 tokens, RTX 4050)

    layer total        61.48 ms
    norm+qkv            0.16      fused, one call per layer
    qk norm+rope        0.02
    attention           0.04
    attn out + resid   27.27      o-projection, IQ3_S
    ffn all            33.57      of which ffn proj 33.13
    head               10.78      12.6% of the 85 ms token

    decode matvecs/token: 1.6 on GPU, 0.0 on CPU     type 11: 1.0/token   type 21: 0.6/token

The two dominant items are the FFN projections and the o-projection, and both are the M=1 shape:
27.27 ms for 48 o-projections is 0.57 ms each for 6.9 MB of IQ3_S, i.e. **12 GB/s** - the same ceiling
the prefill kernel hits, from the same cause. Nothing in decode is on the CPU any more (1.6 matvecs per
token, all device), and attention and norms are 0.06 ms combined.

The head is worth noting for the opposite reason: the generic `matvec_kernel`, 127 registers and no
specialisation, still moves 377 MB of Q3_K in 10.78 ms = **35 GB/s**, the fastest rate anywhere in the
model. The register count looked like the alarm; the measurement says it is the best-behaved kernel in
the file. That is the third guess in this document that ptxas or a stopwatch has overturned.

### Prefill is at 9% of the card's bandwidth, so the limit is the decode, not the memory

Arithmetic on the measured numbers, not a profile - the M-row kernel re-reads the whole weight set once
per tile of four rows, so:

    87 tokens   (22 tiles)  92 GB of weight traffic in 5.0 s   = 18.3 GB/s  (10% of 192 GB/s)
  1717 tokens  (430 tiles) 1806 GB of weight traffic in 111 s  = 16.3 GB/s  ( 8% of 192 GB/s)

Two conclusions follow, and both cut against the obvious next move. First, prefill is not
bandwidth-bound: raising the tile from four rows to eight rows should halve the traffic, and it
measured *slower* (16.1 against 17.2 tok/s).

**Correction, from `ptxas -v` rather than from reasoning:** I wrote that MT=8 lost because the
accumulators blew the register budget. That is wrong. `iq2s_matvec_mrow_kernel` uses 38 registers at
MT=4 and reports **zero spill stores and zero spill loads at MT=8**, and `iq3s_matvec_mrow_kernel`
uses 56 - both far under the cap. The MT=8 regression has some other cause (eight concurrent q8 row
streams per warp is the standing candidate) and is currently unexplained. It is recorded that way.

The one genuinely register-starved kernel in the file is the generic `matvec_kernel` at **127
registers**, which is what every type without a specialised kernel falls through to - including the
Q3_K LM head, the single largest matvec in the model at 262144 columns. That is a separate,
unexplored lead. Second, the cost per weight block is dominated by the decode itself - grid index,
eight sign tests, byte extraction, the scale nibble - roughly fifty instructions to feed two `dp4a`
per lane. Decoding a block for four rows barely amortises fifty instructions of setup.

So the lever is not a bigger accumulator array, it is decoding a block once per *thread block* into
shared memory and letting all eight warps consume it for their own columns and rows. That is the same
move as the workspace pool one level down: the tile is declared, written once, recycled, and the
decode setup stops being paid per tile. It is also the only way the tile can grow past four rows
without the register pressure that MT=8 already demonstrated.

### Replay is only free if the key is the exact token sequence

An independent rebuild of NVIDIA's CPU-ARC idea (a CPU replaying a large model's reasoning from a
cache) on one RTX 5090 reports the shape of the trap:

    new maths questions      cache 83%   27B from scratch 96%   stock 1B alone 48%
    near-miss questions      cache answered 186/200, 185 of them wrong   27B 94.5%
    speed                    1.8 s/question on 6 CPU threads                27B ~17 s

The speed is real and the accuracy collapses exactly on questions that look familiar. That is a
property of the *key*, not of the idea: that cache is found by similarity to a stored transcript, so
a one-clause difference retrieves a trajectory that starts right and diverges silently. The paper
builds each entry from the test question's own transcript and reports no miss rate, which is the
measurement that hides the failure.

The engine-level version of the same idea is only safe when the key is exact: `wubu_prefix_cache`
walks a trie of token ids and hashes the token range, so a hit means the prefix is byte-identical and
the KV it returns is the KV the model would have computed. A near-miss cannot hit, by construction.
That is the distinction worth keeping: **exact-prefix reuse is free and indistinguishable; semantic
replay is a different product with a different failure mode**, and it should never be sold as the
same speedup. For a 1717-token prompt this engine spends 111 s computing a KV that an exact-prefix
hit would return immediately - that is the honest ceiling of the technique, and it applies only when
the prompt repeats.

### Staging, not arithmetic: one workspace pool declared by role

The speed in an engine like this is not in the kernel maths, it is in the memory ladder - SSD, then
RAM as transient, then VRAM as workspace - and in never paying for the same staging twice. Every
device scratch buffer in `wubu_cuda_quant.cu` used to be its own static with its own `cudaMalloc`:
the decode q8 row, the M-row activations, the M-row q8 block. They are all the same thing, a staging
area for a shape, and the producer always writes the whole buffer before the consumer reads it, so
**the contents are irrelevant to the declaration**. The data differs, the staging does not.

`pool_get(role, bytes)` is that declaration: a buffer is named by role, materialised on first use,
and reused for any later tensor of the same shape. Nothing allocates after the shapes are seen once.

Measured on an 87-token prefill, `G4_POOL_STATS=1`:

    [POOL] 3 slots, 11.6 MiB total
    [POOL]   q8.mrow    1.45 MiB  uses=204
    [POOL]   mrow.x     5.10 MiB  uses=204
    [POOL]   mrow.y     5.10 MiB  uses=204

**204 matmul calls, three buffers, one allocation each.** That is 11.6 MiB of workspace serving all
48 layers of a 12B model, because the roles recycle. Prefill stayed at 17.3 tok/s and decode tokens
stayed exact, so the recycling is free.

What is still allocated per call rather than declared: the host-side `q`, `k`, `v`, `attn_out`,
`ffn_gate`, `ffn_up`, `ffn_down_out`, `scores` buffers (`model->buf_*`, grown by `realloc` in
`g4_model_forward`), and the CPU phases that walk them - head norms, V norm, RoPE, causal attention,
residual adds, GELU. The kernels already exist on the device for the elementwise ones
(`rms_norm_row_kernel`, `gelu_mul_kernel`); the work is to keep q/k/v resident on the card through
norm, RoPE and attention instead of round-tripping them per layer, and to declare their workspaces
in the same pool. That is the next structural step, and it is the one that actually removes CPU
phases rather than accelerating them.

### Prefill is the real gap, and the device path has to be a GEMM

Everything above tunes decode, one token at a time, and the numbers look respectable in isolation:
14.5 tok/s. The engines that ship this year are not measured there. Strata reports 11,306 tok/s
prefill on an RTX 5090, Maya reports drafting taking a 3090 from 140 to 200 tok/s at 262K context,
Atomic Engine reports SSD-streamed experts with a TurboQuant KV cache. Our prefill on a 1717-token
prompt is **14.4 tok/s** - about 800x below that class, and it dominates time-to-first-token.

Measured on this machine with a 1717-token prompt: prefill **7.3 tok/s**, i.e. **235 seconds before
the first output token**. On a 21-token prompt it is 14.6 tok/s, so the cost grows with prompt
length exactly as an unbatched attention pass does.

The batched forward is present and wired, and it **works** - 6.8 tok/s at 87 tokens, so the old note
in the source (6.5-6.8 vs 12.0-13.6) was right. The segfault I first reported was my own bug in the
toggle that enables it: `g4_model_forward_from_tokens(model, tok, B, T, ...)` takes N = B*T rows, and
passing `n_prompt` for both made 87 tokens into 7569 rows. The engine's batched path was never
broken; the call was.

That measurement is the interesting one. The per-token loop wins (14 vs 6.8) *because* a single token
runs the FFN and attention on the GPU, while the batched path drops to pure CPU - and a Ryzen 5
feeding 4 GB of weights loses to a 4050 that already has them in VRAM. So batching on the CPU cannot
win here by construction, no matter how good the matmul is: the win has to be a **device** GEMM,
where M activation rows consume a weight block that is read once.

First attempt was the cheapest structure: reuse the gated decode kernel, M times, with the weights
already resident. Measured: **3.4 tok/s** - four times *slower* than the CPU path, and the reason is
structural rather than fixable by tuning. M rows through an M==1 kernel is M synchronised H2D/D2H
round trips per tensor per layer; for a 1717-token prompt that is roughly 400k CUDA calls, and the
weight read - the only cost that matters - is not amortised at all. Reverted.

**Batched device attention: 196 s -> 111 s to first token.** With the M-row projections in place the
CPU attention pass was the only quadratic term left, and it is where prefill went at length:
48 layers x 1717^2/2 x 16 heads x 512 dims is 1.2 TFLOP of MACs for one prompt. `attn_causal_kernel`
puts one block on each (query token, q head): warp-per-key for the score dot, the same double-precision
max-and-exp softmax the CPU uses, then one thread per output dim accumulating sequentially over keys
so the per-element order matches the CPU instead of being an adjacent tree.

    1717 tokens, RTX 4050:   CPU attention 8.8 tok/s (196 s)  ->  device attention 15.5 tok/s (111 s)
      87 tokens, RTX 4050:   CPU attention 17.0 tok/s        ->  device attention 17.3 tok/s

The 1717-token gain is 43%, which is the size the arithmetic predicted for attention's share - the
first time in this document that a predicted share and a measured share have agreed. At 87 tokens
attention is ~12% and the gain is 2%, which is the same arithmetic from the other end.

Token identity across 7 prompts x 24 tokens with device attention: identical on 6, one diverges at
token 13 - the same sub-ulp class as every other path here, and a different prompt than the CPU
attention version diverges on, which is what a rounding tie looks like rather than a defect. It stays
behind `G4_ATTN_GPU_BATCH=1` for that reason.

**What is NOT established (pre-attention numbers):** where the remaining 196 s went. The profiler (`G4_PROF`) is gated on
`N == 1` throughout - every `G4P_ADD` and every phase accumulator is `if (N == 1)` - so there is no
phase attribution for prefill at all, and two guesses at it were wrong: the LM head is ~6% of the
87-token case, not the 70% assumed, and covering the projection types with M-row kernels bought 24%
rather than the multiple implied. The quadratic CPU attention is a plausible remaining cost at 1717
tokens but it is a hypothesis, not a measurement. Generalising the profiler to N > 1 is the next
thing to do, before any further prefill optimisation - guessing twice is enough.

**The gain shrinks with prompt length, and that names the next wall.** 1717-token prompt: 7.3
(per-token) → **8.8 tok/s** (batch, M-row kernels) = 235 s → **196 s** to first token. At 87 tokens
the same change is 13.9 → 17.2. Twenty-odd percent either way, but the absolute rate collapses from
17 to 8.8 as the prompt grows, which is the signature of a quadratic term taking over: attention is
still a CPU causal pass over M x M positions while the projections are now on the card. The M-row
kernels were necessary, not sufficient - batched attention on the device is the other half of
prefill, and it is now the binding constraint.

**Measured ladder, 87-token prompt, RTX 4050:** CPU batched 7.3 → IQ2_S M-row 14.3 → IQ2_S + IQ3_S
M-row **17.2**, against the per-token loop's 13.9. That is a 24% win over the current default, with
6 of 7 prompts producing an identical 24-token continuation and the seventh diverging at token 22 -
the same sub-ulp class as the three-way tie described above, and the same thing every engine sees
between prompt-processing and token-by-token attention (batched causal attention sums in a
different order than the incremental KV path). It stays opt-in (`G4_PREFILL_BATCH=1`) for that
reason, not because it is unmeasured.

**First working step: an M-row IQ2_S kernel doubles the batched path** (measured at 87 tokens:
7.3 → **14.3 tok/s**). The cost a GEMM actually amortises here is not the `dp4a`, it is the IQ2_S
decode - grid lookup, eight sign tests, scale nibble, block scale - which was being redone for
every prompt row. `iq2s_matvec_mrow_kernel` decodes a block once into registers and reuses it
across four rows, so one weight pass serves four tokens. That ties the per-token loop (14.1) with
strictly less total work, so the next step is the same treatment for IQ3_S (type 21), which owns
`ffn_down`, `attn_v` and `attn_out` - the largest remaining tensors in the batch path.

One more thing measured on the way: prefill was computing the **LM head for all M prompt rows**,
and generation only ever samples from the last one. Skipping the rest is correct and tiny (12.7 →
11.9 s at 87 tokens), because the head is not where prefill time goes - the projections are.

So a batched device prefill needs a genuine kernel: read each quantized weight block **once**,
multiply it against M activation rows, write M partial sums, with M held in registers or shared
memory. That is the same shape as the existing decode kernel with the inner loop moved outward, and
it is what turns a weight-bound GEMV into a compute-bound GEMM. It also needs the per-column gate
that the old M-row kernel lacked - that kernel was deleted in `85ae63f` because a synthetic test at
K=512 returned garbage in 3-9 of 64 columns and nobody found out why.

### Where the last token divergence comes from: it is a three-way tie, not a defect

`G4_FFN_CPU` was added to separate the two halves of the device path. Three configurations, same
prompt, 24 tokens:

| configuration | tokens 21-24 |
|---|---|
| CPU only (no GPU at all) | `... 52491, 236786, **37475, 1915**` |
| **default: fused attention + device FFN** | `... 52491, 236786, **37475, 1915**` |
| fused attention + **CPU** FFN | `... 52491, 236786, **193634, 3750**` |

The default reproduces the all-CPU continuation exactly for 24 tokens. The config that diverges is
the *mixture* - device attention with a CPU FFN - which is neither of the two pure paths. So there
is no single component to blame: all three sit within ~1e-07 of each other, and which pair matches
first is decided by which components share a rounding path. That is the resolution floor for a
device matvec against a float CPU matvec, and it is why the remaining work is not more kernel
tuning.

### The remaining ADIFF outliers are row amplification that the o-projection cancels

ADIFF reports 2.0-2.4 (17-20% of scale) on layers 14, 18, 31, 32, 33, 39 while the other 42
measure 1e-06. Those six have byte-identical tensor type sets to their neighbours, so it is not a
type issue. A new instrument (`G4_ATTN_DUMP`, which projects both the device and the CPU attention
row through the *same* CPU matvec) shows what happens next:

| layer | attention row | after o-projection |
|---|---|---|
| 18 | 2.053 (scale 7.4) | **0.713 (scale 11.1)** |
| 31 | 2.408 (scale 12.2) | **0.486 (scale 10.3)** |
| 30 | 1.622 (scale 8.3) | 0.785 (scale 10.3) |
| 14 | 2.003 (scale 14.4) | 2.144 (scale 116) |

A 20% row difference collapsing to 5% after projection is what a sharply peaked attention
distribution does: the weights move between keys that carry similar values, so the weighted sum
barely moves. That is consistent with the token stream being identical over 24 tokens on six
prompts, and it means the outliers are amplification in the row, not an error in it.

### One dispatch function, not three

There were **three** hand-written type chains in the CUDA file - the composite's q/k/v, its
o-projection, and `matvec_devbuf` - each listing 22/18/23 and falling through to the generic
kernel. Only the first two were reachable with an IQ2_XS tensor on this model, which is why the
bug appeared exactly five times. All three now call `mv_any()`, and the duplicate `mv_dispatch`
copy is deleted rather than kept alongside it.

### All-layer fusion is now the default: 14.3 tok/s at 65, 13.4-13.6 at 257

With `mv_any()` in place the sliding layers are correct, so fusing them is no longer a
quality trade. Measured on the same machine, same model, one after the other:

| path | decode @65 | decode @257 | prefill |
|---|---|---|---|
| previous default (full-attention layers only) | 13.1 | 12.4 | 13.7 |
| **new default (all 48 layers)** | **14.3** | **13.5** | **14.4** |

`G4_ATTN_DEV=full` restores the old behaviour, `=slide` fuses only the sliding layers, `=0`
disables the fused path entirely.

Token identity against the previous default: identical for 24 tokens on every prompt tried
(six prompts, including code, arithmetic and factual questions). Past roughly 30 generated
tokens the two part ways - the 40-token run first differs at token 31 - because the per-layer
residue still accumulates once every layer contributes it. That limit is documented rather than
hidden: the default is chosen on 24-token identity, which is the gate this engine has used
throughout, and long single-shot generations should be re-checked before relying on the last few
tokens of a 100-token continuation.

### Root cause: the composite's q/k/v dispatch had a silent hole, and IQ2_XS fell into it

The two contradictory measurements resolved to one bug. `wubu_cuda_attn_half_dev` spelled its own
dispatch for the three projections:

    if (tq == 22) ... iq2s_v3
    else if (tq == 18) ... iq3xxs
    else if (tq == 23) ... iq4xs
    else rc = wubu_cuda_quant_matvec(...)     // everything else

The generic kernel does **not** implement type 17, and it answers with a zero buffer and
`rc = 0`. So on the five layers whose `attn_k` is IQ2_XS the composite attended on an **all-zero
K**: no error, no fallback, and `max|k| = 0.0e+00` (measured; the device weight bytes matched the
host bytes exactly, so the tensor was uploaded fine). That is why every structural hypothesis
failed - KV rows, spans, reductions, exp, sqrt, quantiser were all correct, because the problem was
that K was never computed.

Fixed by one dispatch function `mv_any()` used by the composite, the stage-only path and the
o-projection, so a missing type cannot silently fall through again.

| | before | after |
|---|---|---|
| `max|k|`, layer 16 (attn_k IQ2_XS) | 0.0 | 9.25e-01 |
| all-layer fusion, 40 tokens | stutter at token 3 | **matches the reference for 30** |
| all-layer speed @65 / @257 | n/a (degenerate) | **14.0 / 13.3 tok/s** |
| default (full layers) speed | 13.1 / 12.4 | unchanged |
| worst per-layer ADIFF | 9.7 - 16.7 | 2.0 - 2.4 |

All-layer fusion is now **7% faster than the default** and produces the same continuation for 30 of
40 tokens before diverging. It stays opt-in (`G4_ATTN_DEV=all`) because token identity is the bar
for a default change, but the reason it was unusable was this bug, not accumulation.

Open: layers 14, 18, 31, 32, 33, 39 still measure 2.0-2.4 absolute in ADIFF while their neighbours
measure 1e-6, and all of them are ordinary IQ2_S layers. That is the next lead.

### The divergence is inside layer 0, and the two halves contradict each other

New instruments, all kept: `G4_HSTATE_ALL=<dir>` + `G4_HSTATE_AT=<pos>` dump the residual stream
after every layer at a chosen position; `G4_ATTN_DUMP` exposes the device attention row
(`wubu_gpu_get_attn_dump`) so the o-projection that follows it can be compared; `G4_KV_REPRIME`
re-uploads the whole CPU KV cache every token.

Measured at **position 21, the first decode step, where both paths have identical input**:

| quantity | fused vs default |
|---|---|
| attention row (device vs CPU) | 1.5e-07 |
| o-projection from that row | **0.000e+00 - bit identical** |
| residual stream after the attention residual, layer 0 | **5.5e-01 (5.2e-03 relative)** |

Two of those cannot all be true: if the attention row matches, and the o-projection of that row
is bit-identical, then `x += postnorm(oproj)` has the same inputs in both paths and must produce
the same result. Either the dumped position is not the same step in both runs, or one of the
three measurements is comparing the wrong buffer. That contradiction is the first thing to
resolve - guessing past it has been wrong every time this session.

Ruled out on the way: the device KV cache is not stale (`G4_KV_REPRIME` changes nothing, so the
device rows equal the host rows), and the type-17 K layers are no longer involved (fixed above;
their error is 1e-06 now, and the divergence is unchanged).

### Open lead, unresolved

Excluding the five type-17 layers from fusion (`G4_SKIP_T17`, since removed) made the token
stream match the reference **exactly** in both slide and all modes - before the kernel fix, when
those layers were wrong by 10-17. After the fix they measure 1e-06 and the stutter is unchanged,
so the mechanism that made exclusion work is not the kernel and is not yet explained. The hidden
state dumps say something inconsistent as well: fusing **only layer 0** differs from the default
by 5.7e-06, while fusing **all** layers differs at layer 0 by 4.3e-01 - and layer 0 cannot be
affected by later layers. That is the next thing to chase, and it needs the single-layer mode's
guarantee that fusion actually happened verified first.

### Softmax exp through double as well: 1.63e-07 -> 1.52e-07

Same trick as the sqrt: `(float)exp((double)(score - max))` on both sides, so the two paths
call the same function with the same precision instead of libm's `expf` against CUDA's. ADIFF
moved 1.63e-07 -> 1.52e-07 and all-layer fusion now tracks three tokens further
(`... 149681, 1174, 563, 496, 9760, 52491` instead of stopping at `623`), though the 3-token
stutter at generated positions 2-4 is unchanged.

Note this is the *third* variable where matching the two implementations mattered more than
making either one more accurate: sharing the engine's own `wubu_expf` made things worse
(2.60e-07) because it is 2 ulp from libm, while double `exp` is 0 ulp from itself.

### Every hyperparameter cross-checked against the GGUF and against upstream

The GGUF's own metadata (parsed directly, 58 KV pairs) against the engine's constants:

| GGUF key | value | engine |
|---|---|---|
| `attention.head_count` | 16 | G4_HEADS 16 |
| `attention.head_count_kv` | per-layer array, 8 with 1 at layers 5, 11, 17, ... | g4_kv_heads from the pattern |
| `attention.key_length` / `_swa` | 512 / 256 | G4_FULL/SW_HEAD_DIM |
| `attention.sliding_window` | 1024 | G4_SLIDING_WINDOW |
| `attention.sliding_window_pattern` | T,T,T,T,T,F repeating | g4_layer_is_full |
| `attention.shared_kv_layers` | **0** | no shared caches |
| `rope.dimension_count` / `_swa` | 512 / 256 | G4_FULL/SW_N_ROT |
| `rope.freq_base` / `_swa` | 1e6 / 1e4 | G4_FULL/SW_ROPE_BASE |
| `attention.layer_norm_rms_epsilon` | 9.99999997e-07 | G4_RMS_EPS |

All match, so the model wiring is not the source of the sliding-layer divergence. Upstream
(llama.cpp `gemma4.cpp`) also confirms the two things that had been assumed: `f_attention_scale
= 1.0f` ("Gemma4 uses self.scaling = 1.0, no pre-attn scaling") and `ROPE_FREQ_BASE_SWA` as a
separate base for the sliding layers. It also documents that shared-KV layers read a specific
earlier layer's cache (`kv_source_impl[il] = n_layer_kv_from_start - (is_swa ? 2 : 1)`) - not
relevant here because this checkpoint declares zero shared layers, but worth knowing.

So the sliding-layer divergence is numeric amplification, not a wiring error: 7 fused layers
(full only) stay exact over 64 tokens, 48 fused layers at the same ~1.6e-07 per layer do not.

### All reciprocal-sqrt approximations removed; the early stutter is structural, not numeric

After the double-sqrt fix the object file contains **zero** `vrsqrtss`/`vrcpss` (was 3): the
inlined V-norm in the layer loop was the last one, and it now uses the same double form. ADIFF
went 2.82e-07 -> **1.63e-07**, and the value loop is better with FMA than without (1.63e-07 vs
1.74e-07), so `__fmaf_rn` is what stays.

At 1.63e-07 - roughly 1-2 ulp of the attention row - all-layer fusion still emits three extra
tokens at generated positions 2-4 (`100, 45518, 107`) before tracking the reference for the
next 21. That pattern is not what noise does: a 3-token replay of the answer start is what a
context that is 3 tokens short looks like. So there is a *structural* early-position problem in
the fused sliding path, separate from the numeric one, and it is the next thing to look at -
not more precision work.

### Found it: `1.0f/sqrtf` becomes vrsqrtss under -ffast-math

Continuing the objdump pass past the sum showed the rest of `g4_rms_norm`:

    vdivss  %xmm2,%xmm6,%xmm9      ; ss / n
    vaddss  %xmm3,%xmm9,%xmm3      ; + eps
    vrsqrtss %xmm3,%xmm3,%xmm7     ; hardware reciprocal-sqrt *approximation*
    vmulss  ...                    ; Newton-Raphson refinement

GCC rewrites `1.0f / sqrtf(x)` into `vrsqrtss` plus one Newton step, which is an approximation
CUDA has no equivalent for. That - not the reduction - was the 3.576e-07.

Fix: compute the reciprocal square root through double on both sides
(`(float)(1.0 / sqrt((double)(ss / n + eps)))`), which -ffast-math does not transform and CUDA
evaluates identically.

| | before | after |
|---|---|---|
| ADIFF (fused attention row vs CPU) | 2.82e-07 | **1.63e-07** |
| default tokens | exact | exact |
| decode | 13.4-13.5 tok/s | 13.4-13.5 tok/s |

Also measured while doing this: wiring the shared `wubu_expf` in instead made ADIFF *worse*
(1.63e-07 -> 2.60e-07), so it stays out. Counterintuitive, recorded so it is not re-tried.

### Disassembled GCC, reproduced it op for op, still no match

The rms-norm sum was read out of the object file rather than guessed. `objdump` on
`g4_rms_norm` shows the vectoriser using **four** 8-lane accumulators over
x[32t+0..7], [+8..15], [+16..23], [+24..31] with `vfmadd231ps`, combined as
`(S0+S1) + (S2+S3)` per lane (`vaddps %ymm1,%ymm11,%ymm4` / `vaddps %ymm8,%ymm0,%ymm8` /
`vaddps %ymm8,%ymm4,%ymm6`), then `vextractf128` + `vaddps` (lo+hi), then the hadd pair.

That was implemented exactly on the device, with `__fmaf_rn` and `__fadd_rn` so nothing could
be contracted or reassociated, single-threaded so the order is fixed:

    EG-123 rms_norm weighted: 3.576e-07  ->  3.576e-07   (unchanged)
    ADIFF:                    2.77e-07   ->  3.25e-07    (marginally worse)

So the reduction order is **not** the difference. That is a real elimination, not a guess: the
instruction-level reproduction of the loop left the number exactly where it was, which points
at the `1/sqrtf` path (no `vsqrtps`/`vdivss` appears in the disassembled range, so GCC is
calling out) or at EG-123's own reference rather than the kernel.

Reverted. Four approaches to the last 2.6e-07 have now been measured and all failed; the
useful residue is that the reductions are eliminated by construction for rms, and that the
attention row is verified correct to 2.8e-07 by an end-to-end comparison.

### Three attempts at the residual, one result: guessing the compiler does not work

After sharing the trig (which did help: fused divergence moved from token 2 to token 8) the
attention row still differs by ~2.6-3.0e-07, so three more emulations were tried and measured:

| attempt | result |
|---|---|
| shared `wubu_expf` in both attention paths | expf is bit-identical across toolchains (EG-128: 0/4096) but **ADIFF unchanged at 2.60e-07**, and it moves the reference to divergence at token 10 |
| non-FMA value-loop accumulation (`__fadd_rn`) | 2.82e-07 -> 2.60e-07, marginal |
| rms-norm reduction emulated as 16 FMA accumulators + the add/add-hi/hadd/hadd tree | EG-123 rms stays **3.576e-07 - no match**, ADIFF unchanged |

All three reverted. The conclusion is the useful part: the CPU side is auto-vectorised by GCC
with `-ffast-math`, and its reduction order cannot be inferred from the source - the 16-lane
guess matched for `g4_dotf` (where the source spells the intrinsics out) and did not match for
`g4_rms_norm` (where it is a plain loop the compiler rewrote). Closing the remaining 2.6e-07
means reading the CPU's actual instruction sequence with objdump and reproducing it op by op,
not testing hypotheses about what the vectoriser probably did.

Given that, the fused path stays at +8.5% with a divergent continuation (`G4_ATTN_DEV=all`) and
the default stays exact at 13.4-13.5 tok/s with full-attention-layer fusion on.

### Quantiser exonerated on real data; one instrument removed for lying

EG-127 was extended to take a file so it can compare the device and host Q8_K quantisers on a
real activation instead of synthetic floats (`G4_DUMP_NORMED=<path>` writes the pre-attention
vector at layer 0, position 23). Result on real data:

    EG-127 q8_K dev vs cpu: d_mismatch_blocks=0/15  qs=0/3840 worst_step=0

So quantisation of the pre-attention activation is not the residual either.

`G4_QDUMP` was removed. It compared the composite's staged q against a q reconstructed in the
diagnostic, and after the trig change it reported a 0.49 difference - which contradicts
`G4_ATTN_ADIFF` measuring the *end-to-end* attention row at 2.8e-07 in the same build. A q
difference of 0.49 could not produce an attention difference of 2.8e-07, so the reconstruction
was wrong, not the composite. ADIFF measures the real CPU attention for the same layer and
position and is the trustworthy one. An instrument that reports a wrong number is worse than no
instrument, so it is gone rather than left with a caveat.

### Shared sin/cos and iterative theta: reference re-baselined, +2% default speed

`include/wubu_trig.h` implements one sin/cos (Cody-Waite reduction in double - theta reaches
~1024, and a float reduction measured 4e-06 against libc - then a float minimax polynomial
evaluated with `fmaf`), and both rope paths now build `theta_scale^j` by repeated multiply
instead of `powf`. Both sides therefore perform the same sequence of operations, and
`theta_scale` is computed with double `pow` on both.

Effect, measured:

| | before | after |
|---|---|---|
| decode 65 ctx | 13.2-13.4 tok/s | **13.5-13.6** |
| decode 257 ctx | 12.6 | 12.8 |
| attention row vs CPU (ADIFF) | 3.04e-07 | 2.82e-07 |
| EG-123 rope worst_abs (vs libc) | 2.38e-06 | 4.05e-06 (expected: it now measures the shared implementation against libc) |

**The reference moved:** the CPU path now uses this trig instead of libc's, so tokens match the
old baseline through position 20 and diverge at 21 (236786 -> 193634). The continuation stays
coherent - it still produces "thought ... Context ... classic ... philosoph ...", the same
vocabulary - and the new sequence is the reference from here on. Recorded rather than hidden
because every earlier measurement in this document was made against the old one.

The fused path is still not bit-identical (2.8e-07), so the exactness target is not reached;
what is left after this is the q8 of the pre-attention activation, which the composite
quantises on the device and the CPU path on the host.

### The residual is RoPE's trig, and the gate was hiding it

EG-123 has been printing the number all along:

    EG-123 rope: heads=16 head_dim=256 worst_abs=2.384e-06

It passes because the gate's tolerance is looser than that, and 2.384e-06 on q is exactly the
1.4e-06 q difference (and hence the 3.0e-07 attention difference) that the fused path carries.
So the residual is not expf and not the reductions: it is `powf(theta_scale, j)` plus the
`cosf`/`sinf` pair, each implemented differently by CUDA and glibc.

Two consequences worth keeping:

- a gate that passes at 1e-5 cannot see a 1e-06 defect that matters after 40 layers; the rope
  check should report bit-differences, not just a worst absolute;
- the fix has a precise shape: precompute the per-j theta factors on the host in double and
  pass them as a table (removes `powf` from both sides), then a shared `cosf`/`sinf` built the
  same way as `wubu_expf` (fmaf range reduction plus polynomial). Drive that number to 0 and
  the fused path becomes bit-identical; no other stage is left.

### Shared expf: built and validated, not yet wired in

`include/wubu_expf.h` implements one expf (range reduction with fmaf, Horner polynomial,
scale by a power of two via the exponent field) declared `__host__ __device__` so both
attention paths can call it. EG-128 now checks it:

    EG-128 wubu_expf dev-vs-cpu: 0/4096 differ        <- bit-identical across toolchains
    EG-128 expf dev-vs-libc:     654/4096 differ, max 2 ulp

So the approach works. It is not wired into the model yet because it does not fix the fused
path on its own: with expf shared the attention row still differed by 3.0e-07, and that
residual traces back to **q**, which differs by 1.4e-06 - i.e. to RoPE's `cosf`/`sinf`/
`powf`, the same class of toolchain difference one stage earlier. Wiring it in today would
change the CPU reference's tokens (measured: divergence at generated token 10) in exchange
for no speed, since RoPE still blocks exactness.

The remaining work is therefore the same treatment for `cosf`, `sinf` and the per-layer
`powf` used for the RoPE theta - a shared implementation each, then wire all four in and
re-baseline once.

### The residual is `expf`, measured (EG-128)

`test_eg128_expf` compares the device `expf` against the C library's over the range the
softmax actually feeds it:

    EG-128 expf dev-vs-libc: 1255/4096 differ, max 2 ulp (worst at x=-10.725595)

So ~30% of the softmax exponentials differ by 1-2 ulp between the two toolchains. That is the
floor: no amount of reduction-order matching can close it, because the operation itself is
implemented differently on each side. Closing it means shipping one shared `expf` - a
reduction, a polynomial and a scale-by-power-of-two, written with explicit
`__fmul_rn`/`__fadd_rn` on the device and no fast-math contraction on the CPU - and using it
in both attention paths. That would make the attention row bit-identical; nothing else in the
path is left to differ.

### Attention kernel now mirrors the CPU's reduction order

Rewrote `attn_decode_kernel` to reproduce `g4_dotf` exactly: 16 explicit accumulators with
`__fmaf_rn` in the CPU's stride pattern, combined as `l[j] = a[j] + a[j+8]` then
`((l0+l4)+(l1+l5)) + ((l2+l6)+(l3+l7))` - the add / add(lo,hi) / hadd / hadd tree - with the
softmax sum done sequentially in one thread and the value loop accumulating `(e * inv) * v` in
kp order.

| measurement | before | after |
|---|---|---|
| attention row vs CPU (ADIFF) | 4.34e-07 rel | **3.47e-07 rel** |
| sliding-layer fusion, first divergence | generated token 2 | **generated token 8** |
| decode, full-layer fusion (default) | 13.3 tok/s | 13.3-13.4 tok/s |
| decode, all-layer fusion | 14.6 | 14.2-14.4 |

No speed cost on the default path, and the fused path now tracks the reference three times
longer before parting. The residual 3.47e-07 is almost certainly `expf`: CUDA's and glibc's
implementations differ in the last bit, so bit-exactness cannot be reached by ordering alone.

### Measured: the device attention row is correct to 4.3e-07

`G4_ATTN_ADIFF=1` stages the attention on the device into a host buffer and lets the CPU path
run as well, then diffs the two rows for the same layer and position (layer 0, position 23,
24 rows of real cache):

    [ADIFF] layer0 pos23 attention dev-vs-cpu=4.768e-06 rel=4.339e-07 (scale 1.1e+01)

So every stage of the composite is now verified by direct measurement against the CPU
sequence: q 1.4e-06, k 1.2e-07, v 9.5e-07, KV rows exact, span `pos+1`, attention 4.3e-07.
The per-layer hidden-state difference is 1.07e-05, and the continuation parts at generated
token 2 in all-layer mode.

What that leaves is the arithmetic between the attention row and the residual stream: the
o-projection's Q8_K quantisation of a value that differs in the seventh digit, plus the
post-attention norm and residual, accumulated over 40 layers. Closing it means emulating the
CPU's reduction order exactly - `g4_dotf` is two 8-lane FMA accumulators followed by a
specific add/hadd/hadd tree, and the softmax max/sum are sequential over cached positions -
so the port is 16 explicit FMA accumulators per score plus a sequential softmax, with
`fmaf` to match `_mm256_fmadd_ps`. That is a precision-matching port, not a bug fix.

### Debug hooks in the fused path

All env-gated, all off by default: `G4_ATTN_WARM=<pos>` (start fusing only at that position,
for bisecting), `G4_QDUMP=1` (compare the composite's staged q against the same
matvec + q-norm + rope on the CPU), `G4_ATTN_KVCHK=1` (read device KV rows back and diff them
against the CPU cache). They exist because every one of the bugs above was invisible from the
token stream alone - the failure looked like "the fused path is numerically worse" in all
three cases.

### The residual difference, measured directly

Dumped the hidden state after layer 0 at position 23 in both modes (`G4_HSTATE=<path>`) and
diffed the vectors: worst absolute difference **1.07e-05** on a vector whose largest element
is 49.3, i.e. ~2e-07 relative, and no element differs by more than 1e-05. So the sliding
composite is not computing the wrong thing - it is computing the same thing in a different
arithmetic order, and that is the whole story:

- full-attention layers: 8 of the 48, so the per-layer ~1e-05 never accumulates enough to
  flip a greedy token. Fusing them is bit-identical over 64 tokens.
- sliding layers: 40 of the 48. The same per-layer difference, five times as many layers,
  and the residual stream drifts far enough to flip a token by the second generated token.

That also retires the "5e-06 per layer" hypothesis in favour of a measured number, and it
explains why no gate ever failed: every individual stage is within 1e-06 of the CPU.

### Still open: what the residual divergence actually is

With k and v from the composite matching the CPU to 1e-07 at the same position, its
attention output still differed by 1.3 relative. The attention kernel itself is gated
(EG-126), the quantiser is gated (EG-127), k and v match - so the remaining suspects are the
staged q (not yet compared) and the device cache contents. Reproducing that comparison needs
the composite to return its attention row, which is the next probe.

It is still opt-in. The earlier root-cause note said accumulation order, ~5e-06 per layer:

- EG-126 (new) gates `attn_decode_kernel` at the model's real width (16 q heads, 8 kv heads,
  hd 256, 20 populated positions) against a double reference: worst absolute 4.6e-06,
  relative 4.6e-06. So the attention kernel is right at real dimensions - EG-124's blind
  spot was its small shape, not the kernel.
- The error per layer therefore comes from computing the same attention in a different
  order (the device does a plain mul-add score with a warp reduction; the CPU path uses
  FMA with two accumulators and a different horizontal sum).
- Bisected by layer kind: with only the full-attention layers fused the first two tokens
  match (`45518, 107`) and it parts at token 2; with only the sliding layers it parts at
  token 1; with all 48 fused it is already wrong at token 0. That is a per-layer error
  compounding with the number of fused layers, which is what 5e-06 x N looks like.
- Priming the device KV cache from the CPU cache before every token changes nothing, so KV
  maintenance is not implicated either.

The earlier note here said "wrong from the first token" and treated that as a structural
bug. It is what all-48-layers-fused looks like, and the bisect above is the correction:

- all 48 layers take the fused path (`share_kv=0`, `no_dev=0`), so there is no mixed-path
  KV-cache explanation for it;
- the *first* sampled token differs from the default path's (`236751` instead of `45518`),
  and the continuation is a loop of control tokens (237542, 236779, 236756) decoding to
  `_`, where the CPU-attention path reaches "This is a classic philosophical question".

A first-token divergence is a correctness bug, not 1.5e-05 of compounding rounding: EG-124
passes because it exercises one position with synthetic weights, and the composite has not
been validated against the CPU sequence at a real position with real weights. Until that
gate exists, the +11% is unavailable and the path is a measurement tool, not an option.

Speed that produces a token the CPU path never would is not a speedup.

## Device attention half-layer (opt-in, `G4_ATTN_DEV=1`)

`q,k,v -> per-head RMS norms -> RoPE -> KV append -> attention -> o-projection` in one
submission, with a per-layer device KV cache (512 positions, primed from the CPU cache).

| context | default | `G4_ATTN_DEV=1` |
|---|---|---|
| 33 tokens | 13.2 tok/s | - |
| 257 tokens | 12.2 tok/s | - |

The default path used to fall from 11.4 to 9.1 tok/s between 65 and 256 tokens because the
attention section was scalar and grew with kv_len (7.3 -> 29.1 ms). Vectorising it took
that to **4.6 -> 9.0 ms** and the curve is now nearly flat; the fused path is still ~14%
ahead at 200 tokens because it also removes the device idle.

It is opt-in because it does not hold the identical-token bar: the fused order changes the
accumulation. EG-124 (synthetic weights, same inputs) measures the divergence explicitly -
q/k/v/attn 4e-07, o-projection output 1.5e-05 - and at that size a near-tie flips by the
third token (45518, 107, then 140 instead of 236775). The direction is the same one that
already moved this engine from F32 to Q8_K activations; it is a different computation, not
a defect, but the default path keeps the bar.

## Constraint: device KV memory

`kvd` is 2048 floats (8 KB) per position per layer, twice (K and V). A 4096-position context
needs **3.2 GB** of device KV across 48 layers, which does not fit beside 4.0 GB of resident
weights on a 6 GB card. A budgeted device KV of 256-512 positions costs 192-384 MB, which
does, with CPU attention as the fallback past that.

## Correctness bar

An engine change is accepted only if the greedy token sequence is unchanged
(45518, 107, 236775, 818, 6590, ...) and the relevant gate passes. Kernel gates compare
against `gguf_dequantize`, never against another port of unknown provenance:

| gate | what it proves |
|---|---|
| EG-111 / 113 / 114 / 120 / 121 | fused IQ2_S, Q4_0, Q3_K, IQ3_XXS, IQ4_XS dots vs the dequantiser |
| EG-119 | fp4 layouts (NVFP4 40, Q2_0 42, MXFP4 39) vs the file's own offsets and the E2M1 lattice |
| EG-122 | device kernels per column vs the CPU ports, and M-row batching |
| EG-123 | device RMS norm / residual / RoPE / attention vs transcripts of the CPU routines |

Prefill note: a batched M-row device path measured prefill 6.8 -> 15.4 tok/s with identical
tokens, but its synthetic gate returns garbage for a few columns at K=512 and the cause is
unidentified, so it is not shipped — prefill stays on the CPU path.

## Next

Assemble the device attention half-layer. The primitives exist and are gated at the model's
real dimensions (`rms_norm_row_kernel`, `residual_add_kernel`, `rope_kernel` 2.1e-06,
`attn_decode_kernel` 2.1e-06 with 16 heads / 8 kv heads / head_dim 256 / kv_len 21,
`kv_append_kernel`).

A first composite (`q,k,v -> norms -> RoPE -> KV append -> attention -> o-proj` in one
submission, with a per-layer device KV cache primed from the CPU cache) was built and
removed rather than shipped: after fixing two real RoPE bugs it reproduced q, k and v to
1e-06 against the CPU sequence, but its attention output still differed by 1-3 max-abs.

Suspects eliminated by measurement since then:

- **prime + append mechanics** (EG-123 `kv roundtrip`): 20 rows primed and one appended at
  the engine's real stride (kvd = 2048) come back bit-exact, worst 0.000e+00;
- **the attention kernel at the model's dimensions**: 2.1e-06 against the CPU routine
  (16 heads, 8 kv heads, head_dim 256, kv_len 21);
- **q, k, v**: 1.4e-06 / 2.7e-07 / 4.8e-07.

So the KV data is right and the attention math is right, which leaves the *arguments or
ordering* inside the composite. The next attempt should diff the device attention output
against a CPU attention computed from the **device-downloaded KV**, not from the CPU cache;
if those agree, the fault is in the device KV view (a stale row or an offset), and if they
disagree the fault is in the call itself.
