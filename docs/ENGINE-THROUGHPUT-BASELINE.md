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

### Prefill is the real gap, and the device path has to be a GEMM

Everything above tunes decode, one token at a time, and the numbers look respectable in isolation:
14.5 tok/s. The engines that ship this year are not measured there. Strata reports 11,306 tok/s
prefill on an RTX 5090, Maya reports drafting taking a 3090 from 140 to 200 tok/s at 262K context,
Atomic Engine reports SSD-streamed experts with a TurboQuant KV cache. Our prefill on a 1717-token
prompt is **14.4 tok/s** - about 800x below that class, and it dominates time-to-first-token.

Measured on this machine with a 1717-token prompt: prefill **7.3 tok/s**, i.e. **235 seconds before
the first output token**. On a 21-token prompt it is 14.6 tok/s, so the cost grows with prompt
length exactly as an unbatched attention pass does.

The batched forward that should fix this is present and wired, and it **segfaults** (exit 139) at 37
and 87 prompt tokens; its own comment records it as slower than the per-token loop as well
(6.5-6.8 vs 12.0-13.6 tok/s) from a measurement taken before the fused kernels landed. So there is
currently no working batched prefill path at all, and that - not another percent of decode - is
the top of the queue.

First attempt was the cheapest structure: reuse the gated decode kernel, M times, with the weights
already resident. Measured: **3.4 tok/s** - four times *slower* than the CPU path, and the reason is
structural rather than fixable by tuning. M rows through an M==1 kernel is M synchronised H2D/D2H
round trips per tensor per layer; for a 1717-token prompt that is roughly 400k CUDA calls, and the
weight read - the only cost that matters - is not amortised at all. Reverted.

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
