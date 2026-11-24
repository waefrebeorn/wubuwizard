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

## Device attention half-layer (opt-in, `G4_ATTN_DEV=1`) - why it is not the default

It is the faster path: **14.1-14.3 tok/s** against 12.8-12.9 at 65 tokens of context, and
**13.5** against 12.0 at 257, with the fused q/k/v, FFN and incremental prefill all in
place. Its arithmetic also tracks the CPU sequence closely (q/k/v/attention to 4e-07,
o-projection to 1.5e-05, EG-124).

It is still opt-in because of what it generates, not because of a bookkeeping rule. Same
weights, same prompt, 40 tokens: the fused path collapses into a loop of control tokens
(237542, 236779, 236756 repeating) and decodes to `_`, while the CPU-attention path reaches
"This is a classic philosophical question". A 1.5e-05 per-layer deviation compounding over
48 layers is enough to fall off the model's manifold at IQ2 quantisation. Speed that
degrades the continuation is not a speedup.

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
