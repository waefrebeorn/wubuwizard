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
| decode at 65 tokens of context | **11.4 tok/s** |
| decode at 129 tokens of context | 10.6 tok/s |
| decode at 257 tokens of context | **9.1 tok/s** |
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
- o-projection shape end to end (`wubu_cuda_bench_oproj`, K=2560 x N=3840, 30 iters):
  upload 0.008 + q8k 0.035 + kernel 0.040 + download 0.026 = **0.110 ms/call**, so 48 calls
  per token should cost **5.3 ms** — against the 12.6 ms the engine charges for the same
  section. The kernel is 78 GB/s at that shape, so the shape is not the problem; ~0.15 ms
  per call is unaccounted for and is the next thing to bisect;
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

## Device attention half-layer (opt-in, `G4_ATTN_DEV=1`)

`q,k,v -> per-head RMS norms -> RoPE -> KV append -> attention -> o-projection` in one
submission, with a per-layer device KV cache (512 positions, primed from the CPU cache).

| mode | decode, 65 tokens |
|---|---|
| default (CPU attention, per-tensor matvecs) | **11.4 tok/s** |
| `G4_ATTN_DEV=1` | **13.5 tok/s** |

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
