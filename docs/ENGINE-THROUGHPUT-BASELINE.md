# Engine throughput baseline (2026-11-23)

Measured on this host (Ryzen 5 8645HS, RTX 4050 6 GB, 16 GB RAM, Win11) with the
GGUF model this machine actually runs. Command:

```
./gen_text_gemma4.exe C:/Users/eman5/models/gemma-12b/gemma-4-12b-it-UD-IQ2_M.gguf "The meaning of life is" 12 40
```

Result (exit 0, real output, no errors):

```
[G4] Loading model (667 tensors, 4003 MB data blob)
[G4] 48 layers loaded (~4003 MB quantized + 4.2 MB norms)
Prefill: 21 tok in 53.21s (0.4 tok/s)
Decode:  13 tok in 183.15s (0.1 tok/s)
```

For reference, llama.cpp on the same host and model: pp ~196 tok/s, tg ~21 tok/s
(`data/bench/`, prior matrix). The engine's own forward path is ~200x slower than
the runtime it is meant to replace. Gate coverage is not the bottleneck; the forward
path is.

## Why it is slow: the model's weight types

GGUF header of `gemma-4-12b-it-UD-IQ2_M.gguf`, parsed from the tensor-info block:

| Type | Tensors |
|---|---|
| IQ2_S | 204 |
| IQ3_S | 95 |
| IQ3_XXS | 20 |
| IQ2_XS | 5 |
| IQ4_XS | 4 |
| Q3_K | 1 |
| F32 | 338 |

`src/quantized_matmul.c:510` states it plainly:

```c
/* IQ2_S / IQ3_S have no Q8 vec_dot yet: quantized_matmul dequantizes the
 * weight to F32 and calls SGEMM. */
```

So every token dequantises 299 of the model's large matrices (IQ2_S + IQ3_S) to F32
before the matmul. That is tens of GB of F32 traffic per token on a 4 GB model, and it
matches the measured 14 s/token. IQ2_XXS (20 tensors) already has a fused
`iq2_xxs_vec_dot`; the two dominant types do not.

## After EG-111 (fused IQ2_S/IQ3_S x Q8_K dot, 2026-11-23)

The two dominant types now have a Q8_K vec_dot (`src/wubu_vecdot_iq.c`, ported from
`llama/ggml-cpu/quants.c`) and `quantized_matmul_batched` dispatches to them; the
whole-tensor F32 dequantisation branch is deleted. Same command, same host:

| Run | Prefill | Decode |
|---|---|---|
| before (dequant to F32 per call) | 21 tok in 53.21 s (0.4 tok/s) | 13 tok in 183.15 s (0.1 tok/s) |
| scalar fused Q8_K vec_dot | 21 tok in 31.56 s (0.7 tok/s) | 13 tok in 18.57 s (0.7 tok/s) |
| AVX2 fused Q8_K vec_dot | 21 tok in 20.12 s (1.0 tok/s) | 13 tok in 11.63 s (1.1 tok/s) |
| + fused Q3_K LM head (EG-114) | 21 tok in 4.27 s (4.9 tok/s) | 13 tok in 3.06 s (4.2 tok/s) |
| + fused IQ4_XS dot (EG-116) | 4.9 tok/s | 4.5 tok/s |
| + fused IQ2_XS dot (EG-117) | 6.7 tok/s | **5.2 tok/s** |
| + same, `G4_GPU=1` (device IQ2_S/IQ3_S/Q3_K/Q4_0) | - | 7.6 tok/s |
| + v3 warp-per-column IQ2_S kernel (no atomics) | - | 8.3 tok/s |
| clean rebuild, `G4_GPU=1` (all kernels, no stale objects) | 6.8 tok/s | 8.8 tok/s |
| + device IQ3_XXS kernel (EG-120/122) | 6.8 tok/s | 9.8 tok/s |
| + device IQ4_XS kernel (EG-121/122) | 6.8 tok/s | **10.3 tok/s** |
| clean rebuild, CPU only | 6.6 tok/s | 5.7 tok/s |

Note on rebuilding: the Makefile does not track header dependencies, so editing a
header without `rm -f src/*.o` leaves objects of different struct layouts linked
together. That produced 4.4 tok/s decode and a different greedy prefix once; after
a clean rebuild the same source gives 5.7 (CPU) / 8.8 (GPU) and **CPU and GPU now
agree on every greedy token** (45518, 107, 236775, ...), where the earlier GPU run
diverged from step 10.

The AVX2 port (ggml's x86 kernel) keeps the integer accumulation, so it agrees with the
scalar port and the F32 reference to 2.9e-06 / 7.4e-07 relative, and it produces the same
greedy tokens as the scalar run on this prompt.

Decode is 9.9x faster; prefill 1.7x. Correctness: the kernels agree with the
`gguf_dequantize()` reference to 1e-7 relative on 64 synthetic blocks per type (gate
`test_eg111_iq_fused`). Greedy output matches the old run for 9 samples and then
diverges, because activations are Q8_K-quantised instead of F32 — expected, not a bug.

The head was the real bottleneck, and it is **Q3_K**, not Q4_0: the GGUF carries a single
Q3_K `token_embd` (412 MB) and no code path had a Q3_K vec_dot, so every token
dequantised the whole head to F32. EG-114 adds the fused Q3_K x Q8_K kernel; decode went
1.1 -> 4.2 tok/s and prefill 1.0 -> 4.9 tok/s with identical greedy tokens.

Remaining cost is now the layer weights themselves: 2.6 GB of IQ2_S/IQ3_S per token at
~11 GB/s effective. Thread scaling is flat (4/8/12 threads: 0.9 / 1.0 / 1.1 tok/s before
the head fix), so this is a bandwidth wall on the CPU path — the next real step is GPU
residency with repacked tile layouts, not another kernel tweak.

## Next engine work (replaces further gate work)

1. Port `ggml_vec_dot_iq2_s_q8_K` and `ggml_vec_dot_iq3_s_q8_K` from the in-repo
   reference `llama/ggml-cpu/quants.c` (plus `llama/ggml-common.h` block layouts and the
   IQ2_S/IQ3_S grids already present as `src/iq2s_grid.inc`) to C11 in `src/`.
2. Fuse them into `quantized_matmul_batched` for decode (M small): quantise the
   activation row to Q8_K once, then dot per output column. No F32 weight buffer.
3. Gate: identical logits vs the dequant+SGEMM path on 64 fixed random rows (max abs
   diff under 1e-4 for F32 accumulate), then measure tok/s on the same command above.
   Acceptance: decode > 1 tok/s with identical greedy token sequence on 20 prompts;
   a speedup claim requires the measured number in this file, not a PASS line.

## Residency plan wired into the GGUF loader (EG-110 integration)

`WUBU_RESIDENCY_MB=N` (with `G4_STREAM_WEIGHTS=1`) now runs `wubu_residency_plan` over
the real tensor sizes at load: one plan entry per layer for the attention group (hot)
and one for the FFN group (streamed), with the remaining budget handed to the streamed
weight cache as its cap. Measured, 8-token decode, same prompt:

|Where the decode time goes now (`G4_PROF=1`, per decode token, 12B IQ2 model):

```
[PROF] layer total       99.5 ms      (48 layers)
[PROF] norm+qkv          22.0 ms      q/k/v projections + pre-attn norm
[PROF] qk norm+rope       3.5 ms
[PROF] attention          8.4 ms
[PROF] attn out+resid    13.0 ms      o-proj + post-attn norm + residual
[PROF] ffn all           52.5 ms      gate/up/down + gelu + norms
[PROF] head              10.6 ms      262k-vocab Q3_K tied head
```

FFN is 2.7 GB of weights per token, so 52.5 ms means **51 GB/s effective** - about a
quarter of the card's 204 GB/s. `wubu_cuda_bench_stages(20)` on the FFN shape with three
rotating weight buffers (no L2 reuse) reports the same: 0.247 ms per 18.9 MB matvec =
**76.5 GB/s**, versus 240 GB/s when the same buffer is re-read (L2 hits).

The cause is the access pattern, not occupancy: the warp-per-column kernel walks one
column's 82-byte blocks while the eight warps of a block read columns 15 KB apart, so
DRAM sees 82-byte scattered reads instead of 128-byte lines. Split-K was tried and
measured neutral (21.9 vs 21.5 ms for qkv), which rules out occupancy.

Where the decode time goes now (WUBU_GPU_STATS=1, 9 decode tokens):

```
[GPU stats] type  calls   ms_total  MiB(touched)     [CPU stats] decode_calls  decode_ms
           11       8       85.7     (Q3_K head)      11      0        0.0
           18     160       33.3     (IQ3_XXS)        18      0        0.0
           21     760      181.1     (IQ3_S)          21      0        0.0
           22    1632      360.4     (IQ2_S)          22      0        0.0
           23      32       11.1     (IQ4_XS)         23      0        0.0
                                                      17     40       37.6  (IQ2_XS, 4.2 ms/tok)
```

Every matvec except the 5 IQ2_XS tensors (11 MB) is device-resident during decode. GPU
time is ~23 ms of the 97 ms forward, so the remaining ~70 ms/token is not matvec work at
all: it is the rest of the forward (norms, attention, MoE routing, head quantisation).
That is the next thing to instrument, not another kernel.
|---|---|---|---|---|---|
| mmap, no streaming | - | - | 1.0 tok/s | **1.1 tok/s** | - |
| streamed | 3072 MiB | 2199 MiB | **3.4 tok/s** | 0.4 tok/s | 217 / 2744 |
| streamed | 6144 MiB | 5271 MiB | 3.3 tok/s | 0.9 tok/s | 2850 / 111 |
| streamed | 8192 MiB | 7319 MiB | 3.4 tok/s | 0.9 tok/s | 2850 / 111 |

Streaming is **3.4x faster on prefill** (sequential reads + the one-layer-lookahead
prefetch thread) and slightly slower on decode (0.9 vs 1.1 tok/s) even when the whole
streamed set is cached, because every matmul copies bytes out of the cache instead of
reading the mmap directly. Below ~4.4 GiB of cache the streamed set (2720 MiB) does not
stay resident and decode collapses to 0.4 tok/s.

## Windows GPU lane (blocked, documented)

`Makefile.win` now has a `%.o: %.cu` rule, a `gpu-probe` target and a
`gen_text_gemma4_gpu` target. Status:

- Compile clean with nvcc 13.5 / sm_89: `gpu_gemma4_forward.cu`, `gpu_quant_matmul.cu`
  (needed a `__builtin_memcpy` -> `memcpy` shim for the MSVC host compiler),
  `gpu_quant_matmul_row_major.cu`, `flash_attn_q4_0_opt.cu`,
  `flash_attn_q4_0_prefill_opt.cu`.
- Do not compile on Windows: `src/gpu_gemma4.cu` (passes `cudaStream_t*` where a
  `cudaStream_t` is expected; the Linux GPU target never linked it either) and
  `src/cuda_kernels.cu` (SSM kernels use runtime `__constant__` dims as template
  arguments; tracked separately).
- Link fails by design: nvcc and `cudart.lib` emit MSVC-ABI objects;
  `ld.exe` (MinGW) cannot resolve `__security_cookie`, `__GSHandlerCheck`,
  `_Init_thread_epoch`. A Windows GPU engine needs an MSVC/cl.exe link lane
  (host objects built with cl, or a link.exe step), not a MinGW link. The recipe is
  kept in `Makefile.win` as the starting point for that lane.
- The `gen_text_gemma4_gpu` binary at the repo root is a stale Linux ELF and cannot run
  here (`Exec format error`).
