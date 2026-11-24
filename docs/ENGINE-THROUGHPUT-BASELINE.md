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

The AVX2 port (ggml's x86 kernel) keeps the integer accumulation, so it agrees with the
scalar port and the F32 reference to 2.9e-06 / 7.4e-07 relative, and it produces the same
greedy tokens as the scalar run on this prompt.

Decode is 9.9x faster; prefill 1.7x. Correctness: the kernels agree with the
`gguf_dequantize()` reference to 1e-7 relative on 64 synthetic blocks per type (gate
`test_eg111_iq_fused`). Greedy output matches the old run for 9 samples and then
diverges, because activations are Q8_K-quantised instead of F32 — expected, not a bug.

Remaining gap to llama.cpp (~21 tok/s decode) is now the output head: the logits path
dequantises the whole tied `token_embd` matrix (412 MB, Q4_0) to F32 every token
(`src/wubu_gemma4_model.c:787`), which dominates at 1.1 tok/s. Next: run logits through
`quantized_matmul_batched` (Q4_0/Q6_K vec_dot) instead of dequantising the head, then
thread tuning.

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
