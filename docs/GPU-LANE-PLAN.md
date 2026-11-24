# GPU lane — proven bridge, measured budget, port order (2026-11-23)

The MinGW engine can now call CUDA code at runtime, so the GPU path is no longer blocked
by the MSVC-ABI link wall. This file records what exists, what the device actually has,
and the order in which the remaining work pays off.

## What works today

`src/wubu_cuda_bridge.cu` → `wubu_cuda.dll` (nvcc 13.5, sm_89, MSVC link), loaded by the
MinGW engine with `LoadLibrary` + `GetProcAddress` over `extern "C"` symbols. Gate
`tools/test_cuda_bridge.c` (`mingw32-make -f Makefile.win wubu_cuda.dll test_cuda_bridge`):

```
EG-112 bridge: device = NVIDIA GeForce RTX 4050 Laptop GPU (cc 8.9, 6140 MiB)
EG-112 bridge: gemv worst_abs=0
EG-112 bridge: vram total=6140 MiB free=5080 MiB
test_cuda_bridge: PASS
```

Exported so far: `wubu_cuda_available`, `wubu_cuda_device_name`, `wubu_cuda_sgemv`
(device-resident), `wubu_cuda_sgemv_host`, `wubu_cuda_meminfo`, `wubu_cuda_fit`
(weights + kv + compute <= free - reserve, with overflow guard).

## Measured budget on this host

| Item | Bytes |
|---|---|
| Device total / free at idle | 6140 / 5080 MiB |
| Model (12B IQ2 GGUF data blob) | 4003 MB |
| Headroom after model | ~1077 MiB |
| KV at 4096 ctx (48 layers, 8 heads x 128, F16 K+V) | ~100 MiB |
| Compute scratch (activations, logits row) | ~10 MiB |

So full model residency fits at short context with ~950 MiB slack. The 6 GB target is
comfortable; the 262k-context MoE case is not, and that is exactly what the residency
plan (EG-110) already models in bytes.

## Why the CPU path stops at 4.2 tok/s

Measured after EG-111/113/114: decode 4.2 tok/s, prefill 4.9 tok/s, and thread scaling is
flat (4 / 8 / 12 threads: 0.9 / 1.0 / 1.1 tok/s before the head fix). Layer weights are
2.6 GB per token; at ~11 GB/s effective that is the wall. The IQ2_S/IQ3_S kernels are
grid-gather bound (4 scalar grid lookups per 32 elements), so more threads do not help.

## Port order

1. **Tile repack on upload.** llama.cpp's CUDA path is fast because the weights are
   repacked into tiles at load (see `ggml/src/ggml-cuda/` `dequantize_mul_mat_vec` and the
   `*_tile` layouts). Uploading our GGUF blocks as-is gives uncoalesced reads (column
   stride ~1.2 KB). Repack once into a tile layout on the host, then upload.
2. **`dequant_mul_mat_vec` for IQ2_S / IQ3_S / Q3_K / Q4_0** against Q8_K rows, one warp
   per output column, `__shfl` reduction. This is a direct port of the kernels already
   proven correct on the CPU (`src/wubu_vecdot_iq.c`, `wubu_vecdot_q3_k.c`).
3. **Device-resident weight set** with `wubu_cuda_fit` deciding what fits; everything that
   does not fit stays on the CPU path, layer by layer (the residency plan already produces
   that split).
4. **KV on device**, grow on demand, and a host fallback when a request exceeds the plan.

## What not to do

- Do not upload per matmul: PCIe at ~6 GB/s makes a 4 GB model 0.7 s/token.
- Do not link nvcc objects into the MinGW binary; the DLL boundary is the fix for the ABI
  mismatch and it costs one function-pointer call per kernel launch.
- Do not claim a speedup before the gate measures tokens/s on this host.
