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

## First end-to-end attempt: measured, slower than the CPU path

`G4_GPU=1` now uploads all 337 quantized tensors (4005 MiB) at load and routes decode
matvecs through `wubu_cuda_matvec_dev`, with the activation quantized on the device.

| Run (12-token prompt) | prefill | decode |
|---|---|---|
| CPU, fused kernels (EG-111/113/114) | 4.9 tok/s | **4.2 tok/s** |
| GPU resident, per-matmul host round-trip | 5.1 tok/s | 1.5 tok/s |

Two findings, both real:

1. **The kernel is not the problem, the interface is.** Each matvec uploads the row,
   quantizes, launches and downloads, so every one of ~340 matvecs per token pays a
   host round-trip. Measured per-matmul speedup of the kernel itself is 2.6x
   (`test_eg115_cuda_matvec`: 15360x3840 IQ2_S, 19.9 ms CPU vs 7.8 ms GPU), which the
   round-trips throw away.
2. **The GPU path is not bit-identical.** Greedy tokens match for 9 steps and then
   diverge (818 vs 140 at step 10), because the device Q8_K quantizer rounds through a
   different code path. Any promotion of this path needs a token-agreement gate, not a
   tolerance check.

Two fixes were applied and measured:

- Grids moved from `__constant__` to `__device__` memory: every lane indexes a different
  grid entry, and constant memory serializes divergent addresses. Per-matmul speedup went
  from 2.6x to **5.5x** (`15360x3840` IQ2_S: 20.1 ms CPU single-thread vs 3.7 ms GPU
  including upload).
- The per-call `cudaDeviceSynchronize` calls were dropped (overlapping launches, the
  result download is the only sync). End-to-end decode went 1.5 -> **4.0 tok/s**.

Third fix, the decisive one: a **lane-per-grid-entry** IQ2_S kernel where the 32 lanes of a
warp cover one 256-element block exactly (lane = group*4 + entry), with `__dp4a` for the
4-wide int8 multiply-add. Device-resident cost of a 15360x3840 IQ2_S matvec:

| kernel | ms/call | vs 1 CPU thread |
|---|---|---|
| scalar CPU port | 19.8 | 1.0x |
| warp-per-column, no dp4a, includes upload | 3.6 | 5.5x |
| lane-per-entry + dp4a, includes upload | 3.1 | 6.5x |
| lane-per-entry + dp4a, device-resident | **0.35** | **56x** |

Engine decode with the dp4a kernel and device-resident weights: **5.5 tok/s** (CPU 4.2).
The gap to the 0.35 ms/call ceiling is the per-matvec host round-trip and the IQ3_S/Q3_K
types still on the generic kernel.

Known defect: the IQ3_S lane-per-entry kernel returns non-finite values on 5 of 64 test
columns, so IQ3_S stays on the generic kernel and the gate records the failure instead of
hiding it.

That is still a *tie* with the CPU path (4.2 tok/s), and the arithmetic explains why: the
CPU engine already uses 12 threads at ~36 GMAC/s, while a single GPU matvec of the FFN
shape runs 59 MMAC in ~1.5 ms (~40 GMAC/s). The GPU only wins by 10-20x when the weights
are laid out so warps read coalesced tiles — i.e. after repacking, which is the remaining
work. Tokens still diverge at step 8-10 vs the CPU path because the device Q8_K quantizer
rounds differently.

Until then the GPU path stays opt-in (`G4_GPU=1`) and the CPU path is the default.

## Where the remaining decode time goes (measured)

`WUBU_GPU_STATS=1` prints per-type matvec counts and milliseconds for both the device
path and the CPU fallback, plus a forward-vs-sample split. One 21-token decode run on
the 12B IQ2 GGUF with `G4_GPU=1`:

```
[decode split] forward 188.3 ms/tok, sample+mask 0.1 ms/tok
[GPU stats]   11   20 calls   236 ms   (Q3_K head, 12 ms/call)
[GPU stats]   21 1900 calls   501 ms   (IQ3_S, 0.26 ms/call)
[GPU stats]   22 4080 calls  1253 ms   (IQ2_S, 0.31 ms/call)
```

193 IQ2_S/IQ3_S matvecs per token at 0.26-0.31 ms each = 96 ms, of which the kernel
itself is ~0.02 ms: **the host round-trip (2 memcpys + 3 launches + sync) is 90% of every
call**. The other ~92 ms/token is CPU work inside the forward (norms, attention, GELU,
residuals) that has no device path yet.

So the next step is not another kernel: it is to stop paying a round-trip per matvec —
batch the 2-3 matvecs that share an input per layer into one call, then move the whole
forward to the device. Until that lands, `G4_GPU=1` stays opt-in at 5.4 tok/s vs 4.2 on
the CPU path.

## What not to do

- Do not upload per matmul: PCIe at ~6 GB/s makes a 4 GB model 0.7 s/token.
- Do not link nvcc objects into the MinGW binary; the DLL boundary is the fix for the ABI
  mismatch and it costs one function-pointer call per kernel launch.
- Do not claim a speedup before the gate measures tokens/s on this host.
