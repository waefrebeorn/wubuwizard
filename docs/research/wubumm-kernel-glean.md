# WuBuMM kernel-optimization glean (for wubuwizard inference path)

Source: `C:/Users/eman5/WuBu/repos/WuBuMM` (github.com/waefrebeorn/WuBuMM), pulled 2026-11-27,
HEAD `c6d0611`. This is the "tandem" work the user is researching for math-optimization routes.
It applies to the wubuwizard 6GB-VRAM / faster-than-upstream goal.

## Flagship wins from the git log (bit-identical where noted)
- Tiled CUDA attention: 7.7x per block vs naive. Kernel in `cuda_h3_attn_tiled.cu`:
  shared-mem Q/K/V tiles, every load via `float4`, `H3T_KV` kv tile + `H3T_PAD` to dodge bank conflicts,
  one 128-dim lane dot per score. Reuse this exact staging for our Inkling attention.
- gemm_tiled: coalesce the A/B tile loads -> bit-identical, block 523->387 ms; then BK 16 -> 32 ->
  block 387->342 ms, bit-identical. Rule: **tile loads coalesced + larger BK (32) are free wins when
  numerically identical.** (A "128x128 tile" was tested and reverted — bigger is not always better.)
- F16C f16->f32 conversion in safetensors loading -> -30% block read time; branchless scalar fallback
  when F16C unavailable. Rule: convert f16 on load with F16C intrinsics, keep a branchless path.

## Wing 7 research (docs/research_wing7_kernel_optim_2026-08-23.md) — adoptability for 8GB class GPUs
Ranked by expected gain for an RTX-4050/8GB target:
1. W8A8 + group quant, group=128, per-channel scales -> large speedup, minimal quality loss.
2. Sliding-window + KV block recompute every 4 steps -> ~40% memory cut.
3. Token merge (ToMe/Sinkhorn) 30%, applied only in middle blocks.
4. Q4 aggressive -> avoid (FID +200-300%).
5. bf16 compute + q8 weights for GEMV -> best speed/memory tradeoff (2-3x).

## Concrete configs from the repo
- compute dtype bf16 (faster than fp16 on RTX 40-series), weights q8 primary, q4 only for MLP after a sanity test.
- group_size 128, per_channel_scales true, reordered_group_quant (RGQ) when supported.
- block_recompute_interval 4, sliding_window_size 512, streaming attention beyond 1024 tokens.
- ToMe merge_ratio 0.30, blocks 2-4 of 6.

## Notes
- `shaders/` in WuBuMath (wubumm) holds only test compute shaders (mobius_add, poincare_dist) —
  no real Vulkan infer kernels embedded yet. WuBuMM has the CUDA kernels; the user reports Vulkan
  speeding up some parts this stack is "stuck on" with CUDA, and is learning the hardware for more.
- Latest WuBuMM commits approach specificity: F16C load, branchless scalar; H3 attention tiled fix;
  safetensors f16; gemm_tiled BK tuning. All bit-identical where claimed.

## WuBuMM measured findings (docs/H3_MEASURED_FINDINGS.md) — the load-bearing lessons for our engine
Pulled again 2026-11-27, now at d773abe (611-file sync; TAE decoder, Vulkan shader audit, attention/tensor-core
benchmarks, INT8 tooling, `wubumm_gguf.c` self-contained GGUF reader, `wubumm_fastmath.h` f16/bf16 tables).

- **qknorm+rope was half the "attention" span and uncoalesced** (thread-per-row, 512-byte lane stride,
  3.51 GB/s vs ~450 available) -> coalesced gives ~100x (30.8 -> 0.32 ms/block). Rule for our Norm/Rope:
  coalesce along the lane-safe contiguous dim, never thread-per-row.
- **Attention at 0.9 TFLOPS was NOT instruction-bound — it was 6.39 GB of avoidable K/V traffic.** Six
  scheduling hypotheses, five refuted (ILP-8 slower, float4 +3.7% noise, bank-conflict fix no change,
  occupancy 12%->25% no change, stride-129 39% slower). Rule: optimize reuse/coalesced volume first;
  do-not-burn budget on ILP/occupancy/conflict micro-tuning.
- **Tensor cores matter**: wmma attention 28.3 -> 11.87 ms (2.36x), QK^T 2.404 TFLOPS exact to 1.5e-7,
  both gated on cos >= 0.999. 16 query rows/block = -14% from reuse volume.
- **Residual stream must stay f32** — fp16 cap 65,504 vs residuals peaking near 5e5; block-0 measured
  28,118 (2.33x headroom). Our Inkling path must not fp16 the residual.
- **Turing has no BF16 hardware** (bf16 emulated, f16 hits tensor cores). Ada (RTX 4050) is different —
  verify, don't assume the wing-7 "bf16 faster" rule transfers.
- **W8A8 cannot meet the cos>=0.999 gate** (best SmoothQuant alpha 0.99821); W8A16 can. Q4 degrades FID.
- **Batching M is the largest serving lever** (-42% at B=2), ahead of INT8: weights stream 77 GB/run vs
  ~120 MB activations, weights ~6x a request.
- **Audit conversions exhaustively** — WuBuMM caught parse_dtype testing len==3 for the 2-char "I8" (so
  every INT8 tensor silently parsed as F32), a bf16 bit-pattern numeric cast returning 1.065e9, and a
  f16_to_f32 that approximated 2^-14 and double-negated inf (-inf -> +1e30). All 65,536-pattern exhaustive
  now. Parallel to our gguf_extract `unsupported type 23` blocker: verify the dtype parser switches on
  exact token-length/kind, not a heuristic.
- **WDDM context-switch tax scales with burst duration, not load** — any CUDA work taxes an active stream.
  Measure bursts <1s with standalone CPU-compiled harnesses on random buffers; never load the checkpoint
  just to time a kernel.
