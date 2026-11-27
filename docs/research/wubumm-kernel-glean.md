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
