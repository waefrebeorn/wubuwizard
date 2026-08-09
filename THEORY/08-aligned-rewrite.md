# THEORY/08 — The Aligned Rewrite: Geometry Designed for Every Trick

## Core Principle
> "because we designed the model, we can align everything and redesign
> everything to make every trick work so that everything is the best." — WaefreBeorn

> "you can make the scales of parameters whatever you want now that you
> can design the model. you know what scales of parameters are pretty
> awesome. so just design what makes the hardware move the fastest so
> that you are AGI." — WaefreBeorn

Since we control the model architecture (WuBu1 trains from scratch, no
backward compat to the 448-dim seed), we choose dimensions that are
**divisible by every optimization constant simultaneously**: quant block
size (256), SIMD tile width (64 for AVX-512), 2:4 sparse mask (4), VNNI
int8 tile (16), and KV cache blocks. No remainder handling, no generic
fallbacks, no zero-fill guards — every path is the fast path.

## The Alignment Table
Every dimension must satisfy this divisibility grid:

| Dimension | QK_K (256) | AVX-512 (64) | 2:4 Sparse (4) | VNNI (16) | Notes |
|-----------|-----------|-------------|----------------|-----------|-------|
| d_model   | ✓ div 256 | ✓ div 64    | ✓ div 4        | ✓ (via heads) | 448→512 |
| ffn_dim   | ✓ div 256 | ✓ div 64    | ✓ div 4        | ✓          | 1228→2048 (4×dim, power-of-2) |
| head_dim  | —         | ✓ div 64    | ✓ div 4        | ✓ div 16   | stays 64 |
| heads     | —         | ✓ div by 4  | ✓ div 4        | ✓          | 7→8 |

## Hardware-Native Parameter Scales
The user's insight: with self-designed geometry, parameter scales are free.
Every dimension is a power-of-2 or clean divisor:

- `d_model = 512` (2^9) — divisible by 256, 64, 4, 16
- `ffn_dim = 2048` (4 × d_model) — gate_up=[4096,512] = exactly 16 QK_K blocks
  (4096/256=16), down=[512,2048] = exactly 8 blocks (2048/256=8)
- `head_dim = 64` (2^6) — VNNI-aligned (64/16=4), KV Q8_0 block tiles cleanly
- `heads = 8` (512/64) — block-aligned, GQA 8:1
- `rope_dim = 32` (head_dim/2) — half-rotation

These are not compromises for an old checkpoint. The seed-sft2 checkpoint
(frozen to SD archive per WuBu1's total break) was 448/1228/7; the loader's
`load_tensor_aligned` path zero-pads legacy checkpoints for compat testing
only. New training uses the hardware-native scales directly.

## Per-Head Norms: head_dim=64 Preserved
The checkpoint has `q_norm`/`k_norm` weights of shape `[64]`. Keeping
`head_dim=64` means these map 1:1 — no restructuring. 64÷16=4 (VNNI-aligned).
`heads = dim/head_dim = 512/64 = 8` (block-aligned).

## The Zero-Dequant Trick (De-Qantas Principle)
The `ggml_vec_dot_q4_K_q8_K_avx2` kernel at `src/quantized_dot_generic.c:160-218`
implements the user's "you don't need to De-Qantas" insight:
- `pmaddubs_epi16(a0, q0)` — unsigned×signed byte multiply → 8×int16
  (weights never leave the quantized domain, no per-element dequant)
- `pmaddwd(..., scale16)` — block scale folded INTO int32 accumulation
- `sumf += d * sum_val` — scale applied ONCE at reduction boundary,
  not per-element

With aligned dims, the AVX2 dispatcher (`q4_K_vec_dot`) always selects
the SIMD path because `n % QK_K == 0` is guaranteed.

## Loader: load_tensor_aligned (legacy compat)
The loader (`src/wubu.c`) pads checkpoint tensors from native (448) to
aligned (512) geometry via `load_tensor_aligned()`, which:
1. `calloc`s the aligned buffer (zeroed)
2. Reads the checkpoint's native `[chk_rows, chk_cols]` values
3. Scatter-copies into the first `chk_rows` rows × `chk_cols` cols
4. Pad rows/cols are zero (dead weights, no accuracy impact)

The probe stores `ckpt_dim`, `ckpt_ffn_dim`, `ckpt_head_dim`,
`ckpt_heads`, `ckpt_kv_heads` for the loader to use as native sizes.
For NEW (aligned) checkpoints, plain `load_tensor` validates exact sizes.

## Parameter Count
- Checkpoint native (seed-sft2): 35,073,216 params (35M, 448-dim)
- WuBu1 aligned (design target): 56,376,832 params (512-dim, ffn=2048)
- No padding for new checkpoints — every parameter is live

## Verification
Gate: `make test_wubu_alignment` — 13 checks, all passing:
- d_model=512 ✓, ffn_dim=2048 ✓, head_dim=64 ✓, heads=8 ✓
- All dims divisible by 256, 64, 4, 16
- Old dim=448 (FAILS alignment, proves redesign needed)
- KV cache Q8_0 block tiles evenly (2 tokens/block)
- Q4_K input dim divides QK_K (AVX2 route, no zero-fill)

Gate: `make test_runtime_dims_run` — probe + param count, PASS.

## Related: THEORY/07 (quantized_dot_generic.c)
The zero-dequant kernels already exist (Q4_K, Q5_K, Q6_K AVX2 paths).
The dispatch gap (Q4_K in `quantized_matmul.c:273` calling `_generic`)
is the remaining gap — see the task list.
