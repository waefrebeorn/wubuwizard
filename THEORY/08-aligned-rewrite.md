# THEORY/08 — The Aligned Rewrite: Geometry Designed for Every Trick

## Core Principle
> "because we designed the model, we can align everything and redesign
> everything to make every trick work so that everything is the best." — WaefreBeorn

Since we control the model architecture, we can choose dimensions that are
**divisible by every optimization constant simultaneously**: quant block
size (256), SIMD tile width (64 for AVX-512), 2:4 sparse mask (4), VNNI
int8 tile (16), and KV cache blocks. No remainder handling, no generic
fallbacks, no zero-fill guards — every path is the fast path.

## The Alignment Table
Every dimension must satisfy this divisibility grid:

| Dimension | QK_K (256) | AVX-512 (64) | 2:4 Sparse (4) | VNNI (16) | Notes |
|-----------|-----------|-------------|----------------|-----------|-------|
| d_model   | ✓ div 256 | ✓ div 64    | ✓ div 4        | ✓ (via heads) | 448→512 |
| ffn_dim   | ✓ div 256 | ✓ div 64    | ✓ div 4        | ✓          | 1228→1280 |
| head_dim  | —         | ✓ div 64    | ✓ div 4        | ✓ div 16   | stays 64 |
| heads     | —         | ✓ div by 4  | ✓ div 4        | ✓          | 7→8 |

## Key Insight: Keep the Checkpoint Head Structure
The checkpoint (seed-sft2) has `head_dim=64`. The aligned engine KEEPS
`head_dim=64` (already VNNI-aligned: 64÷16=4) and only pads `heads`
upward from 7→8 and `dim` from 448→512. This means:
- **Per-head weights (q_norm, k_norm, q/k/v projections)** map 1:1 — no
  truncation or restructuring needed.
- **Weight matrices** (q_proj, o_proj, g_proj) are zero-padded: pad rows
  (dead heads 8) + pad cols (dim 449-512). Dead weights contribute zero
  to activations — no accuracy loss.
- **FFN** (gate_up, down) padded similarly: 1228→1280 ffn_dim.
- **Embedding/norms** padded 448→512.

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

## Loader: load_tensor_aligned
The loader (`src/wubu.c`) pads checkpoint tensors from native (448) to
aligned (512) geometry via `load_tensor_aligned()`, which:
1. `calloc`s the aligned buffer (zeroed)
2. Reads the checkpoint's native `[chk_rows, chk_cols]` values
3. Scatter-copies into the first `chk_rows` rows × `chk_cols` cols
4. Pad rows/cols are zero (dead weights, no accuracy impact)

The probe stores `ckpt_dim`, `ckpt_ffn_dim`, `ckpt_head_dim`,
`ckpt_heads`, `ckpt_kv_heads` for the loader to use as native sizes.

## Parameter Count
- Checkpoint native: 35,073,216 params (35M)
- Aligned (zero-padded): 42,221,056 params
- The 7M extra params are zero-padded (dead) — same model behavior,
  different memory layout. New training runs use the aligned geometry
  natively (no padding).

## Verification
Gate: `make test_wubu_alignment` — 13 checks, all passing:
- d_model=512 ✓, ffn_dim=1280 ✓, head_dim=64 ✓, heads=8 ✓
- All dims divisible by 256, 64, 4, 16
- Old dim=448 (FAILS alignment, proves redesign needed)
- KV cache Q8_0 block tiles evenly (2 tokens/block)
- Q4_K input dim divides QK_K (AVX2 route, no zero-fill)

Gate: `make test_wubu35_dims_run` — probe + forward + param count, PASS.

## Related: THEY/07 (quantized_dot_generic.c)
The zero-dequant kernels already exist (Q4_K, Q5_K, Q6_K AVX2 paths).
The dispatch gap was: `quantized_matmul.c` called `_generic` directly.
Now `q4_K_vec_dot` dispatcher routes to AVX2 when `__AVX2__` is defined
and `n % QK_K == 0` (always true with aligned geometry).
