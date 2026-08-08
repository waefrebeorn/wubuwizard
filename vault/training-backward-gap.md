# Training Backward Integration — Findings

## Root Cause of Backward Crash

The SSM backward function (`wubu_ssm_backward`) crashed because all F32 weight pointers (`w->ssm_out_weight`, `w->attn_qkv_weight`, `w->attn_gate_weight`) are NULL — the model only loads quantized versions. Additionally, `beta_flat`/`gate_flat` raw-to-computed conversion needed for backward_recurrence.

## Fix Applied (May 27, 2026)

Three dequant-on-demand fallbacks added to `wubu_ssm_backward`:

1. **ssm_out_weight**: `wubu_ssm_backward_output_proj` now accepts quantized weight params. When F32 weight is NULL, dequantizes via `gguf_dequantize` to temp buffer and frees after.

2. **beta_flat/gate_flat**: When NULL, computed from `beta_raw` (sigmoid) and `alpha_raw + dt_bias` (softplus) × `ssm_a`.

3. **attn_qkv_weight / attn_gate_weight**: In the backward matmul section (steps 1-3), dequantized on-demand before `backward_matmul_nt` calls.

Memory overhead: ~128MB peak (32MB ssm_out + 64MB qkv + 32MB gate), freed after each layer's backward.

## Status: Cell 150 RESOLVED ✅

| Weight | F32 ptr | Quant ptr | Forward uses | Backward needs | Status |
|--------|---------|-----------|--------------|----------------|--------|
| ssm_out_weight | NULL | valid | quantized path | F32 ptr (dequant fallback) | ✅ |
| attn_qkv_weight | NULL | valid | quantized path | F32 ptr (dequant fallback) | ✅ |
| attn_gate_weight | NULL | valid | quantized path | F32 ptr (dequant fallback) | ✅ |
| ssm_beta_weight | ✅ F32 | — | F32 | F32 ✅ | ✅ |
| ssm_alpha_weight | ✅ F32 | — | F32 | F32 ✅ | ✅ |
| ssm_conv1d_weight | ✅ F32 | — | F32 | F32 ✅ | ✅ |

### Fix Applied (May 27)

Dequant-on-demand (Option C) implemented. Forward path unchanged — backward path dequantizes to temp F32 buffers as needed.

### Next Steps for Training

1. ✅ Cell 150 resolved: `wubu_ssm_backward` passes with real model
2. Wire per-layer backward into train_real.c
3. Full training loop with synthetic data validation
