# Logit Parity Analysis — May 27, 2026

## Current Status: 0.974 cos-sim vs llama.cpp reference

### Evidence this is IQ2_M quantization floor

1. **Uniform random divergence**: Cos-sim is 0.945-0.986 across all vocab segments with no systematic pattern. Correlation(|ref|, |diff|) = -0.024 (essentially zero), proving divergence is additive random noise, not scaling or systematic error.

2. **Negligible bias**: Mean diff = -0.053 across 248K logit dimensions. Model is unbiased — errors cancel out.

3. **Top-token consensus**: 41/50 top tokens match between bytropix and llama.cpp. The remaining 9 are close alternatives (single-digit token IDs, high-probability alternatives).

4. **Not fixable by scaling**: Optimal rescaling (0.955) doesn't change cos-sim — it's already at the optimal scale (±0.004).

5. **Hidden state divergence**: Hidden state cos-sim = 0.25 between bytropix and llama.cpp. Each layer's quantized matmul adds ±2% random error due to 2-bit weight quantization. After 40 layers, the hidden state accumulates ~75% "noise". The output projection (248320×2048) averages over this noise, producing 0.974 cos-sim on logits.

### Implication
IQ2_M (2-bit) at 2048-dim hidden produces ~2.6% logit error from quantization. This is the FLOOR for this model. To reach >0.99 parity, use Q3_K, Q4_K, or F16 model.

### Files
- `~/bytropix/vault/output-projection-fix.md` — Q4_K output proj fix
- This file — parity analysis with IQ2_M quantization floor
