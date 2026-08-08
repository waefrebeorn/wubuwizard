# Q4_K Output Projection Fix

## Problem
Bytropix output projection was producing ALL-ZERO logits. The model appeared to work (outputting "!" from "cat") because argmax of all-zero logits = token 0, which decodes to "!" in this tokenizer. The real top token should be [220]="!" with logit ~11.35.

## Root Causes (TWO bugs, both required to produce the symptom)

### Bug 1: GCC -O3 dead-code-eliminates output projection
The output projection was wrapped in:
```c
if (0) { // logit cache disabled
    // HUGE block of cache logic with #pragma omp parallel for
    ...
} else {
    // actual output projection with quantized_matmul
    ...
}
```

GCC with `-O3 -ffast-math -fopenmp` would dead-code-eliminate the entire `else` block. The `#pragma omp parallel for` inside the dead `if(0)` block confused the compiler's control flow analysis, causing it to discard the else branch.

**Fix**: Remove the `if(0) {} else {}` wrapper entirely. Let the output projection run directly.

### Bug 2: Q4_K vec_dot AVX2 path produces zeros
Even when the output projection code runs, the Q4_K vec_dot dispatch via `q4_K_vec_dot()` uses the AVX2 path (via `ggml_vec_dot_q4_K_q8_K_avx2`) which produces zeros on this i5-8365U CPU.

**Fix**: Changed dispatch to use `ggml_vec_dot_q4_K_q8_K_generic` instead of the AVX2 dispatch.

## Verification
- Prefill: 1 tok in ~2.5s with generic vec_dot (was 0.7s with zeros-producing path)
- Cos-sim vs llama.cpp reference: **0.9743** (good but not perfect - IQ2_M quantization limit)
- Top-10 token predictions match reference

## Next Steps for Parity
- 0.974 cos-sim may be IQ2_M quantization floor (2-bit at 2048-dim). Try T10/T50 model for debug cycles.
- Layer-by-layer cos-sim to find exact divergence points
