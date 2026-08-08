# Chunked SSM Recurrence Error Analysis

**Date:** May 27, 2026  
**Tool:** `test_chunked_ssm_error` at `tools/test_chunked_ssm_error.c`  
**Build:** `make test_chunked_ssm_error`  
**Usage:** `./test_chunked_ssm_error [T]` (T=sequence length, default 64)

## Finding: Chunked SSM (CS=2) is Fundamentally Approximate

The chunked SSM recurrence (`wubu_ssm_chunked.c:22`, `#define CS 2`) processes 2 tokens per chunk using a local attention matrix approximation. This is NOT a precision/FP-accumulation issue — it's a formula-level approximation that produces measurably different results from the exact sequential recurrence.

## Error Magnitude

| T | cos-sim (random) | cos-sim (constant) | Interpretation |
|---|---|---|---|
| 4 | 0.026 | 0.961 | Already diverged at T=4 |
| 8 | 0.071 | — | Getting worse |
| 16 | 0.058 | — | No improvement |
| 32 | 0.084 | — | Still poor |
| 64 | 0.288 | — | Slight regression |
| 128 | 0.083 | — | Catastrophic |
| 256 | -0.080 | — | Anti-correlated |

**Key insight:** With identical constant inputs (q=0.1, k=0.1, v=0.2, all same across T), chunked matches sequential at cos-sim=0.961, max-err=0.009. With varying (random) inputs, cos-sim drops to 0.026 at T=4.

## Root Cause

The chunked algorithm computes a LOCAL attention matrix within each CS-sized chunk using a triangular solve approximation. For identical inputs this solve is exact (all chunk elements same), but for varying inputs the approximation introduces layout-dependent errors.

The chunked formula (from llama.cpp `delta-net-base.cpp` `build_delta_net_chunking()`) uses:
1. Decay mask: `M[i][j] = exp(G[j]-G[i])` for causal masking
2. KB, KQ matrices: dot products within the chunk
3. Triangular solve: `(I+L)^T X = -L` → attention matrix A = I+X
4. Intra-chunk attention: various products using A

The triangular solve step (lines 192-200 in wubu_ssm_chunked.c) is where the layout-dependent errors originate. For non-uniform q/k values within a chunk, the solved matrix X differs from the ideal attention that the sequential path computes.

## Implications

- **Cell 074 is a research problem**, not a code bug fix
- The `SSM_CHUNK_MIN=4096` guard in wubu_ssm.c is correct — defaulting to sequential
- Fixing CS>1 requires deriving a more stable chunking formula or accepting the approximation
- For current inference (T << 4096), the sequential path is always used, so this doesn't affect decode speed

## Related Code

- `src/wubu_ssm_chunked.c` — Chunked SSM implementation (CS=2)
- `src/wubu_ssm.c:645-654` — Dispatch to chunked vs sequential (SSM_CHUNK_MIN)
- `tools/test_chunked_ssm_error.c` — Error quantification tool
- `tools/test_chunked_vs_seq.c` — Debug comparison with constant inputs
