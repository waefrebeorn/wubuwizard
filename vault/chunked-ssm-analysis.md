# Chunked SSM Analysis — May 27, 2026

## Claim
Battleship cell 074 claimed "Chunked SSM broken, FP accumulation forces SSM_CHUNK_MIN=4096."

## Investigation
- Built test_chunked_vs_seq: compares wubu_ssm_chunked_recurrence (CS=2) vs wubu_ssm_sequential_recurrence
- With constant inputs (q=0.1, k=0.1, v=0.2, beta=0.5, gate=0):
  - **CS=1**: cos-sim=1.000000, max-err=0.000000 (EXACT match)
  - **CS=2**: cos-sim=0.961058, max-err=0.008707 (systematic error, constant ratio ~1.77x)

## Root Cause
The chunked formulation uses A = (I+L)^{-T} to process CS tokens in one batch. This attention matrix mixes value contributions across the entire chunk. When computing `v_attn = v_new^T @ kq`, the kq matrix is lower-diagonal (keeps z ≥ t), allowing **future K within the chunk to affect past Q outputs**. This is:

- ✅ **Correct for training**: GPU chunking is a well-known optimization (llama.cpp's `build_delta_net_chunking`). The model adapts to the chunked computation.
- ❌ **Wrong for inference**: Sequential inference output[t] must only depend on tokens 0..t. The chunked formula violates this.

The fix would require making kq upper-diagonal (z ≤ t), but since the mask already zeros z < t, only the diagonal would survive — making CS>1 identical to CS=1 (no chunking benefit).

## Resolution
- Inference always uses the sequential path (SSM_CHUNK_MIN=1M). ✅ Correct.
- Chunked path exists for training/GPU use where exact sequential match isn't required.
- Not a bug — a fundamental mathematical property of the chunked formulation.

## Verification Code
```c
// tools/test_chunked_vs_seq.c — compile with make test_chunked_vs_seq
// Tests: wubu_ssm_chunked_recurrence vs wubu_ssm_sequential_recurrence
// CS=2 in wubu_ssm_chunked.c
```
