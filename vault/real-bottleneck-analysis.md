# Real Bottleneck Analysis — May 27, 2026

## Original Claim
Decode speed drops 50% (1.2→0.6 tok/s) as context grows from <1K to ~2K.
Root cause: dense GQA attention O(n²) at short/medium context.

## Measured Reality (PROFILE=1, 4 threads, IQ2_M, i5-8365U)

### Per-token decode time breakdown:

| Component | Short (6 KV) | 200 KV | Growth | Notes |
|-----------|:-----------:|:------:|:------:|-------|
| GQA attn (10 layers) | 37.7ms | 43.5ms | +15% | Sub-linear scaling, ~8% of decode |
| SSM attn (30 layers) | 251ms | 130ms | −48% | Decreases with warm cache |
| MoE (40 layers) | 144ms | 144ms | ±0% | Context-independent |
| Output proj | 245ms | 245ms | ±0% | **CONTEXT-INDEPENDENT** |
| **Total** | **~563ms** | **~563ms** | | **1.8 tok/s steady state** |

### Key Findings

1. **GQA is NOT O(n²)** — grows only 37.7→43.5ms (15%) going from 2→200 KV positions. It's sub-linear because AVX2 dot products and memory bandwidth are the bottleneck, not O(n²) complexity.

2. **Output projection is the real bottleneck** — [2048 × 248320] Q4_K matmul takes ~245ms per decode step, accounting for 43.5% of total decode time. This is context-independent (always the same matmul).

3. **SSM decreases with warm cache** — first few decode steps have cold cache misses (251ms), stabilize at ~130ms after ~20 tokens.

4. **Multi-turn conversation penalty comes from per-turn process restart**, not decode decay:
   - serve_local.py spawns `subprocess.run(gen_text_cpu)` for EACH request
   - Each turn rebuilds KV cache from scratch by pre-filling the full conversation history
   - Turn 3 (~2000 tok context): prefill dominates (~555s at 3.6 tok/s), not decode

### The Real Bottleneck Stack (Decode Only)

```
Output proj [2048×248320 Q4_K]: 245ms (43.5%) ← HARDWARE BOUND
MoE (40 layers):                144ms (25.6%)
SSM (30 layers):                130ms (23.1%)
GQA (10 layers):                 43ms  (7.7%)
```

### Why Sparse Attention WON'T Fix This

Sparse attention at 512+ tokens reduces GQA from `O(L·KV)` to `O(L·(w+g))` where w=512, g=128.
- Dense at 2000 KV: 10 layers × 2000 = 20K Q·K pairs
- Sparse at 2000 KV: 10 layers × (512+128+1) = 6410 pairs
- Savings: 68% fewer Q·K pairs
- But GQA is only 7.7% of decode → saving 68% of 7.7% = 5.2% total speedup
- **Sparse at 512 gives at best 5% overall improvement**

### ACTUAL Fix Approaches (by ROI)

| Priority | Fix | Est. Gain | Effort | Description |
|----------|-----|:---------:|:------:|-------------|
| P0 🔴 | Persistent KV cache across conv turns | 5-10× multi-turn | 8-16h | Keep gen_text_cpu alive across requests, only prefill delta |
| P0 🔴 | Logit cache with speculative verify | 1.5-2.5× decode | 4-8h | Skip output proj when top-1 token stable |
| P1 🟡 | Q4_K output proj → F16 chunked | 1.1-1.3× decode | 4-8h | Partial compute: only top-K then verify |
| P2 🟢 | Lower SPARSE_MIN | ~5% decode | 15min | Already done (4096→512) |
| P2 🟢 | OMP thread scaling | ~5% decode | 1-2h | Vary threads by context size |

## Conclusion

The "context growth penalty" as originally diagnosed (GQA O(n²) causing throughput decay) is incorrect. GQA scales sub-linearly and is dwarfed by the output projection. The real performance issues are:

1. **Process-per-turn architecture** in serve_local.py eliminates KV cache between turns, causing massive prefill overhead 
2. **Output projection is hardware-bound** at ~245ms per decode step

To make meaningful speed gains, focus on persistent KV cache and logit cache speculative verify, not sparse attention tuning.
