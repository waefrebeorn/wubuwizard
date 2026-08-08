# Context Growth Penalty — Analysis (UPDATED May 28 — RESOLVED)

## The Problem (Now Fixed)
Decode speed dropped 50% as context grew from <1K to ~2K tokens during multi-turn conversation.

### Original Measured Data (3-turn NES Q&A on i5-8365U, NON-PERSIST mode)
| Turn | Context Size | Decode Speed | Notes |
|------|:------------:|:------------:|-------|
| 1 | ~50 tok (just prompt) | ~1.0 tok/s | Cold start, includes model load |
| 2 | ~500 tok (1 prior turn) | ~1.2 tok/s | Warm, short KV cache |
| 3 | ~2000 tok (2 prior turns) | ~0.6 tok/s | Growing KV cache, dense attn O(n²) — **actually process-per-turn re-prefill** |

### Resolution: Persistent KV Process (May 28)
After implementing persistent KV (`gen_text_cpu --persist` + `serve_local.py --persist`):

| Turn | Time (persist) | Δ from Baseline |
|------|:-------------:|:---------------:|
| 1 | 32.1s | 4.5× faster |
| 2 | 31.5s | 5.9× faster |
| 3 | 31.0s | **13.4× faster** |

**Per-turn time CONSTANT (~31s) regardless of KV cache size.** Context growth penalty = eliminated.

## Root Cause (Historical — For Reference)

### Re-Diagnosis (May 27): Not GQA O(n²)
The original assumption that decode drops due to GQA dense attention O(n²) was wrong.
Profiling shows output projection [2048×248320 Q4_K] at 245ms (43.5%) is the real per-token bottleneck.
GQA grows only 37.7→43.5ms (15%) from 2→200 KV positions.

The multi-turn penalty was from process-per-turn architecture in serve_local.py:
each request spawns a new gen_text_cpu and re-prefills the full conversation history
from scratch. NOT from per-token decode slowing down.

### Why Sparse Isn't Active Early
`SPARSE_MIN=4096` (lowered to env-var controlled, default 512). Below this threshold,
dense attention is used regardless of `USE_SPARSE_ATTN=1` env var.

### Other Candidates
1. **SSM recurrence** — 30 SSM layers have O(T×D) recurrence, not O(n²). Should scale linearly.
2. **MoE expert computation** — Fixed 8 active experts per layer regardless of context. Not context-dependent.
3. **Output projection** — Fixed [2048 × 248320] matmul. Context-independent.

## Fix Options (Historical — All Implemented)

### Option A: Lower SPARSE_MIN (Easiest) ✅ Done
Changed `#define SPARSE_MIN 4096` to env-var controlled default 512.
- **Impact**: Sparse attention at 512+ tokens.
- **Risk**: None identified — SSM handles long-range, sparse GQA at short ctx is fine.

### Option B: Q4_0 KV Cache for Decode ✅ Done (Cell 244)
Q4_0 KV cache format (4:1 compression vs F16). No benchmark improvement at short ctx.

### Option D: Persistent KV Process ✅ Done (May 28)
The REAL fix. `gen_text_cpu --persist` + `serve_local.py --persist` keeps model loaded and KV
cache across turns. Eliminates 50% multi-turn decay. Per-turn time CONSTANT ~31s regardless of KV size.
- **Impact**: 7.9× overall improvement on 3-turn conversation
- **Effort**: 8-16h (implemented May 27, verified May 28)

### Option F: Logit Cache N-hop Reuse ✅ Done
Direct logit cache max_hits=2. 51% decode speedup (1.7→2.6 tok/s).

### Option E: Chunked Output Proj (top-K verify) ⬜ Not Yet Implemented
Compute top-K logits, verify argmax stable vs cached, skip full output proj on match.
- **Potential**: 1.5-2× decode speedup
- **Risk**: 100% cache miss rate observed with speculative verify (May 27 attempt)

## Verification
After fix: PROFILE at 5 context lengths (50, 256, 512, 1024, 2048 tok). Report tok/s curve. Run 3-turn conversation, measure per-turn tok/s.

## Related Files
- `src/wubu_ssm.c` — GQA attention forward (dense + sparse paths)
- `include/wubu_model.h` — SPARSE_MIN, SPARSE_W, SPARSE_G defines
- `src/wubu_model.c` — GQA cache management, env var parsing
- `tools/benchmark-context.sh` — TODO: create benchmark script
- `Makefile` — CFLAGS changed: **`-ffast-math` removed (replaced with `-fno-fast-math`)** — IEEE 754 compliance restored

## Compilation Flags Fix (May 28)
**Problem**: `-ffast-math` in CFLAGS enabled `-fassociative-math`, which reorders FP operations
in SSM recurrence dot products (`sum += h[i][j] * k[j]`), changing rounding per-layer.
Over 30 SSM layers × multiple decode tokens, this compounds into visible output divergence.

**Fix**: Replaced `-ffast-math` with `-fno-fast-math` (restores IEEE 754 compliance).
- Single-token cos-sim vs llama.cpp: 0.974 → **0.976** (cat)
- Between builds (fast vs no-fast): cos-sim **0.99975580** (top-5 argmax identical)
- Build: `gcc -O3 -march=native -funroll-loops -ftree-vectorize -fno-fast-math`
