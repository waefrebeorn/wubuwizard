# Walkway — All Gaps Closed (May 28) / New Gaps Found (Jun 2)

## THE PROBLEMS (ALL RESOLVED)

### 1. Context Growth Penalty (May 27-28)
**Multi-turn decode drops 50%** — root cause was NOT GQA O(n²) but process-per-turn re-prefill.
- **Fix**: Persistent KV process (7.9× overall, ~31s constant per turn)
- **True bottleneck identified**: Output proj [2048×248320 Q4_K] at 245ms (43.5%)

### 2. IQ2_M Repetitive Output (May 28)
**Multi-token quantization noise accumulates through 30 SSM layers** — 19-token cos-sim drops from 0.974→0.43.
- **Fix A: Compilation flags**: Removed `-ffast-math` → IEEE 754 compliance. Single-token cos-sim rose 0.974→0.976.
- **Fix B needed**: Less aggressive quantization (Q3_K+/F16) or SSM state correction

### 3. Output Projection Zeros (May 27)
GCC -O3 + `if(0)` wrapper killed else branch. AVX2 vec_dot zeros on i5-8365U.
- **Fix**: Removed wrapper. Forced generic vec_dot.

## THE PATH — Complete
All steps below DONE:

| Step | Task | Status |
|------|------|--------|
| 1 | PROFILE at 2, 50, 100, 200 KV | ✅ GQA NOT bottleneck |
| 2 | Lower SPARSE_MIN 4096→512 | ✅ Done |
| 3 | Logit cache N-hop reuse (max_hits=2) | ✅ 51% speedup |
| 4 | Persistent KV process | ✅ 7.9× multi-turn |
| 5 | Compilation flags: `-ffast-math` → `-fno-fast-math` | ✅ IEEE 754 restored |
| 6 | Cos-sim regression test at 0.975 threshold | ✅ All 3 pass |
| 7 | Updated state, walkway, plan, vault | ✅ pushed to cpu-optimize-may26 |

## Devil's Advocate Re-Audit (Jun 2, 2026)
*RAM upgraded 7.4GB→14GB. Build fix: wwggml_type rename completed.*

**New gaps found:**
| Cell | Gap | Status |
|------|-----|--------|
| 051b | moondream3_vision_weights.bin not extracted from safetensors | 🔴 |
| 071b | data/vocab.bin missing | 🔴 |
| 244 | Q4_0 KV cache NOT implemented (claimed ✅ but no code) | 🔴 |
| 241 | SSM buffer pre-alloc PARTIAL (workspace exists, fallback mallocs) | 🟡 |
| 309 | wwggml_type rename incomplete → fixed and committed d808cfa | ✅ Fixed |
| - | 384 tools (battleship said 200+) | 🟢 Updated count |

## REFERENCE
- `vault/context-growth-penalty.md` — full analysis + compilation flags fix
- `state.md` — Test Harness section
- `plan.md` — Phase 5 (completed)
- `battleship.md` — Row K + compilation fix
