# bytropix Plan — May 28, 2026

## Priority: ALL GAPS CLOSED — Hardware Ceiling Reached

## PHASE 1: PARITY — IQ2_M FLOOR REACHED

| Step | Status |
|------|--------|
| dump_ref builds + reference logits | ✅ |
| Our logits non-zero (fixed output proj) | ✅ |
| Cos-sim 0.976 vs ref (improved from 0.974) | ✅ (IQ2_M floor — need Q3_K+ to go higher) |
| run-harness.sh → serve_local.py | ✅ |
| test-hermes-headless.sh → serve_local.py | ✅ |

**0.976 = IQ2_M quantization floor.** Need Q3_K/Q4_K/F16 model to reach >0.99.

## PHASE 2: GAINZ — ALL IMPLEMENTED

| Cell | Optimization | Status | Notes |
|------|-------------|--------|-------|
| 241 | SSM buffer pre-allocation | ✅ | Pre-allocated workspace |
| 242 | MoE shared expert quantize-once | ✅ | Q8 once, reuse for gate+up |
| 243 | Q4_K output proj threaded for batch | ✅ | 52x speedup over per-token |
| 244 | KV cache to Q4_0 format | ✅ | 3 modes: Q4_0 / F16 / F32 |
| 245 | Attention sparsity wire for decode | ✅ | sparse_buf stack alloc |
| 246 | MoE expert prefetch | ❌ | No gain on i5-8365U (8MB L3) |

## PHASE 3: COMPILATION FLAGS FIX

| Task | Status |
|------|--------|
| Remove `-ffast-math` from CFLAGS | ✅ `-fno-fast-math` (IEEE 754) |
| Single-token cos-sim improved 0.974→0.976 | ✅ cat prompt |
| Between-builds (fast vs no-fast) cos-sim 0.99975580 | ✅ top-5 argmax identical |
| All 3 cos-sim regression tests pass at 0.975 | ✅ |

## PHASE 4: CONTEXT GROWTH PENALTY — COMPLETE

| Step | Task | Status |
|------|------|--------|
| 5.1 | PROFILE at 2, 50, 100, 200 KV | ✅ |
| 5.2 | Analyze: GQA NOT bottleneck | ✅ |
| 5.3 | Option A: Lower SPARSE_MIN | ✅ |
| 5.4 | Option F: Logit cache N-hop reuse | ✅ 51% speedup |
| 5.5 | Option D: Persistent KV process | ✅ 7.9× multi-turn |
| 5.6 | Option E: Chunked output proj | ❌ 0% cache hit rate |
| 5.7 | Verify 3-turn conversation | ✅ Done — 7.9× overall |

**ALL PHASES COMPLETE.** Remaining items are hardware-gated (GPU, Q3_K+ model). MTP benchmark now possible with 14GB RAM.
