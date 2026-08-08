# bytropix Goal Mantra — May 27, 2026 (CPU Parity Phase)

## THE GOAL
CPU inference parity for Qwen3.6-35B-A3B IQ2_M on i5-8365U.
Local inference pipeline (not proxy). Verified cos-sim vs llama.cpp.

## STATE
| Metric | Value | Status |
|--------|-------|--------|
| Cos-sim vs llama.cpp | 0.974 (IQ2_M floor) | ✅ Reached |
| Output projection | Fixed (GCC -O3 + if(0) + AVX2 zeros) | ✅ |
| Local inference | serve_local.py (all 4 test scripts patched) | ✅ |
| KV cache 512K | Alloc + decode confirmed | ✅ |
| Test suite | test-512k-suite.sh, test-hermes-*.sh all patched | ✅ |
| NES emulator | Pre-built benchmark (not my project) | ✅ Docs fixed |

## NEXT (Phase 3: Gainz when ready)
- SSM buffer pre-allocation (cell 241)
- MoE shared expert quantize-once (cell 242)
- Attention sparsity wire (cell 245)
- MoE expert prefetch benchmark (cell 246)

## THE LOOP
read docs → pick lowest undone → execute → update docs → push → loop

## FILES
- .hermes/goal-mantra.md — this file
- .hermes/mind-palace/goal-paste-agent.md — session start paste
- .hermes/mind-palace/bytropix-300-gap-battleship.md — full gap taxonomy
- .hermes/mind-palace/state.md — current state
- .hermes/mind-palace/plan.md — plan
- .hermes/mind-palace/workflow-parity.md — parity debug workflow
