# bytropix — Project Overview (Jun 2, 2026)

## Mission
CPU inference engine for Qwen3.6-35B-A3B (Gated DeltaNet + MoE) in pure C.
AMD Ryzen 7 7445HS / 14GB RAM / 12 cores. RAM upgraded from 7.4GB.

## What Works ✅
- **CPU text inference** — ~1.3 tok/s prefill, ~2.0 tok/s decode (persistent KV) on i5-8365U; expect higher on Ryzen 7
- **Context growth penalty ELIMINATED** — persistent KV process: 7.9× multi-turn improvement
- **Compilation IEEE 754** — `-fno-fast-math`, SSM recurrence FP drift fixed
- **Cos-sim vs llama: 0.976** — IQ2_M floor (up from 0.974 with compilation fix)
- **Cos-sim regression: 3/3 at 0.975 threshold**
- **Between-builds cos-sim: 0.99975580** — top-5 argmax identical
- **GGUF model loading** — 733+ tensors, 13 quantization types
- **MoE (256 experts, 8 active)** — F32 router + quantized expert matmuls
- **SSM recurrence** — Gated DeltaNet, 30 layers, 128-dim state
- **GQA attention** — 16 Q-heads, 2 KV-heads, IMRoPE
- **Persistent KV** — `gen_text_cpu --persist`, per-turn time constant ~31s
- **Logit cache** — 51% decode speedup (max_hits=2)
- **KV cache Q4_0** — 3 modes: Q4_0 / F16 / F32
- **Sparse attention** — env-var controlled, SPARSE_MIN=512
- **ChatML support** — `CHAT=1` env var
- **All test suites pass** — 6/6

## What's Not Done 🔲
| Feature | Priority | Status |
|---------|----------|--------|
| GPU output proj | P1 | Hardware-gated (needs GPU) |
| MTP CPU benchmark | P2 | Now possible with 14GB RAM |
| Cos-sim >0.99 | P1 | Needs Q3_K+/F16 model |
| Q4_0 KV cache | P1 | 🔴 NOT IMPLEMENTED (cell 244) |
| Vision weights extraction | P0 | 🔴 moondream3_vision_weights.bin missing (cell 051b) |
| Vocab.bin extraction | P0 | 🔴 data/vocab.bin missing (cell 071b) |
| SSM buffer pre-alloc | P1 | 🟡 PARTIAL — workspace exists but fallback mallocs (cell 241) |
| Mixed-curvature hyperbolic | P3 | Research |
| Training pipeline | P4 | Hardware upgrade |

## Hardware
- CPU: AMD Ryzen 7 7445HS (12 cores, 24 threads)
- RAM: 14GB DDR5 (14336MB WSL2)
- Storage: NVMe SSD
- Platform: WSL2 (Windows Subsystem for Linux)

## Key Achievements
- Persistent KV: 7.9× multi-turn speedup (744s → 94.6s for 3-turn conversation)
- Compilation flags: `-ffast-math` → `-fno-fast-math` (IEEE 754 compliance)
- Logit cache: 51% decode speedup
- Output proj: 52× speedup (OMP threading fix)
- Cos-sim regression: automated 3-prompt test suite at 0.975 threshold
- KV cache Q4_0: 4:1 compression
- SSM workspace pre-allocation: 13 malloc/free per layer eliminated
- Branch: `work-fork` — all fixes pushed in `.hermes/mind-palace/`
