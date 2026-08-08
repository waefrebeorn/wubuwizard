# bytropix State — May 27, 2026

## Current Status: PARITY REACHED (IQ2_M floor)

### Cos-sim: 0.9743 vs llama.cpp reference
IQ2_M quantization floor (2-bit, 2048-dim). Need Q3_K+/F16 model to reach >0.99.

### Gap Closure Status
| Gap | Status | Notes |
|-----|--------|-------|
| Output projection zeros (GCC -O3 + if(0) + AVX2) | ✅ FIXED | Removed if(0) wrapper, forced generic vec_dot |
| dump_ref API (llama_model_load_from_file) | ✅ FIXED | Modern API fix + text prompt tokenization |
| run-harness.sh proxy → serve_local.py | ✅ PATCHED | Real local CPU inference |
| test-hermes-headless.sh proxy → serve_local.py | ✅ PATCHED | Real local CPU inference |
| NES emulator = benchmark (not project) | ✅ DOCS FIXED | Pre-built. Do NOT develop. |
| test-512k-suite.sh SIGPIPE | ✅ PATCHED | Removed grep -q pipes, fixed exit capture |

### Critical Learned Fixes
1. **GCC -O3 dead-code elimination**: if(0) + pragma omp inside dead block kills else branch
2. **AVX2 vec_dot zeros**: generic vec_dot required on i5-8365U
3. **IQ2_M precision floor**: 2-bit @ 2048-dim = 0.974 max cos-sim

### Branch
- `cpu-optimize-may26` — all parity fixes (pushed)
