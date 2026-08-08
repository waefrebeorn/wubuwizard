# bytropix Plan — CPU Parity Phase (May 27, 2026)

## Phase 1: Output Projection Fix ✅
- Zero logits diagnosed: GCC -O3 + if(0) wrapper killed else branch
- Q4_K AVX2 vec_dot zeros on i5-8365U: forced generic vec_dot
- Cos-sim 0.974 vs llama.cpp (IQ2_M quantization floor)

## Phase 2: Infra Parity ✅
- All 4 test scripts patched to serve_local.py (real local inference)
- dump_ref: API fix + text prompt tokenization
- NES emulator: confirmed pre-built, marked as benchmark (not project)
- test-512k-suite.sh: SIGPIPE fix

## Phase 3: Gainz (when ready)
- SSM buffer pre-allocation (cell 241)
- MoE shared expert quantize-once (cell 242)
- Attention sparsity wire (cell 245)
- MoE expert prefetch benchmark (cell 246)
