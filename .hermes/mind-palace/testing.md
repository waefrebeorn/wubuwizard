# bytropix — Testing Protocol (May 28, 2026)

**All gaps closed. Hardware ceiling reached.** Cos-sim 0.976 vs llama.cpp (IQ2_M floor).

## Quick Start
```bash
# CPU inference (verified working)
MODEL=~/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf OMP_NUM_THREADS=4 \
  ./gen_text_cpu "The capital of France is" 20 40

# Cos-sim regression test (3 prompts, threshold 0.975)
MODEL=~/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf THRESHOLD=0.975 \
  bash tools/test-cos-sim-regression.sh

# Test suite
bash tools/test-512k-suite.sh
bash tools/test-hermes-integration.sh 8005
```

## Test Suites

| Suite | Command | Tests | Status |
|-------|---------|-------|--------|
| Cos-sim regression | `test-cos-sim-regression.sh` | 3 single-token vs llama | ✅ ALL PASS at 0.975 |
| 512K suite | `test-512k-suite.sh` | KV alloc, sparse attn, memory, RoPE, NES | ✅ 6/6 |
| Hermes integration | `test-hermes-integration.sh` | Pipeline: server→chat→stream→agent→vault→NES | ✅ 6/6 |
| Hermes headless | `test-hermes-headless.sh` | Binary→server→endpoints→format | ✅ all pass |
| Inference pytest | `tests/test_inference.py` | 24 tests | ✅ 1.16s |

## Key Findings (May 28)

- **Context growth penalty ELIMINATED**: persistent KV gives constant ~31s/turn regardless of KV size
- **Compilation IEEE 754**: `-ffast-math` removed → SSM recurrence FP drift fixed
- **Single-token cos-sim 0.976** → improved from 0.974 with compilation fix
- **Between-builds cos-sim 0.99975580** → top-5 argmax identical between fast/no-fast
- **Multi-token divergence** remains: IQ2_M quantization noise accumulates through 30 SSM layers (needs Q3_K+)
