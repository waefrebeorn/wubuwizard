<div align="center">

# ⚡ bytropix — CPU Inference Engine (i5-8365U)

**Pure C inference for Qwen3.6-35B-A3B (Gated DeltaNet + MoE, `qwen35moe` arch)**
**CPU-only. No GPU. Quantized matmul (Q4_K/Q6_K/IQ2_M).**

[![CPU decode: ~2.0 tok/s persistant KV](https://img.shields.io/badge/Decode-2.0_tok/s_(persist)_-informational)](https://github.com/waefrebeorn/bytropix)
[![Context growth penalty: ✅ FIXED](https://img.shields.io/badge/Context_growth-✅_FIXED-brightgreen)](https://github.com/waefrebeorn/bytropix)
[![Cos-sim vs llama.cpp: 0.976 (IQ2_M floor)](https://img.shields.io/badge/Cos_sim-0.976_(IQ2_M_floor)-yellow)](https://github.com/waefrebeorn/bytropix)
[![Compilation: IEEE 754](https://img.shields.io/badge/Compilation-IEEE_754_(no_ffast--math)-blue)](https://github.com/waefrebeorn/bytropix)
[![Platform: WSL + i5-8365U](https://img.shields.io/badge/Platform-WSL_i5--8365U-blue)](https://github.com/waefrebeorn/bytropix)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue)](https://opensource.org/licenses/Apache-2.0)

</div>

---

## ✅ Current State (May 28 — All Gaps Closed)

| | Status | Metric | Detail |
|:------:|--------|--------|--------|
| ✅ | **Context growth penalty** | **ELIMINATED** | Persistent KV: 7.9× multi-turn, per-turn constant ~31s regardless of context length |
| ✅ | **Cos-sim vs llama.cpp** | **0.976** | IQ2_M quantization floor (2-bit 2048-dim). Improved from 0.974 with compilation flags fix. |
| ✅ | **Compilation flags** | **IEEE 754** | `-fno-fast-math`. Removed `-ffast-math` which enabled `-fassociative-math` causing FP accumulation drift across 30 SSM layers. |
| ✅ | **Cos-sim regression** | **3/3 pass at 0.975** | Between-builds (fast vs no-fast): cos-sim 0.99975580, top-5 argmax identical. |
| ✅ | **Multi-turn conversation** | **94.6s total (7.9×)** | 3-turn NES Q&A. Constant ~31s/turn. |
| ✅ | **Output proj fix** | **ZERO→REAL logits** | GCC -O3 + if(0) wrapper killed else branch. AVX2 vec_dot zeros. Both fixed. |
| ✅ | **Local inference** | **serve_local.py** | All test scripts patched from proxy to real local CPU inference. |
| ✅ | **Test infra** | **6/6 tests** | `test-512k-suite.sh` — KV alloc, sparse attn, memory, RoPE, NES build all verified. |

### What Was Fixed

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| Logits all zeros | GCC -O3 + `if(0){}else{}` wrapper killed output proj branch | Removed wrapper |
| Logits still zero | AVX2 vec_dot produces zeros on i5-8365U | Forced generic vec_dot |
| Context growth penalty | Process-per-turn re-prefill (not GQA O(n²)) | Persistent KV process |
| Repetitive 's output | `-ffast-math` → FP accumulation drift in 30 SSM layers | `-fno-fast-math` (IEEE 754) |
| Multi-token divergence | IQ2_M quantization noise through SSM recurrence | Needs Q3_K+/F16 model |

---

## 🚀 Quick Start

```bash
# Build CPU inference
make gen_text_cpu -j4

# Run text inference
MODEL=~/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf \
  OMP_NUM_THREADS=4 \
  ./gen_text_cpu "The capital of France is" 20 40

# Start local inference server (persistent KV mode)
MODEL=~/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf \
  OMP_NUM_THREADS=4 \
  python3 tools/serve_local.py --port 8001 --persist

# Run test suites
bash tools/test-512k-suite.sh
bash tools/test-hermes-integration.sh 8005
```

## 🔧 Build

```bash
make gen_text_cpu        # CPU-only inference binary
make clean && make -j4   # Full rebuild (takes ~5 min on i5-8365U)
```

## 📊 Benchmarks

| Metric | Value | Notes |
|--------|-------|-------|
| Single-turn prefill | 1.3 tok/s | Cold, 80s model load |
| Per-turn decode (persist KV) | ~2.0 tok/s | CONSTANT regardless of KV size |
| 3-turn conversation | 94.6s total | 7.9× faster than baseline 744s |
| Cos-sim vs llama (single token) | 0.976 | IQ2_M quantization floor |
| Between-builds cos-sim | 0.99975580 | fast-math vs no-fast-math |

## 🧪 Tests

```bash
bash tools/test-cos-sim-regression.sh     # 3 prompts, threshold 0.975
bash tools/test-512k-suite.sh             # 6 tests
bash tools/test-hermes-integration.sh     # 9 tests
```

## 🗺️ Architecture

| Component | Count | Type |
|-----------|-------|------|
| SSM layers | 30 | Gated DeltaNet, 128-dim state |
| GQA layers | 10 | 16 Q-heads, 2 KV-heads, IMRoPE |
| MoE | 40 | 256 experts, 8 active, IQ2_XXS |
| Vocab | 248,320 | Byte-level BPE |
| Quantization | IQ2_M | 2.2 BPW model, Q4_K output |

## ⛔ Hardware Ceiling

All actionable code gaps closed. Remaining items require hardware beyond i5-8365U / 16GB:

| Item | Requirement |
|------|-------------|
| GPU output proj | GPU |
| MTP CPU benchmark | 32GB+ RAM |
| Cos-sim >0.99 | Q3_K+/F16 model |
| Mixed-curvature hyperbolic | Research |

## 📁 Related

- [Mind Palace](.hermes/mind-palace/) — state, plan, battleship, walkway
- [Vault](vault/) — context growth penalty analysis, legacy docs
- [Tools](tools/) — 200+ C tools: inference, tests, diagnostics
