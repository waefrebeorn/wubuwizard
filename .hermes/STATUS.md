# bytropix — True State (June 14, 2026 — Multi-Model Integration)

## Ground Truth
**Multi-model adapter works.** Model auto-detection, dynamic dimension extraction, and dynamic KV cache allocation are all functional. DiffusionGemma-26B loads all 30 GQA layers with correct dimensions.

## What Works
- Multi-model adapter (`wubu_model.c`): auto-detects Qwen/Gemma/DiffusionGemma naming ✅
- Dynamic dimension extraction: `d_model`, `head_dim`, `n_experts` from GGUF tensor shapes ✅
- Dynamic KV cache: per-layer offsets, variable `kv_dim` ✅
- `g_tensor_naming` global: set during init, read by `wubu_is_ssm_layer()` ✅
- GQA forward functions accept `d_model` parameter (no `D_MODEL` macro dependency) ✅
- DiffusionGemma model load: 30 GQA layers, correct per-layer dims ✅
- Qwen3.6-35B forward: coherent output, 3-4 tok/s CPU ✅
- Q4_0 KV cache: 4:1 compression ✅
- MTP spec decode: Qwen draft model ✅
- Build: compiles clean (warnings only) ✅

## What's Broken
- DiffusionGemma forward: crashes "tensor too large (512 elems, max 256)" for LARGE layers ❌
  - Root cause: GQA weight loading uses fixed `GQA_HEAD_DIM=256` buffer, LARGE layers need 512
  - Fix: per-layer buffer sizing from `gqa_layer_weights.head_dim`
- Gemma 4 12B benchmark: not tested with main benchmark binary ❌
- GPU forward: no model has end-to-end GPU path ❌

## Key Commands
```bash
# Build
make gen_text_cpu && make bench_512k_full

# Test DiffusionGemma (loads, crashes in forward)
./bench_512k_full /home/wubu/models/DiffusionGemma-26B-Q4_K_M.gguf 4096 1 0

# Test Qwen3.6 (works)
./bench_512k_full /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf 4096 1 0

# Debug layer loading
DUMP_LAYER_DIR=/tmp/dgemma_layers ./bench_512k_full ... 4096 1 0
```

## Priorities
P0 — Fix DiffusionGemma LARGE layer head_dim buffer sizing
P0 — Complete DGemma forward pass (verify decode)
P1 — Benchmark all 3 models at 4K context
P2 — Gemma 4 12B integration into main benchmark path
P3 — GPU forward for at least one model
