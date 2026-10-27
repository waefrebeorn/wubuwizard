# DiffusionGemma-26B Integration Notes

**Date:** 2026-06-14
**Status:** ⚠️ Model loads, forward crashes
**Model ID:** `google/diffusiongemma-26B-A4B-it` (Apache 2.0)
**GGUF:** `/home/wubu/models/DiffusionGemma-26B-Q4_K_M.gguf` (16.8 GB, Q4_K_M)

---

## Architecture Summary

```
DiffusionGemma-26B-A4B-it
├── 30 layers, ALL GQA (0 SSM)
├── d_model: 2816
├── MoE: 128 experts, top-8 routing, d_ff=704
├── Vocab: 248,320 tokens
├── Tokenizer: GemmaSentencePiece (same family as Gemma 4)
├── Canvas: 256 tokens (diffusion-based parallel decoding)
└── Attention pattern:
    ├── Normal layers (20): Q=4096 (16×256), KV=2048 (8×256), head_dim=256
    └── LARGE layers (10): Q=8192 (16×512), KV=1024 (2×512), head_dim=512
```

---

## Tensor Naming Convention

DiffusionGemma uses **Gemma-style naming** (`tensor_naming=1` or `2`):
```
model.layers.{i}.self_attn.q_proj.weight
model.layers.{i}.self_attn.k_proj.weight
model.layers.{i}.self_attn.v_proj.weight
model.layers.{i}.self_attn.o_proj.weight
model.layers.{i}.self_attn.q_norm.weight     ← head_dim elements (256 or 512)
model.layers.{i}.self_attn.k_norm.weight     ← head_dim elements (256 or 512)
model.layers.{i}.input_layernorm.weight      ← d_model elements (2816)
model.layers.{i}.post_attention_layernorm.weight
model.layers.{i}.mlp.gate_proj.weight
model.layers.{i}.mlp.up_proj.weight
model.layers.{i}.mlp.down_proj.weight
model.layers.{i}.gate.weight                  ← MoE router
```

**NOT** Qwen `blk.{i}.*` naming. Detection: search for `model.layers.0.self_attn.q_proj.weight` vs `blk.0.attn_q.weight`.

---

## Multi-Model Adapter Integration

### Detection
```c
// In wubu_model.c init function:
if (gguf_find_tensor(ctx, "model.layers.0.self_attn.q_proj.weight") >= 0) {
    model->tensor_naming = 2; // pure-GQA (DiffusionGemma or Gemma 4)
    g_tensor_naming = 2;
    // Further distinguish DGemma from Gemma 4 by checking n_layers and n_experts
}
```

### Dimensions (extracted from GGUF)
```c
model->d_model = 2816;      // From q_proj.shape[1] (or input_layernorm.shape[0])
model->n_experts = 128;     // From gate.weight.shape[0]
model->n_active_experts = 8; // From config or expert weights
model->d_ff = 704;          // From up_proj.shape[0] / n_experts (per-expert FFN dim)
```

### Layer Configuration
```c
// All 30 layers are GQA (wubu_is_ssm_layer returns 0 when g_tensor_naming == 2)
// Per-layer:
for (int i = 0; i < 30; i++) {
    layer[i].is_ssm = false;
    layer[i].gqa.is_large = (attn_q_norm.shape[0] == 512);  // detect LARGE
    layer[i].gqa.head_dim = layer[i].gqa.is_large ? 512 : 256;
    layer[i].gqa.q_dim = 16 * layer[i].gqa.head_dim;   // 4096 or 8192
    layer[i].gqa.kv_dim = (layer[i].gqa.is_large ? 2 : 8) * layer[i].gqa.head_dim; // 1024 or 2048
}
```

---

## Current Blocker: LARGE Layer head_dim Mismatch

**Crash**: `"tensor too large (512 elems, max 256)"` for LARGE layers during weight load.

**Root cause**: GQA loading path allocates weight buffers sized by `GQA_HEAD_DIM=256` macro, but DiffusionGemma LARGE layers have `head_dim=512`. The `attn_q_norm.weight` tensor has 512 elements.

**Fix needed**: In the GQA layer loading code path, use per-layer `gqa_head_dim` (from `gqa_layer_weights.head_dim`) instead of the `GQA_HEAD_DIM` macro for buffer allocation and weight reading.

**Files to change**: `src/wubu_model.c` (GQA weight loading section, roughly lines 170-250 in layer loop)

---

## Known Issues

1. **KV cache too small (FIXED)**: Was `10 * GQA_MAX_CTX * 512`, now dynamically sums per-layer kv_dim
2. **SSM/GQA count wrong (FIXED)**: Was hardcoded `n_layers - n_layers/4`, now counts actual `is_ssm` flags
3. **D_MODEL macro (PARTIALLY FIXED)**: Forward pass uses `model->d_model`, but MTP load still uses `D_MODEL` (Qwen-only, correct)
4. **LARGE layer head_dim (OPEN)**: GQA loading buffers sized for 256, need 512 for LARGE layers

---

## Benchmark Command

```bash
cd /home/wubu/bytropix
make bench_512k_full
./bench_512k_full /home/wubu/models/DiffusionGemma-26B-Q4_K_M.gguf 4096 1 0
```

Expected: model loads (30 GQA layers), tokenizer runs, forward pass at 4K context.

---

## Relationship to Gemma 4 Engine

DiffusionGemma and Gemma 4 share the same tensor naming convention (`model.layers.{i}.*`) but differ in:
- **d_model**: 2816 (DGemma) vs 3840 (Gemma 4)
- **layers**: 30 (DGemma) vs 48 (Gemma 4)
- **MoE**: Yes 128 experts (DGemma) vs No (Gemma 4)
- **head_dim**: Heterogeneous 256/512 (DGemma) vs uniform 256 (Gemma 4)
- **Attention**: All GQA (both), but different n_kv_heads patterns

The Gemma 4 dedicated engine in `wubu_gemma4.c` is a separate path. Eventually both Gemma-naming models should share a loader.
