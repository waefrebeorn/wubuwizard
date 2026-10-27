# Paradigm Shift: Qwen3.6 → Gemma 4 12B

**Date:** 2026-06-10
**Status:** ⏳ In progress — model downloading, architecture mapped

---

## Why the shift?

| Aspect | Old (Qwen3.6-35B-A3B) | New (Gemma 4 12B) |
|--------|----------------------|-------------------|
| Architecture | Gated DeltaNet SSM + 256 MoE | Dense Transformer (ISWA) |
| Model size | 35B total, ~3.5B active | 12B dense (100% active) |
| Quality | Strong but quantized to ~2.7bpw | QAT q4_0 (higher precision per param) |
| Attention | 10 GQA layers (rest SSM) | 48 attention layers (6:1 SWA:Full) |
| Multi-modal | Moondream3 via dump | Native mmproj + audio + vision |
| Upstream | Qwen3 fork, no updates | Google, actively maintained |
| Community | Niche | 127K downloads in days |

---

## Gemma 4 12B Architecture (from upstream llama.cpp)

### Hyperparameters (from config.json + llama.cpp source)

```
layers:          48 (40 sliding + 8 full attention)
hidden:          3840
heads:           16 Q / 8 KV / 1 GlobalKV
head_dim:        256 (global KV: 512)
FFN:             15360 dense (GELU tanh approx)
vocab:           262,144 (tied embeddings)
max_ctx:         262,144
sliding_window:  1024
logit_softcap:   30.0
rms_norm_eps:    1e-6
attention_scale: 1.0 (no pre-attn scaling)
```

### Layer pattern (ISWA — Interleaved Sliding Window Attention)

```
Layer 0-5:   sliding_attention (RoPE default, θ=10K, window=1024)
Layer 6:     full_attention    (RoPE proportional, θ=1M, rot=25%)
Layer 7-12:  sliding_attention
Layer 13:    full_attention
Layer 14-19: sliding_attention
Layer 20:    full_attention
Layer 21-26: sliding_attention
Layer 27:    full_attention
Layer 28-33: sliding_attention
Layer 34:    full_attention
Layer 35-40: sliding_attention
Layer 41:    full_attention
Layer 42-47: sliding_attention
```

Full attention layers at: 6, 13, 20, 27, 34, 41 (every 7th, 6 total)

### Attention K == V

Gemma 4 uses `attention_k_eq_v: true` — the V projection tensor (`blk.{i}.attn_v.weight`) is **optional**.
If absent, the K projection weights are reused as V. This is checked at runtime:

```c
ggml_tensor * Vcur = model.layers[il].wv
    ? build_lora_mm(model.layers[il].wv, cur)
    : Kcur; // fallback to K projection
```

### RoPE details

| Layer type | RoPE type | θ | Partial rotary |
|-----------|-----------|---|----------------|
| full_attention | proportional | 1,000,000 | 25% (64/256 dims) |
| sliding_attention | default | 10,000 | 100% |

Full attention layers have **learned rope_freqs** (`blk.{i}.rope_freqs.weight`, shape [128]).

### Per-layer KV sharing

`n_kv_shared_layers` layers at the end share KV cache with earlier layers.
For 12B: `n_kv_shared_layers = 0` (all layers have their own KV).

### Final logit softcapping

```
logits = tanh(logits / 30.0) * 30.0
```

This is applied at the end of the forward pass, before sampling.

---

## Tensor name map

### Global tensors

| GGUF key | Shape | Required | Notes |
|----------|-------|----------|-------|
| `token_embd.weight` | [3840, 262144] | ✅ | Input embedding |
| `output.weight` | [3840, 262144] | ❌ | Tied to token_embd if absent |
| `output_norm.weight` | [3840] | ✅ | Final RMS norm |
| `rope_freqs.weight` | [128] | ❌ | Shared rope freqs (unused, per-layer used instead) |

### Per-layer tensors (i = 0..47)

| GGUF key | Shape | Required | Notes |
|----------|-------|----------|-------|
| `blk.{i}.attn_norm.weight` | [3840] | ✅ | Pre-attention RMS norm |
| `blk.{i}.attn_q.weight` | [3840, 4096] | ✅ | Q projection (16 heads × 256) |
| `blk.{i}.attn_k.weight` | [3840, 2048] | ✅ | K projection (8 KV heads × 256) |
| `blk.{i}.attn_v.weight` | [3840, 2048] | ❌ | V projection — uses K if absent |
| `blk.{i}.attn_output.weight` | [4096, 3840] | ✅ | Attention output projection |
| `blk.{i}.attn_q_norm.weight` | [256] | ✅ | Per-head Q RMS norm |
| `blk.{i}.attn_k_norm.weight` | [256] | ✅ | Per-head K RMS norm |
| `blk.{i}.post_attention_norm.weight` | [3840] | ✅ | Post-attention RMS norm |
| `blk.{i}.layer_output_scale.weight` | [1] | ❌ | Per-layer output scalar |
| `blk.{i}.rope_freqs.weight` | [128] | ❌ | Only full attention layers (6 of 48) |
| `blk.{i}.ffn_norm.weight` | [3840] | ✅ | Pre-FFN RMS norm |
| `blk.{i}.ffn_gate.weight` | [3840, 15360] | ✅ | SwiGLU gate |
| `blk.{i}.ffn_up.weight` | [3840, 15360] | ✅ | FFN up projection |
| `blk.{i}.ffn_down.weight` | [15360, 3840] | ✅ | FFN down projection |
| `blk.{i}.post_ffw_norm.weight` | [3840] | ✅ | Post-FFN RMS norm |
| `blk.{i}.ffn_gate_inp.weight` | — | ❌ | MoE router — NOT present in 12B |

### Not present in Gemma 4 (vs Qwen3.6)

| Qwen3.6 tensor | Reason |
|----------------|--------|
| `blk.{i}.ssm_*` (all SSM) | Gemma 4 has no SSM |
| `blk.{i}.attn_k_b.*` | No bias in Gemma 4 |
| `blk.{i}.attn_q_b.*` | No bias |
| `blk.{i}.attn_output_b.*` | No bias |
| `blk.{i}.ffn_gate.bias` | No bias |
| `blk.{i}.ffn_up.bias` | No bias |
| `blk.{i}.ffn_down.bias` | No bias |

---

## Forward pass (graph structure)

```
inpL = token_embd(token_ids) * sqrt(3840)

for il in 0..47:
    # Attention
    cur = rms_norm(inpL, blk.{il}.attn_norm)
    Q = rms_norm(Q_proj(cur), blk.{il}.attn_q_norm)
    Q = rope(Q, positions, rope_freqs if full_attn else nullptr)
    
    if has_kv(il):  # only non-shared KV layers
        K = rms_norm(K_proj(cur), blk.{il}.attn_k_norm)
        V = V_proj(cur) ?? K  # K=V fallback
        K = rope(K, positions, ...)
        V = rms_norm(V)  # V norm via raw ggml_rms_norm
        
        cur = attention(Q, K, V, sliding_window if swa else full)
    else:
        cur = attention(Q, reuse_kv_from_earlier_layer)
    
    cur = rms_norm(cur, blk.{il}.post_attention_norm)
    cur = cur + inpL  # residual
    attn_out = cur
    
    # FFN
    cur = rms_norm(attn_out, blk.{il}.ffn_norm)
    if is_moe_layer:  # 12B: no
        cur = gelu(gate @ cur) * (up @ cur)
        cur = rms_norm(cur, post_ffw_norm_1)
        ... + moe path ...
    else:
        cur = gelu(gate @ cur) * (up @ cur)
        cur = down @ cur
    
    cur = rms_norm(cur, blk.{il}.post_ffw_norm)
    cur = cur + attn_out  # residual
    
    # per-layer embedding (12B: disabled)
    if per_layer_enabled:
        cur = gelu(inp_gate @ cur) * per_layer_embed[il]
        cur = proj @ cur
        cur = rms_norm(cur, per_layer_post_norm)
        cur = cur + pe_in  # residual
    
    # layer scaling
    if out_scale exists:
        cur = cur * out_scale[0]
    
    inpL = cur

# Final
cur = rms_norm(inpL, output_norm)
logits = output_weight @ cur
logits = tanh(logits / 30.0) * 30.0  # softcap
```

---

## VRAM budget estimate (Q4_K_XL quant)

| Component | Size | Notes |
|-----------|------|-------|
| Weights (Q4_K_XL) | ~6,700 MB | Main GGUF file |
| KV cache (FP16, 4K ctx) | ~100 MB | 48 layers × 8 KV heads × 256 × 4096 × 2B |
| KV cache (FP16, 128K ctx) | ~3,200 MB | 48 × 8 × 256 × 131072 × 2B |
| KV cache (Q4_0, 128K ctx) | ~800 MB | 4:1 compression vs FP16 |
| Scratch buffers | ~500 MB | Temp during inference |
| **Total (4K ctx)** | **~7,300 MB** | Fits RTX 5050 8GB |
| **Total (128K ctx, Q4_0 KV)** | **~8,000 MB** | Tight fit |

---

## What bytropix needs to change

### Ships that stay

| Component | Status | Notes |
|-----------|--------|-------|
| `gguf_reader.c` | ✅ Reuse | Already reads any GGUF |
| KV cache (Q4_0) | ✅ Reuse | Same 4-bit cache |
| `wubu_tokenizer.c` | 🔄 Adapt | Different tokenizer (Gemma 4 = 262K vocab) |
| Thread pool | ✅ Reuse | Same thread pool |
| `api_server.c` | ✅ Reuse | Already OpenAI-compatible |

### Ships that sail under new flag

| Component | Action | Notes |
|-----------|--------|-------|
| `wubu_model.c` | 🔄 Rewrite | Qwen3.6 → Gemma 4 forward pass |
| `wubu_ssm.c` | 🗑️ Archive | No SSM in Gemma 4 |
| `wubu_moe.c` | 🗑️ Archive | No MoE in 12B (keep for larger models) |
| `wubu_nested_ssm.c` | 🗑️ Archive | No nested SSM |
| `wubu_poincare_gqa.c` | 🗑️ Archive | No Poincaré GQA needed |
| `cuda_kernels.cu` | 🔄 Adapt | New attention kernels (ISWA) |
| `gpu_quant_matmul.cu` | ✅ Reuse | Same quant matmul |

### New ships needed

| Component | Priority | Notes |
|-----------|----------|-------|
| `wubu_gemma4.c` / `gemma4_model.c` | P0 | New model forward pass |
| `gemma4_attention.cu` | P0 | ISWA CUDA kernels (sliding + full) |
| `gemma4_rope.c` | P0 | Dual RoPE (proportional + default) |
| `gemma4_softcap.cu` | P1 | Final logit softcapping CUDA kernel |
| Gemma 4 tokenizer | P1 | Load Gemma 4 tokenizer from GGUF |
| Vision mmproj loader | P2 | Load `mmproj-F16.gguf` for native vision |
| MTP draft integration | P3 | Speculative decoding with MTP draft model |

---

## Baseline benchmark (llama.cpp)

Run `llama-bench` with Gemma 4 12B GGUF once downloaded:

```bash
cd ~/HASHMIND/llama-cpp-rotorquant/llama.cppCOPY/build
./bin/llama-bench -m /home/wubu/models/gemma4/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf
```

Metrics: prompt tok/s, generation tok/s, VRAM usage, load time.
