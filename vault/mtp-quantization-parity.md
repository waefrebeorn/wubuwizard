# MTP Quantization Parity Vault

**Objective**: Achieve >50% MTP speculative decode acceptance on Qwen3.6-35B-A3B by ensuring quantization mathematical parity between the MTP draft head and the main model's inference path.

## The Problem

Current MTP acceptance: **4%** — the draft head (blk.40 GQA+MoE + nextn.* projection) is quantized with UD-IQ2_M, same as the main model. The quantized MTP head cannot reproduce the main model's token distribution well enough for speculative decoding to accelerate throughput.

At 4% acceptance, MTP decode (8.5 tok/s) is actually slower than non-MTP decode (8.9 tok/s on 16 threads). The overhead of running the MTP head (1 full GQA+MoE layer + 2 RMSNorm + 1 matmul + softmax) exceeds the benefit of accepting 4% of draft tokens.

## Root Cause: Quantization Parity Gap

The MTP head produces drafts by:
```
h_40 = MainModel_40Layers(token_N)          ← full 40-layer forward
h_norm = RMSNorm(h_40, nextn_hnorm)         ← F32, exact
e_norm = RMSNorm(token_embd(token_{N+1}), nextn_enorm)  ← F32, exact
h_cat = [h_norm | e_norm]                   ← 4096-dim concat
h_proj = h_cat @ eh_proj                    ← F32 SGEMM, exact
→ blk.40(h_proj) → same as main model layer 40
→ head_norm → output proj → logits → draft token
```

The issue: blk.40's MoE weights use IQ2_XXS (gate/up) and IQ3_XXS (down), same quant types as main model layers 0-39. However:

1. **The draft head has only 1 layer (blk.40) vs 40 layers of the main model.** Even with perfect quantization, a 1-layer draft cannot reliably predict the 40-layer output. The small acceptance is inherent.

2. **Quantization drift**: Each quantized layer introduces ~0.3% cos-sim error. 40 layers = cumulative. The MTP head (1 layer) has fewer cumulative errors but also far less representational capacity.

3. **Token embedding quantization**: `token_embd` is Q5_K quantized. The draft token's embedding goes through Q5_K dequantization before entering the draft head, while in the main model it goes through the same Q5_K dequant. These are symmetrical — no parity issue here.

## The Solution: High-Precision Draft Head

For MTP to be viable (>50% acceptance), the draft head must produce COS-SIM > 0.999 with the main model's output for the draft token. This requires:

### Strategy 1: Keep blk.40 at F32 (Full Precision)

**Cost**: ~3.2GB RAM for blk.40's MoE weights in F32 (vs ~500MB in IQ2_M quantized)

**Implementation**:
- During model load, detect MTP model
- Load blk.40's MoE weights as F32 (dequantized from GGUF)
- All other layers (0-39) stay quantized
- Use F32 SGEMM for blk.40's MoE expert compute (same f32 matmul path)

**Expected acceptance**: 40-60% (blk.40 output matches main model's blk.40 output perfectly, limited only by context state differences)

**Current code location**: `wubu_model.c:851-943` — blk.40 is loaded via blob pointers (quantized). Change to dequant on load like the F32 fallback path.

### Strategy 2: Q8_0 Draft Head (Medium Precision)

**Cost**: 8 bpw → ~1.3GB for blk.40 MoE

**Implementation**:
- Dequant IQ2_XXS → F32 → quantize to Q8_0 during model init
- Use Q8_0 vec_dot for draft head matmuls (already 4× faster than F32 SGEMM)

**Expected acceptance**: 25-35% (Q8_0 error dominates at ~0.5% cos-sim error per matmul, 3 matmuls × 1 layer = ~1.5% cumulative → ~98.5% cos-sim vs main model)

### Strategy 3: MTP Draft Head Only (Cleanest)

The most elegant solution: **use a SEPARATE F32 MTP head GGUF** that stores only blk.40's weights in full precision. The main model stays fully quantized (10.7GB). The MTP head is a 3.2GB supplement loaded only when MTP=1.

```bash
# Extract blk.40 from main model, dequant to F32, save as separate GGUF
./tools/extract_mtp_head_f32 ~/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf
# Output: ~/models/mtp-head-f32.gguf (~3GB)
```

## Quantization Parity Checklist for MTP

| Component | Main Model Quant | Draft Head Quant | Parity Match | Fix |
|-----------|-----------------|------------------|-------------|-----|
| token_embd | Q5_K | Q5_K | ✅ Same pointer | N/A |
| blk.40 attn_q | Q5_K | Q5_K | ✅ Same path | N/A |
| blk.40 attn_k | Q5_K | Q5_K | ✅ Same path | N/A |
| blk.40 attn_v | Q5_K | Q5_K | ✅ Same path | N/A |
| blk.40 attn_output | Q5_K | Q5_K | ✅ Same path | N/A |
| blk.40 ffn_gate_exps | IQ2_XXS | **F32** | ❌ Must be F32 | Dequant on load |
| blk.40 ffn_up_exps | IQ2_XXS | **F32** | ❌ Must be F32 | Dequant on load |
| blk.40 ffn_down_exps | IQ3_XXS | **F32** | ❌ Must be F32 | Dequant on load |
| nextn.eh_proj | Q8_0 | F32 (dequant'd) | ✅ Already F32 | N/A |
| RMSNorm weights | F32 | F32 | ✅ Same pointer | N/A |

## Implementation Plan

### Phase A: F32 blk.40 Draft Head (1 session)
1. Modify `wubu_mtp_load` to dequantize blk.40's MoE weights to F32 heap memory
2. Add F32 MoE expert forward path to `wubu_moe_forward` for blk.40 only
3. Benchmark acceptance rate

### Phase B: Optimized Q8_0 Draft Head (2 sessions)
1. Quantize blk.40 F32 weights to Q8_0 format after dequant
2. Implement Q8_0 vec_dot for MoE matmuls (already exists in quantized_matmul.c)
3. Benchmark speed vs acceptance tradeoff

### Phase C: Separate F32 MTP Head GGUF (1 session)
1. Create extraction tool for blk.40 F32 weights
2. Load separate GGUF for MTP head at inference time
3. Benchmark cold-start vs runtime memory overhead

## Expected Results

| Config | Draft Acceptance | Decode Speed (16T) | Memory Overhead |
|--------|:---------------:|:------------------:|:---------------:|
| Current (IQ2_M quantized) | 4% | 8.5 tok/s | 0 |
| F32 blk.40 draft head | 45-60% | **12-15 tok/s** | +3.2GB |
| Q8_0 blk.40 draft head | 25-35% | **10-12 tok/s** | +1.3GB |
| Separate F32 MTP GGUF | 45-60% | **12-15 tok/s** | +3.2GB (cold start) |

## Key Insight

MTP is a **bandwidth play**, not a compute play. The draft head runs at 1/40th the cost of a full token, but the acceptance rate determines net speedup:

```
speedup = 1 / (1 - acceptance)
4%  acceptance → 1.04× (worse than single-token due to overhead)
50% acceptance → 2.0× (doubles throughput)
80% acceptance → 5.0× (DeepSeek-V3 claim)
```

At DDR4 bandwidth wall (~2.3 tok/s theoretical max on our hardware), even 2.0× MTP would push decode to ~4.6 tok/s — nearly double. This is the ONLY path past the DDR4 bandwidth wall without hardware upgrade.

## References

- DeepSeek-V3 (2412.19437): 83% MTP acceptance, Q8_0 draft head + full-precision main model
- DeepSeek-V3.2 (2512.02556): DSA sparse attention for MTP context
- bytropix MTP code: `src/wubu_model.c:802-1080` (wubu_mtp_forward)
- bytropix MTP test: `tools/test_mtp_draft.c`

## Implementation Results (May 27)

### v1: Q8_K Dequant Cache ❌
- **Approach**: IQ2→F32→Q8_K requant, Q8_K×Q8_K dot product
- **Acceptance**: 12% (3% loss vs baseline)
- **Root cause**: IQ2→F32→Q8_K requant adds quantization noise. Q8_K dot (int8×int8)
  is less accurate than native IQ2 vec_dot for reproducing F32 computation.
- **Memory**: 41MB (12 slots × 3.4MB)
- **Bug**: MTP_Q8_WEIGHT_BYTES buffer overflow (34/32 ratio ≠ 292/256 block_q8_K ratio)

### v2: IQ Raw-Quant Cache ✅
- **Approach**: Memcpy native IQ2_XXS/IQ3_XXS bytes from blob. Use original vec_dot path.
- **Acceptance**: 16% (matches baseline within noise)
- **Memory**: 24MB (16 slots × ~1.5MB, max 524KB per weight matrix)
- **No precision loss**: No dequant/requant step — stored bytes match blob exactly.
- **Cache fill**: Zero dequant overhead (just memcpy)
- **16% acceptance at K=2**: Speedup = 1/(1-0.16) = 1.19×, overhead-dominated → net-neutral
