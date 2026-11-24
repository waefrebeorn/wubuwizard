# Inkling-2-Mini GGUF — exact inventory (2026-11-23)

Parsed from the file on this disk, `C:/Users/eman5/models/inkling-2-mini/Inkling-2-Mini-UD-IQ2_XXS.gguf`
(GGUF v3, 734 tensors, 62 KV pairs). This is the implementation target for `inkling2`
architecture support, which `rg -i inkling src include` currently does not contain at all.

## Metadata (`inkling2.*` keys, verbatim values)

| Key | Value |
|---|---|
| general.architecture | inkling2 |
| general.size_label | 128x1.8B |
| block_count | 48 |
| context_length | 1048576 |
| embedding_length | 2304 |
| feed_forward_length | 1152 |
| expert_count / expert_used_count | 128 / 8 |
| expert_feed_forward_length | 1152 |
| attention.key_length / value_length | 128 / 128 |
| attention.key_length_swa / value_length_swa | 128 / 128 |
| attention.sliding_window | 1024 |
| attention.layer_norm_rms_epsilon | 1e-06 |
| **rope.dimension_count** | **0** (no RoPE: relative-bias attention) |
| num_loops / loop.core_start / loop.core_end / loop.gate_rank | 3 / 6 / 18 / 64 |
| shortconv.l_cache | 4 |
| expert_weights_scale | 8.0 |
| activation.situ_beta / situ_linear_beta | 4.0 / 25.0 |
| attn_res.block_size | 6 |
| final_logit_softcapping | 30.0 |
| vocab_size / vocab_size_unpadded | 201024 / 200058 |

## Physical blocks vs logical layers (the detail that changes the loader)

`inkling2.block_count = 48` is the **logical** count. Only **24 blocks** carry tensors:
indices 0-23. The 12-layer core (6..17) is stored once and executed `num_loops = 3`
times, so

```
logical 0..5   -> physical 0..5      (front)
logical 6..41  -> physical 6..17     (3 passes over the core)
logical 42..47 -> physical 18..23    (back)
```

Mapping is implemented as `wubu_inkling2_phys_block()` and asserted by
`test_eg118_inkling_meta`. A loader that assumes 48 blocks finds nothing past blk.23 and
reports 312 missing tensors - exactly what the first version of the gate caught.

Layer 0 also lacks `attn_res_query` (no previous block to attend to), and
`attn_q_inp_norm` / `attn_k_inp_norm` / `attn_v_inp_norm` plus the dynamic-tap
`*_conv_proj` tensors exist only on the full-attention layers (5, 11, 17, 23 - every 6th
layer), not on the sliding-window ones.

## Tensor families (48 kinds, 734 tensors)

| Tensor | Count | Dims | Note |
|---|---|---|---|
| `blk.N.attn_q.weight` / `attn_gate.weight` | 24 / 24 | (2304, 3072) | Q and output gate |
| `blk.N.attn_k.weight` / `attn_v.weight` | 24 / 24 | (2304, 1024) | GQA 8 heads x 128 |
| `blk.N.attn_output.weight` | 24 | (3072, 2304) | |
| `blk.N.attn_rel.weight` | 24 | (2304, 384) | relative-position projection |
| `blk.N.attn_rel_bias.weight` | 24 | (16, 1024) | per-head distance bias table |
| `blk.N.attn_rel_clip.weight` | 24 | (16, 24) | |
| `blk.N.attn_sink_proj.weight` | 24 | (2304, 24) | attention sink logit |
| `blk.N.attn_q_clip.weight` | 24 | (128, 24) | q L2 clip |
| `blk.N.attn_k_conv.weight` / `attn_v_conv.weight` | 24 / 24 | (4, 1024) | 4-tap short convs |
| `blk.N.attn_post_conv.weight` / `ffn_post_conv.weight` | 24 / 24 | (4, 2304) | |
| `blk.N.attn_k_conv_proj.weight` / `attn_v_conv_proj.weight` | 4 / 4 | (2304, 32) | dynamic taps, full layers only |
| `blk.N.attn_k_inp_norm`, `attn_v_inp_norm` | 4 / 4 | (2304,) | |
| `blk.N.attn_norm` / `ffn_norm` | 20 / 24 | (2304,) | |
| `blk.N.attn_res_query` / `ffn_res_query` | 23 / 24 | (2304,) | attention-residual query (block of 6) |
| `blk.N.exp_probs_b.bias` | 24 | (128,) | router selection bias |
| `blk.N.ffn_gate_inp.weight` | 24 | (2304, 128) | router |
| `blk.N.ffn_gate_up_exps.weight` | 24 | (2304, 2304, 128) | **gate and up fused** (2 x 1152) |
| `blk.N.ffn_down_exps.weight` | 24 | (1152, 2304, 128) | |
| `blk.N.ffn_gate_up_exps.scale` / `ffn_down_exps.scale` | 20 / 20 | (128,) | per-expert dequant scales |
| `blk.N.loop_{k,v,attn,mlp}_{gain,forget,down}` | 12 each | (64,2304) / (64,2304) / (4608,64) | looped core, layers 6-17 |
| `loop_kv_res_query` / `loop_res_query` | 1 / 1 | (2304, 24) | |
| `token_embd.weight` / `output.weight` | 1 / 1 | (2304, 201024) | untied |
| `token_embd_norm`, `output_norm`, `output_res_query` | 1 each | (2304,) | |

## Quantization types actually present

F32 508, Q8_0 85, **30 x32**, Q6_K 26, **42 x17**, Q5_K 16, Q4_K 14, **40 x7**, IQ2_XXS 7,
IQ1_S 6, IQ2_XS 6, IQ2_S 5, IQ3_XXS 3, IQ4_XS 2.

Type ids 30, 40 and 42 are not in this engine's type table. ggml has 30 = BF16 and has been
adding 4-bit float types for the NVFP4/MXFP4 family (project-maya's spec for the same model
says the experts ship as **NVFP4 e2m1 x e4m3 per 16 with an amax/2688 scale**, and MXFP8 on
layers 40-47). Treat 40/42 as that family: new block layouts, new dequant paths, and a
correctness gate each before anything is wired.

## What the engine needs, in order

1. `LLM_ARCH_INKLING2` recognition in the GGUF loader plus the four KV groups above
   (loop core, relative bias, sinks, short convs).
2. A forward pass with **no RoPE**: `attn_rel` projection + `attn_rel_bias[16][1024]`
   looked up by distance, sink logit added to the pre-softmax row, sigmoid gate on the
   output, 4-tap convs on k/v and on attention/MoE outputs.
3. MoE with 128 experts, top-8, fused `gate_up` (one matvec producing 2 x 1152), router
   sigmoid + `exp_probs_b.bias`, weights normalised x8 (`expert_weights_scale`).
4. The looped core: layers 6..17 run 3 times with the `loop_*_{gain,forget,down}` gates
   (rank 64). Slot-expanded KV as project-maya describes:
   `n_slot = n_phys + (n_passes-1)*(core_end-core_start)`.
5. Attention-residual queries every 6 layers.
6. New quant types 40/42 (NVFP4/MXFP4 family) if the IQ2_XXS file is not used.

Steps 1-5 are the engine work; step 6 can be deferred by using this IQ2_XXS file, which
still carries 39 tensors of the unknown types (F32 norms aside, the expert blocks are
IQ2_XXS in this file — the 30/40/42 tensors are the small ones).
