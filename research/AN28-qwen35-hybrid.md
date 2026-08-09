# AN28 — Qwen3.5-0.8B hybrid (Gated DeltaNet + Gated Attention) on the wizard

2026-08-09. The Distiller V1 toolset model `Qwen3.5-0.8B-Q8_0.gguf` does NOT
yet load: the lfm2 engine is LFM2.5-specific (conv-GQA). This doc decodes the
architecture from the actual GGUF tensors + prior art so the adapter can be
built without further archaeology.

## The model (from the GGUF, 24 layers, d=1024)

```
token_embd.weight  [1024, 248320]  type=8 (Q8_0)
output_norm.weight [1024]          type=0 (F32)
output.weight      [1024, 248320]? untied head (confirm) 
```

**17 hybrid layers** (blk.{0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22}) —
gated attention + GatedDeltaNet in parallel, then post_attention_norm + FFN:

```
attn_norm         [1024]                  RMSNorm, 1e-6
attn_qkv          [1024, 6144]  Q8_0      FUSED: [q_and_gate | k | v]
attn_gate         [1024, 2048]? Q8_0      gate proj (see split question)
ssm_a             [16]                    decay (-A log)
ssm_alpha         [1024, 16]              GDN alpha proj
ssm_beta          [1024, 16]              GDN beta proj
ssm_conv1d        [4, 6144]               causal conv over the GDN input
ssm_dt.bias       [16]                    dt bias
ssm_norm          [128]                   per-head RMSNorm (value heads)
ssm_out           [2048, 1024]  Q8_0      value_dim(2d) -> d
post_attention_norm [1024]
ffn_gate/up       [1024, 3584]  Q8_0
ffn_down          [3584, 1024]  Q8_0
```

**6 pure-attention layers** (blk.{3,7,11,15,19,23}) — standard GQA with
per-head QK norms:

```
attn_norm [1024]
attn_q    [1024, 4096]   = num_heads * hd = 16 * 256 (q_norm [256] confirms hd)
attn_q_norm [256], attn_k_norm [256]
attn_k/v  [1024, 512]    = 2 kv-groups * 256
attn_output [2048, 1024] = d_out(2048) -> d   (d_out = 8 heads * 256)
post_attention_norm [1024]
ffn_*
```

## Gated attention (raschka gallery + Qwen3Next modular code, verified)

```
q_and_gate = x @ W_qkv[0 : 2*d_out]        # 2*d_out = 4096 for the 0.8B
q, gate    = chunk(q_and_gate.view(..., n_heads, 2*hd), 2, -1)
gate       = gate.reshape(..., d_out)      # d_out = n_heads * hd = 2048
q = q_norm(q); k = k_norm(k); RoPE(q,k)
attn = softmax(q k^T / sqrt(hd)) v          # GQA: 8 q-heads over 2 kv-groups
out  = attn * sigmoid(gate)                  # elementwise, the gate feature
y    = out @ W_o[2048 -> 1024]
```

Open question: the fused attn_qkv is 6144 = 4096 (q_and_gate) + 512 (k) + 512
(v) + **1024 extra**. Two candidates: (a) the extra 1024 is the gate proj
(the attn_gate tensor may be fused into qkv and attn_gate is something else),
or (b) the gate is 2048 and the extra 1024 belongs to a different split.
RESOLVE by checking the Qwen3.5-0.8B HF repo's modeling code or by probing
the GGUF against the V1's llama.cpp (the llama.cpp supports qwen35 —
run the same prompt, dump logits, match).

## Gated DeltaNet (Qwen3_5GatedDeltaNet, transformers modular)

```
in_proj_qkv : d -> key_dim*2 + value_dim      (fused q/k/v)
in_proj_z   : d -> ?                          (the output gate z)
mixed = conv1d(in_proj_qkv(x))               (causal, kernel 4)
q, k, v = split(mixed, [key_dim, key_dim, value_dim])
state update (chunked or recurrent): the delta rule with the decay a,
dt from ssm_dt.bias, alpha/beta the projections
out = ssm_out(state * sigmoid(z) ...)
```

The wizard's `wubu_ssm.h` already models this family (Gemma-4 dimensions:
d=2048, dt_rank=32, conv_dim=8192, d_state=128, value_dim=4096). The Qwen3.5
uses d=1024, dt_rank=16, conv_dim=6144, d_state=128, value_dim=2048 — the
same block, different constants. `ssm_norm [128]` = per-value-head RMSNorm.

## Build plan

1. Loader: detect qwen35 (has_ssm + attn_qkv + attn_gate); per-layer kind
   (hybrid vs pure-attn by the tensors present); dims from the KV
   (embedding_length=1024, head_count=8?, head_count_kv=2, key_length=256,
   feed_forward_length=3584, vocab=248320); roles: fused attn_qkv split,
   attn_gate, ssm_* (roles exist for QWEN in wubu_gguf_names.c); the untied
   output.weight head.
2. Forward: hybrid layer = gated-attn branch (fused split + QK norms + RoPE
   + SDPA + sigmoid gate + W_o) + GDN branch (conv1d state + chunked/recurrent
   delta rule + z gate + ssm_out), sum -> post_attention_norm -> FFN.
   Pure-attn layers use the existing lfm2_gqa path (q=4d, kv=0.5d, hd=256,
   q/k norms present).
3. Parity: the V1's llama.cpp (supports qwen35) is the reference — same
   prompt, top-1 logits must match (MiniCPM5-style verification).
4. The GDN recurrent state: only the attention layers keep the KV cache; the
   GDN layers keep the conv state + the recurrent state (the wizard's
   conv_state pattern from the LFM2.5 incremental decode).

Sources: openvinotoolkit/openvino.genai#4138 (the tensor inventory),
sebastianraschka.com gated-attention gallery, rasbt LLMs-from-scratch
ch05/16_qwen3.5, transformers qwen3_next + qwen3_5 modular code,
kaitchup Qwen3.5-397B breakdown.
