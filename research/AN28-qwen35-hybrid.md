# AN28 — Qwen3.5-0.8B hybrid (Gated DeltaNet + Gated Attention) on the wizard

2026-08-09. STATUS: **WIRED** — loads + generates + top-1 parity with the
V1's llama.cpp on the wubu_model (universal engine, NOT the lfm2 — the
lfm2 is LFM2.5-specific conv-GQA). Verified: `Hello` → top1 id=11 (`,`)
identical to the reference; runtime dims D=1024 GQA=8x2 hd=256 kv=512
SSM(k=16 s=128 v=16 dt=16); 18 GDN layers + 6 gated-attn layers; forward
fully finite. Bugs fixed along the way (commit 084d7c5):
1. gen_text embedding file was hardcoded to the qwen36 path (both prefill
   AND decode loop) → wrong-vocab rows → zeros → RMSNorm(0) = all-NaN.
   Now `data/embeddings_<vocab>_<d>.bin.raw`.
2. `quantized_matmul_from_q8` dispatched Q8_0 weights to q8_0_vec_dot,
   which expects Q8_0-format activations (34-byte blocks) but callers pass
   a Q8_K buffer (292-byte) → garbage scales. Q8_0 now takes the
   dequant-SGEMM path (same as IQ/Q2/Q3).
3. GQA head count: `attn_q` columns are the Q+gate FUSED pair (2·hc·hd) —
   q_dim/plain_hd inferred 16 heads and the Q_full matmul read past the
   4096-col weight. True q_dim = `attn_output`'s input dim (2048 → 8 heads
   × 256). The loader now derives q_heads from attn_output.
4. Makefile: `wubu_dense_ffn.o` was missing from CORE_OBJ.
Remaining: logit-level exactness beyond top-1 (near-tie reorder of ids
0/13 from Q8 activation quantization — acceptable).

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

**RESOLVED 2026-08-09 (AN61)** — the extra 1024 is the gated-attn's OWN
output-gate projection fused at the tail: the config has
`attn_output_gate: true`, so the fused layout is
[q_and_gate (2*d_out) | k (kv*hd) | v (kv*hd) | z (d)] = 4096+512+512+1024.
The separate `attn_gate [1024, 2048]` tensor is the sigmoid gate over the
post-W_o output (2*d = 2048). The GDN branch has its own fused
`ssm_in_proj [d, 8192]` = qk (2*key_dim = 4096) + v (value_dim = 2048) +
z (value_dim = 2048); `ssm_conv1d [4, 6144]` = qk + v; `ssm_out
[value_dim*2, d]` = [4096, 1024]. See include/wubu_qwen35.h (the config-
driven split math, test_qwen35 pins the 0.8B numbers).

**Build next (the loader role math is DONE)**: the hybrid FORWARD
(gated-attn branch + GDN branch) + logit-parity vs llama.cpp — needs the
Qwen3.5-0.8B GGUF on disk (unsloth/Qwen3.5-0.8B-GGUF, Q8_0, 1.19GB).

**PARITY PROTOCOL (2026-08-09, the GGUF is local + llama-server runs)**:
- the reference: `/home/wubu/llama.cpp/build/bin/llama-server -m
  /home/wubu/models/Qwen3.5-0.8B-Q8_0.gguf --port 8090`; the fixture
  tool `tools/qwen35_parity_fixture.py "PROMPT" --port 8090` captures
  the deterministic (temp 0) TOP-5 logprobs per generated position.
- the anchor: `research/qwen35-parity-reference.json` — "The capital
  of France is" -> " Paris." (top-1 ' Paris' -2.062, then '.' -0.618).
- the wizard side must match TOP-1 per position once the hybrid
  loader runs the full 24-layer stack (the MiniCPM5-style
  verification); the TOP-5 logprobs are the richer oracle.

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
