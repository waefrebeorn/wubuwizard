# Inkling 2 Mini: port specification

Derived from the GGUF this box actually has and from llama.cpp's `src/models/inkling2.cpp`
(master, read 2026-11-24). Facts below are measured from the file, not from the model card.

Files on disk:
- `C:/Users/eman5/models/inkling-2-mini/Inkling-2-Mini-UD-IQ2_XXS.gguf` — 8.49 GB, 734 tensors
- `C:/Users/eman5/models/inkling-2-mini/Inkling-2-Mini-UD-Q4_K_XL.gguf` — 17.63 GB, 734 tensors

## Why no existing binary loads it

| build | result |
|---|---|
| llama.cpp 9496 (ours) | no `inkling2` architecture at all |
| llama.cpp b12600 / b12686 | `check_tensor_dims: tensor 'blk.24.attn_q.weight' not found` |

Both GGUFs carry `inkling2.block_count = 48` but store only 24 distinct block tensor sets
(`blk.0` .. `blk.23`), because layers 6..17 are executed three times rather than duplicated. Patching
`block_count` to 24 on a copy moves the error to `key inkling2.attention.head_count has wrong array
length; expected 24, got 48` - i.e. llama.cpp wants the **unrolled** 48-layer form with per-layer arrays
of 48.

And that is consistent with what upstream has: `inkling2.cpp`'s graph constructor is

```cpp
for (int il = 0; il < n_layer; ++il) cur = build_layer(il, il, cur, blocks);   // messages = nullptr
```

- a single pass, no loop driver. That is the "pure model" PR (#32217); the loop machinery (PR #32704,
  "re-share the loop slots after the scale pass") is not in the tree. `control()` and the `loop_layer`
  structs are present but never called. **So upstream is not a reference for the loop yet**, and the
  loop semantics have to be taken from the metadata plus `control()`.

## Architecture (from GGUF metadata)

```
arch inkling2                     block_count 48   physical layers 24   loop: core 6..18, 3 passes
hidden 2304                       heads 24 / kv 8 / head_dim 128        vocab 201024 (gpt2 BPE, pre=inkling2)
MoE 128 experts, top-8, expert_ff 1152, route_scale 8.0, shared_experts 0
attn: relative (rel_len 16, extent 1024, rel_bias table, rel_clip 8.0), qk_clip 22.6274166
      rope.dimension_count = 0  -> no RoPE anywhere
      log-scaled positions: tau = 1 + 0.1*log(max(1, (pos+1)/8192)), multiplies q and r on full layers
extras: shortconv l_cache 4, situ activation (beta 4.0 gate / 25.0 up), attn_res blocks of 6,
        value norm (no gamma), final logit softcap 30, MTP blocks (block_size 8, 4 target layers)
sliding window 1024, pattern [1,1,1,1,1,0] -> full-attention layers are 5, 11, 17, 23
```

Tensor map (per layer, 24 of each unless noted):

```
attn_q, attn_k, attn_v, attn_output, attn_gate          projections (attn_gate gates the o-output)
attn_rel [16 x heads]        relative-position query    attn_rel_bias [16 x ?]  bias table
attn_q_clip, attn_rel_clip   per-channel clip scales     attn_sink_proj  sink logits
attn_k_conv, attn_v_conv, attn_post_conv, ffn_post_conv  shortconv weights (l_cache 4)
attn_norm (20), ffn_norm (24), attn_res_query (23), ffn_res_query (24)
attn_{q,k,v}_inp_norm + attn_{k,v}_conv_proj   (4 layers only: the full-attention ones)
ffn_gate_up_exps (fused gate+up for all 128 experts), ffn_down_exps, ffn_gate_inp (router),
ffn_down_exps.scale / ffn_gate_up_exps.scale (20 layers), exp_probs_b.bias
loop_{attn,k,v,mlp}_{down,gain,forget}   (12 layers = the loop core; down: 2*2304 -> 64,
                                          gain/forget: 64 -> 2304)
global: token_embd, token_embd_norm, output_norm, output, loop_res_query, loop_kv_res_query,
        output_res_query
```

## Block math (from llama.cpp graph, restricted to what is implemented there)

`clip(x, scale, t)`: `x *= scale` (if given), then `x * min(1, t/|x|)` via `t * l2_norm(x, t)`.

`mix(blocks, cur, query)` — attention over depth: stack `[cur, block_n ... block_0]` on the depth axis,
rms-norm, then `scores = states · queries` with learned per-slot queries, `softmax(scores/sqrt(2304))`,
out `= scores · states`. This is how `attn_res_query` / `ffn_res_query` / `output_res_query` mix earlier
block outputs.

`convolve(cur, w, dynamic, state, history, offset)` — causal shortconv, kernel 4:
concat the last 3 inputs (kept in the recurrent state at `offset`) with the current row, `ssm_conv` with
`w`; when `dynamic` is present (full-attention layers) add per-head input-dependent taps, reshaping
dynamic to `[4, heads, tokens]`; result `= (cur + out) / sqrt(n_conv (+n_conv when dynamic) + 1)`.

`control(cur, message, loop, ctrl)` — the cross-pass loop gate:
`gate = down_lora(cat(rms(cur), rms(message)))`, `gain = tanh(gain_lora(gate))`,
`forget = tanh(forget_lora(gate))`, out `= cur - forget*cur + gain*message`.
Control slots: `LOOP_CTRL_Q`, `Q+1` (K), `Q+2` (V), `FFN` — four messages per pass.

Layer, in order:
1. `inpSA = cur`. If `attn_res_query` present: `mixed = mix(blocks, cur, attn_res_query)`; SWA layers use
   one slice for q/k/v, full layers use three separate slices.
2. If looping: `inputs[j] = control(inputs[j], message(Q+j), ...)` for j in 0..2.
3. Normalise: SWA → all three share `attn_norm` (or one norm each while looping); full → separate
   `attn_{q,k,v}_inp_norm`.
4. `Q = wq(x0)`, `K = convolve(wk(x1), attn_k_conv, dyn, off 0)`, `V = convolve(wv(x2), attn_v_conv, dyn, off n_embd_k_gqa)`,
   `R = attn_rel(x0)`.
5. `Q = clip(Q, attn_q_clip, 22.63)`, `K = clip(K, NULL, 22.63)`, `V = rms_norm(V)`,
   `R = clip(R, attn_rel_clip, 8.0)`. On full layers `Q *= tau`, `R *= tau`.
6. `sinks = attn_sink_proj(x0)`; `table = pad(attn_rel_bias) · R`, subtract sinks, gather by relative
   position, giving the per-(head, distance) bias `kq_b`; sink vector zeroed afterwards.
7. attention with that bias, scale `1/sqrt(head_dim)`; then `cur *= sigmoid(attn_gate(x0))`;
   `cur = convolve(wo(cur), attn_post_conv, NULL, off 2*n_embd_k_gqa)`.
8. Residual: every `attn_res_block_size`(=6) layers starts a new block (`blocks.push_back(inpSA)`),
   otherwise `cur = inpSA + cur`.
9. `ffn_inp = cur`; `cur = mix(blocks, cur, ffn_res_query)`; if looping `control(cur, message(FFN))`;
   `rms_norm(ffn_norm)`.
10. MoE: router with sigmoid gating, top-8 of 128, `expert_weights_scale = 8.0`, situ activation,
    fused `ffn_gate_up_exps`.
11. `cur = convolve(cur, ffn_post_conv, NULL, off 2*n_embd_k_gqa + n_embd)`; `cur = ffn_inp + cur`.

Output: `mix(blocks, cur, output_res_query)` → `rms_norm(output_norm)` → `output` → softcap 30 →
**drop the padded vocab** (keep 200058 of 201024).

## What is missing upstream and must be designed here

The loop driver. From the metadata and `control()`: layers 6..17 run three passes; each pass produces
four messages per layer (q, k, v, ffn) that gate the next pass's inputs; the `loop_*` tensors are per
layer, not per pass, so the same gates are applied each pass while the message differs. Layer-6's three
passes share one KV cache slot history (the reason the tweet is about cache sharing: without it the 12
core layers need 3x the KV, which is the 8.76 -> 4.38 GB claim).

## LoopCD (arXiv 2610.02185) - where it plugs in

At decode, run the loop to completion, keep the logits after pass 1 and after pass 3, and contrast them:
the strong-vs-weak difference is what looping improved, so `logits = pass3 + alpha*(pass3 - pass1)`
(exact form from the paper). This forces one design requirement on the engine: **per-pass logits must be
reachable**, not just the final ones. In a 24-layer model with a 12-layer core that is cheap - the head
can be run twice, once per pass of interest.

## Port order

1. GGUF loader for `inkling2`: config fields above, 734-tensor map, MoE expert blocks.
2. Single layer, no loop, no MoE shortcut: verify against llama.cpp semantics by checking shapes and
   intermediate norms on a fixed input.
3. MoE router + experts (situ, top-8, scale 8).
4. Relative attention with the bias table and sinks (no RoPE - the whole attention block is different
   from the Gemma path).
5. Shortconv with the recurrent state.
6. attn_res `mix` blocks.
7. Loop driver over layers 6..17 with `control()`.
8. LoopCD at decode: pass-1 vs pass-3 logits.
9. Decode/prefill kernel reuse from the existing engine (IQ quant matvecs, mma) once the CPU reference
   is token-stable.
