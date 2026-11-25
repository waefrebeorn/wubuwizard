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

## Loop semantics (authoritative)

**Source of truth: the base model, `transformers/models/inkling2_loop/modular_inkling2_loop.py`.** Not
llama.cpp - its `inkling2` is the single-pass "pure model" and has no loop driver at all, so it cannot be
the reference for the part that matters. Recorded here because a spec taken from a partial implementation
is how a port gets a wrong skeleton.

Logical depth is **24 + (passes-1) x 12 = 48** (`model_config` computes
`num_hidden_layers + (loop_num_passes - 1) * num_loop_layers`), which is what `block_count = 48` in the
GGUF means. The three passes are not a chain in the naive sense:

```
pre   : layers 0..5
pass1 : layers 6..17, block_states accumulates at block boundaries
pass2+: hidden_states, block_states = pre_loop_hidden_states, pre_loop_block_states   <- RESTART
        messages = message_input(previous_block_states, previous_hidden_states, loop_layer_idx)
        each layer: gate its q/k/v/mlp inputs with those messages, using its OWN cache copies
post  : layers 18..23
final : final_input(block_states, hidden_states)[0] -> norm
```

Three consequences that a naive "run the layer three times" implementation gets wrong:

1. **Each later pass restarts from the pre-loop state**, so the passes do not compound through the
   residual stream - what carries over is the *messages* from the previous pass.
2. **Messages are recomputed from the previous pass's hidden and block states** by
   `Inkling2LoopMessageAttentionResidual`: learned queries of shape
   `[loop_layers, 4, hidden]` (`[12, 4, 2304]`) attend over `rms_norm(concat(block_states, partial))`
   with scaling `hidden**-0.5`, giving four messages per layer - query, key, value, mlp.
3. **Each pass uses its own KV cache copies** (`cache.layers` is extended by
   `deepcopy(loop_cache_layers) * (passes - 1)`, and `cache_offset` addresses them). That is the 3x KV
   the tweet is about: 24 caches become 48 histories, and sharing them back down to 24 halves KV memory.

Message gate (`Inkling2LoopMessageGate`), per input:

```
gate    = down_proj(concat(rms(h), rms(message)))      # 2*hidden -> rank 64
forget  = tanh(forget_gate_proj(gate))                  # 64 -> hidden
msg_gate= tanh(message_gate_proj(gate))
out     = (1 - forget) * h + msg_gate * message
```

(Norm is unweighted RMS. The projections are float32 in the reference.)

Layer input selection matches the tensor census: sliding layers use one `input_layernorm` for q, k, v;
full-attention layers have separate `attn_{q,k,v}_inp_norm` - and those exist on exactly the 4 full
layers (5, 11, 17, 23).

## Step 2 done: the runtime tensor map resolves with 0 missing

`src/wubu_inkling2_load.c` + `include/wubu_inkling2.h` (step 2), probe `tools/inkling_load.c`:
**`load rc = 0`, 0 missing required, 12 optional absent** over all 24 physical blocks.

Three things the census corrected, each of which would have produced a silently wrong forward pass:

1. **The 4 full-attention layers have no `attn_norm`.** blk.5/11/17/23 are the only blocks without
   `blk.N.attn_norm.weight`; their `attn_{q,k,v}_inp_norm` replace it. My first loader marked
   `attn_norm` required on every layer and reported exactly those 4 as missing, which is how it surfaced.
2. **`attn_res_query` exists on layers 1..23**, not only in the loop core (only blk.0 lacks it). So the
   depth mix runs from layer 1 onward; resolving it only for 6..17 would have dropped the mix across
   layers 1..5 and 18..23 - 11 layers of wrong maths that no shape check would catch.
3. **The per-expert `.scale` companions stop at layer 19** (0..19 have them, 20..23 do not).

Measured shapes with a still-open mapping question:

```
loop_res_query.weight     [2304, 24]  F32
loop_kv_res_query.weight  [2304, 24]  F32
```

The reference declares one query tensor of shape `[loop_layers, 4, hidden]` = `[12, 4, 2304]`, i.e. 48
rows. The file splits that into 24 + 24 = 48 rows, so the total matches but the slot order does not.
By name, `loop_res_query` likely covers the query and mlp slots and `loop_kv_res_query` the key and
value slots - **inferred from the names and the 24+24 split, not yet verified** against a reference
forward pass. This is the first thing to pin down before the block residual is implemented.

## Step 3 started: primitives validated against an independent implementation

`src/wubu_inkling2_fwd.c` now has unweighted/weighted RMS norm, a row-major matvec, a whole-tensor
dequantiser and the embedding gather. Deliberately unoptimised (dequantise to f32, plain dot products):
too slow to generate with, but checkable.

**The instrument:** `tools/inkling_fwd_check.c` embeds token 1234, norms it with `blk.0.attn_norm`,
dumps the activation, and prints `attn_q @ x` for the first four rows. An independent Q6_K
implementation in Python then recomputes those four dot products from the same file bytes:

```
engine C           : -5.45495987e-01  +1.41292453e+00  +1.64359808e+00  +1.79610527e+00
python independent : -5.45496014e-01  +1.41292455e+00  +1.64359807e+00  +1.79610532e+00
relative diff       : 4.9e-08         1.3e-08          5.9e-09          2.5e-08
```

Agreement at f32 accumulation noise. That single comparison validates three things at once: our Q6_K
dequantiser, the ggml `(n_in, n_out)` row-major interpretation, and the matvec.

The first run of that check disagreed completely (relative diffs of 0.6 to 2.9). The fault was in the
*instrument*, not the engine - my Python forgot that ggml advances the Q6_K scale pointer by 8 for the
second 128-element half. Worth recording because the reflex on seeing a mismatch is to suspect the code
under test.

**Embedding path: now validated too.** Dequantising token 1234's column of `token_embd` with an
independent IQ4_XS implementation reproduces the engine's gather exactly (all four printed components
match to six decimals) and gives a column RMS of 1.002, which is where trained embeddings sit. So the
IQ4_XS path, the 136-byte block layout and the strided gather are confirmed, not merely self-consistent.

## Short convolution: semantics read from the reference, then verified

`Inkling2LoopShortConvolution` is not a bare conv. It is a depthwise `Conv1d(kernel=4, groups=channels,
padding=3, bias=False)` followed by

```
out = (conv(x) + x) / sqrt(k + 1)
```

Both the residual add and the `1/sqrt(k+1)` are part of the block; omitting them scales the output by
2.24x. Tap order is `tap index == backward distance`, so `w[0]` weights the current token.

The full-attention layers add a second, dynamic term (`Inkling2LoopDynamicShortConvolution`):
`taps = static[tap] + weight_proj(dynamic_input)` with `weight_proj: input_size -> num_heads * k`, shared
within each kv head. That matches the census exactly: `attn_k_conv` is `[4, 1024]` on *every* layer
(static only) while `attn_k_conv_proj` is `[2304, 32]` on the four full layers only (32 = 8 heads x 4 taps).

**Verified:** `in2_shortconv` against a Python brute-force reference over the same weights and input,
all 6,912 values of a 3x2304 run:

```
max abs diff 5.08e-08   mean 6.87e-10   relative max 4.09e-08
```

## A correction: head_dim is 128 on every layer

An earlier note in this file claimed head_dim 512 for full attention and 256 for sliding window. That is
wrong. `attn_q` is `[2304, 3072]` and `attn_k`/`attn_v` are `[2304, 1024]` on *both* blk.0 (sliding) and
blk.23 (full), so both layer types are 24 query heads and 8 kv heads at head_dim 128. The sliding window
changes which keys are attended, not the head geometry.

## Expert magnitudes: resolved

The MoE expert tensors decode to +-hundreds while the dense tensors decode to +-0.05. That is the file's
convention, not a misread: every expert tensor has a per-expert `.scale` companion (F32, one value per
expert, ~4e-5 to 2e-4), and the product lands in the same range as everything else. The `.scale` is
applied at runtime.

## LoopCD (arXiv 2610.02185) - where it plugs in

At decode, run the loop to completion, keep the logits after pass 1 and after pass 3, and contrast them:
the strong-vs-weak difference is what looping improved, so `logits = pass3 + alpha*(pass3 - pass1)`
(exact form from the paper). This forces one design requirement on the engine: **per-pass logits must be
reachable**, not just the final ones. In a 24-layer model with a 12-layer core that is cheap - the head
can be run twice, once per pass of interest.


## Step 1 done: the engine's own GGUF reader opens the file

`tools/gguf_probe.c` (built against `src/gguf_reader.c`, `src/wubu_win.c`, `src/wubu_dequant_fp4.c`) opens
the IQ2_XXS and resolves every probe:

```
734 tensors, 62 KV pairs, arch fields: block_count 48, embd 2304, experts 128, used 8, loops 3
tokens (unpadded vocab) 200058

token_embd.weight            IQ4_XS  [2304, 201024]
blk.0.attn_q.weight          Q6_K    [2304, 3072]        3072 = 24 heads x 128
blk.0.attn_rel_bias.weight   F32     [16, 1024]          distance table: rel_len x extent
blk.0.ffn_gate_up_exps       Q5_K    [2304, 2304, 128]   fused gate+up per expert
blk.0.ffn_gate_inp.weight    F32     [2304, 128]         router
blk.0.attn_k_conv.weight     F32     [4, 1024]           shortconv: 4 taps x (8 kv heads x 128)
blk.6.loop_attn_down.weight  F32     [4608, 64]          loop gate: concat(rms,rms) -> rank 64
blk.6.loop_mlp_gain.weight   F32     [64, 2304]
blk.23.attn_k_conv_proj      (30)    [2304, 32]          dynamic taps: 4 x 8 heads, full layers only
output.weight                IQ4_XS  [2304, 201024]
output_res_query.weight      F32     [2304]
```

So the reader is not the blocker: the 734-tensor map, the MoE expert blocks and the loop gates all resolve
by name. What the engine lacks is the *math* - MoE routing at 128 experts (the existing
`wubu_gpu_moe_forward_experts` is hard-wired to 8), relative attention with the bias table and sinks (no
RoPE), shortconv with its recurrent state, the attn_res `mix` blocks, and the loop driver.

## Reference class (not llama.cpp)

For throughput and design we compare against **Strata** and **Project Maya**, and for the quant formats
we read what **AtomicChat**, **unsloth** and the **base model** ship. llama.cpp is explicitly not a
reference here: its `inkling2` is single-pass with no loop driver, so matching it would mean matching a
partial implementation of the model. Where the formats disagree between producers, the base model's own
definition decides - which is also why the GGUF inventory above is taken from the file rather than from
another engine's loader.

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
