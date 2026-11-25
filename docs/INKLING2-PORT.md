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

## Relative attention: the whole block, verified

There is no RoPE anywhere (`inkling2.rope.dimension_count = 0`). Position enters as a **token-conditioned
mixture of learned distance profiles**:

```
rel_states  = attn_rel(hidden)                       # 2304 -> 24 heads x 16 (d_rel)
rel_states *= attn_rel_clip[h][r]                    # per-(head, profile) learned scale
per head:  rel_states *= min(1, 8.0 / (|rel_states| + 1e-6))     # relative_clip = 8.0
bias[h][dist] = sum_r rel_states[h][r] * P[dist][r]  # P = attn_rel_bias, 16 profiles x 1024 distances
score[h][q][k] = q·k/sqrt(128) + bias[h][dist(q,k)]  # zero outside 0 <= dist < 1024
```

So `attn_rel_bias` is literally the reference's `proj` parameter `(d_rel, rel_extent)`, and the 384-wide
`attn_rel` is `hidden -> num_heads * d_rel` (24 x 16). ggml stores both transposed relative to PyTorch:
`attn_rel_clip` is `(16, 24)` -> scale indexed `[h*16 + r]`, and `attn_rel_bias` is `(16, 1024)` ->
`P[dist*16 + r]`, so the bias is `rel_states @ P.T`.

**Verified** against an independent implementation (Q8_0 dequant of `attn_rel`, F32 reads of the clip and
the profile bank) for token 1234 at blk.0:

```
rel_states   max abs diff 7.9e-07   (values to 7.6)
bias table   max abs diff 8.1e-07   (values to 8.6)   over all 24 x 1024 entries
```

That one comparison pins the Q8_0 path, the 384-output layout, the clip scale layout, the L2-clip formula
and the profile-bank transpose together.

## The rest of the attention block (read from the reference)

- **q**: `attn_q` -> scaled L2 clip, scale `attn_q_clip (128,24)`, threshold `qk_clip = 22.6274166`.
- **k**: `attn_k` -> static (or dynamic, on full layers) short conv -> **bare L2 clip**, same threshold.
- **v**: `attn_v` -> short conv -> **unweighted RMS norm** (not a clip).
- **output**: `sigmoid(attn_gate(hidden)) * attn_out`, then `attn_output`. The gate is a sigmoid over the
  full 3072-wide attention output, so `attn_gate` is `[2304, 3072]` as the census shows.
- **sinks**: `attn_sink_proj` is `hidden -> num_heads`, giving one extra logit per query whose value
  slice is zero: it only absorbs probability mass, so it lowers attention sharpness without contributing.
- **head geometry**: `swa_head_dim`/`swa_num_attention_heads` are separate config fields, but for this
  model both resolve to 128 / 24 / 8 kv heads, which is why the q/k/v tensors have identical shapes on
  sliding and full layers.

## A real engine bug found on the way: IQ1_S raw size was 8 bytes short

Reading a 5.3M-element IQ1_S expert slice segfaulted inside `dequantize_iq1_s_row`. The cause was not
the new code:

```
gguf_raw_size(GGML_TYPE_IQ1_S) -> n_blocks * 42      // counted qh as 8 bytes
#define IQ1_S_BLOCK_SIZE 50                          // the dequantiser's own comment: 2 + 32 + 16
```

`qh` is `uint16_t qh[8]` = **16** bytes, so a block is 2 + 32 + 16 = 50, which is what ggml's
`block_iq1_s` says. `gguf_raw_size` under-allocated 8 bytes per block. This was invisible on
256-element probes because the overrun landed inside a padded buffer; at 20,736 blocks it overruns by
165 KB and crashes. Any IQ1_S tensor read through that path was being read into a buffer that was too
small - the values decoded correctly, then the heap was corrupted. Fixed to 50.

## MoE: routing and the expert path

Read from the reference and implemented:

```
logits  = ffn_gate_inp @ x                     # 128 experts, fp32
scores  = sigmoid(logits)
pick    = topk(scores + exp_probs_b, 8)        # bias affects SELECTION only
weights = scores[pick] / sum(scores[pick]) * 8.0
per expert: gu = gate_up[e] @ x * gate_up_scale[e]
            gate, up = gu[:1152], gu[1152:]    # chunk(2, -1): gate first
            act = tanh(gate/4)*4 * sigmoid(gate) * tanh(up/25)*25
            out += weights[k] * (down[e] @ act) * down_scale[e]
```

Two details that are easy to get wrong: the sigmoid sees the **raw** gate (only the linear factor and
`up` are softcapped), and the correction bias never touches the contribution weights.

**Verified** for token 1234 at blk.0: selection `e112 e6 e116 e95 e9 e108 e51 e111` and weights
`1.487772 1.109196 0.987676 0.947753 0.924911 0.916767 0.823771 0.802153` match an independent
implementation to six decimals, and the weights sum to exactly 8.000000. The per-expert scales land in
the expected band (`gate_up 7.06e-05`, `down 1.01e-04`).

Note: `exp_probs_b` is non-zero on all 128 experts (max 0.065), but for this token the biased and
unbiased top-8 agree, so this check does not yet discriminate the correction-bias rule. A token where
the two selections differ is needed for that.

## Step 4: a complete sliding layer runs end-to-end

`in2_layer_sliding` now runs the whole block for N tokens: attn_norm -> q/k/v with convs and clips ->
relative-bias + sink softmax over the sliding window -> sigmoid-gated output -> post-attn conv ->
residual rule -> depth mix -> MoE -> post-mlp conv -> residual add.

The layer order is **not** the conventional pre-norm block, and two details would be easy to miss:

- `is_block_boundary` is `il % 6 == 0`, so boundaries are layers **0, 6, 12, 18** (not the 5/11/17/23
  full-attention layers). At a boundary the attention output *becomes* the residual and the previous
  hidden state is banked into `block_states`; elsewhere `residual = block_residual + attention_out`.
- `mlp_input` exists on every layer while `attention_input` only for `layer_idx > 0`, which is exactly
  the `ffn_res_query` / `attn_res_query` pattern in the tensor census.

**First result:** `rc=0`, rms 1.0699, max|h| 6.24, no NaN. A causality check falls out for free: token 0
is bit-identical between an N=1 and an N=2 run (it can only attend to itself, and the conv has no history
at t=0) while the sequence RMS changes - so no future leakage.

**Still unverified:** the layer is not yet compared numerically against the reference. That needs a
Python reference for one full layer, which in turn needs IQ1_S and Q2_0 dequantisers in Python (the
expert tensors). Until then this is a runnable graph with internally consistent pieces, not a validated
one.

### Two bugs found while wiring it up

1. **Engine:** `gguf_raw_size(IQ1_S)` was 42 bytes/block instead of 50 (`qh` is `uint16_t[8]`), which
   under-allocated 8 bytes/block and segfaulted the first large expert slice. Fixed.
2. **Mine:** `in2_dequant_slice` asked for `gguf_raw_size(type, 0)` to get the byte offset of element 0,
   but the fp4 raw-size helper returns -1 for `n_elems <= 0`, so **expert 0 of every MoE tensor** failed
   to decode while experts 1..127 worked. Fixed by special-casing offset 0.

## Step 5: the first complete layer is verified against an independent reference

Every intermediate of the sliding layer (blk.0, token 1234, one token) compared against a Python
reference built from **canonical llama.cpp dequantisers** (IQ1_S grid and algorithm taken from
`ggml-common.h`/`ggml-quants.c`, Q2_0 likewise) and the base model's forward semantics:

```
stage   max abs diff    stage    max abs diff
q       3.9e-07         ao       6.3e-08
k       1.8e-07         res      3.2e-08
v       4.8e-07         moein    8.9e-07
attn    3.4e-07         moe      2.0e-06
gate    2.3e-07         final    8.1e-07
```

Values reach ~3.8, so these are float32 accumulation differences. IQ1_S and Q2_0 had already been
confirmed separately on an expert slice (gate_up and the expert output both matching to six decimals).

### Correction: the earlier short-conv verification was circular, and a layout bug it hid

The 6,912-value short-conv agreement reported earlier proved nothing about layout: the C code indexed
the filter as `w[tap*channels + c]` and the Python used the same reshape, so both were wrong the same
way. ggml stores these as `(k, channels)` with **ne0 = k fastest**, i.e. `w[c*k + tap]`, matching
`nn.Conv1d`'s `[out_channels, 1, kernel]`.

What caught it was a **structure test**, not a cross-check: reading the filter as `[tap][c]` gives flat
per-tap RMS (0.95, 0.72, 0.79, 0.77 - noise), while `[c][tap]` gives a clear profile (0.15, 0.26, 0.70,
1.43). A correct grouping of a learned filter shows structure; a wrong one averages across unrelated
axes. Lesson worth keeping: **an independent implementation must be independent in its assumptions, not
just in its code** - same-language, same-reshape "verification" is a mirror.

Also fixed on the way: the layer was writing its attention output into `h` before the residual was
captured, and the MoE slice reader failed for **expert 0** of every tensor (it asked `raw_size` for the
byte offset of element 0, which returns -1 for `n_elems <= 0`).

## Step 6: the depth path, and where it stops

`src/wubu_inkling2_model.c` runs embed (with the normed-embedding wrap) -> 24 physical layers -> the
`final_input` depth mix -> weighted output norm -> lm_head -> softcap(30). `tools/inkling_logits.c`
drives it for one token.

Two things are deliberately explicit rather than silent:

- `in2_forward_token` **refuses** when `n_loops > 1`, because the message gates are not implemented. A
  partial graph that answers anyway would produce the output of a shallower model and be indistinguishable
  from a bug. `IN2_SINGLE_PASS=1` overrides this to exercise the depth path, and anything it prints is a
  1-of-3-pass number.
- The run currently stops at **blk.5** with `rc=-6`: blk.5 is the first full-attention layer and has no
  `attn_norm` (its per-q/k/v input norms replace it), so the sliding-layer function cannot run it. The
  full-attention variant - separate input norms, dynamic conv taps, `log_scale` on q, attention over all
  keys rather than a window - is not implemented yet. That is the next piece, and the depth path is
  otherwise complete for the layers it can run.

## The full-attention layers are structurally different, not just masked differently

blk.5, 11, 17 and 23 are not "sliding layers with a wider window". From the file, all four:

```
attn_res_query.weight  (2304, 3)    three depth-mix queries   (one elsewhere)
attn_q/k/v_inp_norm    present      three separate input norms
attn_norm              ABSENT       replaced by the three input norms
```

and from the reference, `Inkling2LoopFullDecoderLayer` builds
`attention_input = BlockAttentionResidual(config, num_queries=3)` and feeds three *separately mixed*
states into q, k and v:

```
q_state, k_state, v_state = attention_input(block_states, h)      # 3 queries, 3 outputs
h = attn(q_input_layernorm(q_state), k_input_layernorm(k_state), v_input_layernorm(v_state))
```

So the depth mix is not just a preconditioner for the attention - on these layers it produces three
different inputs, which is why the query tensor has three rows. Implementing them as a sliding layer with
`sliding_window = None` would have run and produced plausible numbers with the wrong structure entirely.

### Dynamic short conv (full layers only)

```
taps[c][j] = static[c][j] + dynamic[h][j]        # dynamic is per HEAD, shared across its channels
out[t][c] = sum_j taps[c][j] * x[t-j][c]
out = (out + x) / sqrt(2*k + 1)                  # sqrt(9) = 3, NOT sqrt(5) as the static conv
```

Two traps: the dynamic term is per kv head while the static term is per channel, and the normaliser is
`sqrt(2k+1)` here against `sqrt(k+1)` for the static post-attention/post-mlp convs. The reference even
comments on the static one that it differs from Inkling by having an output scale.

The four full layers also use `LogScaledL2Clip` on q (the `tau = 1 + 0.1*log(max((pos+1)/8192, 1))` term),
which the sliding layers do not.

## Step 7: token -> logits runs end to end (24 layers, one pass)

`in2_layer_full` is implemented and wired in, so the depth path now reaches the lm_head:

```
forward rc=0  token=1234  d=2304 vocab=201024
logits: max 19.4700  nan=0  logZ 21.8288
  top1 id=196984   top2 id=60159   top3 id=150817   top4 id=64   top5 id=31109
```

Three tokens were run to check the output is input-dependent rather than constant: top-1 came out
`51993`, `196984` and `82` for tokens 2, 1234 and 5000, with different max logits (19.42 / 19.47 / 17.03).

**These are not the model's predictions and must not be read as such.** Two known gaps sit behind them:

1. **One loop pass of three.** `IN2_SINGLE_PASS=1` is set; the message gates are still unimplemented, so
   this is the output of a shallower model with the same layers.
2. **Sliding layers ignore `block_states`.** They still use the single-block approximation where
   `mlp_input` degenerates to a norm. The full layers do use the real 3-query mix. That inconsistency
   means the two layer classes see different depth information, which is wrong in a way the logits cannot
   reveal.

So: the plumbing is complete end to end, and the numbers are structurally unverified. The next steps
close the two gaps rather than adding features.

## Gap closed: sliding layers now use the depth mix

The sliding layer had never applied `attention_input` at all - it normed the raw hidden state - and its
`mlp_input` was hard-wired to a bare RMS norm. Both now go through the real depth mix over
`[block_states; partial]`, so sliding and full layers finally see the same depth information. The full
layers keep the 3-query variant.

**The change is safely bounded, and that is checkable:** blk.0 is bit-identical afterwards (rms 0.913700,
h[0..3] = 0.045412 0.754640 -0.048702 -0.061992), which is what must happen - layer 0 has no
`attn_res_query` and no banked blocks, so the mix degenerates to the same norm. The verified reference for
blk.0 therefore still stands rather than being invalidated by the refactor.

The model-level output does change, as it should: top-1 moves from 196984 to 66 for the same token, since
every layer above 0 now mixes depth instead of ignoring it.

## The three-pass loop runs

`in2_messages` (four learned queries over the previous pass's `[block_states; hidden]`) and
`in2_message_gate` are implemented, the layer takes messages into its q/k/v/mlp inputs, and the model
forward now does the real structure: pre 0..5 -> pass 1 over 6..17 -> passes 2 and 3 restarting from the
pre-loop state with messages -> post 18..23. `IN2_SINGLE_PASS=1` still forces one pass for A/B.

```
3 passes : rc=0  max logit 23.27  min 20.28  top1 150417
1 pass   : rc=0  max logit 20.31  min  6.73  top1 66
```

The loop materially changes the output, which is the point - but **the logit spread collapsed** from
14 points to 3. A near-constant logit vector is not what a working model of this size should produce, and
it is not explained by anything measured so far. Two candidate causes, neither proven:

1. **The message-query slot order is an assumption.** The file splits the reference's `[12, 4, 2304]`
   query tensor into `loop_res_query` and `loop_kv_res_query`, 24 rows each. I assumed res = (query, mlp)
   and kv = (key, value). If that is wrong, all four messages are permuted every pass after the first.
   `IN2_MSG_ORDER=1` flips the alternative so the two can be compared once there is a reference to compare
   against.
2. **Full-attention layers ignore messages.** `in2_layer_full` takes no message argument, so in passes 2
   and 3 the four full layers (5, 11, 17, 23 - which are inside the core three times) run ungated while
   their sliding neighbours are gated. That is a known hole, not a subtlety.

Until one of those is closed, the three-pass numbers are structurally complete but numerically suspect,
and the flat logits are the visible symptom.

## The flat-logit symptom was the ungated full layers

Before: three passes, logits spanning 3 points (20.28..23.27). The suspicion was the full layers ignoring
messages. The file settled whether that mattered: **blk.11 and blk.17 carry all four loop-gate tensors as
well as the three input norms**, while blk.23 (full but outside the loop) has norms and no gates. A
converter does not emit 24 gate tensors per layer for decoration, so those gates have to be applied - and
they gate the three depth-mixed states plus the mlp input.

Applying them:

```
ungated full layers : rc=0  max 23.27  min 20.28  span  3.0   top1 150417
gated full layers   : rc=0  max 20.97  min  7.77  span 13.2   top1 190472
```

The span recovers from 3 to 13 points. That is not proof of correctness - it is evidence that the flat
distribution was caused by the ungated layers rather than by something else, because removing that one
defect moved the symptom in the expected direction.

### Message-query slot order: still an assumption, now measured

`IN2_MSG_ORDER` flips the 24+24 interpretation of the split query tensor:

```
order A  res=(query, mlp) / kv=(key, value) : span 13.2  top1 190472
order B  res=(query, key) / kv=(value, mlp) : span  4.7  top1 386
```

Order A gives the more structured distribution, which is what a working model should produce, so A stays
the default. **Suggestive, not conclusive** - a second defect elsewhere could flatten B just as easily.
This is exactly the kind of question that needs a reference forward pass to settle, and it remains the
single largest unverified assumption in the port.

## The output ids are real vocabulary, and what multi-token still needs

The engine has no detokeniser (only `gguf_tokenizer_token_count`), so `tools/gguf_detok.py` walks the KV
section for `tokenizer.ggml.tokens` and expands the Gemma-family `\u0120` space marker and `<0xNN>` escapes.
Decoding the top-1 ids from earlier runs:

```
190472 -> ' telescope'    150417 -> ' group'     289 -> 'ch'
66     -> 'c'             60159  -> 'dji'        386 -> 'ly'
```

Every one is a real subword in the 200,058-token vocabulary. Weak but real evidence: a graph that was
badly wrong would be expected to rank control or degenerate ids at the top. It does not say the words are
*right* - these are context-free single-token predictions from an unverified message-slot mapping.

**What multi-token generation needs, concretely.** `in2_layer_full` and the model forward are single-token:
the full layer's attention loop and buffers, and the model's block-state handling, are sized for one token.
Sliding layers already take `n_tok`. So the path is:

1. Generalise `in2_layer_full` to `n_tok` with causal masking (the sliding layer is the template).
2. Generalise `in2_forward_token` to a token sequence, with block states per position rather than per token.
3. Add the KV cache so a step costs one token instead of the whole prefix - which is also the first point
   where throughput means anything.

That is a real refactor, not a switch, and it is the next body of work.

## A sequence-capable full layer, and a bug it exposed

`src/wubu_inkling2_fullseq.c` generalises the full-attention layer to `n_tok` with causal masking, so
prefill and generation can share one path. The rewrite was expected to reproduce the single-token results
exactly at `n_tok == 1`. It does not:

```
single-token layer : max 20.9703  min 7.7672  span 13.2  top1 190472
sequence layer     : max 19.8403  min 8.9541  span 10.9  top1 87
```

The difference has one identified cause. The single-token version applied the slot-3 (mlp) message gate by
writing the gated value **into `h`**, so the residual add two lines later
(`if (!boundary) res[i] = h[i] + res[i]`) added a message-gated vector instead of the layer input. The
reference applies that gate to the mlp input *after* the depth mix, leaving the residual untouched. The
sequence version does it correctly, and the numbers move as a result.

So the sequence layer is not merely a generalisation - it fixes a defect that was silently corrupting the
residual stream on four layers per pass. Worth stating plainly: **the earlier three-pass numbers were
produced by a graph with this bug in it**, and the span 13.2 figure that made the full-layer gating look
right was measured through that bug. The gating conclusion still stands (it moved the symptom the right
way), but the absolute numbers before this commit are superseded.

## The model forward takes a sequence

`in2_forward_seq` replaces the single-token entry point: it embeds every token, keeps block states as
16 entries x `n_tok` positions (the mix grew a `block_stride`/`pos_off` pair to address them), runs the
layers over the whole sequence, and emits logits for the last position.

**The regression that matters passed this time:** at `n_tok == 1` the output is byte-identical to before
the generalisation (`max 19.8403`, `min 8.9541`, `top1 87`). Unlike the last refactor, this one changed no
arithmetic - it only threaded a stride through.

```
1 token  : rc=0  max 19.8403  min 8.9541  span 10.9  top1 87
2 tokens : rc=0  max 20.1966  min 2.7856  span 17.4  top1 5195
```

The span widening with context is the expected direction, not evidence of correctness.

**Still not exercised by any of this:** the sliding window (needs > 1024 tokens before it clips anything)
and `tau` (needs position > 8192 before `log` stops returning 0). Both are implemented and untested at
their operating point.

## Greedy generation, without a cache

`tools/inkling_gen.c` prefills a prompt and takes the argmax repeatedly, re-running the whole prefix each
step. That is seconds per token - far too slow to be the final path - but it cannot hide a cache bug,
which is exactly why the first output should come through it. The KV cache comes after, and any divergence
between cached and uncached decoding is then a cache bug by construction rather than a needle in the graph.

## First generated tokens, and what they say

Generation before this run selected id **200567**, which is beyond the 200,058 real tokens - a padded vocab
row with a meaningless embedding. The reference drops the padded vocab before sampling; that mask was
missing. With it applied the model can no longer pick padding, and the first generation reads:

```
ids: 1234 87 779 11 1234   ->   "utexot,ute"
gen logits: 19.84, 17.18, 19.09, 27.90
```

**It is degenerate**: the fourth token is the prompt token itself, at a logit of 27.9 against ~19 for
everything else. A model returning to its own input with a commanding margin is a symptom, not a
prediction - it is what a broken or misindexed graph tends to do, and the output projection returning one
token far above the rest is consistent with that.

So there are now two definite bugs found and fixed by generating output (padding mask, and earlier the
residual corruption from the slot-3 gate writing into `h`), and the output itself is still wrong. That is
the expected state: no stage above blk.0 has ever been compared against a reference, and the message-slot
order is still an assumption with only suggestive evidence behind it.

**Next diagnostic, and it is cheap:** the LM head is the last stage and the easiest to check in isolation.
Dump the final hidden state, dequantise the matching rows of `output.weight` independently, and confirm the
logit for the top token reproduces. If it does, the head is exonerated and the defect is upstream in the
loop; if it does not, the head or the final norm is the problem and everything above it was noise.

## The LM head is correct, so the defect is upstream

`tools/head_check.py` dequantises output.weight rows with a hand-written IQ4_XS decoder (canonical layout,
kvalues_iq4nl from ggml-common.h) and dots them with the dumped final hidden state, undoing the softcap(30)
the engine applies. Five rows, including the top token:

```
   id     python dot   engine raw        diff
   87     +23.855974   +23.855975    -6.5e-07
  779      +3.559044    +3.559045    -7.1e-07
   11     +13.309932   +13.309932    -4.6e-07
 5195      +5.146991    +5.146991    -6.3e-08
 1234      +6.985292    +6.985292    -9.8e-08
```

All at float32 precision. So the final norm, the output projection, the IQ4_XS dequantiser, the row
indexing, the dot product and the softcap are all **correct**. Every wrong logit comes from the transformer
stack above blk.0, and the remaining suspect list is short: the depth mix, the message gating, slot order,
block states, and the loop structure.

This is the first stage above blk.0 to be compared against an independent implementation at all. It also
shows the method works: one comparison, and the graph is split in half with no guessing.

**Correction to my own first attempt at this check:** the initial run disagreed by 10-14 units on every row
and reordered the top tokens. That was not a head defect - my Python indexed `qs[0:16]` for every
sub-block instead of advancing 16 bytes per sub-block as `dequantize_row_iq4_xs` does. The engine was right
and the instrument was wrong, which is the third time in this port that the check, not the code, was at
fault.

## Per-layer dumps, and the shape they show

`IN2_DUMP_LAYERS=<dir>` writes the hidden state after every layer, so the stack can be bisected. The first
read of those dumps needs no reference at all - just whether the statistics follow the architecture:

```
layer  rms    max|x|   kind
  0   2.3879   9.7919  slid BOUND
  5   3.7198  18.3012  FULL
  6   0.5376   3.3903  slid BOUND    <- reset
 11   1.8474  17.8182  FULL
 12   0.6169   2.2679  slid BOUND    <- reset
 17   2.3897  23.4495  FULL
 18   1.5433   5.1558  slid BOUND    <- reset
 23   2.2524   8.0898  FULL
```

RMS grows smoothly within each 6-layer segment and drops hard at every boundary. That is exactly what the
boundary rule predicts: `is_block_boundary` means the attention output *becomes* the residual and the
previous state is banked, so the accumulated energy is replaced rather than carried. Nothing explodes,
nothing collapses, and the resets land on 0/6/12/18 - the boundaries the census says they should.

This is a structural sanity check, not a correctness proof. It rules out "a layer is scaling itself to
death", which is the cheapest failure to spot and the one most likely to be behind degenerate output.

## The boundary bank was in the wrong place

Reading the reference's layer forward settled a placement question my code had guessed at:

```python
block_residual = hidden_states
if self.attention_input is not None:
    hidden_states = self.attention_input(block_states, hidden_states)[0]   # mix WITHOUT the new entry
...
if self.is_block_boundary:
    block_states = torch.cat([block_states, block_residual.unsqueeze(-2)], dim=-2)   # bank HERE
else:
    residual = hidden_states = block_residual.float() + hidden_states
hidden_states = self.mlp_input(block_states, hidden_states)[0]             # mix WITH the new entry
```

The bank sits **between** the two mixes: the attention mix must not see the entry, the mlp mix must. Mine
banked in the model *after* the layer returned, so on boundary layers 6, 12 and 18 the mlp mix was one state
short - it was mixing over the previous segment's block states but not the one that layer had just banked.
Every boundary layer, every pass.

Fixed by banking before the layer call and passing `n_blocks - 1` to the attention mix when
`il % attn_res_block == 0`. Result: `top1 87 -> 14776`, span 10.9 -> 11.9. A placement bug, found by
reading the reference rather than by experiment, and it changed the output when fixed.

Also confirmed while reading: the mix really is attention (`scaling = hidden_size ** -0.5`) over
unweighted-RMS-normalised states with the values being those same states - which is what `in2_mix` does, so
the formula was right and only the input set was wrong.

## The depth mix verified with real block states

blk.0 is the only layer with no `attn_res_query` and no banked states, so the mix had never run with a
populated block list - even though it runs 24 times per pass. `tools/mix_check.py` checks blk.1's mix
against an independent implementation over the verified blk.0 output:

```
weights [0.5074748 0.49252525]
rms engine 0.702816   python 0.702816
max abs diff 1.907e-06   corr 0.9999999999999954
```

Two things this establishes. The banked block state is the **normed embedding** (IQ4_XS row +
`token_embd_norm`) - confirming what blk.0 banks. And the weights sit near 0.5/0.5, a soft blend; without
the `hidden_size**-0.5` factor the 2304-dim dot products would differ by orders of magnitude and the softmax
would collapse to a hard 1/0 pick, so the scaling is load-bearing, not cosmetic.

Also resolved by listing the actual tensor names: `blk.1.attn_res_query.weight` is `[2304]`, a single
vector, so using row 0 is correct - and blk.1 carries only the static `attn_k_conv`/`attn_v_conv`, with no
`*_proj`, matching "dynamic convs are full-layer only". My first probe missed these because I queried names
without the `.weight` suffix.

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
