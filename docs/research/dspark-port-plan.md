# DSpark drafter — port plan for wubuwizard

Source of truth: `strata/engine/src/inkling2/cuda/ink_dspark.cuh` (1,843 lines, MIT) and the pack manifest.
Weights: 54 tensors, 216 MB, already extracted to `C:/Users/eman5/models/inkling-2-strata/dspark-f32` as f32 by
`tools/strata_extract.py`.

## Architecture, quoted from their own header

    Taps [5, 23, 41, 48] mixed via softmax(fuse_weight[0..3]) + dflare_norm
        -> context K/V cache (SWA-1025, Global, SWA-1025) + f_rnn
    3 non-causal block-attention layers (24 Q, 8 KV, head_dim 128, NeoX half-split RoPE theta=1e6, SwiGLU 9216)
    Fast base logits over B=8 slots via 230 MB 4-bit screening plane (or BF16/Q8_ROW)
        + 4-pass radix Top-K_cand (K_cand = 256/512)
    Persistent 7-step markov_rank=256 RNN head scoring ONLY K_cand rows of unembed
    Supports BOTH Block-8 linear drafting AND Tree-16 2-branch beam drafting with exact target verification

So one drafter step emits **8 candidate tokens** (B=8). Verification then accepts the longest agreeing prefix against the
target's own greedy argmax, exactly as our `IN2_LOOKUP` loop already does.

## Why this is the whole speedup, and why it is safe

The measured facts from our engine make the size of the prize explicit: MoE visits cost 1.4 ms each at M = 1 and scale poorly
with threads, so eight tokens per forward instead of one is worth up to ~2.8x forward count with 100 % acceptance, and
upstream reports 1.7-3.1x. **Parity is unaffected by construction**: the drafter only proposes, and a proposal is accepted only
when the target's argmax at that position equals it. Our 15-token oracle must keep passing with `IN2_DSPARK=1` exactly as it
does with `IN2_LOOKUP=1`.

## Tensor inventory mapped to the three stages

    stage 1  context fuse      fuse_weight [4,4] F32, mask_embed [2304], norm [2304], dflare_norm per layer [2304]
    stage 2  3 x block attn    q_proj [3072,2304], q_proj_ctx [3072,2304], k/v_proj [1024,2304] + _ctx variants,
                               o_proj [2304,3072], mlp gate/up [9216,2304], down [2304,9216],
                               q_norm/k_norm/k_norm_ctx [128], input/post layernorm [2304]      (all S5_K or F32)
    stage 3  markov head       markov_w1 [201024,256] Q8_ROW (52 MB), rnn_proj [768,5120] Q8_ROW,
                               hidden_out [2304,256] Q8_ROW, hidden_out.bias [2304], rnn_proj.bias [768],
                               dflare_norm [2304]

Note the two `_ctx` projections: the drafter attends to a **context cache** built from the four tapped layers, separate from
its own block state. That is the cross-attention and it is the part with no analogue in our engine today.

## Integration points in wubuwizard

1. **Tap capture.** Our forward loops visits 0..71 over 48 physical blocks (`is_loop_layer` at `load.c:78`). The taps are
   physical layer indices 5, 23, 41, 48, so capture the post-layer residual for those four blocks into a `[4][d]` buffer per
   token position. Cost: four `memcpy`s of 2304 floats per token — nothing.
   **Unresolved and must be settled first:** the manifest's `target_layer_ids` ends at 48 while our GGUF reports
   `block_count = 48` (indices 0..47). Either the ids are 1-based or the distributions differ. Settle by checking whether
   index 48 exists in the Strata trunk tensor list (`blk.48.*` present or not) before writing any code.
2. **Drafter forward.** New file `src/wubu_inkling2_dspark.c` + header: 3 layers of 24-head/8-KV block attention over the 8
   slots, SwiGLU 9216, NeoX half-split RoPE theta 1e6 (we already have `apply_half_rope`-equivalent code in the full-attention
   path to crib from).
3. **Markov head.** Rank-256 RNN over 7 steps, scoring only the top-K_cand rows of the 201,024-row unembed. Without the 4-bit
   screening plane we can do the simple version: take the target's own top-K from the base logits we already compute.
4. **Draft/verify loop.** Already exists behind `IN2_LOOKUP`; add `IN2_DSPARK=1` to source drafts from the drafter instead of
   the suffix matcher. Rollback infrastructure already committed: row tags + `in2_kv_rewind`, `in2_kv_conv_save/restore` over
   both caches, and `in2_state_save/rewind` wrappers. Their `CpuSlot::sconv_snaps` keeps snapshots for steps 0..8 - a ring of
   eight, where we keep one. For B=8 verification we should match their ring.

## Staging, with a gate at each step

1. **Settle the layer-index discrepancy** (one check of the trunk tensor names). Blocks everything else.
2. **Port stage 1 + 2 into a standalone C tool** that reads the f32 tensors and emits 8 draft ids for a fixed input, and check
   them against the target's own continuations for a prompt we already have. No engine changes yet.
3. **Wire the taps and the draft/verify loop** behind `IN2_DSPARK=1`; gate on the 15-token oracle, which cannot move because
   the drafter only proposes.
4. **Measure acceptance and tok/s** on non-repetitive text - the number that has been missing all session.

Step 2 is the real work: roughly the size of the existing full-attention layer plus a small RNN. Everything before it is
settled.
