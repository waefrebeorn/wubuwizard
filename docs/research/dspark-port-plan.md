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

## Tap capture already exists; step 1 of the staging list is done without new code

`IN2_DUMP_LAYERS=<dir>` already dumps the post-layer hidden state for every layer and every pass
(`L%02d_p%d.bin`), and `IN2_DUMP_L=<il>` dumps a single layer. Verified against the oracle run:

    IN2_DUMP_LAYERS=$T/taps bash tools/oracle_gate.sh   ->  GATE PASS 15/15, 54 files
    file sizes: 27,648 B = 6,912 floats = 3 tokens x 2,304   (prefill pass)
                 9,216 B = 2,304 floats = 1 token  x 2,304   (decode pass)

So `d = 2304` is confirmed from the dumper as well as the manifest, and the four drafter taps can be produced today for the
visits that map to distinct (layer, pass) pairs.

**One real gap.** The dump keys on physical layer index plus pass counter, so the loop core's repeated visits 6..17 collide
inside a pass and overwrite each other. `target_layer_ids` wants visits 5, 23, 41 and 48, which in a 0..47 visit space are not
all distinguishable this way - visit 23 and visit 41 both land on core layers with the same pass counter. Closing it needs a
visit-indexed dump: a counter incremented by `RUN_LAYER` and a filename that carries it, roughly ten lines, and it does not
touch the maths at all.

That is now the next concrete step, and it is small. After it, the drafter's inputs are reproducible from a single gate run.

## Visit-indexed dump added, and it corrects my own account of why it was needed

`IN2_DUMP_VISITS=<dir>` writes `V%03d_il%02d_p%d.bin` for every visit. One insertion in `RUN_LAYER` (a counter and a
conditional write), no maths touched, oracle unmoved: **GATE PASS 15/15**.

What it proves, from the runtime rather than from tensor names:

    visit range observed: 000 .. 047        -> exactly 48 visits, 0..47
    V005 -> il05 p0      V023 -> il11 p1      V041 -> il17 p2
    no visit 048 exists

Three consequences, in increasing order of importance:

1. The 24-physical-layer / 48-visit accounting is now confirmed **behaviourally**, not inferred from `blk.N` names.
2. `target_layer_ids [5, 23, 41, 48]` - since 47 is the last visit and nothing beyond it runs, **48 must denote the final hidden
   state after the loop**. The earlier inference now has runtime support.
3. **Retraction: my stated reason for needing this dumper was wrong.** I wrote that visits 23 and 41 "both land on core layers
   with equal pass counters" and would collide. They do not - V023 is `il11 p1` and V041 is `il17 p2`, distinct keys, so
   `IN2_DUMP_LAYERS` could already have produced all four taps. The visit dumper is still worth having (it makes visit identity
   explicit and it is what proved 0..47), but the collision argument I gave for it was false, and I should have checked the
   actual mapping before asserting a gap. This is the same error pattern as the Q8_ROW 260-byte stride: a plausible structural
   argument asserted without a lookup.

Taps are now reproducible exactly, by visit index, from one gate run.

## Stage 1 ported and running on real taps (`tools/dspark_ref.py`)

`k_dspark_mix_taps` (:52-86) is simple and fully specified:

    w = softmax(fuse_w[row][0..3]);  v = sum_j w[j]*tap_j;  out = v * rsqrt(mean(v^2) + eps) * norm_w

Implemented as a numpy reference and run against the four real tap dumps:

    visit 05 <- V005_il05_p0.bin     visit 23 <- V023_il11_p1.bin
    visit 41 <- V041_il17_p2.bin     visit 47 <- V047_il23_p2.bin

    row  softmax(fuse_w[row])                            rms_pre  rms_post
      0  [0.093, 0.113, 0.688, 0.106]                     3.0843    2.9928
      1  [0.330, 0.222, 0.348, 0.100]                     3.3417    3.6074
      2  [0.180, 0.060, 0.287, 0.473]                     2.5961    3.5469
      3  [0.202, 0.015, 0.748, 0.035]                     3.4655    3.0477

    rms(norm_w) = 3.0923 - the post-norm scale lands in that neighbourhood as the algebra requires.

Two practical findings: the taps live at **mixed pass suffixes** (`V023` is `p1`, `V041` is `p2`), so any loader must
match on visit number only, not on `p0`; and each fuse row is a genuinely different mixture (row 3 puts 0.748 on the
third tap), which is the mechanism the header calls "taps mixed via softmax(fuse_weight[0..3])" - four fused contexts,
not one.

Stage 1 is done. Stage 2 (3 block-attention layers + context cross-attention) is next, and its kernels
(`k_dspark_block_attn` :127, `k_dspark_split_qkv` :218, `k_dspark_swiglu` :246) are the ones to read next.

### Correction: the `rms_pre` column in the first commit was a tautology

The first version printed `pre` as the RMS of `outs[r] / norm`, and since `outs = v/rms(v)*norm`, that quantity is
`v/rms(v)`, whose RMS is **exactly 1.0000 by construction for every row** - which is precisely what it printed. It looked
like a passing check and confirmed nothing. Fixed: `fuse_taps` now returns the true pre-normalisation RMS of the fused
vector, and the corrected numbers are

    row  rms_pre  rms_post
      0   3.0843    2.9928
      1   3.3417    3.6074
      2   2.5961    3.5469
      3   3.4655    3.0477

Same failure pattern as the degenerate `oracle exact len=0` comparisons earlier in this project: a number that cannot fail,
reported as if it had been checked.
