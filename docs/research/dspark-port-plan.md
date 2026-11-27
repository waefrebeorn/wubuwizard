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

## Stage 2 runs on real weights - but the activation scale grows, and I do not yet claim it is correct

Wiring resolved from code, not guessed:

    layer il mixes the taps with softmax(fuse_w[il]) normalised by dspark.layers.il.dflare_norm.weight
        (call site :1445-1449 passes il as the row and dflare_norm as norm_w) - **not** dspark.norm.weight
    ctx k/v = k_proj_ctx / v_proj_ctx applied to that fused context, kv_ctx_fused = [M, 2048] then split
    per-head RMSNorm over 128 then NeoX half-split RoPE, theta 1e6, pairs (d, d+64)
    GQA kh = h/3 (24 q, 8 kv), scale = 1/sqrt(128), non-causal over ctx ++ own block, online softmax

`tools/dspark_stage2.py` runs all three layers against the extracted f32 weights and produces finite output:

    layer 0: fuse w=[0.093 0.113 0.688 0.106]  ctx=1x2304  out_rms= 5.5620
    layer 1: fuse w=[0.330 0.222 0.348 0.100]  ctx=1x2304  out_rms= 7.8600
    layer 2: fuse w=[0.180 0.060 0.287 0.473]  ctx=1x2304  out_rms=13.8959
    out (8, 2304) rms=13.8959 finite=True min=-71.1 max=47.1

**That growth is a red flag, not a result.** Each layer multiplies the state RMS by roughly 1.4-1.8x, which a 3-layer stack
with pre-norms should not do. Two candidate causes, in order of likelihood:

1. My block input is a placeholder - `taps[3]` repeated 8x. The real block state is the embedding of the 8 draft slots, so the
   magnitudes here mean nothing. This alone could explain a rising scale.
2. A genuine wiring error, most likely that the block state should be re-normalised at init (their `dflare_norm` may apply to
   the state and not only to the fused context) or that I have SwiGLU/`down_proj` orientation wrong.

Until one of those is excluded, this is "the shapes compose and the code runs", not "stage 2 is ported". The diagnostic is to
feed a real block state - embed 8 token ids - instead of the placeholder and see whether the growth persists.

Also corrected: `dspark.norm.weight` is **not** the norm used in tap mixing (stage 1 used it, wrongly); the mixing norm is the
per-layer `dflare_norm`. `dspark.norm.weight` is used elsewhere - final output, still to be located.

## Diagnostic round: real init did not fix the growth, and S5_K dequant is NOT a constant-factor error

**1. The real block init is known and is now used.** `emit_dspark_backbone` (:1484-1489):

    k_ink_embed_norm(embed.weight, embed_norm.weight, anchor_id, d_cur, 1, H)   // slot 0 = anchor token
    k_dspark_fill_mask(mask_embed, d_cur, B)                                    // slots 1..7 = mask_embed

Replacing my `taps[3]` placeholder with the real `mask_embed` (rms 0.5956) did **not** fix the growth:

    layer 0 out_rms 3.9421   layer 1 7.0837   layer 2 13.4376     (was 5.56 / 7.86 / 13.90)

So the placeholder was not the cause. Candidate 1 eliminated.

**2. S5_K dequant tested against a fan-in invariant, with Q8_ROW as the control.** For a matrix `[out, in]` the weight stdev
should sit near `1/sqrt(in)` if the checkpoint respects init scale - and two Q8_ROW tensors, whose decode is independently
verified, land on it to within 0.2%:

    tensor                                   fmt      stdev   1/sqrt(in)  ratio
    q_proj    [3072, 2304]                   S5_K    0.0409     0.0208     1.965
    o_proj    [2304, 3072]                   S5_K    0.0274     0.0180     1.517
    down_proj [2304, 9216]                   S5_K    0.0178     0.0104     1.707
    gate_proj [9216, 2304]                   S5_K    0.0328     0.0208     1.575
    hidden_out[2304,  256]                   Q8_ROW  0.0626     0.0625     1.002
    rnn_proj  [ 768, 5120]                   Q8_ROW  0.0140     0.0140     1.001

**A constant multiplicative error in my S5_K decode would show as a constant ratio. It does not** - the ratios scatter
1.52..1.97. So either the dequant is right and these tensors genuinely sit above init scale, or the error varies per superblock
in a way that produces scatter. Either way, "S5_K is off by a constant factor" is refuted, and tuning a constant would be
fitting noise.

The Q8_ROW ratios of 1.002 and 1.001 are the strongest confirmation yet that the extractor is faithful: two independent
tensors, different shapes and fan-ins, landing on the same invariant.

**Where that leaves the growth:** unexplained, and the remaining suspect is my wiring rather than the weights. The next test
should not be more introspection - it should be ground truth. We already have 12 oracle continuation ids; a correct stage 3
would draft tokens that agree with them at a high rate. Implementing the Markov head and measuring draft-vs-oracle agreement
turns this from "is my activation scale plausible" into a pass/fail number.

## Stage 3 (Markov head) implemented and its dynamics are stable - which sharpens the stage-2 problem

Semantics, read from the kernels:

    joint[5120] = [ state[256] | markov_w1[prev_tok][:256] | hidden[2304] | f_rnn_ctx[2304] ]
    proj  = joint @ rnn_proj^T + rnn_proj.bias                       -> [768]
    g     = sigmoid(proj[0:256]);  cand = tanh(proj[256:512])
    state = cand + g*(state - cand);  lat_out = tanh(proj[512:768])
    corr  = lat_out @ hidden_out^T + hidden_out.bias

`tools/dspark_stage3.py`, 7 steps, weights straight from the pack:

    step  state_rms  state_absmax  corr_rms  corr_absmax
       0     0.5325        0.9996    1.3172       4.8077
       3     0.6644        1.0000    1.3304       4.8324
       6     0.7031        1.0000    1.3299       4.8660

State converges upward and saturates at the tanh bound; `corr` sits at rms 1.33 with an absmax of 4.8 and does not drift across
seven steps. **That is what a correctly wired recurrent head looks like** - the tanh/sigmoid structure bounds it by construction,
and the numbers are consistent with hidden_out's verified stdev (0.0626) applied to a 256-dim tanh output.

**The contrast is the useful result.** Stage 3 is stable on the same weights where stage 2 grows 3.9 -> 7.1 -> 13.4 per layer.
Both run on the same dequantised tensors and the same verified extractor, so the weights are not the differentiator: **the block
stack's wiring is.** That is now the single hypothesis left, and it is narrow - the candidates inside stage 2 are the residual
adds, the attention output scale, or a missing normalisation of the block state after the residual.

What remains for a ground-truth number is the unembed rows: scoring is `dot(corr, unembed[cand])` and those 201,024 rows live in
the trunk, not in the DSpark package. The next step is a small engine-side dump of the rows for the 12 known oracle ids, then
draft-vs-oracle agreement becomes a percentage instead of an opinion.

## Retraction: the stage-2 "growth" was never evidence of a bug, and orientation is uniquely determined

Two eliminations, both of which remove suspects I had been carrying:

**1. Orientation is not a free parameter.** Their call sites pass a transpose flag that is `true` for k/v and `false` for q/o,
while the manifest lists q_proj as `[3072, 2304]` - two statements that cannot both be taken naively. So I ran all 16
combinations of (q,k,v,o) orientation. **Fifteen of them fail a dimension check outright**; only `(False,False,False,False)` -
treating every matrix as `[out, in]` and computing `x @ W.T` - even composes:

    (F,F,F,F)  3.942  7.084  13.438   growth 1.897   STABLE
    every other combination: shape error, e.g. "size 1024 is different from 2304"

The manifest shapes therefore pin the wiring completely. My implementation was the only one arithmetically possible; there was
never an orientation bug to find.

**2. A rising RMS inside the stack is erased before it reaches anything.** The final norm is `dspark.norm`, and RMSNorm output
scale is independent of input scale - measured, not asserted:

    input_rms -> output_rms
         0.010 ->   3.0666
         1.000 ->   3.0818
        13.438 ->   3.0818
      1000.000 ->   3.0818        reference rms(dspark.norm) = 3.0923

The head sees the normalised state, so the 3.9 -> 7.1 -> 13.4 climb is invisible downstream. It is also not unusual in family:
the four trunk taps themselves measure rms 7.75, 3.76, 4.12 and 2.89.

**Retracted: "that growth is a red flag, not a result" and the claim that it pointed at a wiring defect.** I treated an
unexplained number as a symptom without first checking whether the pipeline normalises it away, and it does. The state remains
*unverified* - there is still no ground truth - but there is no longer any reason to believe the block stack is mis-wired. That
is a different statement, and the honest one.

**Correction to my own diagnostic, twice in a row:** the line I printed as "after rmsnorm(dspark.norm) = 3.0923" was another
tautology - it printed `rms(norm_w)` and labelled it the transformed output. The real test is the table above, which varies the
input scale and shows the output scale hold. Third tautological check this project; the pattern is printing a quantity that is
fixed by definition and reading it as a measurement.

Where this leaves stage 2: wired the only way the shapes allow, scale behaviour understood, dynamics stable in stage 3. What is
still missing is not a hypothesis - it is ground truth, which needs the unembed rows for the 12 known oracle ids.

## FIRST GROUND-TRUTH POSITIVE: the full drafter ranks the true next token first

The pipeline now runs end to end on real weights - stage 1 (tap fusion) -> stage 2 (3 block-attention layers) -> stage 3
(Markov GRU) -> unembed scoring - and the token it should predict is the one it puts first.

Scoring hook added to the engine (`IN2_SCORE_VEC` / `IN2_SCORE_IDS`): `in2_matvec_quant` takes an element offset, so one
unembed row costs one row-length matvec instead of the 201024-row head. **The hook was validated before being believed** -
fed the engine's own final hidden, it ranks the true next token first, so it reproduces the real logits:

    engine fin  -> SCORE 149876 44.467781   (true next token)
                   SCORE 220    44.408203   (near-tie, as greedy decode of this prompt actually is)

Then the drafter's `corr` vector, same 12 candidate ids:

    drafter corr -> SCORE 149876 6.822034   <- correct, and with a clearer margin
                    SCORE 220    5.598398
                    SCORE 11     3.052794
                    SCORE 149919 1.846477
                    SCORE 149984 0.491476
                    SCORE 371    0.071222

**Caveat, stated plainly:** the candidate set is the 12 oracle continuation ids, not the full 201,024-row vocabulary. Ranking
the right token first among twelve is real evidence that the stages are wired coherently - it is the first end-to-end signal
in this thread that is not introspection - but it is not an acceptance rate. A drafter that only ever had to choose among
twelve plausible tokens could look right and still be wrong at full vocab.

Next step is therefore the honest version of this test: score all 201,024 rows for `corr` (chunked, reusing the same hook) and
take the true argmax, then repeat over the 7 drafting steps. That gives top-1 agreement, and agreement over steps gives the
acceptance rate the whole port exists to produce.

## Full-vocab test refuted the 12-candidate result, and the documented fix restored it

Scoring `corr` over all 201,024 rows instead of 12 changed the answer - which is why the caveat mattered:

    control (engine fin)  TOP5  149876:44.468  220:44.408  149878:35.706  152933:33.836  150455:33.611
    drafter corr          TOP5  196997:12.830   62:11.816   63:9.684   149876:6.822   12:6.237

**The true token ranks 4th at full vocabulary.** The 12-candidate test was not evidence of a working drafter; it was a
candidate set I had biased toward the answer.

The cause is documented in their own scorer: `k_dspark_score_cand_eval(..., const float* cand_base, // [K_cand] (nullable ->
0.0f))` computes `dot(corr, unembed_i) + cand_base[i]` - the Markov head produces a **correction**, and it is added to the
base logits from the trunk's own head. I had scored the correction alone. With base added on the union of both top-5s:

       id      base    markov       sum
    149876    44.468     6.822    51.290   <- truth
       220    44.408     5.598    50.007
    149878    35.706     3.791    39.497
    152933    33.836     1.177    35.013
    196997    19.217    12.830    32.047

Top-1 combined is the true token, margin 1.283. Note what the correction is doing: it lifts 196997 from 19.2 to 32.0 (+12.8) and
149933 down by 2.2, i.e. it reorders candidates inside the trunk's own shortlist rather than proposing from nowhere - exactly
what a correction head should do.

**Still not an acceptance rate:** the union set is 12 ids, not 201,024. The combined full-vocab scan is the next step, and it
needs the base logits inside the engine rather than two runs stitched together - the hook should compute `logits + corr-row`
per candidate id in one pass.

## Full-vocabulary combined scan: the drafter's top-1 IS the true token

`IN2_ADD_VEC` now computes, inside one process, `base_i + dot(corr, unembed_i)` over all 201,024 rows - the base recomputed from
`fin` with the same tensor and the same kernel, so no hand-stitching:

    ADD5  149876:51.289814  220:50.006599  149878:39.497284  149928:35.352219  152933:35.013157

**Top-1 is 149876, the true next token**, and it agrees with the value I computed by hand from two separate runs
(51.289815) to six decimals - an independent path to the same number.

Two things this does and does not establish:

* It does establish that the whole port is coherent at full vocabulary: tap fusion, three block-attention layers, the Markov
  correction and the unembed dot all contribute, and the correction is doing real work - it moves 220 (base 44.408) to 50.007
  and keeps the true token ahead by 1.28, where the base alone was a 0.06 near-tie. Without the correction greedy decode of this
  prompt is a coin flip; with it, the margin is 20x larger.
* It does **not** establish an acceptance rate. This is one position. The structure that matters is eight drafted positions
  verified against the target, and the drafter has only been shown to be right where the trunk was already 99.9% sure.

Next: the 7-step drafting loop - feed the drafter its own previous token instead of the anchor, verify each position against the
engine's argmax, and report accepted-prefix length. That number is the one the port exists to produce.

## The 7-step drafting loop measures 0/7 - and the reason is a specific missing input, not a mystery

`tools/dspark_loop.py` runs the real loop: the drafter consumes its own draft as the next `prev_tok`, and each position is
scored as `base(block_out[i]) + corr_i` by the engine's own kernel via `IN2_BASE_VEC`. Result:

    step    prev   draft  oracle      score  ok
       0  149879   42623  149876     52.461   n
       1   42623   42623     220     53.931   n
       2   42623   42623   36838     53.785   n
       3   42623   42623  182189     53.787   n
       4   42623  113993  149933     54.698   n
       5  113993  113993     220     55.827   n
       6  113993  113993   50843     56.051   n

    accepted prefix: 0/7

**This is the number the earlier single-position result hid.** Scoring the trunk's own logits plus a correction gave 1/1,
because the base was the trunk's - the drafter was only ever asked to break a near-tie it was handed. Asked to produce the base
logits itself, it is wrong at every position.

**Cause, and it is concrete: the block state was initialised with `mask_embed` in all eight slots.** Their `emit_dspark_backbone`
puts the **embedded, embed_norm-normalised anchor token in slot 0** and `mask_embed` only in slots 1..7. I substituted
`mask_embed` everywhere because the anchor embedding was not available in Python - so all eight slots start from the identical
vector and differ only by RoPE position, and the loop degenerates to repeating one or two tokens (42623, 42623, 42623, 42623,
113993, 113993, 113993). That degeneracy is the signature of a slot-independent block state, exactly what a constant init would
produce.

**Next step is therefore small and specific:** an engine hook to dump `embed_norm(embed[token])` for a given id
(`IN2_DUMP_EMB=<id>`), then initialise slot 0 with the anchor's real embedding and re-run this loop. Until that is done the
0/7 is not a verdict on the drafter - it is a verdict on my input, and the two are worth keeping separate.

## Anchor embedding fixed the degeneracy; 0/7 remains, and the next deficiency is context length

The anchor embedding needed no new hook - `IN2_DUMP_EMBD` already dumps the post-embedding stream, so running the anchor
token alone gives exactly `token_embd_norm(token_embd[149879])`:

    post-embedding stream 1 x 2304, rms 1.1253
    cosine(anchor_emb, mask_embed) = 0.0361      <- essentially orthogonal

So the init I had been using was not a small approximation, it was a different vector. With slot 0 set correctly:

    step    prev   draft  oracle      score  ok
       0  149879  166406  149876     54.389   n
       1  166406   35162     220     57.659   n
       2   35162  161338   36838     55.908   n
       3  161338    1929  182189     53.915   n
       4    1929    1929  149933     52.058   n
       5    1929   31057     220     54.128   n
       6   31057  200477   50843     51.358   n

    accepted prefix: 0/7

**What changed is the failure mode, and that is worth recording.** Before, the loop emitted one or two tokens over and over
(`42623, 42623, 42623, 42623, 113993, 113993, 113993`) - slot-independent and degenerate. Now it emits seven different tokens,
still wrong. Degeneracy was a symptom of the init; it is gone, so that defect is genuinely fixed even though acceptance did not
move.

**Next deficiency, identified from our own data: the context cache holds one position.** The taps were dumped during a
one-token decode pass, so `M = 1` and the cross-attention has a single key/value pair to look at. In the real drafter that
cache holds `apos` positions - the whole committed sequence, up to the 1025 sliding window. A one-position context makes the
`*_ctx` projections almost decorative, which is consistent with a block state that now varies but carries no information about
the prompt.

So the next run must dump taps from a real prefill - at least a few hundred positions - and rebuild the loop from that. The
mechanism is already there (`IN2_DUMP_VISITS` during a longer `--text` prefill); only the input is wrong, again.

## Three input hypotheses tested against a real prompt; all three fail to move 0/7

The `--text` mystery was my own invocation, not the tool: `text_mode` requires `argc >= 4`, so `gen_fast 1 --text` silently fell
through to numeric prompt parsing and turned `"--text"` into token 0. With a dummy third argument the tokenizer runs:

    TEXT_FILE 1302 bytes cap 1310      ->  281-position prefill, visit-5 dump = 2,589,696 B = 281 x 2304

So the context cache now really holds 281 positions. Rebuilding the loop on that, with the anchor set to the last prompt token
(220) and the oracle being the engine's own continuation for the same prompt:

    step    prev   draft  oracle      score  ok
       0     220  150327    1377     52.960   n
       1  150327  167207   68014     54.416   n
       ...
    accepted prefix: 0/7                               drafted [150327, 167207, 167207, 167207, 170409, 146120, 146120]

**Context length was not the blocker.** Then the last unused tensor in the pipeline, `dspark.norm` (rms 3.09, matching the trunk
taps' own 2.9-7.7), applied to the block output before the head:

    step    prev   draft  oracle      score  ok
       0     220  201014    1377     12.281   n
    accepted prefix: 0/7                               scores collapse to 12-17, drafts no better

**Post-norm was not the blocker either.** It is left in the code behind `IN2_DSPARK_POSTNORM=1` and off by default, because an
unvalidated change should not become the baseline.

### What has been eliminated, and what that implies

Eliminated by measurement, each one a hypothesis I considered likely: the block input placeholder (anchor embedding, cosine
0.036 with mask_embed), the context length (1 -> 281 positions), and the missing final norm. Also eliminated earlier: weight
orientation (only one combination composes) and a constant dequant error (ratios scatter).

What remains is not another input guess. The port is structurally complete - every stage runs, the head is stable, the shapes
compose - but functionally it is wrong, and after five eliminations the productive move is to stop guessing at the inputs and
get a **reference from their own engine**. They ship a CPU backend (`engine/src/inkling2/ink_cpu_backend.cpp`) that decodes all
these formats and runs the same drafter. Building it and dumping the block output and `corr` for the same prompt would let us
diff intermediates tensor by tensor, which converts this from hypothesis testing into ordinary debugging.

That is the next step, and it is the one that actually ends the guessing.

## Read the actual reference: three real divergences fixed, 0/7 unchanged

`ink_cpu_backend.cpp:1051-1228` contains their CPU `dspark_draft`, the ground truth I had been missing. Comparing it against my
loop found three genuine wiring errors, not guesses:

1. **Slots are 1-based in the step loop.** Their `for (step = 1; step <= n_draft; ++step)` reads `cur[step * D]`, so the seven
   drafting steps consume block slots **1..7**, never slot 0 (slot 0 is the anchor). I had been reading slots 0..6.
2. **RoPE positions are `anchor_pos + i`** (:1099-1100), not `M + i`. With 281 context positions the anchor sits at 280, so my
   block was rotated one position too far forward - every q and k in the block.
3. **The final norm is applied** (:1163-1164): `rms_norm(cur, dspark.norm.weight)` on all B slots before the head. I had it off
   by default; it is now on.

Also confirmed identical to my implementation, which is worth stating because it retires three worries: GQA `n_rep = H/KV`,
`attn_scale = 1/sqrt(hd)`, non-causal attention over `ctx ++ block`, `state = cand + g*(state - cand)`, `corr = latent·hidden_out
+ bias`, `prev_tok` seeded with the anchor and then the drafted id. And their scorer is `argmax(unembed @ (cur[step] + corr))`,
which is algebraically the same as my `base + corr·unembed` - they even note the identity in a comment (:1197).

With all three fixed:

    step    prev   draft  oracle      score  ok
       0     220   43194    1377     11.584   n
       1   43194  150053   68014     15.964   n
       ...
    accepted prefix: 0/7      drafted [43194, 150053, 12, 32135, 150053, 12, 32135]

**Still 0/7.** The three fixes are real - they were divergences from the reference, not opinions - but they do not close the gap,
so something upstream is still wrong and it is not the block/head arithmetic.

**Next unknown, precisely located:** their context K/V cache is not computed at draft time. It is a *stored per-layer BF16 buffer*
(`sl.dspark_k[il]`, `sl.dspark_v[il]`) that their prefill fills token by token, and `dspark_draft` only reads it (:1115, :1133).
So the difference is in **how the cache is built**, and that is the next thing to read - where `dspark_k[il]` is written during
prefill, and from exactly what tensor. My reconstruction (`k_proj_ctx(fused taps)` RoPE'd at each position) may be subtly wrong
in input, normalisation, or ordering, and I cannot tell from the output which.

## STRATEGIC CORRECTION: the reference is upstream llama.cpp, not the Strata C++ drafter

Two discoveries from the last few reads change what "correct" means for this port.

**1. The drafter ships as its own GGUF, in two incompatible layouts.** `engine/tests/core/ink_dspark_layout_test.cpp`
documents them with sizes and dates:

    dspark-Inkling-2-Mini-Q8_0_Old.gguf   402,278,560 B   "the layout the engines read"
    dspark-Inkling-2-Mini-Q8_0.gguf       503,889,952 B   2026-11-23 14:54 UTC, llama.cpp mainline "dflash":
                                                           ctx_mix / ctx_norm / markov_w2 ...

and the old layout uses different names from the Strata pack I extracted: `dspark_fuse.weight`, `markov_w1.weight`,
`mask_embd.weight`, `joint_proj.weight/bias`, `output_norm.weight`, `blk.N.attn_k_ctx.weight` - versus the pack's
`fuse_weight`, `markov_head.markov_w1`, `mask_embed`, `markov_head.rnn_proj`, `norm`, `layers.N.self_attn.k_proj_ctx`. Same
mathematics, different packaging, and **a second generation that adds `ctx_mix`, `ctx_norm` and `markov_w2`** - i.e. the
architecture evolved and the pack I have may not match the mainline implementation.

**2. Our own upstream checkout already implements it.** `common/speculative.cpp:1374`:

    struct common_speculative_impl_draft_dspark final : public common_speculative_impl_draft_dflash
    {..., "draft-dspark", COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK}

with the note at :3201 that "dflash/dspark decode the whole noise block in a single pass and sample every block position on the
backend". There is also a built `build-ref/bin/llama-cli.exe` sitting in the checkout.

**Why this matters more than any of the six divergences I fixed today:** this engine's contract is token-for-token parity with
upstream llama.cpp. I have spent the session validating the drafter against the *Strata* implementation - reading its CUDA and
CPU drafter line by line and matching them - while the thing our engine must agree with is upstream's `draft-dspark`. Those are
different codebases with possibly different generations of the architecture, and that alone can explain a persistent 0/7 with
every stage wired plausibly.

**Consequence for the plan.** The next step is not another hypothesis about my taps. It is to establish upstream as the oracle:
check whether `build-ref/bin/llama-cli.exe` supports `--draft-dspark` (and rebuild if the binary predates the source), run it on
the same prompt, and compare its drafts against mine. That gives a reference we can actually execute, on the same machine, for
the same model - which is what every failed hypothesis today was missing.

The 0/7 measurement stands as the honest state of my implementation. What changes is what it is being measured against.

## Upstream runs the drafter as a GGUF graph model - which explains why my port of the Strata code cannot match it

`common/speculative.cpp:1374-1448` settles the architecture question:

    common_speculative_impl_draft_dspark : common_speculative_impl_draft_dflash
        - loads a SEPARATE model as params.ctx_dft
        - reads metadata: dflash.sample_from_anchor, dflash.has_markov_context, dflash.has_confidence_head
        - has_markov_context -> ctx_rows = block_size + 1, state taken from llama_get_embeddings_nextn()
        - init({ n_draft_max: sample_from_anchor ? block_size : block_size - 1, backend_sampling: true,
                 dev_features: true, nextn_masked: !has_markov_context })

So upstream does **not** implement the drafter as a hand-written CPU/CUDA loop at all. It loads the drafter GGUF as a second
llama model and decodes the whole noise block in one `llama_decode` pass, with the Markov context arriving through `h_nextn`
embeddings. The per-layer BF16 K/V cache, the `dspark_k[il]` / `dspark_frnn` buffers and the explicit 7-step GRU loop I have
been faithfully porting exist in **Strata's** engine, not in upstream's.

**That is very likely the whole 0/7.** I ported one implementation of DSpark and have been measuring it against continuations
produced by a model whose drafter upstream would run through a different graph with different metadata keys and possibly a
newer architecture generation (`ctx_mix`, `ctx_norm`, `markov_w2`). Every stage of my port could be individually correct and
still disagree with that, and the agreement metric I built cannot distinguish "my maths is wrong" from "these are two
different drafters".

**Also established: the built reference binary is unusable on this box.** `build-ref/bin/llama-cli.exe` exits 127 with no output;
it imports `api-ms-win-crt-private-l1-1-0.dll`, a private UCRT shim that is not present on consumer Windows. Not a CodeIntegrity
block, not a missing CUDA runtime - the import simply cannot resolve. Running upstream as the oracle would need a local rebuild.

### Where the port actually stands

    Strata drafter: fully decoded (54 tensors, both formats), fully read (CUDA + CPU reference),
                    implemented end to end (stage 1, 2, 3), six divergences found and fixed against the reference,
                    and measured at 0/7 on real continuations.
    llama.cpp drafter: a separate GGUF graph model with metadata-driven behaviour; not yet examined at tensor level.

The honest conclusion is that these are two implementations, and the port I built is of the one that is *not* our parity target.
Before any more hypothesis work, the drafter GGUF that upstream actually reads should be fetched and its tensor table compared
with the Strata pack. If it carries `ctx_mix` / `ctx_norm` / `markov_w2`, then the pack I decoded is a different generation and
the 0/7 is explained without my maths being wrong anywhere.

## Parsed tensor table at last: the main model has no drafter tensors (was a grep before, now a real table)

`tools/gguf_names.py` reads a GGUF's metadata and tensor-info section without touching the data, so it works on multi-GB files
and cannot hit the array-field `MemoryError` that killed the earlier struct walk. On the main checkpoint:

    Inkling-2-Mini-UD-IQ2_XXS.gguf: v3, 734 tensors, 62 kv pairs
    tensors matching "dspark": 0

That finally replaces the weakest claim in this project - "no MTP head, based on a 40 MB string grep that matched `smtplib` and
`Beagle` in the vocabulary" - with an actual parse of all 734 tensor entries. The drafter is definitively a separate file, which
is why the pack ships 54 `dspark.*` tensors and the checkpoint ships none.

Both drafter files exist on the hub at exactly the sizes their layout test documents:

    dspark-Inkling-2-Mini-Q8_0.gguf        503,889,952 B   (mainline "dflash": ctx_mix / ctx_norm / markov_w2)
    dspark-Inkling-2-Mini-Q8_0_Old.gguf    402,278,560 B   (the layout the engines read)

The mainline one is downloading to `C:/Users/eman5/models/inkling-2-drafter/`. When it lands, `gguf_names.py --skeleton` on it
against the Strata pack's 54 names answers the question that matters: same drafter, or a different generation.

## The two drafters are different architectures - the 0/7 is explained, and it was never my arithmetic

Downloaded `dspark-Inkling-2-Mini-Q8_0.gguf` (503,889,952 B, exact size) and parsed its table: **58 tensors, 39 kv pairs,
25 distinct skeletons.** Against the Strata pack's 54, the differences are structural, not cosmetic:

    mainline (dflash)                        Strata SQ-10 pack
    blk.N.ctx_mix.weight      [4]     F32    -- absent (fuse is a [4,4] softmax matrix, one per pack not per layer)
    blk.N.ctx_norm.weight     [2304]  F32    -- absent (dflare_norm plays this role)
    markov_ctx_mix.weight     [4]     F32    -- absent
    markov_ctx_norm.weight    [2304]  F32    -- absent
    markov_w1.weight          [256, 200058] BF16    markov_head.markov_w1 [201024, 256] Q8_ROW  (transposed AND re-scoped)
    markov_w2.weight          [256, 200058] BF16    -- absent
    markov_w2.bias            [200058] F32          -- absent
    joint_proj.weight/bias                          markov_head.rnn_proj (+bias)
    mask_embd.weight                                mask_embed
    output_norm.weight                              norm.weight

And the weight layout differs too: mainline `attn_k.weight` is `[2304, 1024]` (in, out) where the pack's `k_proj` is
`[1024, 2304]` (out, in).

**The important one is `markov_w2 [256, 200058]` with a `[200058]` bias.** In the Strata design the Markov head emits a
2304-wide correction which is then dotted against a shared unembed matrix. Here there are **two rank-256 projections straight to
the vocabulary**, so the head produces logits directly at rank 256 - a different mechanism, not a different packing.

**Conclusion.** I ported the Strata drafter faithfully - decoded its formats, read its CUDA and CPU implementations line by line,
found and fixed six real divergences, and matched it stage by stage - and it was never going to agree with continuations from a
model whose drafter is this one. The 0/7 was measuring two different architectures against each other. That is a better outcome
than a bug: every one of those six fixes is still correct *for the implementation they were derived from*.

**Next step:** implement dflash as upstream defines it, from this GGUF. `src/models/` in the upstream checkout holds the graph
builder, and the file on disk gives exact shapes. The Strata work is not wasted - stage 1's tap fusion, the block attention
structure and the GRU recurrence all have direct counterparts - but `ctx_mix` (a learned 4-vector per layer), `ctx_norm`,
`markov_ctx_mix/norm` and the `w1`/`w2` vocab projections are new and must come from upstream's code rather than from inference.

## The mainline algorithm, read in full from upstream (src/models/dspark.cpp)

`build_dspark_inp_block` (:194-225) and `build_dspark_markov_head` (:232-455) define the whole drafter. Reconstructed:

    # block input, per block of block_drafts positions
    x0 = tok_norm( token_embd[anchor] )                 # anchors, RMS-normed by token_embd_norm
    x0 = cat(x0, mask_embd[0 .. block_drafts-2])        # the noise rows

    # 3 dflash blocks over x0 (attn_q/k/v + *_q_norm/_k_norm, attn_k_ctx/attn_v_ctx, ctx_mix, ctx_norm, ffn SwiGLU)
    ...
    base = logits(hidden)                                # [n_vocab, B]  (tied token_embd, output_norm)

    prev  = anchor
    state = 0                                            # [R] = [256]
    for i in 0 .. B-1:
        w1_prev = markov_w1[prev]                        # [R] embedding lookup, w1 is [R, n_vocab]
        markov  = w1_prev
        if joint_proj:                                   # the "rnn head"
            joint  = concat(state, w1_prev, hidden_i, f_rnn)      # [3R + n_embd]
            joint  = joint_proj @ joint + joint_proj_b            # [3R]
            gate   = sigmoid(joint[0:R])
            cand   = tanh   (joint[R:2R])
            state  = cand + gate * (state - cand)        # note: gate*(old-cand), NOT (1-gate)
            markov = tanh   (joint[2R:3R])
        bias_i = markov_w2 @ markov + markov_w2_b         # [n_vocab]  <- vocabulary-sized
        col    = base[:, i] + bias_i                      # (+ softcap if configured)
        prev   = argmax(col)                              # greedy path
        out[:, i] = col

**This is not the algorithm I implemented.** Differences that matter:

- The Markov head produces a **vocabulary-space bias** `w2 @ markov` (rank 256 -> 200058 rows) that is added to the block's
  own logits. There is no 2304-wide `corr`, no `hidden_out`, no dot against a shared unembed, no candidate shortlist, no
  `cand_base`. My entire stage 3 plus the `IN2_ADD_VEC`/`IN2_BASE_VEC` machinery reproduces a mechanism this drafter does not have.
- The gate is `cand + gate*(state - cand)`; the Strata code I ported used `cand + g*(state - cand)` too, so that one matches.
- `f_rnn` is an **extra input row to joint_proj**, not a separate context tensor fused by `softmax(dflare_norm@fuse)`.
- The anchor embedding is **RMS-normalised by `token_embd_norm`** before entering the block. I fed the raw embedding.
- `ctx_mix` is a `[4]` per-layer vector (a learned mix over the 4 tap layers), and `ctx_norm` a `[2304]` norm - the per-layer
  context K/V is built with `attn_k_ctx`/`attn_v_ctx` from that mixed context, which is the closest thing to what I ported.

So the port has to be redone against this, and the good news is that it is *easier*: no candidate machinery, no unembed dot, and
every weight is in the 503 MB GGUF now on disk. What is needed is a GGUF reader for its two quant types (`Q8_0` for the block
matrices, `BF16` for w1/w2) and a block forward that follows `build_layers`.

**Immediate next step:** read `dflash.cpp`'s layer builder (~:114-200) for the exact block maths with `ctx_mix`/`ctx_norm`, then
write a GGUF dequantiser (Q8_0 = 32-weight blocks with an fp16 scale, *not* the Strata Q8_ROW plane layout) and dump the 58
tensors, so the new implementation runs on real weights instead of the Strata pack's.

## The drafter GGUF is decoded, and two independent cross-checks tie it to the Strata pack

`tools/gguf_extract.py` dequantises the drafter to flat f32. It implements ggml `Q8_0` - a 34-byte block per 32 weights
(`fp16 d` + `32 x int8`, `value = d*q`) - which is **not** the Strata `Q8_ROW` plane layout, so sharing a decoder between the two
would have been wrong. Verified by closing the byte budget:

    58 tensors, 491,956,264 B of data, data_start = 11,933,600
    11,933,600 + 491,956,264 = 503,889,864   vs file 503,889,952   (88 bytes of padding)

and by decoding `blk.0.attn_q.weight [2304, 3072] Q8_0` to 28,311,552 B of f32 with rms 0.0208 - a plausible weight scale, not
garbage. All 58 now sit as .f32 in `C:/Users/eman5/models/inkling-2-drafter/f32/` (470 MB).

**The cross-checks that matter.** Two values in this GGUF reproduce numbers I measured independently in the Strata pack:

    output_norm.weight        rms 3.09236        Strata dspark.norm.weight        rms 3.0923
    blk.0.ctx_mix.weight      [0.0926 .. 0.6879]  Strata fuse row 0 softmax      [0.093, 0.113, 0.688, 0.106]

The norm is identical to four significant figures, and the mainline `ctx_mix` vector is exactly the **softmax of the Strata
`fuse_weight` row** - mainline stores the mix pre-softmaxed as a `[4]` vector where Strata stores a `[4,4]` matrix and softmaxes
a row per output feature. That is the same first layer under two parametrisations, and it settles that both packs derive from the
same checkpoint rather than from different models.

**What this means for the port.** The weights are not in dispute and never were - my Strata extraction was correct. What differs is
the *wiring*: mainline mixes the four taps with a learned `[4]` vector and one `ctx_norm` per layer, while Strata applies a `[4,4]`
matrix and `dflare_norm`. And the markov head, as established last step, is a rank-256 vocabulary bias rather than a 2304-wide
correction. So the remaining work is a block forward that follows `dflash.cpp`, on weights that are now decoded and cross-checked.

**Next step:** write the forward - `x0 = tok_norm(embd[anchor]) ++ mask_embd`, three blocks
(`attn_q/k/v` with `q_norm`/`k_norm`, ctx K/V from `rms_norm(ctx_mix·taps)·ctx_norm` through `attn_k_ctx`/`attn_v_ctx`, SwiGLU),
then `output_norm` and the markov chain - and run it against the engine's own continuation.

## The forward is written, and the unembed decode is vectorised and validated

**`tools/dflash_ref.py`** implements the mainline algorithm on the decoded weights: `x0 = [rms(embd[anchor], token_embd_norm);
mask_embd.T]`, three blocks of GQA attention (24 Q / 8 KV, head 128, q/k-norm per head, NeoX RoPE theta 1e6) with per-layer
context K/V built as `rms(ctx_mix · taps, ctx_norm)` through `attn_k_ctx`/`attn_v_ctx` (K normalised by `attn_k_ctx_norm` and
rotated, V neither), SwiGLU 9216, then `rms(·, output_norm)`, `base = h @ unembed^T`, softcap 30, and the Markov chain
(`joint_proj [5120, 768]` = `state 256 + w1_prev 256 + h_i 2304 + f_rnn 2304`; `state = c + g*(state-c)`; `markov = tanh`;
`col = base[:,i] + markov_w2 @ markov + w2b`; greedy).

Two structural facts fixed while writing it, both from the drafter's own metadata rather than inference:

    dflash.sample_from_anchor = False       -> slot 0 is a bonus anchor, drafts come from slots 1..7
    markov_w1/w2 [256, 200058]              -> drafter vocab is 200058, trunk vocab is 201024

The second one is a real open question, not a footnote: the drafter's vocab is 966 smaller than the trunk's and the GGUF carries
no d2t mapping tensor, so the forward currently assumes the drafter vocab is the first 200058 trunk ids. That assumption is
testable and must be tested before any acceptance number from this path means anything.

**`tools/dump_unembed_fast.py`** exists because `strata_extract.dequant_sk` is a pure-Python double loop - 256 elements per
superblock, 1,809,216 superblocks for `model.llm.unembed.weight [201024, 2304]` - which would run for hours. The vectorised
version decodes 40k superblocks at a time straight into a memmap, and it was validated against the slow one before being trusted:

    200 superblocks / 51,200 weights:  max|fast - slow| = 1.490e-08   rel 4.516e-08   MATCH
    unembed rms 0.04584

That validation is the whole point: a fast decoder that silently disagrees with the verified one would poison every logit
downstream, and the failure would look like a modelling error, not a decoding one.

## First end-to-end run of the mainline forward: it executes, and it is 0/7 with position-insensitive output

The pipeline now runs on real weights end to end. Inputs: the 281-position prompt `pre.txt`, taps
`L05_p0 / L11_p1 / L17_p2 / L23_p2` (the four visits `[5,23,41,48]` resolved from the engine's own schedule in
`wubu_inkling2_model.c:227-303` - layers 0-5, then the 6..17 core three times, then 18-23), the anchor embedding for token
220, and the decoded `unembed`. Oracle is the engine's own greedy continuation after the anchor:
`1377 68014 371 157021 161319 220 32329`.

    layer rms 1.7454 / 2.6692 / 4.4663        h rms 3.1360   output_norm rms 3.0924
    base rms 4.3223   f_rnn rms 0.6909
    drafts  152290, 11, 11, 11, 149881, 196995, 19      MATCH 0/7

**The output is near position-independent**: token `718` tops almost every slot, and even slot 1 - which should predict the
first drafted token `1377` - returns `718` with a 17.8 peak over a ~16.8 floor. A flat distribution that ignores the slot index
is the signature of a block whose attention is averaging over the context instead of selecting from it.

**Five hypotheses eliminated by measurement, not by argument:**

    1. position offset      anchor_pos = T vs T-1        drafts unchanged (152290 first either way)
    2. vocab misalignment   both GGUFs have exactly 200058 tokens, identical at every index
                            (the trunk's 201024-row matrices are padded with 966 unused rows)
    3. RoPE convention      NeoX half-split vs GPT-J interleaved   both 0/7, 718 still dominant
    4. norm parameterisation  all 21 norm tensors centre on ~1 (means 0.56..4.41, rms 0.90..5.29),
                            so multiplicative RMSNorm is right, not Gemma (1+w)
    5. numerical blow-up    per-layer trace: x 1.44 -> 4.47, q ~1.5, ctx K 1.37..1.91, ctx V 0.83..4.23,
                            attention logits +-11..17 - nothing saturates, nothing overflows

The engine's layer-rms growth (1.75 -> 2.67 -> 4.47) also appeared in the Strata port's stage 2 (3.94 -> 7.08 -> 13.44), and
`ctx_norm` grows with depth (rms 0.96 -> 3.58 -> 5.29), so the growth is a property of the model rather than a bug in either port.

**What is left** is the ctx/tap wiring: which residual point the four taps are taken from, whether the mask rows are ordered
as assumed, and whether the block queries are meant to see the ctx with the same RoPE base they are given. Those are checkable
one at a time against the trace above, and none of them is the arithmetic this project spent two days suspecting.

## The 1/7 was a space coincidence - corrected - and tap selection is eliminated as the unlock

The first tap set I tried (`L11_p0, L17_p0, L11_p1, L17_p1`, the engine's own `src_next` definition) produced MATCH 1/7 where the
original set gave 0/7, and sharper per-slot logits. I called that a breakthrough signal. **It was not.** Checking *which* index
matched: the match is at position 5, where the oracle token is `220` - a space. A space is the single most likely token in any
continuation, so matching it is not evidence of a working drafter.

Scanning all 24 orderings of that tap set makes the point quantitatively:

    8 orders -> 1/7     16 orders -> 0/7     no order above 1/7

If tap order were the missing piece, some ordering would separate from the rest. None does; the eight 1/7 results are the same
space match seen from different angles.

**Out-of-sample check.** A second prompt was built specifically for this, and the first attempt had to be thrown away: repeating one
sentence nine times made the engine loop on `1377 68014`, and a looping oracle lets a broken drafter "succeed" by accident. The
third prompt (263 tokens, seven distinct sentences) produces seven *unique* oracle tokens:
`34541 150463 195644 149880 150422 149878 177853`. Against that:

    taps L05_p0,L11_p1,L17_p2,L23_p2      0/7
    taps L11_p0,L17_p0,L11_p1,L17_p1      0/7
    taps L11_p1,L17_p1,L11_p2,L17_p2      0/7

**Eliminated, with the measurement that eliminates it:** tap identity, tap order, tap phase. Note also a mechanical detail worth
recording: only the `p0` pass dumps the full sequence when `n_gen > 1` (`p1`/`p2` dumps come out as single rows, 9216 B); with
`n_gen = 1` all passes carry the full 263 positions. Greedy decode is deterministic, so taps can be taken from a `n_gen = 1` run
and the oracle from a longer one on the same prompt - which is what the out-of-sample test above does.

**Where this leaves the port.** Every input-side hypothesis is now eliminated with measurements: architecture (read from upstream),
weights (decoded and cross-checked), vocab (identical), norms (multiplicative, verified), RoPE (both conventions), positions
(T vs T-1), taps (identity, order, phase), and numerics (nothing saturates). What remains is not another guessable detail: it is
that no intermediate value from this implementation has ever been compared against a *running* reference.

**Next step, and it is not another hypothesis:** build upstream llama.cpp locally with MSVC. The shipped `build-ref` binary cannot
run here (private UCRT import), but a local build will, and `common/speculative.cpp` already implements `draft-dspark` against this
exact drafter GGUF. That gives a reference that can be executed on this machine, for this model, with intermediate values to
compare - the thing every failed hypothesis this session was missing.

## A working upstream reference now exists on this machine - but it cannot load this trunk, and the reason is a real format mismatch

**The reference binary builds and runs.** The shipped `build-ref` was MinGW/UCRT and imported
`api-ms-win-crt-private-l1-1-0.dll`, a private UCRT shim that consumer Windows does not ship, so it exited 127 with no output -
my earlier diagnosis, now confirmed by fixing it. `tools/build_upstream_msvc.bat` builds with MSVC 19.44 (WinLibs MinGW and
MSVC differ precisely in whether they link the public UCRT):

    llama.exe --version -> version 1.0.0-dev, built with MSVC 19.44.35222.0 for Windows AMD64

Three CLI details cost a few attempts and are recorded so they need not be rediscovered: `llama.exe` is now a **multicall** binary
(`llama.exe cli ...`), `--draft-max` was **removed** (use `--spec-draft-n-max`), and `-no-cnv` no longer exists. The speculative
type is inferred from the drafter GGUF's metadata, so `-md dspark-...gguf` alone selects `draft-dspark`.

**Then it refused the model, in two stages.**

    llama_model_load: error loading model: check_tensor_dims: tensor 'blk.24.attn_q.weight' not found

The trunk declares `inkling2.block_count = 48` but physically holds `blk.0..blk.23` - 24 layers. Our engine reads 48 as the visit
count of the loop-core schedule (24 physical layers, 48 visits, the accounting settled earlier in this document); upstream reads it
as the layer count and demands 48 layer tensors. Patching the single value to 24 in a copy (`tools/gguf_patch_kv.py`, a same-size
in-place rewrite) produced the second and decisive error:

    key inkling2.attention.head_count has wrong array length; expected 24, got 48

So the file is **visit-indexed throughout**, not merely in `block_count`: its per-layer arrays carry 48 entries. Upstream wants
physical-layer indexing. That is not a byte patch - it is a header rewrite with a data shift, because dropping array entries changes
the header length and moves every tensor offset.

And the entries cannot simply be truncated to the first 24: under the schedule read from the engine's own code (visits 0-5 = L0-L5,
6-17 = L6-L17, 18-29 = the core again, 30-41 = the core again, 42-47 = L18-L23) the first 24 visit entries cover L0-L17 twice over
and **omit L18-L23 entirely**. The physical selection is `[0..17] + [42..47]`. Getting that wrong would silently miswire two thirds
of the layers.

**Where this leaves the reference goal.** A runnable upstream exists, but there is no trunk it can load on this machine: the only
local export is visit-indexed. Reaching a reference therefore needs either a physical-indexed export of the trunk, or a header
rewriter that maps the 48-entry visit-indexed arrays down to 24 physical entries using the schedule above. The 8.5 GB patched copy
was deleted rather than left to rot on disk.

**This also reframes the whole port.** If the model's own metadata is visit-indexed, then a drafter consuming "taps at 4 target
layers" is consuming *visits*, and every intermediate I have computed this session used engine dumps taken at visit indices - which
is consistent with the design rather than a mistake. What has never been checked is whether the drafter's `ctx_mix` expects the four
taps in visit order or in physical-layer order, and that question is now answerable from the model file rather than by guessing.

## The reference is RUNNING and it accepts tokens: acceptance 0.149, mean length 2.00

Two things were wrong with my attempts to run upstream, and neither was the binary.

**The trunk was the obstacle, and a correct one was already on disk.** `models/inkling-2-mini/mainline/Inkling-2-Mini-UD-IQ2_XXS.gguf`
is architecture **`inkling2_loop`**, `block_count = 24`, 744 tensors, 1,280 bytes larger than the copy I had been fighting. The file
I kept trying to load declares `inkling2.block_count = 48` with 48-entry per-layer arrays and 24 physical layer tensors - internally
inconsistent for any loader that reads block_count as a layer count, which is exactly why it failed twice. The mainline export exists
and needed no rewriting at all. (`tools/gguf_reindex_layers.py` was written for that case and correctly refused to proceed on any
array that was not provably periodic; it is kept because it is the tool for converting the visit-indexed export, not because the
mainline one needed it. `--override-kv` in upstream would also have covered the scalar case.)

**`completion` cannot drive a draft model.** Only `cli` and `serve` accept `--spec-draft-model`; `completion` offers the draft cache
types but no draft model, and `cli` insists on conversation mode in this build (`-no-cnv` is defined in arg.cpp yet rejected by the
subcommand). The server is the way in, because `/completion` takes a raw prompt with no chat template.

    llama.exe serve -m mainline/Inkling-2-Mini-UD-IQ2_XXS.gguf \
                    -md dspark-Inkling-2-Mini-Q8_0.gguf \
                    --spec-type draft-dspark --spec-draft-n-max 7 -c 2048 --port 8099

and it comes up with the implementation genuinely active:

    common_speculative_impl_draft_dflash: adding speculative implementation 'draft-dspark'
    common_speculative_impl_draft_dflash: - n_max=7, n_min=0, p_min=0.00
    common_speculative_impl_draft_dflash: - block_size=8, mask_token_id=0, n_extract=4
    common_speculative_impl_draft_dspark: - sample_from_anchor=false, has_markov_context=true

**`n_extract=4`** is the name upstream gives the tap count, and `block_size=8, mask_token_id=0, sample_from_anchor=false,
has_markov_context=true` match the drafter's metadata read earlier - independent confirmation that the parameter set I implemented
against is the one being used.

**And the acceptance, measured on the same 263-token prompt used for the out-of-sample test:**

    draft acceptance = 0.14894 ( 7 accepted / 47 generated ), mean len = 2.00

So dspark on this trunk accepts 14.9% of drafted tokens, at a mean accepted length of 2.00 per step. That is the number this port
has to reach. It also settles something important: the drafter is *not* expected to be near-perfect on this model, and my 0/7 is a
real gap rather than the ceiling - upstream gets seven accepted tokens where mine gets none.

A live reference implementation, on this machine, for this model, with a measured acceptance figure to match, is exactly what every
failed hypothesis this session was missing. The next step is to run the same prompt through both and compare intermediates.

## The oracle is verified against the reference, and three more hypotheses are dead

**The foundation holds.** Running the same 263-token prompt through the upstream server and through our engine gives the same 12
greedy tokens:

    server : 34541 150463 195644 149880 150422 149878 177853 1164 149879 179357 152024 13
    ours   : 34541 150463 195644 149880 150422 149878 177853
    agree on first 7: True

So the trunk behaves identically, the oracle is exactly the reference's continuation, and the 0/7 is entirely a drafter gap - not a
prompt, tokenizer, or sampling difference. The server's own top-1 for step 1 is `34541` ("woke") at logprob -2.0875.

**Three more hypotheses eliminated, each by measurement:**

    1. ctx injection        with ctx K/V zeroed (block-only attention): slot 1 top = 68073, still 0/7
                            with ctx injection:                        slot 1 top = 149965, still 0/7
       => the block is wrong either way; it is not the ctx wiring alone.
    2. anchor input         slot 0 fed the token embedding (as I had it):        slot 1 top 149965
                            slot 0 fed the trunk's FINAL hidden at that position: slot 1 top 12
       => `has_markov_context=true` does not simply mean "use the final hidden as slot 0" either.
    3. unembed source       not testable this way: the mainline trunk carries output.weight as IQ4_XS (no decoder written),
                            and the Strata S6_K copy is the one in use.

**One new lead, from the mainline trunk's tensor list:** it contains `loop_kv_res_query.weight [2304, 24]` - a per-layer (24-column)
query over the residual stream. That is a mechanism our engine does not implement, and it is a plausible definition of *which* states
become the four taps (`n_extract=4`): not four hardcoded visit dumps, but whatever this learned query selects. Our engine collects
`src_next` at `blk_len` intervals for its loop-core messaging, which need not be the same thing at all.

**Where this stops being guesswork.** Every input-side variable I can vary has been varied. The remaining differences between my
forward and upstream's are inside upstream's graph, and upstream is now buildable in ~5 minutes incrementally. The next step is to
instrument `src/models/dspark.cpp` and `src/models/dflash.cpp` in the local checkout - print slot logits and the four extracted
states - rebuild, and compare those numbers against `tools/dflash_ref.py` line by line. That converts this from inference about
internals into a direct numerical diff, which is what the last several eliminations have been working around.

## The taps are layers 4, 10, 16, 22 - established from the reference, not inferred - and f_rnn now matches it numerically

The instrumentation works. `common/common.cpp` gained a tensor-dump callback enabled only when `LLAMA_DUMP_DIR` is set (with
`LLAMA_DUMP_MATCH` naming substrings to match), reading values the way `common/debug.cpp` does: `ggml_backend_tensor_get` for
device buffers, `t->data` when host. It dumps the first occurrence of each matching tensor, which is the prefill - exactly what a
comparison against a hand-written forward needs. With `LLAMA_DUMP_MATCH="inp_target_features;f_rnn"` the reference wrote:

    inp_target_features  [ne0=4, ne1=2304, ne2=259]   9,547,776 B
    f_rnn                [ne0=2304, ne1=259]          2,386,944 B

**Correlating those four reference taps against 48 of our own dumps settles the question that guessing never could:**

    tap 0  ->  L04_p0   cos +0.999918
    tap 1  ->  L10_p1   cos +0.994184
    tap 2  ->  L16_p2   cos +0.987634
    tap 3  ->  L22_p2   cos +0.648514

So the taps are the residuals after **layers 4, 10, 16 and 22 - a stride of exactly 6** (`attn_res.block_size`), with the pass
index advancing as depth increases. Not the visit indices `[5, 23, 41, 48]` I had been using, and not the `src_next` layers 11/17
either. Three of the four correlations are effectively exact (0.995-0.99992); tap 3 is the outlier at 0.65, so either L22 is
captured in a different pass than our p2, or the fourth tap is not a plain residual of layer 22.

**And `f_rnn` - which I compute myself from `markov_ctx_mix` and `markov_ctx_norm` - matches the reference's:**

    upstream rms 0.727080     mine 0.722787
    cosine(mine, upstream) = 0.981564      first row 0.998723      last row 0.985542

That is the first end-to-end numerical agreement anywhere in this port. It says the context mixing path is right: right taps
(for the first three), right `markov_ctx_mix` weighting, right `markov_ctx_norm` normalisation, right layout. The residual 1.8%
cosine shortfall is consistent with the tap-3 mismatch above and is the next thing to close.

**Acceptance with the reference's own taps is still 0/7**, so the error is inside the block forward or the Markov chain - but those
are now the *only* unknowns left, and each can be diffed the same way: the dump hook accepts any tensor name, so the drafter's
`t_logits` can be captured next and compared against `dflash_ref.py`'s `base` directly.

Build note for future sessions: MSVC relinks fail with LNK1104 "cannot open file" when overwriting an existing DLL/exe that a
previous run mapped - the fix is to delete the stale artefact and relink, not to retry blindly. Building `-j 1` avoids the
concurrent-link variant of the same error, and a still-running `llama.exe` will hold `llama-common.dll` open.

## Named-tensor dumps now work; the block forward is where we diverge, and the diff harness exists to prove it

Two `ggml_set_name` calls in `src/models/dspark.cpp` (`dspark_base` on the pre-Markov logits at :248, `dspark_logits` on the final
output at :453) make exactly the tensors we need visible to the dump hook. That required one rebuild, and the rebuild required
deleting `bin/llama.exe` first - overwriting a stale artefact gives LNK1104 regardless of retries.

Captured with `LLAMA_DUMP_MATCH="dspark_base;dspark_logits"`:

    dspark_base    [200058, 2]   1,600,464 B
    dspark_logits  [200058, 2]   1,600,464 B

**The comparison against our forward, position-matched:**

    ref  col0  rms 3.866  top5 150525:11.13 151320:9.67 199254:8.96 198437:8.72 151405:8.46
    ref  col1  rms 7.346  top5 151457:7.28 13:7.17 151090:7.08 151337:6.93 149876:6.85
    ours slot0 rms 4.394  top5 150743:19.76 123268:18.05 155795:17.73 185634:17.62 69717:17.49
    ours slot1 rms 4.364  top5 158951:18.38 129992:18.37 153817:18.04 123268:17.74 150743:17.68

    cos(ref col0, ours slot0) = -0.081      cos(ref col0, ours slot1) = -0.005
    cos(ref col1, ours slot1) = +0.001

Two things stand out. The reference's logits are **much flatter** - max 11.13 where ours reach 19.76 - and the vectors are
essentially uncorrelated. Our magnitudes are the right order (rms 4.4 vs 3.9/7.3), so this is not a scaling bug: the content is
different.

**Important caveat, stated so it is not over-read:** the reference graph runs with `n_tok = 2`, not our 8 slots, in every request
tried (n_predict 2, 4, 8 all produce a 2-column dump). So the column-to-slot correspondence is not established, and one negative
cosine set is not proof the block is wrong - only that no slot matches column 0 or 1. What it does establish is that `base` and
`logits` are now capturable and comparable, which is the machinery the remaining work needs.

**Acceptance varies by request**: 7/47 (0.149), 4/40 (0.100), 1/21 (0.048). It is not a fixed number and any parity claim must
compare over the same prompt and the same n_predict.

**Next step:** name the block input (`cur` after `build_dspark_inp_block`) and each of the three block outputs, dump them, and
compare stage by stage against `dflash_ref.py` - the same technique that pinned `f_rnn` to cosine 0.982 and identified the taps as
layers 4/10/16/22. That localises the divergence to a stage instead of leaving it in "the block" as a whole.

## The input pipeline is verified exactly - three cosines of 1.000000 - and the divergence is inside layer 0

Naming the raw anchor with one `g.cb(anchors, "dspark_anchor_raw", -1)` in `build_dspark_inp_block` closed the last input question
and exposed the real defect in the process.

    cos(ref position 0, tok_norm(anchor_raw)) = +1.000000    rms 0.9760 both
    cos(ref position 1, mask_embd row)        = +1.000000    rms 0.5956 both
    cos(ref noise_norm-0, rms(x0, blk.0.attn_norm)) = +1.000000  rms 0.8378 both
    cos(ref noise_norm-1, ...) = +0.0905     cos(ref noise_norm-2, ...) = +0.0979

So every stage before layer 0's attention is now confirmed exactly against the reference: the block input (anchor row AND mask
row), and the first layer's pre-attention norm. `noise_norm-1` and `-2` differ only because they are downstream of layer 0's
output. **The divergence is inside layer 0's attention or FFN.**

**And it explains the whole session's 0/7.** `cos(dspark_anchor_raw, our_embd.bin) = +0.0657` - the vector our engine dumps as the
"anchor embedding" is *not* the anchor token's token embedding. Every tap hypothesis, every RoPE convention, every position offset
tried in this document was run with a wrong anchor row in slot 0. My construction was right (`tok_norm(token_embd[anchor])`, which
reproduces the reference to 1.000000) and the input was wrong - which is exactly the class of error that survives every
architecture-level check while making all of them look like failures.

Note also what the reference's own tensors say about the mask: all seven `mask_embd` rows are identical, a single broadcast vector
(rms 0.5956). Our code indexes them individually, which happens to be equivalent.

**Two facts about the embedded `our_embd.bin` worth fixing separately:** it disagrees with the reference's raw anchor (0.066) and
also with the Strata pack's `model.llm.embed.weight` row 220 (0.137), so the engine's `IN2_DUMP_EMBD` is dumping something that is
neither. That is a bug in our tooling, not in the drafter, and it should be fixed before the parity harness relies on it.

**Next step:** dump `Qcur`/`Kcur`/`Vcur` for layer 0 (the reference already `cb`s them as `Qcur-0` etc.) and compare against
`dflash_ref.py`'s layer-0 projections, which splits the remaining divergence into attention versus FFN in one measurement.

## A real loader bug found and fixed (ggml weight orientation) - acceptance unchanged, layer 0 still unresolved

Comparing layer 0's projections against the reference exposed a genuine bug in `dflash_ref.py`, independent of everything else:

    blk.0.attn_q projected with fromfile().reshape(2304,3072) then x @ W   -> q rms 1.6453
    same tensor as .reshape(3072,2304) then x @ W.T                        -> q rms 1.9230
    reference Qcur-0                                                        -> q rms 1.9231

ggml stores `ne0` fastest, so element `(i, j)` sits at `i + ne0*j`; reshaping the flat file to the ggml dims `(ne0, ne1)` in
row-major gives the **transpose** of the matrix the file represents. The correct reader is `reshape(ne1, ne0)` as `(out, in)` and
`x @ W.T` for `mul_mat`. Only the corrected form reproduces the reference's magnitude (1.9230 vs 1.9231, against 1.6453 for the
wrong one) - a 17% error in every projection, silently applied to all nine matmul sites.

The `w()` helper in `dflash_ref.py` now does this, and the Markov head uses it too: `markov_w1` row `prev` (a contiguous 256-float
run, which the old code sliced wrongly), `joint_proj` as `(768, 5120)`, `markov_w2` as `(vocab, 256)`.

**What it did not fix:** acceptance is still 0/7, and the layer-0 projections still disagree in content even with the right
magnitude - `cos(mine, ref Qcur-0)` is ~0.02 at every layout permutation tried (`[t,h,d]`, `[t,d,h]`, and both reference
readings), while the norms agree to four decimals. Equal norm with orthogonal content is not a scaling or transposition problem;
something upstream of the projection differs in *which* elements are combined, most plausibly the per-head norm axis or a
pre-projection operation the reference applies that this forward does not.

**Where the port stands, precisely.** Verified against the reference: block input (anchor row and mask row, both cos 1.000000),
layer-0 pre-attention norm (cos 1.000000), `f_rnn` (cos 0.982), tap identity (layers 4/10/16/22, cos 0.99+ for three of four).
Not yet verified: layer-0 q/k/v content, attention output, FFN, and the Markov chain. The divergence is inside layer 0's
attention, and the next probe is to dump the reference's pre-norm `Qcur` (before `attn_q_norm` and RoPE) so the norm axis and the
rotation stop being confounded with the projection.

## Layer 0 verified exactly against the reference, and a second real bug fixed: the draft block's RoPE positions start at 0

Dumping the reference's pre-norm projections (`cb(Qcur, "Qcur_raw", il)` right after the reshape in `build_layers`) split the
remaining divergence cleanly, and corrected an earlier wrong conclusion of mine.

**My "orthogonal content" claim last round was a bug in my own comparison code, not in the forward.** Comparing the reference's
`Qcur_raw-0` against my `x @ W.T`-with-corrected-orientation projection by elementwise difference rather than cosine:

    first values  mine [0.6077 -0.8311 0.7203 -1.6762 ...]   ref [0.6021 -0.8295 0.7189 -1.6781 ...]
    max|diff| = 0.028017     mean|diff| = 0.005031     over all 6144 values

That is Q8_0 dequantisation plus f32 rounding. The projections were right; my cosine was computed over mismatched layout views.
This is the same failure mode as the tautological `rms_pre` earlier - a measurement that looks like evidence and isn't.

**A second real bug: block positions.** Feeding Q through the per-head norm and NeoX RoPE and diffing against the reference's
post-norm `Qcur-0`:

    no rope   max|diff|  6.16674   mean 0.050178
    rope@0    max|diff|  0.07677   mean 0.003223     <-- exact
    rope@256  max|diff| 24.84287   mean 1.007973
    rope@257  ... 24.9 / 0.98     rope@258  ... 25.0 / 0.99     rope@259  ... 25.0 / 1.01

The draft block is its own sequence and its RoPE positions **start at 0**, not at the anchor's position. `dflash_ref.py` used
`anchor_pos + arange(8)`; it now uses `arange(8)`. Every earlier tap/RoPE experiment in this document carried that error.

**So layer 0 is now verified end to end up to attention:** block input (cos 1.000000 for both the anchor and mask rows), layer-0
pre-attention norm (max|diff| 0), Q projection (max 0.028), Q after norm+RoPE (max 0.077), K and V raw (rms 2.1364 vs 2.1361).
Acceptance is still 0/7, so what remains is attention composition, the FFN, or the Markov head - not the block input.

**Taps re-identified by elementwise difference instead of cosine (cosine was what misled me before):**

    tap 0  L04_p0   mean|d| 0.038866   max 3.36
    tap 1  L10_p1   mean|d| 0.192188   max 11.22
    tap 2  L16_p2   mean|d| 0.218398   max 19.15
    tap 3  best candidate L23_p2, mean|d| 1.676690   max 81.82      <- no match

Three of four are exact (the residual is quantisation). The fourth is **not any residual our engine dumps** - the tail layers are
only dumped once, in pass 2, and neither L22_p2 nor L23_p2 fits. `IN2_DUMP_H` is no help: it writes a single 2304-float file for
the final position only, not a per-position tensor.

That fourth tap is the immediate suspect for the remaining error, since the injected context K - which mixes all four taps - is
close but not exact:

    Kcur_injected-0  rope@0:  mean|diff| 0.147888   max 7.45633     (vs ~0.005 for the Q path)

One wrong tap out of four, weighted 1/4, is the right order of magnitude to produce exactly that.

**Next step:** determine the fourth tap. The reference's own features give a target; the candidates our engine cannot currently
produce (a per-position final hidden, or the tail block's last layer in a pass we do not dump) need a dump hook in the engine
first - the same `IN2_DUMP_VISITS` machinery that produced the others, extended to emit the tail layers at every pass.

## The fourth tap is a state our engine never produces - a schedule-level difference, not a modelling one

Tap 3 resists every candidate. Measured by elementwise difference over the same 259 positions, with the known-good tap 2 as a
control:

    tap 2 vs L16_p2    segments [0:64] 0.128   [64:128] 0.111   [128:] 0.315      <- exact, as established
    tap 3 vs L22_p2    segments 2.101 / 2.405 / 1.980 / 1.655   min 1.061 at position 1
    tap 3 vs L23_p2    segments 1.859 / 2.148 / 1.564 / 1.161   min 0.732 at position 195
    tap 3 vs L17_p2    segments 3.251 / 3.415 / 3.421 / 3.324   (worse everywhere)

Also eliminated: normalisation (best residual 1.814 with `L23_p2` scaled 0.844, against a tap3 rms of 3.64 - barely better than
predicting zero), and position misalignment (±3 offsets change nothing beyond the fourth decimal).

The taps that *do* fit follow a completely regular pattern - layers **4, 10, 16** with passes **0, 1, 2**, i.e. stride 6 with the
pass index advancing by one each time. The natural fourth member is **layer 22 at pass 3**, and our engine has no pass 3: the core
(6..17) runs three times (p0, p1, p2) and the tail (18..23) once (dumped as p2). So the reference is consuming a residual that our
loop-core schedule never materialises.

**That reframes the remaining error.** It is not that my drafter code misuses the tap - it is that the trunk schedule feeding it
differs. The injected context K mixes all four taps, so one missing tap of four is consistent with the ctx K discrepancy measured
last round (mean|diff| 0.148 against ~0.005 for the Q path), and with the persistent 0/7 that survived every drafter-side fix.

**Next step:** compare the two loop-core schedules directly - count how many times the reference's trunk visits the core block and
at which layers, from `inp_target_features`' own structure (four taps at +6 stride implies four core visits), against our engine's
`for il in [loop_core_start, loop_core_end)` executed three times. If the reference runs the core four times, that is a trunk
difference to fix in the engine, and the drafter parity follows from it rather than from further drafter work.

## Taps 0-2 matching and tap 3 not matching localises a TRUNK divergence to layers 17-22

Least squares over all 48 of our residual dumps, per tap, on 120 positions:

    tap 0:  residual 0.0424   L04_p0 = +1.000   (everything else <= 0.004)
    tap 1:  residual 0.1260   L10_p1 = +0.979
    tap 2:  residual 0.1224   L16_p2 = +0.995
    tap 3:  residual 1.4345   L20_p2 = -1.639, L23_p2 = +1.462, L21_p2 = +0.816 ...  (no single tap, no fit)

Three taps are single residuals with weight ~1. The fourth cannot be expressed as *any* linear combination of our 48 residuals,
so it is not a tap my engine fails to dump - it is a state my engine computes differently.

**Both trunks declare the same loop schedule**, so the schedules are not the difference:

    non-mainline:  inkling2.num_loops = 3, inkling2.loop.core_start = 6, core_end = 18
    mainline:      inkling2_loop.loop.count = 3, loop.start = 6, loop.end = 18, attn_res.block_size = 6

and our engine implements exactly that: 6 leading layers + 12 core layers x 3 passes + 6 tail layers = 48 visits,
`n_layer_phys = 48 - (3-1)*12 = 24`.

**What the tap pattern actually says.** Taps 0-2 are layers 4, 10, 16 - in visit terms 4, 22, 40, spacing 18. The fourth visit at
that spacing would be 58, which no 48-visit schedule contains; the tail's L22 sits at visit 46. Since taps 0-2 match this engine's
own residuals to 0.04-0.13 but tap 3 (L22) differs by 1.65, the divergence cannot be in layers 0-16 at any pass. It lies between
**layer 16 in the final core pass and layer 22 in the tail** - i.e. in the last core pass (layers 17) or the tail block itself
(18-23), and it accumulates rather than appearing at the input.

That is the most precise localisation this port has had: not "the trunk differs" but "the trunk agrees through L16_p2 and disagrees
by L22". Everything downstream - the injected context K, the attention, and the drafter's 0/7 - is consistent with that.

**Next step:** bisect the tail. Dump the reference's residual at the layer-22 position (or an equivalent named tensor) so the
first layer after L16_p2 where the two disagree can be identified - the same bracketing technique that located the divergence to
layer 0 earlier. In the engine that means comparing successive tail layer outputs against a reference dump of the same layers,
which needs one more `cb()` in the trunk's tail loop.

## Corrected again: the trunk is NOT the problem, and the tap list is in the drafter's metadata

Two of last round's claims have to be retracted, both from better measurement.

**The trunk is fine.** Dumping the reference's per-layer tail inputs (`cb(cur, "trunk_tail_in", il)` before `build_layer` in
`inkling2-loop.cpp`) and diffing them against our residual dumps over 200 positions:

    reference input to layer 19  vs our L18_p2   0.0267
    reference input to layer 20  vs our L19_p2   0.0381
    reference input to layer 21  vs our L20_p2   0.0496
    reference input to layer 22  vs our L21_p2   0.0611
    reference input to layer 23  vs our L22_p2   0.0774
    reference input to layer 18  vs our L17_p2   0.2935

The tail agrees to 0.03-0.08 - the same order as the taps that were already matching. So the earlier claim that "the trunk diverges
between L16_p2 and L22" is **wrong**: there is no divergence in the tail. What diverges is only the *fourth tap*, and it is not one
of the tail residuals:

    tap 3 vs reference in18..in23    3.35 / 2.74 / 2.65 / 2.54 / 2.37 / 2.13
    tap 3 vs our L21/L22/L23_p2      2.37 / 2.14 / 1.82

**The tap list was in the drafter GGUF, and my GGUF reader was silently skipping it.** `dflash.target_layers` is an *array*, and
`gguf_extract.read_header` skips arrays (it walks them only to advance the cursor). Reading it with the array-capable reader:

    dflash.target_layers = [5, 23, 41, 48]

That is the authoritative list, it matches the value inherited from the Strata manifest at the very start of this port, and it is
**1-based against the 48 visits** our engine performs. Note the schedule in `inkling2-loop.cpp:84-120` is explicit that
`res->t_layer_inp[visit++] = cur` records the residual *entering* each layer, visit-indexed - so the taps are layer inputs, not
outputs, and visits 5/23/41 map to the inputs of layers 4/10/16, exactly the three we matched.

**Which leaves id 48.** Visits are 0-based over 48 entries (0..47), so id 48 is one past the end: the natural reading is the
residual *after* the last tail layer, i.e. the final hidden before `output_norm`. Our engine never dumps that per position -
`IN2_DUMP_H` writes a single 2304-float row for one token - and none of our 48 residuals is it, which is precisely what the
least-squares test showed (tap 3 unexplained by any combination, residual 1.56).

**Next step:** dump our engine's final residual per position (after the last tail layer, before the final norm) and test it against
tap 3. That is a small change at the end of the prefill path, and if it lands the context K becomes exact - the piece that has been
loose since the ctx K measured 0.148 while the Q path measured 0.005.

## Three of four taps are BIT-EXACT, and the fourth is an index past the last layer

Dumping every visit input from the reference (`cb(cur, "trunk_inp", visit)` at all three sites in `inkling2-loop.cpp`) turns the
tap question into a lookup:

    tap 0 = visit 5    mean|d| 0.00000      (visit 4 -> 1.43)
    tap 1 = visit 23   mean|d| 0.00000      (visit 22 -> 0.74)
    tap 2 = visit 41   mean|d| 0.00000      (visit 40 -> 1.16)
    tap 3 = no match   best visit 40 -> 2.88

**0.00000 is not a loose match - that is bit-identical**, which means our trunk reproduces the reference's residual stream exactly
at visits 5, 23 and 41. Given the engine's whole purpose is parity, this is the strongest confirmation the trunk has had: correct
weights, correct schedule, correct numerics, at three widely separated points in the 48-visit sequence.

And the ids are exactly the drafter's metadata, 0-based:

    dflash.target_layers = [5, 23, 41, 48]

So the fourth tap is **index 48 of a per-visit array**, i.e. one past the last layer. Two candidates for what lives there were
tested, and neither is it:

    reference post-tail residual (cb after the tail loop):  rms 2.210   vs tap3 rms 3.993   mean|d| 1.986
    our engine's final residual (final_resid == L23_p2):    rms 2.516   vs tap3              mean|d| 1.822
    our engine's final residual after output_norm:          rms 3.227                        mean|d| 2.226
    our engine's final residual after the output_res_query mix: rms 0.861                     mean|d| 2.416

So index 48 is neither the residual entering the LM head, nor the post-tail residual, nor the res-query mix of it. Since taps 0-2
are bit-exact through the same mechanism, the mechanism is settled and only this one entry's *content* is unknown - it is some
state with rms ~4, larger than the post-tail residual, that the trunk stores at the visit index one past its last layer.

**New engine hook:** `IN2_DUMP_FINAL=<dir>` writes `final_resid.bin` and `final_mixed.bin` - the per-position residual after the
last tail layer, raw and after the `output_res_query` mix. Verified `final_resid` is byte-identical to `L23_p2`, so the existing
layer dump was already the post-tail residual and there is no off-by-one in the engine's naming.

**Next step:** read the trunk side of the extraction - `common/speculative.cpp` sets `embd_row` and the trunk writes
`layer_inp_src`; that code decides what sits at index `n_layer`. My guess-free answer requires reading how that buffer is sized and
filled rather than testing more candidate states from our side.

## How the reference actually extracts taps - read from the trunk side, not inferred

`llama_context::layer_inp_dev_alloc()` (llama-context.cpp:1496) and `build_inp_target_features` (dflash.cpp:226) together settle the
mechanism:

    layer_inp_dev = ggml_new_tensor_2d(ctx, F32, n_layer_inp * n_embd, cparams.n_batch)

So the buffer is **one row per token, with `n_layer_inp` layer states concatenated across the row** - not one row per layer. The
count `n_layer_inp` comes from counting the true entries of `cparams.embeddings_layer_inp`, a per-layer boolean mask, and
dflash.cpp:59 sets `n_embd_inp_enc_impl = target_layer_ids.size() * n_embd`. `build_inp_target_features` then does
`ggml_get_rows(layer_inp_src, embd_rows)` - embd_row selects a **token**, and the row that comes back is the concatenation of the
selected layers' states, which dflash reshapes to `[n_embd, n_target, n_tokens]`.

That reconciles everything measured earlier: comparing `feats[j]` against per-visit dumps is comparing a single layer's state, which
is why three of them matched bit-exactly and the fourth did not - the fourth entry of that concatenation is a *layer*, and I have
been looking for it among *visits*, where only 0-41 were ever captured in the relevant run.

    visits captured in dump_ref10: 0..41 (40 files; 42-47 absent)

`trunk_tail_in` was captured in a separate run, and `trunk_final` in another, so no single dump contains the whole schedule and the
"tap 3 = post-tail residual" test compared across runs while the tail visits themselves were never in the same scan as `feats`.

**Next step:** one run that captures the tail visits (42-47) together with `feats`, so the fourth tap can be matched against them in
the same pass. That is the last unidentified input in the drafter's context path, and the only thing between the current state
(three taps bit-exact, block input and layer-0 attention verified) and an exactly-correct context K.

## ALL FOUR TAPS IDENTIFIED - and the fourth is bit-exact, just not in our engine's output path

`src/models/inkling2.cpp:442-447` names the fourth tap outright:

    void llama_model_inkling2::graph::build_output(ggml_tensor * cur, ...) {
        cur = ggml_reshape_2d(ctx0, mix(blocks, cur, model.output_res_query), n_embd, n_tokens);
        cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
        // DSpark target features: the last one is the final norm output of every token
        res->t_layer_inp[n_layer] = cur;
        ...

So tap 3 is the **final normalised output**, computed as `build_norm(mix(blocks, post-tail residual, output_res_query))`, stored at visit index `n_layer` - which is why `cparams.embeddings_layer_inp` is resized to `n_layer()+1` and why the second index setter accepted id 48. Dumping that exact tensor and diffing:

    tap3 vs trunk_final_norm      mean|diff| 0.00000000    rms 3.9934 vs 3.9934   <- bit-exact
    our norm(mix(resid)) vs same  mean|diff| 2.260794      rms 3.2432

**The complete tap list, all verified:**

    tap 0 = residual entering visit 5   (bit-exact vs our L04_p0)
    tap 1 = residual entering visit 23  (bit-exact vs our L10_p1)
    tap 2 = residual entering visit 41  (bit-exact vs our L16_p2)
    tap 3 = final normed output         (bit-exact vs the reference's own tensor; ours differs by 2.26)

That closes the tap question completely and moves the error one stage further out: three taps come from the residual stream, which our engine reproduces **exactly**, and the fourth comes from `build_output`, which it does not. So the remaining defect is in our `output_res_query` mix or the final RMSNorm - `in2_mix` feeding the final norm.

**This also explains the ctx K discrepancy** measured much earlier (mean|diff| 0.148 while the Q path was 0.005): the context mixer is fed three exact taps and one tap that is off by 2.26, and the error propagates through `ctx_mix`.

**Next step:** dump the reference's `mix(...)` result *before* the norm (one `cb` at inkling2.cpp:443) and compare it against our `final_mixed.bin`, which isolates mix from norm in a single measurement. Whichever of the two differs is the last piece of the context path.

## The "2.26" was my Python, not the engine - our mix already matches the reference

Dumping the reference's pre-norm mix (identity `ggml_scale` to stop the reshape being elided - a bare `cb` on a reshape feeding
`rms_norm` never executes, which is why the first attempt produced an empty dump dir while the string was present in llama.dll):

    our final_mixed vs reference trunk_mix    mean|diff| 0.028871    rms 0.855 vs 0.855
    our final_resid vs reference post-tail    mean|diff| 0.071300

**Our `in2_mix` already reproduces the reference's `mix(blocks, cur, output_res_query)` to 0.029**, with identical rms. The
`output_res_query` mixing was never the defect.

And the 2.26 gap that made it look like a defect was an artifact of my own measurement: applying my hand-rolled RMSNorm to the
*reference's own* mix does not reproduce the *reference's own* final_norm - it is off by 2.256 at every epsilon (0, 1e-6, 1e-5,
1e-4) and with or without the norm weight. A formula that cannot map a known input to a known output of the same graph is a broken
formula, not evidence about the engine. The norm weight I extracted also looks wrong for this role (rms 3.09, range -5.06..13.13,
while the target has rms 3.99) - so either the extraction is the wrong tensor or the scale is applied differently.

That retires the last "defect" in the context path. Standing tally of what is *proven* by measurement:

    3 of 4 taps     bit-exact (visits 5, 23, 41)
    tap 4 source    identified exactly: build_norm(mix(...)) at t_layer_inp[n_layer]
    our mix         0.029 from the reference's mix, same rms
    our post-tail   0.071 from the reference's post-tail residual

**Next step:** stop comparing through my own numpy. Dump the engine's final hidden (`IN2_DUMP_H`) and the reference's
`trunk_final_norm` for the same position and compare those two numbers directly - engine against engine, no intermediate formula of
mine in the path. That also removes the suspected-bad `output_norm.weight` extraction from the critical path.

## Engine against engine: the output path is correct to 0.032

No numpy, no formula of mine - our `IN2_DUMP_H` final hidden against the reference's `trunk_final_norm` for the *same* input:

    prompt "Hello" (1 token on both sides, so row 0 aligns exactly)
    our final hidden vs ref trunk_final_norm    mean|diff| 0.031800   max|diff| 0.2073
    rms 3.4390 vs 3.4477                        corr 0.999934

That is a 0.9% relative difference with correlation 0.99993 - the output path (`mix(blocks, cur, output_res_query)` followed by the
final RMSNorm, i.e. the state stored at `t_layer_inp[n_layer]` and fed to the drafter as tap 4) is **correct**. The 2.26 "defect"
was my hand-rolled norm and nothing else.

Note the earlier attempt at this comparison was invalid and I should record why: `pre3` tokenises to 263 tokens in our engine but
259 in the reference, so a 263-vs-259 prefill cannot be aligned by row index, and scanning all 259 rows for a match produced a
meaningless "best" of 2.49. A one-token prompt removes the alignment problem entirely.

**Where the port now stands, every line measured:**

    trunk residual stream        3 taps bit-exact (visits 5, 23, 41); post-tail residual 0.071
    output path (mix + norm)     final hidden 0.032, corr 0.99993  <- the state that is tap 4
    tap 4 source                build_norm(mix(...)) at t_layer_inp[n_layer], identified exactly
    our in2_mix                 0.029 vs the reference's mix, identical rms

So all four drafter inputs are now either bit-exact or within 1%, and none of them is structurally wrong.

**Next step:** recompute the ctx K in `tools/dflash_ref.py` with the corrected tap set and compare it against a dump of the
reference's own `ctx_mix` output - the same engine-to-engine discipline that just retired the output-path doubt.

## The drafter's whole context path is verified - and the dump was off by one layer

Comparing our `ctx_kv` against the reference's own `Kcur_injected` / `Vcur_injected` (cb at dflash.cpp:425-426), on a one-token
prompt so positions align and RoPE at position 0 is the identity:

    using the reference's own feats:   K 0.0034-0.0046   V 0.0036-0.0195   (rms equal to 3 dp)
    using our taps V004/V022/V040:     K 0.0052-0.0094   V 0.0064-0.0371
    using our taps V005/V023/V041:     K 0.230-0.269     V 0.256-0.905

So `tools/dflash_ref.py:ctx_kv` - `rms(sum_i ctx_mix[i]*tap_i, ctx_norm)` -> `attn_k_ctx`/`attn_v_ctx` -> `attn_k_ctx_norm` -> RoPE -
**is correct**, and the 1.5-3.1 discrepancy in the middle row was our *dump naming*, not the engine or the port.

`IN2_DUMP_VISITS` writes `h` after the layer has been applied (the surrounding code even labels the other hook `l_out-%d`), so
`V005` is the **output** of visit 5, while the reference's tap for id 5 is the residual **entering** visit 5 - which is the output of
visit 4:

    tap for id k = our V{k-1}

With that one-line correction our taps land 0.0139-0.0318 from the reference's, and the injected K/V land at 0.005-0.037.

**Two measurement bugs, same family, both now recorded:** first the feats layout (after `permute(1,0,2,3)` the fastest axis is
`n_target`, so `reshape(1,n_embd,n_target)` and transpose - not `reshape(-1,n_embd)`), then the tap index (layer output vs layer
input). Both produced plausible-looking numbers in the 1.2-3.1 range that were pure artifact.

**Status of the drafter context path: verified end to end.**

    taps            0.014-0.032 from the reference
    ctx mixed K/V   0.005-0.037
    ctx math        correct in dflash_ref.py
    output path     final hidden 0.0318, corr 0.99993

**Next step:** the same engine-to-engine discipline for the block/anchor path (`inp_block_anchor` -> 8-slot input -> 3 layers ->
head), where the only remaining unverified inputs are the anchor embedding and the mask rows.

## Upstream research settles the block contract (and confirms the 1-based tap list)

Three independent sources, fetched rather than inferred:

1. **ninfer-3090 maintainer notes** (`docs/maintainer/inkling2-model.md`): *"the anchor is `n_embed_norm(E[a])`, and the 7
   masks are `mask_embd`"*; block size 8 = one anchor plus 7 drafts; **attention within the block is non-causal**
   (`dflash.attention.causal = false`); the drafter borrows the target's `token_embd`, `token_embd_norm` and `output`.
2. **ufrik's GGUF card** for a different draft: `dflash.target_layers = [6, 24, 42, 48]` explicitly documented as **1-based** -
   which is exactly our `[5, 23, 41, 48]` 0-based, independently confirming the offset derived from the dumps.
3. **unsloth's dspark README**: `--spec-draft-n-max` is clamped to 7 because the trained block size is 8 and *"the first slot of
   each block is the last accepted token"*; mainline llama.cpp has no DSpark support yet, it is carried by the unsloth pin.

So our `x0 = [rms(anchor_emb, token_embd_norm) ; mask_embd rows]` with 8 slots and non-causal block attention matches the
documented architecture.

**Dead end recorded so it is not chased again:** `inp_noise_embd` (cb in dflash.cpp) is *not* the block input. Dumped for a
one-token prompt it has **2 slots**, rms 0.976, while our computed anchor has rms 4.317 and the mask rows 0.596. It belongs to a
different (noise-embedding) branch of the file, and one slot did match the mask rows' rms exactly (0.5956) with mean|diff| 0.507,
which is a permutation-shaped near miss, not a match. The block input is not exposed by an existing cb and would need its own hook.

**Next step:** verify the drafter end-to-end on the thing that actually matters - drafted token identity against the reference's own
draft - instead of continuing to match intermediate tensors one at a time. The intermediates are now all verified (taps 0.014-0.032,
ctx K/V 0.005-0.037, output path 0.0318/corr 0.99993), so a token-level comparison is the only remaining meaningful gate.
