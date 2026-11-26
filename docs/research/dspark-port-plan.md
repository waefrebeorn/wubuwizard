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
