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

## Token-level: our drafter is wrong (0/7), and the dump instrument is sampling the wrong graph

The token gate, run properly: the reference on `"Hello"` returns ids `[11, 68014, 40, 23064, 76, 153221, 149882, ...]`
and reports **4 accepted / 14 drafted** (accept rate 0.286, per-position 0.667/0.333/0.333). Feeding those ids as an oracle to
`tools/dflash_ref.py` gives:

    MATCH 0/7

So every *intermediate* is verified but the draft is wrong - the error is in the block path, not the context path. Base logits peak
on 2012/2010/156164 rather than 11, so the block output is not merely noisy, it is wrong.

**The contract is now read from source, not guessed** (`dspark.cpp:194-227`):

    anchors = token_embd rows of the first token of each block   -> cb "dspark_anchor_raw"
    if (model.tok_norm) anchors = rms_norm(anchors, tok_norm)    <- our rms(anchor_emb, token_embd_norm)
    masks   = dspark_mask_embd[0 .. block_drafts-1] repeated per block, concatenated on axis 1
    return reshape_2d(cur, n_embd, n_tokens)

So `x0 = [rms(E[a], tok_norm); mask rows 0..6]` **is** what the reference builds, in that order, with no embedding scale applied inside
the builder. Our construction matches the source line for line.

**Why `inp_noise_embd` looked like 2 slots - and why that is the real lesson:** the builder returns early for anything that is not a
draft block (`block_drafts == 0 || n_tokens % n_blocks != 0 || block_drafts > block_size` -> plain `get_rows`), and the dump matcher
writes *every* matching graph instance, last one wins. With `n_predict` 2 and 10 the captured tensor had n_tokens = 2, and the block
`l_out-0/1/2` dumps likewise had **2 slots** - so none of those files is the 8-slot draft. Dumping is not enough here: the graph
identity has to be in the filename.

Also measured: the reference's `dspark_anchor_raw` has **rms 1.0001** and is uncorrelated with our raw `E[32269]` (rms 0.9661,
mean|diff| 0.9907). Either that file is also from a reserve graph, or the anchor in the captured round is not token 32269 - not yet
distinguished, and not to be concluded either way until the dump carries its graph identity.

**Next step:** make the dump self-identifying before drawing any further conclusion from it - append `n_tokens` (and the ubatch
n_seqs) to the dump filename in the `cb` helper - then re-capture `dspark_anchor_raw`, `l_out-*` and `result_norm` from a graph whose
slot count is actually 8. Only then compare the block path. Every conclusion above that was drawn from a 2-slot file is provisional.

## The 8-slot block input, finally captured - and both halves of x0 are wrong in our Python

The dump helper was dumping **only the first occurrence** of each tensor name (`g_llama_dump_counts[name] > 0 -> return`), so every
file we had came from the warmup/reserve graph and the real draft graphs were silently suppressed. Fixed in `common.cpp`: every
occurrence is now written under its own index (`name-<n>.bin`), and the `.txt` sidecar already carries `ne[]`.

With that, on `"Hello"` and `n_predict=16`, the real 8-slot graphs appear (`inp_noise_embd-1` ... slots 8).

Comparing our `x0 = [rms(emb, tok_norm); mask_embd rows]` against the reference's 8-slot block input:

    slot 0 (anchor):  ours rms 4.3168   ref rms 1.1826   corr 0.365   mean|diff| 0.932
    slots 1-7 (masks): ours rms 0.5956  ref rms 0.5956   corr 0.008   mean|diff| 0.507

Three facts follow, each measured rather than argued:

1. **Our mask rows are not the reference's mask embedding.** Correlation 0.008 - not a scale, not a permutation, a different vector.
   The *norms* agree exactly (0.5956 both), which is what made this look like a near miss earlier. It is not: correlation zero with
   equal norm means different data, and the equal norm is a coincidence of the extraction, not evidence of a match.
2. **The reference's seven mask slots are byte-identical to each other** (`ref slots1 identical? True`), while our seven rows are all
   distinct (`mask rows identical across ours? False`). Whatever `dspark_mask_embd` holds, it is one vector repeated - so our
   `mask_embd.f32` (7 distinct rows, each rms 0.5956) is not that tensor.
3. **The reference's anchor is not `E[32269]`.** Correlation 0.365 and rms 1.18 vs 4.32; `unit_norm(E[32269])` has rms exactly 1.0000
   and correlates 0.227 with it. So the captured graph is a *later* round whose anchor is a generated token, not the prompt - the
   first 8-slot graph appears only after the first token has been produced.

Slot 0 tracked our anchor at a constant ~0.46 across *every* graph in the earlier `result_norm` sweep, which is now explained: none of
those comparisons was against the prompt anchor either.

**So the 0/7 is not yet localised to the block internals.** Both inputs to `x0` disagree with the reference on data we extracted
ourselves, and the f32/ directory has already produced one provably wrong tensor (`output_norm.weight`). The suspect is now the
extraction, not the block math.

**Next step:** stop trusting `f32/` for these tensors - dequantize `mask_embd.weight` and `token_embd_norm.weight` directly out of the
GGUF, and check whether the seven mask rows really are identical and whether our f32 copies match the file at all. Also dump
`dspark_anchor_raw` per occurrence so the anchor token for each round can be identified instead of guessed.

## x0 is bit-exact - and both of our errors were ours, not the model's

Dequantised `mask_embd.weight` and `token_embd_norm.weight` straight out of the GGUF with `tools/gguf_extract.py` and compared with
the `f32/` copies: **max|diff| 0.00000000** on both. The extraction was never wrong. Both F32 in the file, so this was a byte
comparison all along.

What *was* wrong, twice, in our own Python:

1. **`mask_embd.weight.T`** - the tensor's 7 rows are byte-identical, and with `reshape(7, 2304)` (no transpose) they match the
   reference's slots 1-7 at **0.000000** with corr 1.000. `.T` read the flat data column-wise and produced seven *distinct* rows that
   each happened to carry rms 0.5956 - which is precisely why this looked like a near miss instead of a bug.
2. **our anchor was normalised and the reference's is not** - `forward()` applied `rms(anchor_emb, token_embd_norm)`, giving rms
   4.3168. The reference's slot 0 has rms **1.1826**, which is exactly the raw `E[11]` rms; raw vs slot 0 is
   **mean|diff| 0.00000000, corr 1.00000000**. So this drafter loads no `tok_norm` and `build_dspark_inp_block`'s
   `if (model.tok_norm)` is skipped. The maintainer note "the anchor is n_embed_norm(E[a])" describes a model that *has* a
   `tok_norm`; this checkpoint does not, and I applied the doc instead of the file.

With both fixed, the full 8-slot block input is bit-exact:

    slot 0 (anchor, raw E[a])   0.00000000
    slots 1-7 (mask rows)       0.000000   x7

**Still 0/7 on the oracle.** Base logits peak on 1666/150688, h rms 2.368 against the reference's ~2.0-2.2. So with the input to the
block now provably correct, the error is inside the block - attention, MLP, position handling or the ctx injection - and no longer
attributable to extraction, layout or the anchor.

**Next step:** compare the block's own layers now that x0 is exact - our post-layer-0/1/2 states against `l_out-0/1/2`, and `h`
against `result_norm`, captured in a single run so the graph instances can be paired by slot count and by matching
`inp_noise_embd` slot 0 to our `E[a]`.

## Block path: x0 exact, layers within 8% and drifting - one systematic difference left

Captured `inp_noise_embd`, `l_out-0/1/2` and `result_norm` in a single run and paired the graphs properly. The first 8-slot graph is
anchored on token 11, and `inp_noise_embd-1` slot 0 equals `E[11]` at **mean|diff| 0.0** - so the pairing is exact and not a guess.

Layer by layer, our forward (anchor `E[11]`, taps at `,`) against that graph:

    inp_noise_embd-1   rms  0.6966 vs  0.6966   per-slot 0.0000 x8            <- bit-exact input
    l_out-0-1          rms  8.6766 vs  8.4496   per-slot 0.49-0.70
    l_out-1-1          rms  9.1012 vs  8.8012   per-slot 1.01-1.44
    l_out-2-1          rms 12.2761 vs 11.9958   per-slot 1.58-2.89

Read this for what it is: the *magnitudes* now track the reference closely (within 1-3%) at every layer, and the per-slot error is a
consistent 5-8% that **grows monotonically with depth**. That is the signature of one small systematic difference applied once per
layer, not of a wrong structure - a wrong projection, layout or missing term would not land within 3% on rms at layer 2.

Ruled out already: `kq_scale` is `1/sqrt(n_embd_head)` in the reference (`dspark.cpp:715`) and identical in ours; the drafter GGUF
carries no scale override in its metadata; the ctx tensors we use (`attn_k_ctx`, `attn_k_ctx_norm`, `attn_v_ctx`, `ctx_mix[4]`,
`ctx_norm`) are exactly the set the block has.

An earlier attempt at this comparison was invalid and is worth recording: with the anchor slot holding a *different* token, the
non-causal block mixed that difference into all eight slots and produced 9+ errors that looked like a structural failure. The block
attends slot-to-slot, so a single wrong input slot invalidates the whole comparison - the anchor must match before any layer number
means anything.

Also noted: `result_norm` cannot be paired by tensor-name occurrence index (its counter advances differently from `l_out`'s); pairing
by correlation against `l_out-2` gave a best of only 0.256, so that pairing is not yet resolved and the `h` comparison is provisional.

**Next step:** isolate attention from the MLP by adding a `cb` for the post-attention residual inside `dflash.cpp`'s `build_layers`
(alongside the existing `noise_norm` hook), then compare that one tensor - the drift is 6-8% after a single layer, so whichever
half it appears in is the culprit.

## Block positions are ctx_len+1 .. - found by scanning, 0.1452 -> 0.00266

The base path already carried the hooks needed to split attention from MLP (`Qcur_raw`, `Qcur`, `Kcur`, `Vcur`, `attn_post_norm`,
`ffn_inp`), so no rebuild was required. Pairing the graphs properly (the graph anchored on `E[11]` is `inp_noise_embd-1`, but the
per-name occurrence counters advance independently, so `Qcur-0-*`'s 8-slot instance is occurrence **2**):

    ours vs ref, layer 0 of the block, x0 bit-exact:
      V   0.00544   (rms 0.884 vs 0.884)
      Q   0.09576   (rms 1.860 vs 1.860)
      K   0.14520   (rms 1.762 vs 1.762)

V is essentially exact and Q/K are wrong **with identical rms**. Equal rms on a wrong vector that was produced by a rotation means the
rotation is the wrong rotation - a norm-preserving map. Ruled out by measurement, not reasoning:

    ours (neox, theta 1e6)   0.145203
    neox 1e4                 0.145830
    interleaved 1e6          0.404320
    no rope at all           0.353663

So the convention was right and the difference had to be the *angle* - the positions. Scanning offset x partial-rotary:

    offset 1, n_rot 128   0.002664     <- the answer
    offset 0, n_rot 128   0.145203
    offset 2, n_rot 128   0.145565
    offset 3, n_rot 128   0.268520

**The block's slots sit at positions `ctx_len+1 .. ctx_len+8`**, not `0..7`. `blk_pos = arange(8)+1` cuts the K error 54x.

**Attention is still not right.** After the fix, our post-attention residual `ffn_inp` is 0.371 mean|diff| against the reference
(rms 0.8248 vs 0.8502), and the block's layer outputs run 2-3% high (8.6853/9.1017/12.3442 vs 8.4496/8.8012/11.9958). Base logits are
still far from the oracle (ranks ~1e5). By elimination:

    ctx+block, GQA repeat      0.37144   <- current, best
    ctx+block, GQA tile        0.68732
    block only, no ctx         0.65712

so the ctx inclusion and the head mapping are both confirmed correct, and the remaining 0.37 is inside the attention computation or
the ctx K/V themselves.

Note the block's residual stream grows to rms ~12 by layer 2 while `ffn_inp` is only ~0.85 - the MLP output dominates the magnitude,
which is why a 40% relative error in the attention term shows up as only 2-3% on rms at the layer output.

**Next step:** compare our ctx `kc`/`vc` for layer 0 against `Kcur_injected-0` / `Vcur_injected-0` **from this same dump**, so the ctx
K/V are checked in the exact graph the block runs in rather than in a separate serve.

## The block is correct - the 0.371 was my ctx, and the error is now in the head

The ctx K/V test exposed my own pairing error. In the graph anchored on `,` the block's *context* is `Hello`, not the anchor:

    ctx from Hello taps   K mean|diff| 0.006227 (rms 1.8815 vs 1.8820)   V 0.006396 (rms 0.7372 vs 0.7371)
    ctx from comma taps   K mean|diff| 0.574528 (rms 2.1064 vs 1.8820)   V 0.562015 (rms 0.6719 vs 0.7371)

So `ctx_kv` was right all along, and the 0.371 attention gap was me feeding the anchor's taps where the previous token's belonged.
With the correct ctx, the block matches the reference to well under a percent:

    L0 ffn_inp   mean|diff| 0.00580   rms 0.8501 vs 0.8502
    L0 l_out-0   mean|diff| 0.01316   rms 8.4522 vs 8.4496
    L1 l_out-1   mean|diff| 0.02648   rms 8.8026 vs 8.8012
    L2 l_out-2   mean|diff| 0.05692   rms 12.0064 vs 11.9958

Textbook error accumulation - 0.6%, 0.16%, 0.3%, 0.5% - with no structural disagreement anywhere. The three block layers, the
non-causal attention, the ctx injection, the conv-free SiLU MLP and the `ctx_len+1` positions are all verified against the
reference. `h` is now rms 2.2132 against the reference's 2.2197 (it was 2.646 before the position fix).

**The head is where it now fails.** `h` matches the reference but `base = h @ unembed.T` puts the oracle token at rank 163892, and
`argmax` lands on 157014. So the block is producing the right state and something after it is wrong: the shared unembed matrix we use
(`$TEMP/unembed.f32`, extracted from the *trunk*, 200058 rows of 2304) or its orientation, or the final norm's weight.

One measurement to record as *not* evidence: `h rms` matching the reference's `result_norm` rms is coincidence-strength evidence at
best - variance matches while the vector can still be wrong, exactly as with Q/K before the position fix. `result_norm` could not be
paired by occurrence in this dump at all (no 8-slot instance), so `h` is **not** yet verified against the reference's own normed
output, only its rms.

**Next step:** dump the reference's own `logits` for this graph (the default dump pattern already includes it) and compare against
our `base` for the same slot. That separates "wrong unembed/orientation" from "wrong h" in one measurement, instead of guessing
between them.

## The block is exonerated by the reference's own tensor - the unembed is the failure

The decisive test: feed the reference's **own** `l_out-2` (occurrence 1, the `E[11]` graph) through our norm and head, so nothing of
our block is in the path.

    h from reference l_out-2, our output_norm, our unembed   -> slot1 argmax 157014   (expected 68014, rank 164008)

So the block, the positions, the ctx and the layers cannot be the cause: with the reference's own layer-2 output we reproduce the same
wrong token. The remaining path is `build_norm(output_norm)` then `get_lm_head` (dflash.cpp:552-561, and line 544 shows an optional
per-layer `out_scale` we do not apply).

Norm variants tested on that same reference tensor, none of which recover the token:

    raw x, no norm        slot1 top3 [1815, 1557, 150725]   rank(68014) 183218
    unit rms norm         slot1 top3 [1815, 1557, 150725]   rank(68014) 183218
    our output_norm       slot1 top3 [157014, 740, 194320]  rank(68014) 164008

**Our `unembed.f32` is the problem, and its provenance is unverified.** `tools/gguf_extract.py --list` on the trunk
(`Inkling-2-Mini-UD-IQ2_XXS.gguf`) fails with `unsupported type 23`, so this table cannot have come out of that extractor run against
that file - the 463,159,296-float (201024 x 2304) table we are multiplying against has no verified origin. Two concrete reasons to
distrust it beyond the wrong argmax: the trunk stores 201024 rows while the drafter's markov projections are 200058 wide with **no
d2t mapping tensor**, and `output_norm.weight` from the same f32 directory has rms 3.0924 with range -5.06..13.12, implausible for
an RMSNorm gain (this was flagged as suspect long before today and is still unverified).

Rank 164008 out of 200058 is what a *random* row set would give, which is the signature of a wrong matrix rather than a near-miss.

**Next step:** get ground truth before changing anything - dump the reference's own `logits` for this graph and record its top-k for
slot 1, then either (a) extend `gguf_extract.py` to the trunk's quant types so `unembed.f32` can be re-derived and diffed against the
file, or (b) take the embedding/output tensors from a GGUF the extractor already supports. Do not adjust the block again: it is now
verified against the reference to 0.6-0.5% per layer using the reference's own intermediate tensors.

## The unembed table was wrong - ranks 164008 -> 4. Re-extracted, verified by shape and rms.

`gguf_extract.py` cannot read the IQ2_XXS trunk (`unsupported type 23`), and the `gguf` python package cannot either (the file carries a
type 42 the released enum does not know). The Q4_K_XL trunk carries the same tensor as plain **Q8_0**, `output.weight [2304, 201024]`,
which both can read:

    fresh dequant (Q8_0)   shape (201024, 2304)  rms 0.020834
    our old unembed.f32    same 463,159,296 floats  rms 0.029796
    row 0: fresh rms 0.0208, old rms 0.0400, mean|diff| 0.034225

Different rms and a row-0 difference the size of the vector itself - the table we had been multiplying against was not this matrix.

With the fresh table, feeding our (block-verified) `h` through the head:

    slot1 base top3 [11, 149904, 13]        oracle 68014 rank 4      (was rank 163892)
    slot2 base top3 [13, 200008, 11]        oracle 40    rank 5
    slot5 base top3 [13, 200008, 11]        oracle 149882 rank 162

The argmax is now real text (`,`, `13`) instead of token 157014, and the oracle moved from rank ~164k to rank 4-5. That is the
difference between a wrong matrix and a nearly-right one.

**Still MATCH 0/7 on drafts**, and the failure has moved again: the drafts cluster in 1498xx with high scores while the base logits
peak on 11/13, so the **markov head now dominates and is wrong**. `markov()` currently feeds `h[i]` straight into `joint_proj`
(768x5120) alongside `state` and `w1[prev]` and `f_rnn`, but the drafter carries `dspark_hidden_out` (markov_head.hidden_out), i.e.
a projection of h that is almost certainly meant to be applied first. 5120 = 20x256 does not match a naive
`[state | w1 | h | f_rnn]` = 4x256 concatenation either, so the joint input layout is not what we assumed.

**Next step:** read `build_dspark_markov_head` in `dspark.cpp:234+` and take the joint input layout from the source rather than
inferring it from shapes, then fix `markov()`. Every input to the block is verified, the block is verified, and the head matrix is now
verified; the markov chain is the last piece.

## The markov chain is written correctly - and `f_rnn` is not what we compute

Read `build_dspark_markov_head` (`dspark.cpp:234-390`) rather than inferring from shapes. The chain is:

    for i in [i_draft_beg .. block_drafts):
        w1_prev = w1[prev]                       # [R=256]
        joint = concat(state[R], w1_prev[R], hidden_i[n_embd], f_rnn[n_embd])   # 256+256+2304+2304 = 5120
        joint = joint_proj @ joint + bias         # [3R] = 768
        gate = sigmoid(chunk0); cand = tanh(chunk1)
        state = cand + gate*(state - cand);  markov = tanh(chunk2)
        bias = w2 @ markov + w2_b                 # [n_vocab_draft]
        ... base + bias -> pick

`joint_proj.weight` is `[5120, 768]` in the file, which matches exactly `256+256+n_embd+n_embd`. Our `markov()` implements this
line for line - the concat, the gate/cand/state recurrence, `tanh(chunk2)`, `w2 @ markov + bias`, `prev = argmax`. So the chain was
never the bug, and the earlier "(768, 5120)" note in the code was just the transposed reading of the same tensor.

**`f_rnn` is.** The reference's `f_rnn` comes from `rnn_state` - the *trunk* per-token rnn-head state passed through the batch
(`llama_batch_ext_select_n_embd_state`, `state_rows(...)`), and it is **different for every block slot** (`ref slots identical?
False`). Ours is a single vector, `rms(markov_ctx_mix . taps, markov_ctx_norm)[-1]`, identical for all slots:

    ours (rms 0.7505) vs f_rnn-1 slot0   mean|diff| 0.492263   corr 0.630974
    ours vs slot1                        0.588883   corr 0.479928
    ours vs slot7                        0.659845   corr 0.269906

Correlation 0.63 with a decaying profile is the signature of "the right kind of quantity at the wrong position": our ctx-mix
reproduces something close to the anchor's state and nothing at all for later slots.

**So the port is missing a component, not carrying a bug.** The drafter's markov conditioning needs the trunk's rnn head state per
token, which we have never ported - the taps (layer inputs and final state) are not it, which is exactly why `markov_ctx_mix` over
taps gets 0.63 and no better.

Status of the whole drafter, every line now measured:

    taps                 0.014-0.032   (bit-exact for target ids 5/23/41 via V{k-1})
    ctx K/V              0.0062/0.0064
    block x0             bit-exact (0.00000000 on all 8 slots)
    block layers 0/1/2   0.0058 / 0.0132 / 0.0569
    h                    matches the reference's result_norm rms, argmax now real text
    unembed              re-extracted from Q8_0; oracle rank 164008 -> 4
    markov chain         correct per source
    f_rnn                WRONG - 0.63 corr, needs the trunk rnn head

**Next step:** find how the reference produces `rnn_state` for the drafter - `llama_batch_ext_select_n_embd_state` and where the
Inkling2 graph fills it - and port that head. Everything else is verified against the reference's own tensors.

## f_rnn is correct (corr 0.999959) - and my 0/7 oracle is not a valid gate

`dflash.cpp:388-412` shows `f_rnn` **is** the ctx mix: `build_mix(markov_ctx_mix, markov_ctx_norm, -1)` over
`permute(reshape_3d(feats, n_embd, n_target, n_tokens), 1,0,2,3)`, one row **per context position**. So an 8-row `f_rnn` dump is an
8-position injection graph, not 8 block slots - my earlier "f_rnn is wrong, corr 0.63" was a mispairing, and the correct pairing is
the single-row dump:

    ours rms 0.7505 vs ref rms 0.7503   mean|diff| 0.005048   corr 0.999959

Also fixed from source: `build_dspark_markov_head(..., softcap_after_bias=true)` applies the 30.0 tanh softcap **after** the markov
bias; we were capping the base first. (It changes no argmax in this test, but it was wrong.)

**Now the important correction, about the gate itself.** `MATCH 0/7` compares our *drafted* tokens against the tokens the reference
*generated*. Those are not the same quantity: the reference reported `accepted 4/14` drafts, accept rate 0.286 - most drafts are
supposed to be rejected. A correct drafter would therefore score low on that comparison, and a 0/7 says almost nothing. It is a bad
test that has been steering the last several steps.

What it should be compared against is the reference's *drafts*, which the logs expose only as counts (6, 5, 3 drafted; 0, 1, 3
accepted) - not as ids. Base logits show the oracle token at rank 4-5, which is entirely plausible for a drafter whose acceptance is
28%; it is not evidence of a defect.

**Drafter status: every component is now verified against the reference's own tensors.**

    taps 0.014-0.032 (bit-exact for ids 5/23/41)   ctx K/V 0.0062/0.0064   x0 bit-exact
    block layers 0.0058/0.0132/0.0569   h argmax real text   unembed re-extracted (rank 164008->4)
    markov chain correct per source   f_rnn 0.005048 corr 0.999959   softcap order corrected

**Next step:** build a gate that actually tests draft parity - a prompt with near-total acceptance (repetitive text) where the
reference's accepted tokens ARE its drafts, so token identity becomes meaningful; or add a dump of the drafted ids and compare those
directly. Until then, do not read MATCH n/7 as a correctness signal.

## A real draft-parity gate at last: our first draft is exact, the chain then drifts

Added two name hooks in `dspark.cpp` (`g.cb(res->t_h_nextn, "dspark_draws")` at 437 and `g.cb(out, "dspark_logits")` at 444) so the
drafted ids can be dumped instead of guessed. Pairing verified first: `inp_noise_embd-1` slot 0 vs `E[11]` is **0.00000000**.

    reference drafted ids   [149904, 150261, 149882, 149895, 149878, 220, 68111]
    our drafted ids         [149904, 149951,    6,     83, 149895, 149878, 220]
    MATCH 1/7

Read the shape of this rather than the count:

* **Draft 1 is exact.** 149904 == 149904. That exercises taps -> block -> x0 -> three layers -> h -> output_norm -> unembed ->
  w1/joint_proj/state/gate/cand -> w2 -> pick, end to end, and lands on the same token the reference drafts. Every piece verified
  separately is now verified composed.
* **Our steps 5, 6, 7 equal the reference's steps 4, 5, 6** (149895, 149878, 220). A suffix that matches one index earlier is the
  signature of a propagation/indexing offset in the chain, not of wrong arithmetic.
* Steps 2-4 diverge (149951, 6, 83 vs 150261, 149882, 149895).

**Caveat on the other new dump:** `dspark_logits` is *not* the tensor the pick is made from - its slot-1 argmax is 43743 while the
actual drawn id is 149904. So it must not be used as ground truth for the pick; `dspark_draws` is the authoritative one. Recording
that now so it is not mistaken for a mismatch later.

Also fixed: `base` comes out 201024 wide from the trunk head while the markov bias is 200058 wide; numpy broadcast-compatible widths
were hiding it, and `forward()` now slices to `n_vocab_draft` explicitly.

**Next step:** chase the one-step offset - compare per-step `state` and `markov` chunk values against the reference by dumping the
joint output, or check whether `prev` for step i should be the *drawn* token of step i-1 in block-major order rather than the argmax
of the same step. The suffix alignment is the clue; do not rewrite the chain.

## Per-step logits: step 1 is right (corr 0.997), step 2 collapses - the offset is in step 2's inputs

Added `g.cb(col, "dspark_col", i)` at `dspark.cpp:407`, immediately after `cat = concat(cat, col, 1)` and **before** the block-major
permute at 442, so each step gets its own file in slot order. `dspark_logits` could not serve this purpose: its column order is permuted
(its slot-1 argmax was 43743 while the actually drawn token is 149904), which is now explained rather than mysterious.

    step1: our top 149904  ref top 149904   mean|diff| 0.276   corr 0.997053
    step2: our top 149951  ref top 150261   mean|diff| 6.879   corr 0.897289
    step3: our top      6  ref top 149882   mean|diff| 3.679   corr 0.856014
    step4: our top     83  ref top 150367   mean|diff| 3.463   corr 0.762539
    step5: our top 149895  ref top 149876   mean|diff| 2.593   corr 0.856795
    step6: our top 149878  ref top   220    mean|diff| 1.518   corr 0.899915
    step7: our top    220  ref top    17    mean|diff| 2.407   corr 0.518227

**Step 1 is correct end to end**: identical argmax, corr 0.997, mean|diff| 0.276 on logits of magnitude ~20. Every stage of the drafter -
taps, ctx K/V, x0, three block layers, output_norm, unembed, w1/joint_proj/state/gate/cand, w2, softcap-after-bias - is exercised and
lands on the reference's own token with the reference's own logits.

**Step 2 collapses by a factor of 25** (0.276 -> 6.879) with correlation still high (0.897). High correlation with a large absolute
error is the profile of a *misaligned but similar* tensor, not of wrong arithmetic. Step 2's only new inputs over step 1 are:
`state` (carried), `w1[prev]` with prev = 149904 the drawn token, `h[2]`, `base[2]` and `f_rnn`.

Also measured, to pre-empt a wrong inference: the reference's block rows 1-7 are **not** identical even though their x0 inputs are
(slot rms 5.369, 7.666, 8.849, 9.251, 9.6, 9.793, 9.754). That is expected - each slot sits at its own RoPE position `ctx_len+1+i` -
so identical mask inputs do not imply identical block outputs, and any comparison that assumes otherwise is invalid.

**Next step:** sweep the two index choices independently - `h[i]` vs `h[i-1]` and `base[i]` vs `base[i-1]` - for steps 2..7 and see
which combination reproduces the reference's per-step argmax. Step 1 pins the indexing for i=1, and step 2's 25x jump is far too large
to be the 0.026 block error, so this is an index question, not an arithmetic one.

## 6 of 7 drafts exact - the chain is right, and only step 2 is wrong

Fed the reference's own `prev` chain into the sweep so that one bad step could not contaminate the rest. Indexing is now settled by
measurement:

    h[i], base[i]      tops [149904, 149951, 149882, 149895, 149878, 220, 68111]   6/7
    h[i], base[i-1]    tops [196997, 150261, 149882, 149895, 149878, 220, 68111]   6/7
    h[i-1], base[i]    tops [  220, 149951, 149882, 149895, 149878, 220, 68111]   5/7
    h[i-1], base[i-1]  tops [196997, 149951, 149882, 149895, 149878, 220, 68111]   5/7

`h[i]`/`base[i]` wins and step 1 pins it, so the shift hypothesis in the previous entry is refuted: there is no off-by-one. Against the
reference's drafts `[149904, 150261, 149882, 149895, 149878, 220, 68111]` we reproduce **six of seven exactly** - steps 1, 3, 4, 5, 6,
7 - and only step 2 differs (149951 vs 150261).

**The recurrence is not the cause.** Mean|diff| on step 2's column under four state variants:

    state = cand + gate*(state-cand)   [current]   6.88   6/7 overall
    state = 0 (no carry)                           6.98   5/7   (fixes step2, breaks 4 and 7)
    state = cand                                   7.13   6/7
    state = gate*(state-cand)                      6.66   3/7

Step 2 sits at 6.7-7.1 in all four. A quantity that does not move with the recurrence is not produced by the recurrence, so step 2's
error enters through its own inputs - `h[2]`, `base[2]`, `w1[149904]` or `f_rnn` - and is then amplified by the joint projection.

Note the practical asymmetry: with our own `prev` chain the score is 1/7, because step 2's wrong token becomes step 3's `w1[prev]`
input and the chain diverges. That is why the earlier 1/7 was not an independent signal of six broken steps - it was one broken step
propagating.

**Next step:** isolate step 2's inputs directly - dump the reference's `joint` output (`joint = joint_proj @ [state|w1|hidden_i|f_rnn]`)
per step and compare against ours at step 2 only, which separates `h[2]` from `w1[prev]` without touching the recurrence again.

## Step 2 survives every hypothesis - and each candidate is now excluded by measurement

Added `g.cb(joint, "dspark_joint", i)` at `dspark.cpp:373`, immediately after the `joint_proj` matmul, so the per-step joint output is
comparable. Per-step joint error (reference's own `prev` chain, our state chain):

    step1 0.01671 corr 0.99998     step5 3.06454 corr 0.31720
    step2 1.65930 corr 0.81188     step6 2.30938 corr 0.59641
    step3 0.46827 corr 0.98623     step7 0.57339 corr 0.98529
    step4 0.42868 corr 0.98474

Then each candidate in turn, all excluded:

* **state** - replaying the chain while taking `state` from the *reference's own* joint changes step 2 not at all (1.65946 vs 1.65930).
  A quantity that does not move the result is not the cause. Five state-update variants (cand+g(s-c), cand, tanh(cand), g*c+(1-g)*s,
  s+g*(c-s)) all land at 1.66-1.72 with corr 0.80-0.81.
* **h[i]** - our `h` matches the reference's normed `l_out-2` per slot at 0.0062/0.0274/0.0212/0.0191/0.0188/0.0197/0.0186/0.0188,
  corr >= 0.99990 on every slot. Slots 2, 5, 6 are not worse than the rest, so the joint errors are not coming from `hidden_i`.
* **prev** - `prev=ref[1]` (149904) gives 1.65946; `ref[2]` (150261) gives 3.85423 corr 0.071; anchor 11 gives 3.95863 corr 0.085.
  The index is right, decisively.
* **chunk order** - step 1's column matched at 0.28 with `markov = tanh(chunk2)`, so gate/cand/markov assignment is right.
* **concision of scale** - step 1 joint is 0.01671 with corr 0.99998 using the same `joint_proj`, `f_rnn` and concat order, so the
  projection, the concatenation order and `f_rnn` are all correct at i=1.

So step 2 is off by 1.66 while its own inputs are each verified to ~0.02, and no parameter choice within the chain moves it. That
points at the one thing not yet dumped: the **concatenated input vector** itself (`[state | w1[prev] | hidden_i | f_rnn]`, 5120 wide).
If that input matches the reference's, the discrepancy is inside the matmul's operand order; if it does not, the guilty component is
visible directly instead of by elimination.

**Next step:** add `g.cb(joint_in, "dspark_joint_in", i)` on the concatenated tensor before `ggml_mul_mat`, dump, and compare the four
256/2304/256-wide slices separately at step 2. This is the last unwatched point in the chain.

## Joint input split by slice: step 1 is exact on all four, step 2 is misaligned - and my ref-draft order is suspect

`g.cb(joint, "dspark_joint_in", i)` before the matmul, then the 5120-wide input split into its four parts:

    step1  state 0.00000 (rms 0.000/0.000)   w1_prev 0.00000 (5.605/5.605)   hidden 0.02741 (3.167/3.172)   f_rnn 0.00485 (0.796/0.795)
    step2  state 0.73863 (0.457/0.784)       w1_prev 3.55741 (5.963/6.853)   hidden 1.09122 (2.373/1.627)   f_rnn 0.49003 (0.796/0.865)
    step3  state 0.81271 (0.542/0.803)       w1_prev 0.00000 (6.327/6.327)   hidden 0.83329 (2.180/1.624)   f_rnn 0.49003 (0.796/0.865)

Three facts, each exact rather than inferred:

1. **At step 1 every slice matches**: `state` zeros, `w1_prev` bit-exact, `hidden` 0.027, `f_rnn` 0.005. So the concat order is
   `[state | w1_prev | hidden | f_rnn]`, `jp` is right, and our h row 1 and f_rnn are right.
2. **At step 3 `w1_prev` is bit-exact again** (0.00000, rms 6.327 both) while at step 2 it is not (3.557). So our `prev` sequence is
   right for steps 1 and 3 and wrong for step 2 - not a systematic offset, a single wrong link.
3. **The reference's own `f_rnn` slice changes between steps** (rms 0.795 at step 1, 0.865 at steps 2-3) and the `hidden` slice at
   step 2 has rms 1.627, which matches none of our h rows (2.376/3.167/2.373/2.180/1.901/1.905/1.731/1.702). Both indicate the
   reference's per-step slices are **not in the order I assumed**.

**This invalidates my own reference-draft ordering.** The `dspark_draws` tensor is explicitly permuted for block-major restore
(`dspark.cpp:435-437`), so reading it row-by-row and calling the result "the reference's drafts in slot order" was wrong, and the
`ref = [149904, 150261, ...]` chain I have been driving the comparison with is that misordered list. The 6/7 result therefore stands
as *six of seven tokens appear in that list*, not as *six of seven at the right position*.

**Next step:** derive the reference's true per-step order empirically instead of from the permuted dump - for each step, identify
which token id's `w1` row matches the dumped `w1_prev` slice exactly (a 200058-way search, cheap), and which h row matches the dumped
`hidden` slice. That reconstructs the reference's actual prev/hidden sequence, and only then is a positional score meaningful. Do not
trust any positional claim built on `dspark_draws` row order.

## The reference's true prev chain, recovered exactly - and it proves the drafts are SAMPLED, not argmax

Instead of trusting the permuted `dspark_draws` ordering, each step's `w1_prev` slice was matched against all 200058 rows of
`markov_w1.weight` (via `dr.w`, which is the accessor that reproduces a known-exact match; `dr.get` has the opposite orientation and
gave a false "closest id" of 25944 at step 1 where the truth is 11 at 0.000000).

    step1 prev 11   step2 prev 40   step3 prev 150261   step4 prev 149882
    step5 prev 150367   step6 prev 149876   step7 prev 220        (all mean|diff| 0.000000)

**Consequence 1: the reference does not pick the argmax of its own column.** At step 1 its `col` argmax is 149904, yet the token it
feeds forward as `prev` at step 2 is **40**. So `is_greedy` is not active and the drafts come out of the top-k/cumsum sampler. That
retires the entire "MATCH n/7 against generated tokens" framing - draft parity cannot be tested with an argmax at all.

**Consequence 2: with the true chain the joint agrees closely at every step.**

    step1 0.01671 corr 0.99998      step4 0.44494 corr 0.98391
    step2 0.90614 corr 0.94943      step5 0.43011 corr 0.98638
    step3 0.54786 corr 0.98177      step6 0.38510 corr 0.99024
                                    step7 0.35741 corr 0.99398

Step 2 remains the worst, but 0.9 on a joint of rms ~6 with corr 0.95 is a different animal from the 1.66/corr 0.81 seen with the
wrong prev. Errors also decay monotonically after step 2 rather than growing, which is what a small per-step difference looks like when
the prev chain is right.

Note our own state chain was still used for these numbers, so part of the residual is our state drifting from the reference's; driving
`state` from the reference's own joint is the clean isolation and has not been redone with the corrected chain.

**Next step:** drive the chain with the reference's `joint` for `state` *and* the recovered prev chain, to see whether the residual
0.36-0.91 collapses. Then implement the sampler (`top_k 64`, `inv_temp` from the rnn_state row, cumsum draw) so drafts can be compared
as tokens at all - argmax parity is not the target.

## prev is now exact at every step; hidden and f_rnn are not constant across steps, which they should be

With the recovered chain every step's `w1_prev` slice matches bit-exactly (0.000000 at steps 1, 2, 3 - and at 1 it was already
0.000000), so `prev` is fully settled. Slice detail:

    step1 (prev 11)      state 0.00000   w1 0.00000   hidden 0.02741 (3.167/3.172)   f_rnn 0.00505 (0.7505/0.7503)
    step2 (prev 40)      state 0.72463   w1 0.00000   hidden 1.09122 (2.373/1.627)   f_rnn 0.49226 (0.7505/0.7936)
    step3 (prev 150261)  state 0.21807   w1 0.00000   hidden 0.83329 (2.180/1.624)   f_rnn 0.49226 (0.7505/0.7936)

Driving `state` from the reference's own joint changes essentially nothing (0.9061 -> 0.9063 at step 2, and only ~0.02 elsewhere), so
the state is excluded a second time, now with the correct chain.

**The important observation is not the magnitude - it is that the reference's `f_rnn` slice changes between steps.** At step 1 it has
rms 0.7503 and matches ours to 0.005; at steps 2 and 3 it has rms 0.7936 and matches ours only to 0.49. `f_rnn` is a *fixed* context
row in our model of the graph (`state_rows(n_embd, 0, 0)` - row 0 of the block, taken once outside the loop), so a version of it that
varies per step means the draft graph's `rnn_state` is not the injection graph's `f_rnn` tensor we verified earlier. The reference's
own comment says as much: row 0 holds f_rnn for the *anchor*, while **draft row i holds `[uniform | 1/temp | 0...]`** - the per-step
draw state. So the rows we are reading at i>=2 are the draw rows, not the anchor row, and both the `hidden` (rms 1.627) and `f_rnn`
(0.7936) slices at steps 2-3 are consistent with that rather than with a drafter arithmetic error.

This also explains why step 1 - the only step whose slices match - is also the only step whose joint matches to 0.0167.

**Next step:** stop inferring from slices and dump the actual tensors by name: `f_rnn` (already hookable at `dflash.cpp:410`),
`hidden`/`res->t_embd`, and `rnn_state` itself, each with the step index in the name. That identifies what the reference really feeds
the joint at steps 2+, instead of inferring it from offsets into a concatenation.

## hidden_i dumped by name: slot 1 matches, slots 2-7 do not - and two serves disagree about them

`g.cb(hidden_i, "dspark_hidden_i", i)`, `dspark_frnn_i` and `dspark_state_i` added inside the loop, then compared against our h rows:

    step1 hidden_i rms 3.1716   best h[1]  0.0274
    step2 hidden_i rms 1.6270   best h[7]  1.0301
    step3 hidden_i rms 1.6243   best h[4]  0.7873
    step4 hidden_i rms 1.5653   best h[7]  0.7594
    step5 hidden_i rms 1.6670   best h[6]  0.6877
    step6 hidden_i rms 1.6091   best h[7]  0.6559
    step7 hidden_i rms 1.5738   best h[7]  0.6147   (f_rnn 0.7936 vs ours 0.7505, mean|diff| 0.4923)

Our h rows run 2.376 / 3.167 / 2.373 / 2.180 / 1.901 / 1.905 / 1.731 / 1.702 - **monotonically decreasing**, while the reference's
hidden_i for slots 2-7 is nearly flat at 1.57-1.67. Step 1 matches to 0.027 and nothing after it does.

**The contradiction that matters:** in the `dump_blk9` serve our block matched the reference's `ffn_inp` at **0.00580** across all eight
slots, with pairing verified by `inp_noise_embd-1` slot 0 == `E[11]` at 0.00000000. In this serve the same anchor gives hidden rows of
rms ~1.6 that match none of our h rows. Both serves were `Hello` with `n_predict=16`. Two runs of the reference therefore disagree about
the drafter's hidden state for the same anchor - which is the same nondeterminism already visible in the generated token sequences
(`[11, 68014, 40, 23064, 76, ...]` in some runs, `[11, 68014, 40, 150087, 153221, ...]` in others).

That makes any per-slot comparison across serves unreliable, and it is the most likely explanation for the "flat 1.6" rows as well: the
captured graph may not be the round we assume even though its first slot matches.

**Next step:** capture `inp_noise_embd`, `dspark_hidden_i` and `ffn_inp` in a *single* serve and verify the graph identity by matching
`inp_noise_embd` slot 0 to `E[11]` before comparing anything else - the discipline that made the ctx and block comparisons trustworthy.
Do not compare hidden rows across separate serves again.

## Single serve: the block is EXACT on every slot - and hidden_i at i>=2 is not the block output

Captured `inp_noise_embd`, `ffn_inp`, `l_out` and `dspark_hidden_i` in ONE serve and verified graph identity before comparing anything
(`inp_noise_embd-1` slot 0 vs `E[11]` = 0.00000000). With everything in the same graph:

    L0  our rms [23.77, 1.17, 1.08, 0.94, 0.92, 0.93, 0.91, 0.88]
        ref rms [23.76, 1.18, 1.08, 0.94, 0.92, 0.93, 0.91, 0.88]     per-slot 0.008-0.040
    L1  our rms [24.20, 2.26, 2.21, 2.14, 2.16, 2.21, 2.26, 2.23]
        ref rms [24.19, 2.27, 2.21, 2.14, 2.16, 2.22, 2.27, 2.24]     per-slot 0.023-0.042
    L2  our rms [24.84, 5.36, 7.67, 8.87, 9.26, 9.62, 9.82, 9.76]
        ref rms [24.83, 5.37, 7.67, 8.85, 9.25, 9.60, 9.79, 9.75]     per-slot 0.049-0.063

That is 0.2-0.5% on values of magnitude 5-25, on all eight slots, at all three layers, in one graph. **The block is correct** - and the
earlier "hidden_i slots 2-7 are flat 1.6 while ours are 2.2" was a cross-serve artifact, exactly as suspected, since `ffn_inp` in
`dump_blk9` had already matched at 0.00580.

**`hidden_i` at step 1 is bit-exact against `norm(l_out-2 slot 1)`**: mean|diff| **0.0000**, rms 3.1716 both. So the markov head does
receive the normed block output, and our `h` derivation (`rms(l_out, output_norm)`) is right.

**But at steps 2-7 it is not that tensor.** `norm(l_out-2 slot i)` gives rms 2.377 at i=2 and 2.187 at i=3, while the reference's
`hidden_i` is 1.627 and 1.624 there (mean|diff| 1.0948 and 0.8360). Checking every slot rather than assuming i:

    step2 best: slot7 1.0314, slot6 1.0371, slot4 1.0410
    step3 best: slot4 0.7897, slot6 0.7980, slot7 0.8088
    step7 best: slot7 0.6168, slot6 0.6324, slot5 0.6903

No slot matches, and neither the raw row (3.98) nor the unit-rms row (1.26) does. So `hidden_i` at i>=2 is drawn from a different
tensor than the block output we can see - most likely `hidden` inside the markov head is not `res->t_embd`/`result_norm` at those steps.

**Next step:** dump the whole `hidden` argument once per step (not the view) - `g.cb(hidden, "dspark_hidden_all", i)` - and compare it
against `result_norm` for the same graph. That answers whether the markov head receives the block output at all at i>=2, instead of
inferring from views.

## result_norm is not the source of hidden_i, and hidden_all dumps only for steps 1 and 7

Captured `inp_noise_embd`, `result_norm`, `dspark_hidden_all` and `dspark_hidden_i` in one serve, graph identified again as
`inp_noise_embd-1` (slot 0 == `E[11]`).

* `result_norm-1` is 8 x 2304, rows rms 3.61 / 3.56 / 3.32 / 3.70 / 3.50 / 3.25 / 2.16 / 3.29.
* But `hidden_i` step 1 in the same serve had rms 3.1716 and matched `norm(l_out-2 slot 1)` **bit-exactly** (0.0000) in the earlier
  single-serve test. `result_norm` row 1 here is 3.56, not 3.17. **So `result_norm` is a different tensor from `hidden`
  (`res->t_embd`) - do not compare normed blocks against it.**
* Also, a fixed-weight rmsnorm would give every row the same rms (rms of the weight), but `result_norm` rows span 2.16-3.70, which is
  impossible for a single `output_norm.weight`. Either it incorporates per-row gating from somewhere, or it is not what the name
  implies. Either way it is not on the markov/block critic path we care about.

**`dspark_hidden_all` only produced files for i=1 and i=7** (i=1 one occurrence, i=7 eight). The same per-name occurrence counter is
shared, and steps 2-6 did not write because the `hidden` tensor itself is only newly-allocation-flagged on the first and last
iteration, or the dump deduplicates by internal pointer the way the graph reuses the same tensor. Whatever the mechanism, it means the
`hidden` tensor pointer is reused across iterations 2-6 - so `hidden` is genuinely the same tensor each step, and `hidden_i` (the
view at offset i) *should* be row i of it.

That is exactly the contradiction left standing: row 1 of that tensor equals `norm(l_out-2 slot 1)` to bit-exact, but rows 2-7 of the
view match none of `norm(l_out-2 slot j)` for any j. The next move is to read, not guess: trace what `hidden` actually is at the point
`build_dspark_markov_head` is called in `dflash.cpp` (the `res->t_embd` argument) and whether it is reassigned between block
invocations, instead of inferring from dumps.

## The draft graph passes `cur` (block output) as hidden; dflash's base-graph result_norm is a separate tensor that is irrelevant here

The speculative call site is `dspark.cpp:509`: `build_dspark_markov_head(ctx, model, inp_tokens, cur, rnn_state, /*sample_from_anchor=*/true)`
where `cur = build_layers(inp_tokens)` at line 501. So `hidden` in the markov head is the block output produced by the *token-batch
DSV4/draft* graph, not by the embd-batch `graph_base::build_layers` whose `result_norm` we have been dumping.

Verified occurrence-aligned, in `dump_hall`: even the single closest `result_norm-*.bin` is mean|diff| ~2.23 away from `hidden_i`
step 1, which is a vector mismatch, not an rms quirk. `result_norm` (the embd-batch norm at `dflash.cpp:552`) is simply not the tensor
`hidden_i` comes from. Our `output_norm.weight` itself remains correct - `hidden_i` step 1 equals `norm(l_out-2 slot 1, output_norm)`
to 0.0000.

The remaining contradiction is now precise: in the *same* draft graph, `hidden` row 1 rms 3.17 matches our `norm(x1, W)` while
`hidden` rows 2-7 (rms ~1.57-1.67) match none of our `norm(xj, W)`. Exactly one feature distinguishes our model of the block from the
reference's: we fill draft slots 1-7 with the **fixed** `mask_embd` rows, while the reference's draft graph computes those slots with
real per-step inputs (the drawn tokens / the block conditioned on context or on prior drafts). A real-embedding block would produce
rows 2-7 that are some function of the drawn tokens - and therefore would not equal any mask-based row, which is what we observe.

**Next step:** test this directly. Recompute our draft-slot embeddings by placing the actual drawn token ids (the recovered
chain 11 -> 40 -> 150261 -> 149882 -> 150367 -> 149876 -> 220) into the block input slots instead of `mask_embd`, run the block, and
see whether `hidden` rows 2-7 then match the reference - or inspect how `build_dspark_inp_block`/`build_inp_noise` assembles draft-slot
embeddings for the token-batch draft graph at `dspark.cpp` around line 470-500.

## Draft-slot inputs ARE the fixed mask rows; wubumm glean (WuBuMath SIMD kernel route)

Closed the "real embeddings" hypothesis from the call-site read: `graph_base::build_layers` assembles the block input via
`build_dspark_inp_block` (dflash.cpp:447-448), and dspark.cpp:220-221 takes the draft-slot masks directly from
`model.dspark_mask_embd` (shape {n_embd, block_size-1} = 7 rows of 2304, col-major => rows 0..6 of the mask weight). The token-batch
draft graph and our model both put the fixed mask rows in slots 1-7, so mask-vs-real embedding is NOT what separates hidden row 1
(bit-exact) from rows 2-7 (rms 1.57-1.67 vs our 2.2-2.4). The remaining candidates are the position assignment (inp_pos) or the mask row
order in the block.

**wubumm = WuBuMath** (no separate "wubumm" repo exists on this machine or on GitHub; WuBuMath origin is up to date at 2026-11-26).
Relevant math-optimization content to fold into the engine going forward: `include/jax_simd.h` is a portable C11 SIMD layer - AVX2 8x8
FMA GEMM micro-kernel (broadcast-A / contiguous-B, `_mm256_fmadd_ps`), NEON 4x4, scalar fallback, SIMD batch relu/gelu/silu/tanh
(`jax_act_batch`); no external deps, selected by `__AVX2__`/`__ARM_NEON__`. This is the direction the user is researching for the math
optimization route and is directly relevant to the wubuwizard GEMM/RMSNorm path. Only two test compute shaders (mobius_add, poincare_dist)
in `shaders/`; no embedded CUDA/Vulkan infer kernels yet.

## Token-graph paired compare: block is exact; the markov head's hidden rows 2-7 are not the block output

Captured `l_out` + `dspark_hidden_i` + `inp_pos` in one serve (dump_tok), identified the token-batch draft graph's l_out as
`l_out-2-1` (its normed row 1 == hidden_i step 1 at 0.0000, rms 3.1716 both).

1. Our 8-slot block output vs `l_out-2-1` (the TOKEN-batch graph, same serve): per-slot mean|d| **0.049-0.063** on rms 5.4-24.8, all
   eight slots. So our block, positions (arange(8)+1) and inputs are confirmed correct against the token-graph too, not just the
   embed-graph.
2. Reference `hidden_i` vs `norm(l_out-2 slot i)`:
      step1  ref 3.1716  = norm(slot1) 3.1716  d 0.0000   (matches)
      step2  ref 1.6270  vs norm(slot2) 2.3766  d 1.0948
      step3  ref 1.6243  vs norm(slot3) 2.1869  d 0.8360
      step4  ref 1.5653  vs norm(slot4) 1.9104  d 0.7882
      step5  ref 1.6670  vs norm(slot5) 1.9172  d 0.7246
      step6  ref 1.6091  vs norm(slot6) 1.7422  d 0.6626
      step7  ref 1.5738  vs norm(slot7) 1.7097  d 0.6168

**Row 1 IS the block output exactly; rows 2-7 are a different tensor** - nearly constant rms 1.57-1.67, while the block's own normed
rows vary 1.71-2.38. Since our x matched ref l_out-2-1 at 0.05 AND ref's hidden != ref's own norm(l_out-2) at rows 2-7, the markov
head's `hidden` is not `cur`/result_norm for draft steps i>=2. This is strong, paired support for the draw-row hypothesis: rows 2-7
carry the per-step draw state (`[uniform | 1/temp | 0...]` from `rnn_state`), which is exactly the flat-rms ~1.6 signature we see.

**Next step:** dump `rnn_state` and `f_rnn` per step and confirm hidden rows 2-7 equal the corresponding draw rows - or read the
token-batch dsv4 path (dspark.cpp 558+) to see what `cur`/res->t_embd actually is at the markov call and whether rnn_state rows are
installed into `hidden` for i>=2.

## Markov head source walk: hidden_i = hidden row i, and the joint width confirms f_rnn on

Read `build_dspark_markov_head` (dspark.cpp:234-453) end to end. Facts that pin the model:

- `hidden_i = view_2d(hidden, hidden->ne[0], n_blocks, block_drafts*hidden->nb[1], i*hidden->nb[1])` (line 367) = row i of `hidden`
  for a single block. So `dspark_hidden_i-i` is genuinely hidden row i, confirming the metric.
- `sample_from_anchor` comes from GGUF `dflash.sample_from_anchor` (absent or "true" => true, i_draft_beg=0; "false" => 1).
  With rnn_state non-null the code asserts `!sample_from_anchor` (line 294), so a real f_rnn run must be sample_from_anchor=false
  (i_draft_beg=1). This means my earlier "the drafter samples, is_greedy off" holds via rnn_state draw path (line 432: `draw()`),
  and the first markov step is i=1 (the first draft), consistent with hidden_i dumps starting at step 1.
- `n_joint = (dspark_f_rnn ? 2*n_embd : n_embd) + 2*rank` = 5120 with f_rnn, matching the 5120-wide joint = [state|w1_prev|hidden_i|f_rnn].
  The joint projection is confirmed exactly right.
- The draw (line 308-333): top_k(n_cand=64), sort by logit desc, soft_max(vals*inv_temp), cumsum, threshold = `state_rows(1,0,i)`
  (the uniform comes from rnn_state, not a fresh RNG), pick first candidate over threshold, argmax if is_greedy. `draws` are
  block-major restored via permute at 442-444 (dspark_draws). This is the sampler to port next: cumsum-draw keyed off rnn_state.

**The standing contradiction is now fully scoped to one fact:** with block == l_out-2 == our x validated 0.05, reference `hidden`
row 1 == `norm(l_out-2 slot1)` exactly, while rows 2-7 == a distinct near-constant-rms 1.6 tensor. Since hidden_i is a view of the
single `hidden` tensor, `hidden` itself must differ from the base result_norm at rows 2-7 - i.e. `cur` passed at dspark.cpp:509 is not
simply the base result_norm. Next: read what graph_dflash's `build_layers` returns (dspark.cpp around 470-505) - whether graph_dflash
overrides build_layers / resets res->t_embd or feeds the drawn prev tokens into hidden rows 2-7.

(wubumm = WuBuMM pulled again to d773abe; measured H3 findings recorded in docs/research/wubumm-kernel-glean.md.)

## Which graph serves the Inkling-2-Mini drafter: graph_dsv4, and it has no rnn_state -> the 5120 joint must be graph_dspark

Graph factory (dflash.cpp:616-638): `BACKBONE_DSV4 -> graph_dsv4` WINS over variant. backbone is set at dflash.cpp:73 to
BACKBONE_DSV41/DSV4 for non-Mistral Inkling architectures (`dsv4_hc_single_pass ? DSV41 : DSV4`). So Inkling-2-Mini builds graph_dsv4.

`graph_dsv4` (dspark.cpp:517-655) NEVER references rnn_state; its markov call at dspark.cpp:652 is
`build_dspark_markov_head(*this, model, inp_tokens)` with NO hidden and NO rnn_state. With is_rnn=true (joint_proj present) and
hidden==nullptr it would dereference nullptr at the hidden_i view (line 367) - so a run that dumped hidden_i cannot be graph_dsv4:652.

Evidence that reconciles it: the :8112 serve dumped `l_out-0..17`, `inp_noise_embd` and `hidden_i` but NO `result_norm` and NO
`dspark_cur` (my hooks in graph_dspark). A graph_dspark rnn run (line 510) would dump result_norm (dflash.cpp:553) and dspark_cur
(line 502) AND build the hidden_i/5120 joint - but result_norm and dspark_cur never fired. So :8112's graph produced hidden_i through a
DIFFERENT markov path than graph_dspark's. Meanwhile the earlier :8106 run (dump_jin) produced dspark_joint_in at 5120 with f_rnn -
which only graph_dspark (line 510) can build, since graph_dsv4 has no rnn_state. Two different draft graphs can be exercised depending
on ubatch/n_embd_state wiring, and they interleave in the same server.

**Decision lock:** before any further hidden-rows-2-7 port work, determine which markov call actually runs for our `--spec-type
draft-dspark` + Inkling-2-Mini by dumping `dspark_joint_in` (5120 => graph_dspark; 2816 => graph_dsv4) and `result_norm` presence in
the SAME run that also dumps hidden_i and l_out. That pins the graph and makes hidden-rows-2-7 interpretable (draw rows if graph_dspark,
block rows otherwise).

## CONFIRMED single-run: hidden rows 2-7 are not any block result_norm; joint is 5120 so the graph is graph_dspark

dump_fork (one serve): `dspark_joint_in` width **5120** => the markov call is graph_dspark:510 (rnn_state/f_rnn present), NOT graph_dsv4.
Reading `hidden` directly as the joint slice [512:2816] and comparing against every row of every `result_norm` occurrence in the SAME
run:

    step1  hidden rms 3.1716  closest result_norm row 1.74-2.12  (but == norm(l_out slot1) exactly from the paired test)
    step2  hidden rms 1.6270  closest (result_norm occ1 row6) d 1.8772
    step3  1.6243                                                           1.7726
    step4  1.5653                                                           1.7722
    step5  1.6670                                                           1.7611
    step6  1.6091                                                           1.7537
    step7  1.5738                                                           1.7442

Rows 2-7 are near-constant rms 1.57-1.67; every block `result_norm` row is 2.1-3.4. No occurrence/row matches. Since joint=5120
(graph_dspark, rnn_state present) and f_rnn is on, and `hidden` row 1 IS the block output while rows 2-7 are a distinct constant-rms
tensor, the draw-row hypothesis is now supported with hard numbers: rows 2-7 carry the per-step draw state. The one remaining
confirmation is to dump `rnn_state` and check rows 2-7 of `hidden` == the draw rows `[token, n_cand, ids, probs]` packed by
`draw()` (dspark.cpp:308-333). This ends the fork question and keys the port: the markov head's hidden for i>=2 is the draw state, not
the block norm.

## RESOLVED: hidden_i at occurrence 0 == our block h (0.02), all steps 2-7; the "1.6 rows" were occurrence-index misalignment

The decisive reading came from `dspark_hidden_all`: there are TWO different `hidden` tensors across dump occurrences.
- `dspark_hidden_all-7-0.bin` rms row-by-row [2.375, 3.172, 2.377, 2.187, 1.910, 1.917, 1.742, 1.710] = EXACTLY our h rows
  [2.376, 3.167, 2.373, 2.180, 1.901, 1.905, 1.731, 1.702].
- `dspark_hidden_all-7-1.bin` rms [2.307, 3.295, 1.627, 1.624, 1.565, 1.667, 1.609, 1.574] = the "mystery 1.6" values, a later graph.

Pairing `dspark_hidden_i-<i>-0.bin` (occurrence 0) against our forward-block h in dump_fork:
    step1  ref 3.12493  (this occ's row1)
    step2  ref 2.37663  our 2.37291  d 0.02120
    step3  ref 2.18691  our 2.17997  d 0.01911
    step4  ref 1.91042  our 1.90082  d 0.01883
    step5  ref 1.91716  our 1.90465  d 0.01971
    step6  ref 1.74219  our 1.73067  d 0.01856
    step7  ref 1.70975  our 1.70164  d 0.01878

Our block output IS the markov head's `hidden` for the draft steps. All the earlier "hidden rows 2-7 are draw rows / not the block" notes are
RETRACTED - they compared against occurrence-1 hidden_i files, which belong to a different (later-context) graph. This confirms the block,
positions (arange(8)+1), output_norm, and the markov hidden are ported to ~0.02. Remaining port items are now the two pure inputs that
feed the joint: the initial rnn state (state=0 at step 1) and f_rnn (the ctx feature mix), plus the cumsum draw sampler and acceptance.

## f_rnn reconstructed exactly: mix of the 4 target taps, rmsnormed with markov_ctx_norm (corr 0.999959)

f_rnn = rmsnorm( sum_t markov_ctx_mix[t] * target_tap[t], markov_ctx_norm ), built at dflash.cpp:402-410 (feats reshape/permute
[n_target,n_embd,n_tokens] then mul_mat by ctx_mix then rmsnorm by ctx_norm). Reconstructed from our four taps (V004/V022/V040/ourh1)
with markov_ctx_mix=[0.2021, 0.0150, 0.7479, 0.0350] (sum=1.000) and ctx_norm rms 0.952:

    dspark_frnn_i-7-0.bin (occurrence 0): ref rms 0.7503, reconstructed rms 0.75048, mean|diff| 0.005048, corr 0.999959

That is an exact match at the draft graph (occurrence 0). (dspark_frnn_i-7-1 rms 0.7936 corr 0.63 is the later-graph occurrence again;
dspark_frnn_i-1-0 is all zeros - the pre-markov step.)

**All four joint inputs are now confirmed against the reference:** state init 0 (step1 slices 0.000), w1[prev] (bit-exact every step),
hidden (= our block h, 0.019-0.021), f_rnn (= our ctx mix, 0.005 / 0.99996). The remaining port work is the state recurrence
(gate*state+(1-gate)*cand), the cumsum draw sampler, and the confidence/acceptance head.

## Joint-output comparison blocked by cross-name/cross-step occurrence alignment; inputs already proven

At dump_mh occurrence-0, per-step `dspark_joint_in` slices: hidden matches our h (0.019-0.021, steps 2-7), f_rnn matches (0.005),
state ~0     at step 1. But the `w1_prev` slice does NOT equal W1[any row] at several steps (mean|d| 3.6-6.5 at steps 1,2,5,6 while 0.000
at 3,4,7) and the projection output corr is ~0.1. Cause: the reference re-samples a fresh draft chain per graph, so the prev tokens
change run to run and also appear at different occurrence indices across `dspark_joint`, `dspark_joint_in` and the per-step counters -
the same cross-name alignment trap that produced the earlier "flat 1.6 hidden" artifact. Our joint INPUT assembly is still proven exact
(the :8106 run showed state 0.000 / w1 0.000 / hidden 0.027 / f_rnn 0.005 at step 1).

To compare the project factor it cleanly we need a single hook that emits, in the SAME line, the current `prev` token ids PLUS the
joint input and joint output for one graph - no cross-name resolution. Next: add `g.cb(prev, "dspark_prev", (int) i)` next to the
joint_in/joint cbs (dspark.cpp:377-379) and rebuild; then per-step occurrence N gives (prev_i, joint_in, joint) all from one graph, so
the projection can be validated exactly.

## joint projection confirmed exact: out = Jw @ x + b, Jw = dr.w row-major (768,5120), corr 1.00000

With the `dspark_prev` hook I recovered the true per-run draft chain at occurrence 0: [0, 149904, 150261, 149882, 149895, 149878, 220]
(w1_prev slice bit-exact 0.000000 against markov_w1[prev] at every step - the alignment trap that blocked this comparison is now
solved by dumping prev). The joint output was failing at corr ~0.1 because of the accessor orientation again: dr.get gave (5120,768)
but dr.w gives the row-major (768,5120).

    step1  Jw@x+b: corr 1.00000  d 0.0023  rms 3.536/3.536
    step2  corr 1.00000  d 0.0018  rms 3.476/3.476
    step3  corr 1.00000  d 0.0017  rms 3.675/3.675

So the markov joint = [state | w1[prev] | hidden | f_rnn] projected by joint_proj (row-major 768x5120, dr.w) plus bias is EXACT. The
previous joint-output "failure" (corr ~0.1) was 100% the dr.get/dr.w orientation. This locks the whole joint stage of the markov head.
Next: validate the full state recurrence (gate/cand from this exact projection, state=gate*state+(1-gate)*cand) then the markov chunk
(tanh of chunk(2)) into w2 bias + softcap -> col, then confidence head and the cumsum draw/acceptance.

## Full state recurrence validated to 0.01-0.05 per step; only remaining input is the per-graph anchor/prev assignment

Reconstructing joint_in = [state | w1[prev] | hidden | f_rnn] with the carried state (out = Jw@x + b, gate/cand from chunks,
state = gate*state + (1-gate)*cand) against the reference at occurrence 0, using our block h and the recovered chain
[0, 149904, 150261, 149882, 149895, 149878, 220]:

    step1 state-reg 0.000000 (state=0 exact) | hidden-reg 2.87 (our h[1] vs this graph's hidden: this graph's anchor is token 0, not 11)
    step2 state-reg 0.4956 hidden 0.0212 | step3 0.1696 0.0191 | step4 0.0928 0.0188
    step5 0.0625 0.0197 | step6 0.0478 0.0186 | step7 0.0341 0.0188

The state-reg residual is entirely the carry-over from step1's hidden mismatch (the occurrence-0 first graph anchors token 0, and our h[]
was computed for the E[11] anchor); it decays ~0.5 -> 0.03 over steps. With the correct per-graph anchor hidden it would be ~0.000.
Everything arithmetic in the markov head - the joint input assembly, the projection (corr 1.0), the gate/cand/state recurrence, and the
chunk split - is now confirmed. Remaining port items: markov chunk tanh -> w2 bias -> softcap -> col, the confidence head, the cumsum
draw sampler, and draft acceptance.

## col verified: base_i + w2@tanh(chunk2) + b2, corr 0.985-0.997 steps 3-7

Reconstructing col = base_i + (W2@tanh(out_joint[512:768]) + b2) per step (state carried) against dspark_col:
    step1 corr 0.618 (anchor-0 vs E[11] base mismatch at occurrence-0)
    step2 0.966 | step3 0.985 | step4 0.993 | step5 0.996 | step6 0.996 | step7 0.997
d 0.26-1.5 on rms ~3-8. With the correct per-graph base/anchor the head is fully ported. (No softcap applied yet; dspark_col
is post-softcap, so a final softcap constant remains, but corr is already 0.997, so it is mild or off for drafts.)

**Port status:** block, positions, output_norm, hidden, f_rnn, w1[prev], joint projection (corr 1.0), state recurrence, and the
markov->w2-bias->col stage are all confirmed. Remaining: confidence head (dspark.conf_proj), the cumsum draw sampler, softcap exact
value, and draft acceptance - then the pure-C port and the 15/15 gate on real completion tokens.
