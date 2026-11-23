# Front I — speculative decoding & optimisation source mining (cloned code, not papers)

Scope: the eight clones under `C:/Users/eman5/WuBu/research/engine-sources/`
(`EAGLE`, `AngelSpec`, `AdaFlash`, `SpecPilot`, `loopspec`, `MTPLX`, `rec`,
`inference-optimisation-skill`). Every row below is read from **actual source or the
skill markdown**, not from a marketing page. Paths are relative to that root unless
absolute. Rule honoured: **reimplement in C11; never vendor, link or wrap third-party
code**. Claimed numbers are the authors'/repo's, not local measurements.

Read first: `C:/Users/eman5/WuBu/repos/wubuwizard/docs/ENGINE-GAP-REGISTER.md`
(EG-01..EG-70). Already recorded and therefore **not** re-reported as new: EG-11
grammar, EG-12 sampler chain, EG-24 EAGLE-3, EG-25 speculative families, EG-42 TVP,
EG-43 draft KV, EG-55 UNO, EG-56 DFlash, EG-62 DSpark. Where a mechanism is an
instance of one of these it is mapped; `NEW-gN` labels are *proposed* next-free ids,
subject to the dedupe pass before entering the register.

T1 = the current 6 GB VRAM / 16 GB RAM / SSD, single-process, no-dependency C11 box.
HIGH = pure policy/index/maths, portable now. MEDIUM = needs trained drafter weights
or a host sync. CONDITIONAL = only with a non-stock architecture. LOW = hardware-bound
kernel concept.

---

## A. EAGLE (SafeAILab/EAGLE, Apache-2.0; arXiv 2401.15077 / 2406.16858 / 2503.01840)

**M1 — EAGLE-1/2 feature-extrapolation drafting + cum-logprob tree selection.**
Source: `EAGLE/eagle/model/cnets.py:Model.topK_genrate` (lines 670-819); `Model.forward`
(586-665); `EAGLE/eagle/model/choices.py` (`mc_sim_7b_63`, 24 fixed paths).
Algorithm: a single `LlamaDecoderLayeremb` consumes target hidden state + token
embedding; `stable_kv` is retained across steps (`/cnets.py:690-696`); each level takes
`top_k` successors from `logsoftmax(lm_head(norm(h)))` and scores them by cumulated
log-probability `cu_scores = topk_p + scores[:,None]`, keeps global top-`total_tokens`
by that score, then builds an ancestor chain by `mask_index = searchsorted(top_scores_index,
draft_parents-1)` and leaf-to-root `retrieve_indices`. `tree_position_ids = sum(tree_mask,dim=1)-1`.
This is EAGLE-2's confidence-shaped dynamic tree: draft probabilities *choose the tree
shape*, not just the tokens. Numbers (README §Why EAGLE / EAGLE-2): EAGLE-1 3× vanilla
13B, 2× gpt-fast, 1.6× Medusa; EAGLE-2 4× vanilla, 1.4× EAGLE-1; EAGLE-3 (M5) 5.6×
vanilla, 1.8× EAGLE-1, measured on 2×RTX 3090 fp16 Vicuna-13B.
T1 fit: HIGH (the score/product tree builder is plain arithmetic over a candidate list).
Gap: **NEW-g1** (EAGLE-2 dynamic tree); EG-24 covers only EAGLE-3 direct prediction and
tri-layer fusion. `src/wubu_eagle.c` is EAGLE-1 truncated-layer only per register.

**M2 — Static tree buffers for one-pass tree verify.**
Source: `EAGLE/eagle/model/utils.py:generate_tree_buffers` (89-207), `pad_path` (62-86);
companion `utils_c.py:Tree` (57-99).
Algorithm: sort paths by `(len, path)`, build `tree_attn_mask` by marking ancestor
indices from each prefix, emit `tree_indices` with depth-packed layout
`cur[-1] + TOPK*(depth+bias) + 1`, `tree_position_ids` per depth, `retrieve_indices`
= per-leaf root→leaf index path padded with `-2`, sorted with a `maxitem` sentinel so
valid rows order first. Output is a 4-tensor dict consumed unchanged by the verify pass.
T1 fit: HIGH — this is the exact buffer layout a C11 `wubu_tree_verify` needs.
Gap: supporting primitive for `test_eg21_medusa_tree` / EG-25; **NEW-g2** as a named
tree-layout gate if EG-21 stays Medusa-specific.

**M3 — Lossless posterior evaluation (longest-prefix greedy + rejection sampling).**
Source: `EAGLE/eagle/model/utils.py:evaluate_posterior` (337-415).
Algorithm: greedy — `posterior_mask = candidates[:,1:] == argmax(logits[:,:-1])`,
`candidates_accept_length = cumprod(posterior_mask).sum(1)`, take `argmax` of that
count; if zero accept, fall back to candidate 0 and sample the bonus from the target
row. Non-greedy — walk positions while `i == accept_length`, take the first matching
candidate `x`, accept with `r <= px/qx` (draft probability cancels, `qx = 1` on the
tree's top-k); on reject zero that token, renormalise `gtp`, keep sampling; the
correction distribution is the adjusted `gtp`, otherwise a fresh target softmax.
T1 fit: HIGH, and directly the "correction distribution" requirement of EG-61.
Gap: maps to **EG-61** (already recorded); use as its reference implementation.

**M4 — Tree-KV compaction and draft-state restore.**
Source: `EAGLE/eagle/model/utils.py:update_inference_inputs` (419-473).
Algorithm: `select_indices = retrieve_indices[best,:accept+1] + prev_input_len`;
gather the accepted prefix out of the tree KV into contiguous slots with
`dst.copy_(tgt, non_blocking=True)`, set `current_length_data`, slice
`hidden_state_new[:, retrieve_indices][best, :accept+1]`, draw the bonus token, then
re-run `topK_genrate` from those accepted hidden states. No recompute of the accepted
prefix.
T1 fit: HIGH (a gather + hidden-state handoff).
Gap: **NEW-g3** (spec tree KV compaction / draft-state restore); EG-38 is decode-side
prefix-page reuse for distributed HiCache, a different mechanism.

**M5 — EAGLE-3 tri-layer fusion + training-time test.**
Source: `EAGLE/eagle/traineagle3/cnets.py:Model.__init__` (480-534, `fc = Linear(3H,H)`),
`dataprepare` (714-731), `forward` (733-868).
Algorithm: teacher is frozen; stack hidden states 1/2/3 (`cat(hidden_states[0..2])`)
into the draft `fc`; `self.length = 7` sequential draft layers reuse one `midlayer`
with `cache_hidden`; loss `-(position_mask * softmax(target[:,d2t]) *
log_softmax(draft_logits)).sum(-1).mean()` with `target = padding(target, left=False)`
(training-time test = each draft step is trained against the teacher's argmax at that
step); `t2d/d2t` map a reduced draft vocabulary; `acces` logs argmax agreement.
T1 fit: MEDIUM (needs an EAGLE-3 adapter checkpoint; the inference-side fusion path is
already `initialize_tree` in `utils.py:248-253`).
Gap: **EG-24** (already recorded) — this is its exact upstream mechanism.

---

## B. AngelSpec (AngelSpec License; multi-family training/inference framework)

**M6 — DSpark confidence head trained on the accept-probability identity.**
Source: `AngelSpec/angelspec/models/dspark.py:DSparkModel._accept_rate` (125-147),
`_compute_extra_loss` (149-207), `_compute_draft_logits` (87-116).
Algorithm: per position `Σ_v min(p_v,q_v) = 1 - 0.5·L1(p,q)` between student draft and
detached teacher, accumulated in vocabulary chunks to bound fp32 memory; a confidence
head on corrected draft hidden states is trained with BCE against that soft accept
rate (`confidence_detach` stops the gradient reaching the drafter); `confidence_mae`
is logged as the calibration readout; a `markov_head` biases draft logits with the
teacher-forced previous token. `DFlashModel` supplies decay/D-PACE/KL/LK/e2e-TV losses
composably (README §Features).
T1 fit: MEDIUM (policy head is small; weights are trained).
Gap: **EG-62** (confidence head + Markov correction) and **EG-42** (TV acceptance
identity) — both recorded; this is the concrete loss that implements them.

**M7 — DFly / DFlare fused multi-layer context drafter.**
Source: `AngelSpec/angelspec/models/draft/dfly.py:DFlyDraftModel` (52-166), `DFlyConfig` (37-49).
Algorithm: `layer_context_i = RMSNorm(FC(concat(target_hidden)) + fusion_i(target_hidden))`
where `fusion_probs = softmax(layer_fusion_weights[i])` and the residual is
`einsum('t,bstd->bsd')`; DFlash's shared `k_proj/v_proj` are kept, DFlare's separate
context projections are dropped; `enable_hidden_correction` defaults True (TreeFlash
formula (1)) and is applied in the wrapper's `_compute_draft_logits` hook. Chunked
context calls bound memory; per-layer fusion weights let the drafter select which
target layer feeds which draft layer.
T1 fit: MEDIUM (trained drafter family; the fusion-weight idea is a C11 policy).
Gap: **NEW-g4** (layer-fusion drafter family) — EG-55 is gated-LoRA-on-verifier + Psi-Spec;
EG-56 is block diffusion; DFly is a distinct fused-context family.

**M8 — FP8/Hadamard coarse scoring for gradient-free drafter selection.**
Source: `AngelSpec/angelspec/models/ops/opd_indexer.py:quantize_keys` (21-27),
`fp8_coarse_logits` (30-59).
Algorithm: FWHT quantise keys/query (`fwht_quantize`), pack heads to 32/64 for the
`fp8_fp4_mqa_logits` kernel, derive MX scales by `(scale.view(int32) >> 23) * 0x01010101`,
chunk queries by a 16 MiB temporary budget, mask columns past each row's `ends` to
`-inf`. Used by `angelspec/models/draft/path_selector.py` for on-policy draft selection.
T1 fit: LOW on this hardware (vLLM NVIDIA FP8/FP4 MQA op), MEDIUM as an algorithm
(quantised coarse-then-exact selection).
Gap: **NEW-g5** (coarse quantised draft-selection indexer), adjacent to EG-44's
trained sparse indexer but proposal-side, not attention-side. Mark NON-FIT for the
specific CUDA op; keep the two-stage score-then-exact idea.

---

## C. AdaFlash (arXiv 2607.19223; SpecForge-derived, vendored subset only upstream)

**M9 — DFlash block-diffusion masking and anchor sampling.**
Source: `AdaFlash/specforge/core/dflash.py:create_dflash_sdpa_mask` (63-88),
`create_dflash_block_mask` (91-134), `OnlineDFlashModel.forward` (275-400).
Algorithm: KV = `[context S | block_0 .. block_{n-1}]`, Q = the blocks. Four mask
rules: a block sees context strictly before its anchor (`kv < anchor`, so the anchor
hidden state is unavailable), intra-block attention is bidirectional, blocks cannot
see each other, invalid blocks see nothing. Up to `num_anchors=512` anchors sampled
per sequence; noise input puts the real anchor token at each block start and MASK
elsewhere; position ids = `anchor + arange(block_size)`; loss decay
`exp(-(k-1)/gamma)` weights the first predicted token at 1.0; CE plus optional
teacher-KD on the same positions.
T1 fit: MEDIUM-HIGH (the mask + same-position label rule is pure index maths).
Gap: **EG-56** (already recorded) — this is the exact training-side mechanism.

**M10 — Adaptive candidate-length head (the confidence-scheduling mechanism).**
Source: `AdaFlash/specforge/modeling/draft/dflash.py:ThreshHeadSubsequent` (86-112),
`ThreshHeadTwoModel` (115-130), `_prob_head_candidate_len` (133-145),
`DFlashDraftModel.spec_generate` (445-560).
Algorithm: per position form `[max_prob, entropy, top_k_probs]`, mean-pool over the
block, MLP → sigmoid threshold (variant: mean+last draft hidden → residual bottleneck
→ sigmoid). At decode, **always** draft the full block, then
`cum_accept = cumprod(probs[1:])`, truncate at the first k with
`1 - cum_accept > threshold`, clamp to `[len_min=2, block_size]`; run the target only
on that prefix, greedy-accept via `cumprod(block_ids[:,1:] == posterior[:,:-1]).sum()`,
crop target KV to accepted+1 and re-extract context features. Cost of a wrong (too
long) threshold is wasted target rows; too short leaves accepted tokens on the table.
T1 fit: HIGH — the truncation rule needs no weights (a fixed threshold is a valid
C11 first gate); the learned head is a later refinement.
Gap: **NEW-g6** — EG-56 covers the block-diffusion family, EG-62 a per-position
verify budget; *adaptive candidate length from predicted cumulative acceptance* is
neither. This is the single best "confidence scheduling" finding in this front.

**M11 — On-policy RKL labelling of the length head + async hot-swap loop.**
Source: `AdaFlash/pipeline/offline/init_thresh_head.py:compute_thresh_head_labels_rkl`
(173-223), `README.md` §Training (102-137).
Algorithm: from realised `accept_lens = cumprod(argmax(draft)==target).sum`, either
label `cand_len = clamp(expected_accept + lookahead)/block_size`, or label the
calibration target `1 - cum_conf[accept_lens + lookahead]` (with a `>= block-1 → 1`
override). Async pipeline: inference worker writes target trajectories into a swap
buffer; trainers consume and publish versioned drafter weights for hot reload.
T1 fit: MEDIUM (offline labels are data; hot-swap is a serving concern).
Gap: labelling supports **NEW-g6**; the live-update loop maps to **EG-59** (already
recorded).

---

## D. SpecPilot (engine patches as per-setting `files/` overlays; RTX PRO 6000 / SGLang settings)

Numbers unless noted: `SpecPilot/settings/pro6000_c1_qwen38-27b_dflash2/README.md`
(Qwen3.8-27B + DFlash2, batch 1, greedy, 2048 tokens, dev split): stock 4.46× over
AR → final 10.87×; tau 5.23 → 12.80 tokens per verify step; final vs stock 2.44×.
"tau counts tokens per exact-weight pass under TLV." Hardware sm_120 only for the
CUDA kernels.

**M12 — Two-Level Verify (TLV): quantized pre-verify builds a chain, exact pass commits.**
Source: `.../speculative/specpilot_d8_tlv.py` (module docstring 1-19, `q_mode` 73-81,
`compose_chain_tree` 109-149).
Algorithm: step = Q-step + exact step. Q-step drafts a tree and verifies it with the
target's quantized view `d8q`, **greedy accepts without commit**; its accepted path
becomes chain C and its bonus becomes the next draft anchor. Exact step drafts below
that chain, truncates to `N-|C|` nodes, hangs the subtree under the chain
(`par = where(sub_par<0, C-1, sub_par+C)`, ancestor mask rebuilt from the sub-mask),
and the exact target verifies + commits in one pass. Lossless: chain tokens are only
proposals. `SPECPILOT_D8R_TLV_QNOISE` is a test-only mismatch injector.
T1 fit: HIGH (two-pass verify + chain composition is C11-able; "Q view" can be any
already-present quant path).
Gap: **NEW-g7** — refines EG-25/EG-43 but is a distinct algorithm (cheap view proposes
a prefix for the exact verifier).

**M13 — Pointer-jumping tree acceptance + KV-slot permutation.**
Source: `.../speculative/specpilot_d7_accept.py:_d7_tree_accept_kernel` (33-95),
`d7_tree_accept_commit` (98-116).
Algorithm: `ceil(log2 N)` simultaneous gather rounds compute depth, all-edges-match
and an ancestor bitmask; the deepest fully-matching leaf is selected by a max key
`depth*(NP+1)+(NP-nidx)`; path reconstruction is one reduction over the bitmask;
the KV-slot permutation loop runs only to accept length. Claimed: 35 µs → 9 µs at
N=32 (parent's N sequential parent-gathers + N scalar steps, 4 warps → 8 warps); the
d3w register-gather variant is ~7 µs vs ~50 µs at N=48, 7.9 vs 11.5 µs at N=32.
T1 fit: HIGH (pure index algorithm, no GPU-specific primitive required).
Gap: **NEW-g8** (log-depth acceptance); supports EG-21/EG-25.

**M14 — Split-KV tree-verify attention.**
Source: `.../speculative/specpilot_d5_splitkv.py` (docstring 1-22).
Algorithm: parent `extend_attention_fwd` launches one CTA per (request, q head) over
the whole prefix — 24 CTAs on 188 SMs at batch 1, ~0.13 µs per context token per
layer. Replacement packs the G query heads of a KV head as rows so each KV block is
read once per KV head, splits the prefix into S BN-aligned chunks with online fp32
softmax and bf16 tensor-core dots, adds one split for the N fresh draft keys, then a
combine kernel merges S+1 partials. Constraints: page size 1, bf16 KV, N ≤ 64.
T1 fit: HIGH concept (matches the skill's "verify must use decode-style attention"
lever: 47 → 104 ms at 16K when the prefill kernel is used for k-query verify).
Gap: **NEW-g9**; the skill's `gotchas.md` recommends this but the register has no row.

**M15 — Empirical acceptance calibration table for best-first tree ordering.**
Source: `.../speculative/specpilot_d3m_calib_table.py` (whole file: `LOG_Q[10,7,18]`,
`BIN_EDGES`, `BLEND=0.5`), applied by `.../specpilot_tree.py:d3m_calibrate` (186-205)
and `d3m_calib_table` (136-149).
Algorithm: log empirical P(candidate is true next token) keyed by (slot 1..10,
sibling-rank bucket, drafter-probability bin), fitted on 5624 dev verify steps with
prior weight 5 and request-level held-out splits; blend `log p` with `LOG_Q` at 0.5
before the best-first tree selector; override via JSON at runtime.
Claimed: +0.75% / +0.55% tokens per verify on the two held-out halves.
T1 fit: HIGH (a static table + dot product; no training at runtime).
Gap: **NEW-g10** (empirical acceptance prior for tree construction) — a confidence
scheduler distinct from EG-41's offline config search.

**M16 — Position-keyed exponential-race sampling for trees (lossless T>0).**
Source: `.../speculative/specpilot_k001_sample.py` (docstring 1-44).
Algorithm: `x[b,i] = argmin_v E(seed_b, pos, v) / p[b,i,v]` with `pos = prefix_len_b + d_i`;
`E` is a counter-based i.i.d. Exp(1) field keyed by (seed, absolute position, vocab id)
— Gumbel-max in exponential-race form. Greedy tree accept then matches children to
`pred = x`. Proof sketch in the file: a committed token at position n is always the
argmin under its *committed* prefix, so by induction the stream equals AR exponential
race with the same noise field, independent of tree shape. Q/exact passes share the
noise key (`COUPLE=1`) so their chains coincide; `COUPLE=0` gives independent noise.
T1 fit: HIGH — the cleanest C11-compatible lossless sampling rule found here; belongs
beside the EG-12 chain seam.
Gap: **NEW-g11**; EG-12 is a sampler-stage chain, EG-61 is verification exactness.

**M17 — Exact-tie acceptance ("argmax-set greedy").**
Source: `.../speculative/specpilot_d1_tieacc.py` (docstring 1-20).
Algorithm: accept if the child's exact bf16 logit *equals* the row max bitwise;
rewrite `pred'[j]` to the first tied child in node order, bonus stays plain argmax;
counters check `violations == 0`. Strict no-op at T>0.
T1 fit: HIGH (tiny change to a C11 verify kernel; avoids invalid rejections on ties).
Gap: **NEW-g12** (small, well-scoped gate).

**M18 — Q-pass LM head restricted to a certified vocabulary subset.**
Source: `.../speculative/specpilot_d2q.py` (1-62),
`.../speculative/dflash_de_head.py` (`build_vocab_subset`).
Algorithm: only the Q pass uses a subset head of `SPECPILOT_DE_VOCAB_SIZE=98304` ids;
a `CertifiedArgmaxHead` over a contiguous subset copy (bf16 1.0 GB + INT8 screen 0.5 GB)
streams 0.5 GB instead of 1.27 GB. Claimed −0.26 ms per two-level step, identical
outputs 98/100, 0/108k dev positions had a top-20 without a subset token; full-vocab
exact pass unchanged so lossless.
T1 fit: MEDIUM (memory-trade-off variant; 6 GB box benefits from the smaller head).
Gap: **EG-57** (already recorded: high-frequency draft head, full vocab at verify).

**M19 — Adaptive Q-pass count and prefix window.**
Source: `.../speculative/specpilot_pa.py` (docstring 1-28).
Algorithm: `SPECPILOT_PA_QPASSES=n` loops Q passes, each drafting below the chain built
so far and extending it; `QWIN=F,W` makes Q's verify attention read only first F + last
W prefix positions; `QADAPT=len:L` reads chain length after each pass (one host sync)
and stops early at L. Exact pass is untouched → lossless.
T1 fit: MEDIUM-HIGH (adaptive number of cheap passes is a good C11 policy knob).
Gap: **NEW-g13** (adaptive pre-verify depth) — candidate with M10/M12.

**M20 — Certain-finish trailing-step skip + predictive Q gate.**
Source: `.../speculative/specpilot_d4_tailskip.py` (1-19) and
`.../speculative/specpilot_d2g.py` (1-31).
Algorithm: under the overlap scheduler step N+1 is launched before N's result is read,
so the final verify step's output is discarded. Copy step N's commit to pinned host,
wait before replaying N+1's verify graph, and if N's tokens certainly finish (OOV,
stop/eos without ignore_eos, length cap) reuse N's static graph outputs. D2G predicts
possible finish *before* exact step k starts from the already-composed tree of step k
(stop id in tree, or `output_count(k-1) + max_tree_depth + 2 >= max_new_tokens`), and
only then takes the host gate; false positive costs ~1 ms idle GPU.
T1 fit: MEDIUM (needs an overlap scheduler; EG-09 dependency).
Gap: **NEW-g14** (speculation-specific trailing-step skip), refines EG-09.

**M21 — Online block-accept ceiling estimator and lossy steering (labelled).**
Source: `.../speculative/dspark_components/dspark_block_accept_estimator.py`
(`_OnlineCeiling` 100-160, `observe_verify_step` 278-340, `BlockAcceptEstimateRecorder`
221+), `.../speculative/specpilot_lossy.py` (docstring 1-44),
`.../speculative/specpilot_force.py` (1-30, A/B instrument).
Algorithm: chunked gather of token logprobs per pending block, settle on finish,
windowed online ceiling estimate keyed by `forward_ct` with eviction; discontinuities
drop pending blocks. `specpilot_lossy.py` rewrites the per-node prediction for
`relax`/`nucleus` steering with `norep` loop guard — **explicitly NOT lossless**.
`specpilot_force.py` forces the target prediction to a recorded reference sequence so
two engines emit identical text for paired speed A/B (measurement only, not a feature).
T1 fit: MEDIUM for the estimator; the lossy path needs an accuracy gate first.
Gap: estimator maps to **EG-62** (DSpark confidence budget); lossy steering is a
deferred **NEW-g15** behind a quality gate; force-replay belongs in the measurement
protocol (§H), not an engine gate.

---

## E. loopspec (pipelined self-speculation for looped transformers)

Numbers: `loopspec/README.md:11` — "up to **6.83×** decoding speedup"; requires a
looped transformer (Ouro/Raven) and CUDA.

**M22 — S/K/X pipelined self-speculation schedule.**
Source: `loopspec/sglang_recurrent/loopspec/pipeline.py:PipelineSchedule` (125-230),
`continue_pipelined` (254+); README §Inference (first proposal depth must divide total
recurrent depth; second must be a multiple of the first).
Algorithm: S iterations per token, K iterations per cycle, optional q2 at K·X;
`depth = S//K`, endpoints `{K-1:q1, K·X-1:q2, S-1:final}`. Full-tree lane births follow
`b_t = b_{t-1} + b_{t-X}` (with `b_0..b_{X-1}=1`); `max_rows = Σ b_t`; the request-pool
capacity adds a delayed frontier plus one logical root and the original prefill rows;
`role_max_batch_sizes` bounds pre/recurrent/post graph-capture shapes. Verify is the
model's *own* intermediate prediction at q1/q2, so no trained drafter.
T1 fit: CONDITIONAL — needs a recurrent/looped checkpoint, not a stock GGUF. The
birth recurrence and batch bounds are portable to any staged recurrent design.
Gap: **NEW-g16**, distinct from EG-69 ReplaySSM (rollback object) and EG-21 Medusa.

**M23 — One-cycle forward + single-sync compact decision readback.**
Source: `loopspec/sglang_recurrent/loopspec/executor.py:LosslessLoopSpecExecutor.forward`
(293-367), `_prepare_cycle` (205-237), `retain_branches` (369-389);
`.../loopspec/decisions.py:DecisionReadback.stage/read` (64-89).
Algorithm: pre run for stage-0 lanes → K× recurrent iterations over fixed token ids
and state-bank slots → post only at endpoint stages → stage compact decisions to a
pinned host buffer (1 int/row greedy, 5 ints/row sampling) with one CUDA event; do CPU
bookkeeping and *deferred* branch cleanup while the GPU runs; then `read()` once with
`event.synchronize()`. A `_StateSlotPool` acquires/retains slot ownership; KV branch
release is delayed one cycle so `pipeline.py` updates lane objects first.
T1 fit: HIGH for the single-sync readback pattern; MEDIUM for the pipeline itself.
Gap: **NEW-g17** (compact one-sync endpoint readback) — host/sync discipline, useful
alongside EG-09/EG-32.

**M24 — Lossless sampling under pipelined proposals.**
Source: `.../loopspec/sampling/policy.py:SamplingDecisionPolicy` (19-308),
`.../loopspec/sampling/rejection.py:RejectionSampler` (18-182),
`.../loopspec/sampling/distributions.py`; `decisions.py:DecisionOutcome` (32-43).
Algorithm: q1 proposes argmax/sampled drafts with a retained `[vocab]` probability
vector, q2 may choose "argmax" or "residual", the final endpoint compares against q1
then q2 and records `first_rejected/second_rejected/accepted_stage`; rejection is
captured once and replayed so proposals do not perturb the target distribution.
T1 fit: MEDIUM-HIGH (the outcome record is a good C11 struct design).
Gap: supports **NEW-g16**; shares intent with M16/EG-61.

---

## F. MTPLX (MLX native-MTP engine, depth-adaptive policies)

**M25 — Hysteresis depth controller (streak).**
Source: `MTPLX/mtplx/adaptive.py:AdaptiveDepthPolicy` (14-65).
Algorithm: full accept increments `_full_accept_streak`; after `increase_after = 4`
consecutive full accepts raise depth by 1; a rejection at `accepted+1 <=
max(1, previous_depth//2)` increments `_early_reject_streak`; one such early reject
(`decrease_after = 1`) lowers depth. Clamped to `[min_depth,max_depth]`.
T1 fit: HIGH (≈30 lines of C11, no weights).
Gap: **NEW-g18** (simple hysteresis depth step) — a first gate candidate.

**M26 — Expected-value depth gate (the strongest confidence scheduler).**
Source: `MTPLX/mtplx/adaptive.py:ExpectedValueDepthPolicy` (69-236).
Algorithm: per-depth acceptance EWMA (`alpha=0.12`, priors `(0.92,0.64,0.32)`);
`prefix = ∏ EWMA[0..d-1]`, `expected_extra_accept = prefix * EWMA[d] *
confidence_factor(draft_metrics)`; continue iff that exceeds
`extra_cost_s * baseline_tok_s * (1 + safety_margin=0.10)`, with a floor
`min_extra_accept_probability = 0.18`. Warm up 4 full-depth cycles; probe every 32
cycles; after ≥4 cost samples at both depths, observed draft+verify times replace the
priors and `baseline_tok_s` becomes `expected_base_tokens / base_cost`. Measurement
source logged (`configured_prior` vs `observed_draft_and_verify`).
T1 fit: HIGH — no training, only an acceptance EWMA and a cycle timer; the direct
answer to "confidence scheduling" for wubuwizard.
Gap: **NEW-g19** (expected-value speculative-depth controller).

**M27 — Cost-model depth selection with stale probes.**
Source: `MTPLX/mtplx/adaptive.py:CostModelDepthPolicy` (319-524).
Algorithm: `score(d) = (1 + Σ_{j<d} ∏_{i<j} p_i) / t_est(d)` (expected tokens per
second); per-depth time EWMA with warmup-min and spike damping (`SPIKE_RATIO`,
`SPIKE_DAMP`, `TAU_MS`), linear marginal-time extrapolation for unmeasured depths,
hysteresis on switching, and explicit probing of the most stale depth or the best
rival (`PROBE_MARGIN`).
T1 fit: HIGH (same inputs as M26).
Gap: **NEW-g20** (throughput/cost-model depth search) — offline counterpart of EG-41.

**M28 — Verify-specialised low-bit kernels + A/B harness.**
Source: `MTPLX/docs/turbo-verify.md` (whole file), `MTPLX/vllm_metal/metal/kernels_v2/turboquant.metal`.
Algorithm: route 4-bit affine projections in batches of 4–16 rows (the native-MTP
verify shape) to verify-specialised kernels: 4-row K-split on any Apple Silicon,
16-row Metal-4 tensor-op tile on M5-class G17 for depths >3; single-token decode,
drafting and prefill stay stock/bit-identical. Claimed M5 Max / Qwen3.6-27B: 1k decode
48.3 → 65.5 tok/s (4 matched seeds), flappy 55.7 → 64.5, live 55.0 → 66.7, 10k
sustained 59.5; argmax-identical, TV 0 / sample agreement 1.0 at temp 0.6 / top_p
0.95 / top_k 20, but **not bit-exact**. `MTPLX_NAX_VERIFY` is the A/B control and
`MTPLX_KERNEL_SELFCHECK` the load-time diagnostic.
T1 fit: LOW for the Metal kernel; HIGH for the discipline (verify-shape-specific
kernel + explicit A/B switch + distribution gate + "not for bit-exact QA").
Gap: **NEW-g21** (verify-shape specialised matmul); apply the verification discipline,
not the Metal code (never vendor).

---

## G. rec (Recursive OS, single-session MLX engine; 1.4–2.6× llama.cpp on M4 Pro)

**M29 — Bounded-checkpoint prefix rollback for sliding-window KV.**
Source: `rec/engine/src/rec_mlx/prefix.py:PromptCache.checkpoint/fork` (29-71),
`select_prefix` (82-118); `rec/engine/tests/test_prefix_rollback.py` (26-84).
Algorithm: keep at most 2 checkpoints (`checkpoints[-1:]`); each stores position,
deep copies of the sliding K/V ring buffers plus `_valid` and `_ring_index`, and the
logits. On a mismatched prompt take the newest checkpoint with `position <= common`,
restore each layer's ring and offset, truncate tokens/logits, discard later
checkpoints, then replay the suffix. Crucial invariant in a source comment:
*"Never stage decode KV: its arithmetic differs from batched prefill and changes
probabilities."* The test proves the physical ring after rollback+replay is
byte-equal to a fresh prefill.
T1 fit: HIGH — this is the 6 GB single-session case (turn-2 prefix reuse without
re-prefill), and the arithmetic-difference warning matters for a C11 KV design.
Gap: **NEW-g22** — EG-08 is a radix tree, EG-38 is page ownership; neither is
checkpoint-based ring rollback.

**M30 — Single-session resident/team kernels (leads, pointers only).**
Source (not read in depth, listed as leads): `rec/engine/src/rec_mlx/kernels/kv_teams.py`
(167 lines), `resident_block.py` (88), `megakernel.py` (1337), `expert_teams.py`,
`rank_routing.py`; tests `test_shared_kv.py`, `test_kernel_routing.py`,
`validate_resident.py`. No numbers claimed here because the implementation was not
audited — do not turn this into a gap without reading it.
T1 fit: unknown. Gap: lead only.

---

## H. inference-optimisation-skill — the hill-climbing recipe (process, not a code gap)

Source: `inference-optimisation-skill/skills/inference-optimisation/SKILL.md`,
`references/{ceilings,methodology,levers,gotchas,publishing}.md`,
`scripts/{bench,ceiling,mem.sh,queue.sh}`.

**M31 — Ceiling-first loop.**
`SKILL.md` §0-3: baseline → compute ceiling → one change → A/B with controls → keep or
reject with a reason → repeat; plateau after three consecutive levers inside noise.
Working files `PLAN.md`, `THEORY.md`, `ledger.jsonl`, `configs/`, `best-config.json`.
Ground rules: ≥6 decode samples and the **median**; interleaved control/candidate/control
and accept only if the candidate beats both neighbours and regresses nothing by >3%;
fixed power/clock/host state; one GPU job behind `queue.sh`; memory watchdog at the
cap; long-context needle + concurrent soak + greedy equivalence as stability gates;
never force-kill mid-batch.
`ceilings.md` / `scripts/ceiling.py:decode_ceiling` (20-25):
`verify = dense + lm_head + experts·(1-(1-top_k/n_experts)^((drafts+1)·n))`,
`draft = k·n·(draft_layer + draft_head)`,
`bytes_per_emitted = (verify+draft)/(accept·n)`, `c1 = bandwidth / bytes_per_emitted`.
Worked example (Ornith-1.5-35B-A3B, B390, 115 GB/s): 68 tok/s no speculation → 83
with 1 draft → 97 with a pruned draft head → ~103 with 2 drafts; achieved 53 tok/s
(~55% of ceiling). Prefill `≈0.5·peak_FLOPs/(2·active_params + attention term)`.
T1 fit: HIGH — apply verbatim to every new wubuwizard gate's evidence row.
Gap: process; **no gap id**. This is the "how to measure a gate" companion to the
register's "no gate means not absorbed".

**M32 — Measured lever order (priors, not promises).**
`references/levers.md`: persistent JIT/graph caches (startup 113 s → 43 s on Arc B390;
22 min → 7 min on 4× PRO 6000); decode graphs (14.4 → 28.4 tok/s, 16 → 32, 11 → 27);
speculative decoding — prune the draft LM head (c1 37.9 → 48.2, +27%, 512/1940 blocks
at 98.6% coverage; +14% on B390), sweep draft length per concurrency (k=4 at c1, k=3
at c4, k=2 at c16; at 128K one draft 29.6 vs three 15.5 tok/s), measure acceptance per
depth (Ornith-1.5 ~14% at depth 2), verify must use decode-style attention (47 → 104 ms
at 16K with the prefill kernel; 75.7 → 82.6 tok/s on B70 c1), keep sampling on GPU
(~3 ms/step, +10% c1); KV pool = N×context with a full-length soak; prefix caching
confirmed with `cached_tokens>0` (66K turn-2 TTFT 36 s → 0.45 s); prefill chunk size
(8192 → 2048 cut worst stall 7.1 s → 1.95 s for −4% decode); int8 prefill 1.89× with
NLL +0.17% and top-1 agreement 97.2%. Under graphs, fusion attempts gave 0%.
T1 fit: HIGH (all config knobs; relevant devices include the 6 GB class).
Gap: process/priors; use to order the gate backlog, not as evidence of a local feature.

**M33 — Measurement traps specific to speculation.**
`references/methodology.md`: count tokens from usage stats, not SSE chunks (an engine
emits several tokens per chunk under MTP); size prompts with the real tokenizer; build
unique prompts before the clock starts; decode tok/s = completions after first token ÷
time since first token; report cold and warm prefill separately; per-step time and
accepted length separately because **`tok/s = a / step_time`** and higher acceptance
can be slower (7-token draft accepted 3.26 vs MTP 2.85 yet 11–14% slower); acceptance
counters may only update at request completion; content/temperature change acceptance;
greedy equivalence can diverge on a 0.016-nat top-2 tie without being a bug; bootstrap
95% CI before claiming "B is faster". `gotchas.md` adds the hybrid-model Mamba state
slot trap, acceptance counters on SGLang updating at finish, and prefix caching off by
default for hybrid models.
T1 fit: HIGH. Gap: process; this is why the register demands quoted numbers, not
speedup claims.

---

## Cross-cutting mapping to the existing register (do not re-report)

| Source mechanism | Register row |
|---|---|
| EAGLE-3 tri-layer fusion / training-time test (M5) | EG-24 |
| EAGLE-2 accumulate-logprob tree builder (M1) | NEW-g1 (EG-24 is EAGLE-3 only) |
| Longest-prefix accept + correction distribution (M3) | EG-61 |
| DFlash block diffusion (M9), AdaFlash on-policy loop (M11) | EG-56, EG-59 |
| DSpark confidence head / TV identity / verify budget (M6, M21) | EG-62, EG-42 |
| DSpark/DFlash CE+KL+LK+e2e-TV composable losses | EG-42 |
| Restricted Q-pass LM head (M18) | EG-57 |
| Overlap scheduler interaction (M20) | EG-09 (dependency) |
| Per-position grammar masks in MTPLX drafting (CHANGELOG:159) | EG-11 |
| Sampler stages / keyed tree sampling (M16) | EG-12 seam + NEW-g11 |
| Draft-KV memory trade (M18 is a *head* trade, not KV) | EG-43 is NOT covered by M18 |
| UNO shared-KV diffusion drafter | EG-55 (no instance found in these sources) |

## Prioritised new gaps (T1 first)

1. **NEW-g19** Expected-value depth controller (M26) — no weights, one EWMA + cycle timer; highest immediate fit.
2. **NEW-g6** Adaptive candidate length from cumulative acceptance (M10/M11) — one scalar threshold, no weight needed for v1.
3. **NEW-g11** Position-keyed exponential-race tree sampling (M16) — clean lossless T>0 rule.
4. **NEW-g1** EAGLE-2 confidence-shaped dynamic tree (M1/M2/M4) — completes the existing EAGLE-1 partial.
5. **NEW-g8** Pointer-jumping tree accept + KV permutation (M13) — pure index algorithm.
6. **NEW-g7** Two-level quantized pre-verify with chain composition (M12/M19) — needs an existing quant path.
7. **NEW-g22** Checkpoint-based sliding-KV prefix rollback (M29) — single-session 6 GB win.
8. **NEW-g10** Empirical acceptance-calibration table (M15) — data-only, pairs with g1.
9. **NEW-g18/g20** Hysteresis and cost-model depth policies (M25/M27) — A/B controls for g19.
10. Conditional / deferred: g16-g17 (looped architectures), g4/g5 (trained drafters/NVIDIA ops), g21 (Metal kernel), g15 (lossy steering, accuracy gate first).

Recommended first gate naming, if adopted: `test_egNN_ev_depth`, `test_egNN_adapt_len`,
`test_egNN_keyed_sample` — check against **both** Makefiles before reserving.

## Verification status

Read in full or in targeted spans: EAGLE `README`, `model/cnets.py`, `model/utils.py`,
`model/utils_c.py`, `model/ea_model.py`, `traineagle3/cnets.py`, `choices.py`,
`evaluation/speed.py`; AdaFlash `README`, `core/dflash.py`, `modeling/draft/dflash.py`,
`offline/init_thresh_head.py`; SpecPilot setting READMEs and the `speculative/` modules
named above; loopspec `README`, `decisions.py`, `pipeline.py`, `executor.py`,
`sampling/{policy,rejection}`, `branching/manager.py`; MTPLX `docs/turbo-verify.md`,
`docs/specs/2026-08-08-qwen35b-eight-way-mtp-design.md`, `adaptive.py`, `CHANGELOG.md`;
rec `README`, `prefix.py`, `test_prefix_rollback.py`; the whole inference-optimisation
skill directory. AngelSpec was inventoried (`models/draft/*`, `ops/*`, `training/*`,
`README`) and read at `dspark.py`, `draft/dfly.py`, `ops/opd_indexer.py`; the tree-layout
ops were indexed but not audited (M30-style leads only). No code was compiled or run;
all numbers above are quoted from the sources with their paths, not reproduced here.

## Sources (all local, retrievable)

- `C:/Users/eman5/WuBu/research/engine-sources/EAGLE/` (Apache-2.0; arXiv 2401.15077, 2406.16858, 2503.01840)
- `C:/Users/eman5/WuBu/research/engine-sources/AngelSpec/` (AngelSpec License)
- `C:/Users/eman5/WuBu/research/engine-sources/AdaFlash/` (arXiv 2607.19223; SpecForge subset MIT)
- `C:/Users/eman5/WuBu/research/engine-sources/SpecPilot/settings/pro6000_c1_{qwen38-27b_dflash2,pear-qwen38-27b_dflash2_er}/`
- `C:/Users/eman5/WuBu/research/engine-sources/loopspec/` (Ouro/Raven; up to 6.83× claim)
- `C:/Users/eman5/WuBu/research/engine-sources/MTPLX/` (MLX native-MTP, turbo-verify, adaptive.py)
- `C:/Users/eman5/WuBu/research/engine-sources/rec/` (Apache-2.0; 1.4–2.6× vs llama.cpp claim)
- `C:/Users/eman5/WuBu/research/engine-sources/inference-optimisation-skill/skills/inference-optimisation/`
- `C:/Users/eman5/WuBu/repos/wubuwizard/docs/ENGINE-GAP-REGISTER.md` (gap ids and T1 target)
