# C — Sparse and sub-quadratic attention for long context (2025-2026), judged against a frozen checkpoint

Scope: methods that could take Inkling‑2‑Mini from 262k to 1M prefill, and the single question
that decides the answer — what works on a **frozen** checkpoint that already has 64 of its 72
attention visits on a 1024-token sliding window.

Local target (given, not re-derived here): 48 logical layers / 24 physical blocks, d=2304, 72
attention visits = 8 full-attention visits (physical blocks 5, 11, 17, 23) + 64 sliding-window
visits at window 1024; NH=24, NKV=8, HD=128; no RoPE (relative-bias attention); f32 KV measured at
0.11 s/token, 8.0 h projected for 262k.

Citation ids 81–120, ledger at `docs/research/LEDGER.md`.

---

## 0. The arithmetic that decides the question

Derived from the local constants only (no new measurement). MAC counts are exact for the shapes
given; times use the rate implied by the local 0.11 s/token at ~120 tokens (≈4.47 GMAC/token →
≈40.6 GMAC/s), which reproduces the stated 8.0 h for 262k of *linear* work.

| Quantity (per 262,144-token prefill) | MAC | Derived time |
|---|---|---|
| Non-attention (Q/K/V/O projections over 72 visits + 8-of-128 MoE, 48 layers) | 1.16e15 | 7.9 h |
| 64 sliding-window visits at w=1024 | 1.06e14 | 0.72 h |
| **8 full-attention visits (QKᵀ + PV), = 49,152·N²** | **3.38e15** | **23.1 h** |
| Total, dense | 4.64e15 | 31.7 h |

Reconciliation with the given 8.0 h: 0.11 s/token was measured at short sequences, where the
quadratic term is negligible, so 8.0 h is the *linear* cost of 262k tokens and the eight
full-attention visits add ~23 h on top of it at the same service rate. This is the same error the
repo's own ledger already retracted once ("the earlier '35 h at 262k' was wrong in the optimistic
direction: it came from a per-token cost measured at 49 tokens"). Treat 8.0 h as a floor, not a
measurement, and the dense 262k figure as ~32 h.

At 1M the same decomposition gives **5.90e16 MAC ≈ 404 h**, of which 5.40e16 (369 h) is the eight
full-attention visits. The 8/72 visits are 73% of prefill compute at 262k and 91% at 1M; the 64
sliding-window visits are a rounding error by then (0.7 h at 262k, 2.9 h at 1M). **Every viable
change is a change to those eight visits.** Nothing about the 64 windowed visits needs touching.

Two consequences that filter the whole literature:

1. **Prefill cost is set by the number of key-blocks each query reads, not by KV bytes.** Methods
   that shrink or evict the KV cache reduce *decode* bytes and leave prefill FLOPs untouched
   (H2O, SnapKV, FastGen, StreamingLLM, ShadowKV [109], DuoAttention's cache side). They cannot answer
   this question, however good their numbers look.
2. **There is no exact sub-quadratic prefill for a frozen softmax-attention checkpoint.** Exact
   attention costs 2·NH·HD·N² MACs; the only exact accelerations (FlashAttention-style tiling,
   chunking) are constant-factor or bandwidth wins. Every prefill reduction below is an
   approximation, so the real question is *which approximation has the tightest error bound per
   MAC saved*.

---

## 1. Native Sparse Attention (NSA) — the trainable-vs-inference-only distinction

Mechanism: three parallel branches per query — coarse block **compression**, fine-grained token
**selection**, and a local sliding **window** — merged by a learned gate, with a Triton kernel whose
block layout matches the sparse pattern [81].

Claimed numbers, 64k context, 8×A100, GQA g=4, h=16, d_k=192: **9.0× forward, 6.0× backward,
11.6× decode** versus full attention, all measured against Triton FlashAttention‑2 on the same
backend [81]. Memory-touch model at 64k: 5,632 equivalent tokens versus 65,536, hence the ~11.6×
[81, Table 4].

Training: from scratch. The evaluation backbone is a 27B model pretrained on 260B tokens with NSA
in the architecture [81]. Accuracy: NSA *beats* full attention on average across general, long-context
and reasoning suites, and gets perfect 64k NIAH at every position [81].

Why it is not retrofit-able, in the authors' own words: the branches are trained to cooperate, and
"applying sparsity post-hoc forces models to deviate from their pretrained optimization trajectory …
top 20% attention can only cover 70% of the total attention scores" [81, §6.1]. That single sentence
is the empirical case against bolting NSA's selection branch onto a checkpoint that never saw it.

**RETROFIT VERDICT: needs-retraining.** The three-branch structure and its gate are trained
parameters; a frozen checkpoint has none of them, and the paper's own ablation shows post-hoc
sparsity loses far more than it saves.

---

## 2. MoBA (Mixture of Block Attention)

Mechanism: split the KV into blocks (B=4096 at 1M), mean-pool each block to a single key, score
queries against pooled keys, attend to the top-k blocks only — MoE-style routing applied to
attention, with causality enforced by masking future blocks [82].

Numbers: **6.5× prefill speedup at 1M tokens**, and **16× attention-compute reduction at 10M**
(block 4096, top-k=12 → 95.31% sparsity) [82]. At 128K the same setting is 62.5% sparse and RULER
scores **0.7818 vs 0.7849** for the full-attention model — a 0.3-point drop [82].

Training: the published result is a Llama‑8B‑1M that received 1M-token continued pretraining and
then 100B tokens with MoBA active [82]. So the headline numbers belong to a *trained* model.

But note the split that matters here: **MoBA's routing rule contains no learned parameters** — mean
pool, dot product, top-k. The thing that needs training is the model learning to live with 95%
sparsity, not the selection rule itself. A frozen checkpoint can adopt the router and expect the
"model was never trained for it" penalty, not an architectural mismatch.

**RETROFIT VERDICT: needs-retraining as published; its selection rule is adoptable-post-hoc as a
scoring kernel** (see §12 — this is the cheapest family on the list).

---

## 3. Quest (query-aware sparsity)

Mechanism: keep, per KV page (16 tokens in the paper), the **element-wise min and max of the keys**;
bound each page's contribution to a query by `Σ_d max(q_d·kmin_d, q_d·kmax_d)`, take top-k pages,
compute exact attention only inside them [83].

Numbers: **7.03× self-attention latency reduction** and **2.23× end-to-end** at 32K with a
2048-token budget; top-k filtering overhead is 5–10 µs below 128K [83]. Accuracy: near-perfect
passkey retrieval at **~1% of sequence length** (64 and 1024 tokens of budget on 10k and 100k tests),
with "negligible accuracy loss" on long-dependency tasks [83]. The same table shows H2O, TOVA and
StreamingLLM scoring near zero on passkey because they evict the answer before the question arrives
[83] — the cleanest published demonstration that eviction ≠ selection.

Training: none. It is a cache layout plus a top-k kernel; weights untouched.

Caveat for this engine: the bound is only valid if the positional term is included. Quest's corpus
has RoPE inside K, so the bound is over the full score. Inkling‑2‑Mini has **no RoPE** and a
relative-bias term b(i−j), which is not in K — a page bound must therefore add `max_j b(i−j)` over
that page's distance range, or the bound is unsound. See §12.

**RETROFIT VERDICT: adoptable-post-hoc.** No parameters, no profiling pass, and the only cost is
2·HD floats per page per KV head.

---

## 4. MInference 1.0

Mechanism: offline, classify each head into A-shape / Vertical-Slash / Block-Sparse from a few
samples, then at prefill time build the sparse index online and run only the selected diagonal,
vertical, slash and block tiles [84].

Numbers: **up to 10× prefill speedup on a single A100 at 1M tokens**, where dense prefill of an 8B
model takes 30 minutes; index construction is **5–20%** of the runtime [84]. Training: none —
"directly applied to existing LLMs without any modifications to the pre-training setup or additional
fine-tuning" [84]. Accuracy: claimed maintained on InfiniteBench, RULER, PG‑19 and NIAH [84].

The independent number that matters is in §11: at 128K on Llama‑3.1‑8B, MInference's official
Vertical-Slash configuration scores **54.12 on RULER versus 76.89 dense** — a 22.8-point collapse,
while its 4k–64k numbers look fine [85, Table 1]. Pattern-based prefill sparsity is cheap and
training-free and it falls apart exactly at the lengths this engine cares about.

**RETROFIT VERDICT: adoptable-post-hoc, but do not adopt its patterns.** The engineering (offline
head classification + online index) transfers; the A-shape/VS heuristic does not survive 128K.

---

## 5. XAttention (antidiagonal scoring)

Mechanism: score each candidate block by summing attention weights along its **antidiagonals**
(stride S=8/16), because in softmax attention mass concentrates on slashes; a dynamic-programming
pass picks a per-head minimum threshold so the block budget adapts per head [85].

Numbers: **up to 13.5× speedup at 256K** versus FlashInfer's FlashAttention, training-free and
plug-and-play [85]. Accuracy at 128K on Llama‑3.1‑8B‑Instruct, RULER average over 4k–128k:
**XAttention 88.47 (S=8) / 88.08 (S=16) versus dense 87.52**, with the 128K point at **72.31 / 71.11
versus 76.89** [85, Table 1]. So: −4.6 points at 128K, slightly *better* than dense on the average
because it wins at 16k–64k. This is the only training-free block-sparse method in that table that
holds up at 128K (FlexPrefill −2.2, MInference −22.8, SeerAttention −25.3 [85]).

Caveat: the antidiagonal trick leans on RoPE's angular structure. With relative-bias attention and no
RoPE there are no stable slashes to sum, so the *scoring function* does not transfer even though the
*thresholding policy* (per-head budget from a cheap proxy, chosen by DP) does.

**RETROFIT VERDICT: approximation-must-be-measured.** Training-free and the best 128K accuracy in
its own comparison, but its scoring signal is RoPE-specific; port the budget policy, re-derive the
score.

---

## 6. SpargeAttn

Mechanism: two-stage online filter — a fast prediction of the attention map from block-level
statistics (similarity, L1 norms, block maxima) plus a softmax-aware second filter — skipping whole
QKᵀ and PV tiles; combined with SageAttention quantization [86].

Numbers: "**2.5× to 5× faster** than existing dense and sparse attention" across language, image and
video models, with **1.83× end-to-end on Mochi** (L40) and no end-to-end metric loss [86]. Training:
none, "any model" by construction [86]. Accuracy evidence is mostly generative-quality metrics
rather than RULER/LongBench v2.

**RETROFIT VERDICT: approximation-must-be-measured.** Training-free and model-agnostic, but its
thresholds are tuned to GPU tile shapes and its published accuracy evidence is not long-context
retrieval.

---

## 7. LServe (unified static + dynamic sparsity)

Mechanism: convert half the heads into **static streaming heads** (window + sink only, "nearly
free") and give the rest dynamic, query-centric KV-page pruning; one kernel runs both patterns for
prefill and decode [87].

Numbers: **up to 2.9× prefill and 1.3–2.1× decode over vLLM**, maintaining long-context accuracy;
finding that "only a constant number of KV pages is required to preserve long-context and reasoning
capabilities, irrespective of context length" [87]. Training: none.

This is the closest published shape to Inkling‑2‑Mini's own design — most heads cheap, a minority
carrying global context — and it says the minority can be pruned to a *constant* page count. For a
model that already has 64/72 visits at window 1024, that is a direct statement that the remaining 8
visits do not need all N keys either.

**RETROFIT VERDICT: adoptable-post-hoc** for the page-pruning policy (weights untouched; the
static/dynamic head split is decided by profiling, and this checkpoint's split is already given).

---

## 8. Star Attention

Mechanism: two-phase block-sparse approximation — phase 1 runs blockwise-local attention over
sequence shards in parallel (each shard sees only itself plus an anchor block), phase 2 has the query
and generated tokens attend globally to all cached keys [88].

Numbers: "reducing memory requirements and inference time by **up to 11×** while preserving
**97–100% of accuracy**", ICML 2025, training-free, works with any globally-trained Transformer [88].

For a single device, phase 1 is exactly "chunked prefill with a local window plus anchors" — the N²
term disappears because no chunk ever attends to the whole prefix; the global information is
recovered only for the (few) query tokens in phase 2. That is a real, measurable 97–100% accuracy
retention on a frozen checkpoint, at the cost of a second pass.

**RETROFIT VERDICT: adoptable-post-hoc**, with the caveat that the accuracy guarantee is stated for
QA-style workloads where the queries come after the context; it is not a drop-in for tasks that need
full attention *inside* the context.

---

## 9. SeerAttention / SeerAttention‑R

Mechanism: add a small learnable **AttnGate** that pools Q and K along the sequence, projects them,
and predicts which blocks matter; train only the gate by self-distillation from the dense attention
map, leave every original weight frozen [89].

Numbers: **7.3× speedup at 128K at 90% sparsity**; gate training uses **0.5B tokens ≈ 40 A100-hours**
for Llama‑3.1‑8B‑Instruct [89]. SeerAttention‑R removes query pooling for decoding, trains the gate on
**0.4B tokens**, and keeps near-lossless reasoning accuracy at a 4K budget with block sizes 64/128
[90]. In the independent RULER comparison, SeerAttention's published configuration scores **51.61 at
128K vs 76.89 dense** [85] — 25 points down, i.e. the gate is only as good as its distillation data
and the evaluator's sparsity setting.

**RETROFIT VERDICT: approximation-must-be-measured** (backbone frozen, but ~40 GPU-hours of gate
distillation and a measured 128K accuracy cliff). For a pure-C CPU engine, 40 A100-hours is a
plausible one-off; the accuracy risk is the blocker.

---

## 10. RetrievalAttention

Mechanism: build an ANNS index over KV vectors in CPU memory and retrieve the top candidates per
query; the paper's contribution is handling the query/key **out-of-distribution** gap that makes
off-the-shelf indexes useless for attention [91].

Numbers: at 128K on an RTX 4090, **4.9× and 1.98× decode-latency reduction** versus exact-KNN and
traditional-ANNS retrieval, "while maintaining the same accuracy as full attention"; first 8B model
served on a single 24 GB card [91]. Without OOD handling, ANNS must scan **30–50% of all keys** to
reach acceptable accuracy [91] — the honest cost of index-based selection.

**RETROFIT VERDICT: approximation-must-be-measured**, and decode-shaped. It is a retrieval index,
not a prefill schedule; it also needs a trained index per checkpoint, which is a build step this
engine does not have.

---

## 11. KV eviction: FastGen, H2O, SnapKV, StreamingLLM (+ sinks)

These are the methods most often proposed for "long context on a small box", and all four are
**decode-phase** memory policies. Their prefill FLOP count is unchanged: they compute the full
attention matrix, then decide what to keep.

- **FastGen** [92]: profile per-head attention structure offline, then evict long-range context on
  local heads, non-special tokens on special-token heads, keep full cache only on broad heads. No
  fine-tuning. ~40% / 30% / 20% KV-memory reduction on Llama‑65B / 30B / 13B–7B at <1% win-rate
  loss; end-to-end speedup rises from 16.04% to 55.0% as generation grows 512 → 16k at batch 1 [92].
- **H2O** [93]: keep 20% "heavy hitter" tokens by accumulated attention. Up to **29×/29×/3×**
  throughput over DeepSpeed‑Zero‑Inference / HF‑Accelerate / FlexGen and **1.9×** lower latency at
  equal batch [93].
- **SnapKV** [94]: select KV positions per head from an observation window at the end of the prompt;
  fine-tuning-free; **3.6× generation speed and 8.2× memory efficiency at 16K** [94].
- **StreamingLLM** [95]: keep a few initial "sink" tokens plus a rolling window; stable to **4M
  tokens**, **22.2×** faster than sliding-window recomputation [95]. The sink phenomenon itself is
  now well characterised as a training-emergent fixed point, not a quirk [96].
- **ShadowKV** [109]: stores a low-rank key cache and offloads values to CPU, reconstructing a
  minimal sparse KV set per query; evaluated on RULER, LongBench and NIAH. Same category: it buys
  batch size and decode throughput, not prefill FLOPs.

Accuracy: passkey/retrieval tasks are where these break — Quest's Table 1 shows H2O, TOVA and
StreamingLLM at effectively zero on 10k/100k passkey because the answer is evicted before the
question is seen [83]. On LongBench-style QA the same methods look nearly free.

**RETROFIT VERDICT: not-applicable to the prefill question** (they cannot reduce O(N²) prefill by
construction), adoptable-post-hoc for decode memory. Do not let a decode speedup be reported as a
prefill fix.

---

## 12. DuoAttention

Mechanism: split heads into **retrieval heads** (full KV) and **streaming heads** (sink + window),
identify which is which by optimising a per-head gate on synthetic passkey data, then run two
different attention paths [97].

Numbers: memory **2.55× (MHA) / 1.67× (GQA)**, decoding **2.18× / 1.50×**, prefill **1.73× / 1.63×**,
with NIAH parity at a 25% (MHA) / 50% (GQA) full-attention ratio and better LongBench accuracy-memory
trade-off [97]. Training: no weight updates, but a synthetic-data optimisation pass to find the heads.

This is the closest thing in the literature to Inkling‑2‑Mini's existing structure — and the paper's
finding is that a *minority* of heads need full context. Here that minority is already fixed at 8 of
72 visits by architecture, so DuoAttention's contribution reduces to "the 8 full visits can also be
windowed" — which is precisely the approximation §13 quantifies, and which costs 1.6–1.7× prefill at
most because it leaves the 8 visits quadratic but short.

**RETROFIT VERDICT: approximation-must-be-measured** (head identification needed; 1.63× prefill on
GQA is real but small compared with what block-sparse selection on those 8 visits buys).

---

## 13. Lightning Attention and MiniMax‑01's hybrid linear attention

Mechanism: linear attention with a decay, computed in a chunkwise "intra-block quadratic +
inter-block linear" form so it can be tiled and parallelised like FlashAttention [98]; MiniMax‑01
stacks it with softmax attention in a fixed ratio and an MoE (32 experts, **456B total / 45.9B
activated**), trained at **1M** context and extrapolating to **4M** at inference [99].

Numbers: MiniMax‑01's contribution is architectural at scale, not a speedup table on a fixed model;
Lightning Attention‑2's own paper is explicit that the O(n) win is theoretical and that wall-clock
requires the IO-aware tiling it introduces [98].

Training: **mandatory**. A linear-attention layer is a different function class — it has no softmax
and no per-query key set; the QKV projections of a softmax checkpoint cannot be reinterpreted as a
linear-attention state. Accuracy on retrieval is only preserved because the hybrid keeps some
full-softmax layers [99].

**RETROFIT VERDICT: not-applicable** to a frozen checkpoint. This is the one family on the list
where the answer is "buy a different model", and it is the reason MiniMax‑01 exists as a
from-scratch training run rather than a patch.

---

## 14. 2026 work: sinks, streaming, and frozen-backbone routing

- **SinkRouter** (2026) [101]: argues the attention sink is a *training-constructed fixed point*,
  detects the sink signal and skips the computations that would produce near-zero output, with a
  Triton kernel doing block-level branching; training-free; **2.03× at 512K** with competitive
  accuracy on LongBench, InfiniteBench, CVBench, MileBench, MMVP. Decode-side.
- **SpotAttention** (2026) [102]: the most on-point 2026 result for a frozen checkpoint — a
  lightweight selector attaches to a **frozen** pretrained transformer and is trained by KL
  distillation to predict its attention distribution, then picks top-K keys per query with a
  top-p-derived per-layer budget. On Qwen3 (4B–32B) and Qwen3.5 (hybrid linear/full, 4B–9B) it
  **matches dense accuracy up to 128K, eight times its training length**; decode is **3.9× faster
  than FlashAttention and 1.8× faster than Twilight, the strongest training-free baseline**; its
  K-cache quantises to INT4/FP4 at 3.5× shrink with no accuracy cost [102]. Two lessons: (a) a
  distilled selector generalises ~8× beyond the length it was trained on, which is exactly the
  extrapolation a 32k-trained checkpoint needs to reach 262k; (b) even with distillation, parity is
  claimed to 128K, not beyond.
- **HiSA** (2026) [103]: sparse in the *hidden* dimension of the dot product rather than over
  tokens, using a static position-based rule tied to RoPE structure; computes ~1/3 of the QK
  products; trained from scratch, and the paper reports it *improves* on the full product. Requires
  training and requires RoPE — doubly inapplicable here.
- **HISA** (2026) [104]: hierarchical indexer for token-level sparse attention (as in DeepSeek Sparse
  Attention) — block-level coarse filtering first, then fine scoring — because the flat indexer scan
  is itself O(N) per query per layer. Relevant as a *design constraint*: whatever selector this
  engine adopts must be sub-linear in N, or the selection cost re-quadraticises the win.
- **Fine-tuning for sparse attention** (2026) [105]: shows a model can be fine-tuned to co-adapt with
  any KV policy on a single 40 GB A100, "often outperforming models trained with exact attention"
  [105]. This is the cheapest credible path *if* a measured accuracy loss forces a training step:
  policy first, 1–2B tokens of co-adaptation second — not a from-scratch run.
- **Block Sparse Flash Attention** (NeurIPS 2026) [106]: a drop-in block-sparse attention
  replacement for long-context inference; relevant as an existence proof that block-sparse can be
  packaged as a kernel swap rather than an architecture.
- **The Sparse Frontier** (ACL Findings 2026) [100] is the calibration study for all of the above:
  six training-free methods, sequences to 128K, sparsity to 0.95, nine tasks. Its findings: (i)
  sparser big models beat denser small ones at equal cost; (ii) **"for the training-free methods we
  study, fine-grained per-query importance estimation during prefilling remains impractical — due to
  both the cost of estimation and the lack of sparse kernels that translate fine-grained sparsity
  into wall-clock gains"**; (iii) **longer sequences tolerate higher sparsity**, so fixed budgets are
  sub-optimal [100].

Point (ii) is a GPU-kernel statement — in a scalar C engine, skipping a block *is* the wall-clock
win, and with this engine's decode-on-read TQ path it also skips the dequantisation that costs 7.5×.
Point (iii) is the licence to be aggressive at 262k–1M, which is the opposite of the regime MInference
was tuned for.

---

## 15. The answer: cheapest inference-only change, and its price

**Adopt per-page key-bound block selection (Quest's rule, MoBA's pooling as fallback) on the eight
full-attention visits only.** Sources: Quest [83] for the min/max bound, LServe [87] for "a constant
number of pages suffices", XAttention [85] for the per-head budget policy and for the only 128K
accuracy evidence that survives, The Sparse Frontier [100] for the length/sparsity trade-off.

Concretely, per full-attention visit, per KV head:

1. Chunk the prefix into pages of B=64 keys; store per page the element-wise min and max key vector
   (2·HD floats per page per KV head — 1 KB at f32, 1.6% of the 64 KB of f32 K+V in that page).
2. Always include the first page (attention sink [95, 96]) and the last 1024 keys (the window the
   model's other 64 visits already see; matching it keeps the local distribution in-distribution).
3. Score the remaining pages with the Quest bound `Σ_d max(q_d·kmin_d, q_d·kmax_d)`, plus the
   relative-bias term (§16), take top-k with **k=32 → 2048 keys/query**.
4. Run exact attention inside the selected pages. No weights, no gates, no profiling pass, no index
   build.

Why this and not the others: it is the only family that is (a) parameter-free, (b) validated at 128K
with a small measured loss, (c) expressible as "skip these block loops", which in a scalar engine
converts 1:1 into skipped MACs and skipped TQ dequantisation, and (d) sub-linear in selection cost
(2·HD MAC per page instead of HD per key).

Price, derived from §0's arithmetic at the same 40.6 GMAC/s:

| | 262k dense | 262k with k=32, B=64 | 1M dense | 1M with k=32, B=64 |
|---|---|---|---|---|
| Full-visit attention MAC | 3.38e15 | 2.64e13 (128× less) | 5.40e16 | 1.06e14 (512× less) |
| Page scoring overhead | — | 6.6e12 (0.2% of what it replaces) | — | 1.06e14 (0.2%) |
| Total prefill, derived | 31.7 h | **8.9 h** | 404 h | **36.3 h** |

So: 3.6× at 262k, 11× at 1M, and — the actual deliverable — **the curve becomes linear**: 4× the
tokens now costs 4.1× the time (8.9 h → 36.3 h) instead of 12.7×. The residual 8.9 h at 262k is
89% projections + MoE and 8% the existing 64 sliding-window visits; there is nothing left to win in
attention after this, which is why no further sparse trick helps until the linear path is faster.

Accuracy cost, and it must be measured: bound-style block selection at 1/128 sparsity has no
published measurement on this checkpoint. The nearest evidence is XAttention at 128K on
Llama‑3.1‑8B: **−4.6 RULER points at 128K at its default per-head budget, +1.0 on the 4k–128k average** [85],
with pattern-based selection at the same budget losing 22.8 (MInference) and 25.3 (SeerAttention)
[85]. Expect 2–5 RULER points, worst on aggregation/multi-hop (CWE, FVT) and best on single-needle
retrieval; LongBench v2 [107] is the honest second gate because it is reasoning-heavy rather than
retrieval-heavy. The Sparse Frontier's length result [100] says 262k should tolerate 1/128 better
than 128K does, and SpotAttention's 8× extrapolation [102] says a *distilled* selector would hold
better than a bound — that is the upgrade path if 2–5 points is too much.

If even 2 points is too much, the fallback is Star Attention's two-phase schedule [88] (97–100%
retention, at the cost of a second pass and a weaker guarantee for tasks needing full attention
inside the context), not KV eviction, which cannot touch prefill at all [92–95].

---

## 16. Two implementation details that decide whether this works here

**Relative bias breaks the bound unless it is included.** Inkling‑2‑Mini has `rope.dimension_count =
0` and a relative-position bias b(i−j) added to the logits. Quest's bound is over q·k only, valid
because RoPE lives inside K. Here, a page's maximum possible score is
`Σ_d max(q_d·kmin_d, q_d·kmax_d) + max_{j∈page} b(i−j)`. Over a 262k context the bias term can
dominate the q·k term for distant pages, so the bound must be evaluated per page-distance class
(bias is a function of i−j only, so it can be tabulated once per page offset and added during
scoring). Skipping this makes the bound unsound and the selection silently wrong — the failure mode
is not slowness, it is a model that ignores its context.

**Selection must stay sub-linear.** HISA's 2026 result [104] is that a token-level indexer scanning
the whole prefix per query re-quadraticises the problem it was meant to solve. Page-level scoring at
2·HD MAC per page keeps the overhead at 1/64 of the QK it replaces (0.2% of dense attention at
k=32), which is the reason to use pages of 64 and not tokens.

---

## 17. Verdict matrix

| Method | Speedup claimed (context) | Training needed | Long-context accuracy evidence | RETROFIT VERDICT |
|---|---|---|---|---|
| NSA [81] | 9.0× fwd / 6.0× bwd / 11.6× decode @64k | from scratch, 27B/260B tok | beats full attn; perfect 64k NIAH | needs-retraining |
| MoBA [82] | 6.5× prefill @1M; 16× @10M | 1M continued pretrain + 100B tok | RULER@128K 0.7818 vs 0.7849 | needs-retraining (router adoptable) |
| Quest [83] | 7.03× attn / 2.23× e2e @32k | none | passkey ~1% budget near-perfect | adoptable-post-hoc |
| MInference [84] | 10× prefill @1M (A100) | none | 54.12 vs 76.89 @128K [85] | adoptable-post-hoc, patterns rejected |
| XAttention [85] | 13.5× @256K | none | 72.31 vs 76.89 @128K; avg 88.47 vs 87.52 | approximation-must-be-measured |
| SpargeAttn [86] | 2.5–5× vs dense/sparse | none | generative metrics, not RULER | approximation-must-be-measured |
| LServe [87] | 2.9× prefill / 1.3–2.1× decode | none | long-context accuracy maintained | adoptable-post-hoc (policy) |
| Star Attention [88] | 11× time+memory | none | 97–100% of dense | adoptable-post-hoc |
| SeerAttention [89]/‑R [90] | 7.3× @128K, 90% sparse | gate only, 0.5B tok ≈40 A100-h | 51.61 vs 76.89 @128K [85]; AIME near-lossless @4K [90] | approximation-must-be-measured |
| RetrievalAttention [91] | 4.9×/1.98× decode @128K | index build | same as full attention | approximation-must-be-measured |
| FastGen [92] | 16%→55% e2e latency | none | <1% win-rate loss | not-applicable (decode) |
| H2O [93] | 29× throughput, 1.9× latency | none | fails passkey [83] | not-applicable (decode) |
| SnapKV [94] | 3.6× speed, 8.2× memory @16K | none | comparable on QA-style | not-applicable (decode) |
| StreamingLLM [95] | 22.2× vs window recompute, 4M tok | none | needs sink tokens; retrieval weak | not-applicable (decode) |
| DuoAttention [97] | 1.73×/1.63× prefill; 2.18×/1.50× decode | head search only | NIAH parity at 50% full heads (GQA) | approximation-must-be-measured |
| Lightning / MiniMax‑01 [98, 99] | 1M train / 4M inference | from scratch, 456B/45.9B | hybrid keeps retrieval | not-applicable |
| SinkRouter [101] | 2.03× @512K | none | competitive on LongBench/InfiniteBench | not-applicable (decode) |
| SpotAttention [102] | 3.9× decode @128K (vs FA) | selector distillation, backbone frozen | dense parity to 128K = 8× train length | approximation-must-be-measured |
| HiSA [103] | 1/3 of QK products | from scratch + RoPE | improves over full product | not-applicable |
| HISA [104] | indexer cost fix | none (indexer design) | — | adoptable-post-hoc (design constraint) |

---

## 18. What to measure before believing any of this

1. Baseline the eight full-attention visits alone at 8k and 16k on this engine (they should scale
   4× per doubling; if they do not, the O(N²) claim is about something else).
2. Implement §15 with k=8/16/32 and B=32/64; report RULER at 8k–32k (the lengths this engine can
   actually run today) and the derived 262k number, not a measured 262k number.
3. Gate on LongBench v2 [107] as well as RULER [108] and NIAH; the failure mode of block-sparse
   is aggregation and multi-hop, which NIAH cannot see.
4. If the loss exceeds 2–3 points, either raise k (cost is linear in k: k=64 still gives 64× at
   262k) or take the distilled-selector path of [102], which needs a training step but keeps the
   backbone frozen.

---

## Citations (ids 81–120)

| id | source | what it is cited for |
|---|---|---|
| 81 | Yuan et al., *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention*, arXiv:2502.11089 (Feb 2025) | 9.0× fwd / 6.0× bwd / 11.6× decode @64k; 27B/260B-token pretraining; post-hoc sparsity loses (top 20% attn = 70% of scores); perfect 64k NIAH |
| 82 | Lu et al., *MoBA: Mixture of Block Attention for Long-Context LLMs*, arXiv:2502.13189 (Feb 2025) | 6.5× prefill @1M; 16× @10M; 95.31% sparsity; RULER@128K 0.7818 vs 0.7849; 1M continued pretraining |
| 83 | Tang et al., *Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference*, arXiv:2406.10774 (ICML 2024) | min/max page bound; 7.03× attn / 2.23× e2e @32K; passkey near-perfect at ~1% budget; eviction baselines fail passkey |
| 84 | Jiang et al., *MInference 1.0*, arXiv:2407.02490 (NeurIPS 2024) | up to 10× prefill @1M on A100; 30 min dense 8B/1M; index build 5–20%; training-free |
| 85 | Xu et al., *XAttention: Block Sparse Attention with Antidiagonal Scoring*, arXiv:2503.16428 (2025) | 13.5× @256K; RULER Table 1 (dense 76.89 / MInference 54.12 / SeerAttn 51.61 / XAttn 72.31 @128K) |
| 86 | Zhang et al., *SpargeAttn*, arXiv:2502.18137 (ICML 2025) | 2.5–5× vs dense and sparse attention; 1.83× e2e Mochi; training-free, any model |
| 87 | Yang et al., *LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention*, arXiv:2502.14866 (MLSys 2025) | 2.9× prefill / 1.3–2.1× decode vs vLLM; half the heads to streaming; constant page count irrespective of length |
| 88 | Acharya et al., *Star Attention*, arXiv:2411.17116 (ICML 2025) | two-phase block-sparse; up to 11× time+memory; 97–100% accuracy retained |
| 89 | Gao et al., *SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs*, arXiv:2410.13276 | 7.3× @128K at 90% sparsity; gate-only self-distillation, 0.5B tokens ≈40 A100-hours |
| 90 | Gao et al., *SeerAttention-R*, arXiv:2506.08889 | gate trained on 0.4B tokens; near-lossless reasoning at 4K budget, block 64/128 |
| 91 | Liu et al., *RetrievalAttention*, arXiv:2409.10516 | 4.9×/1.98× decode @128K on RTX4090; 8B on 24 GB; ANNS needs 30–50% scan under OOD |
| 92 | Ge et al., *FastGen / Model Tells You What to Discard*, arXiv:2310.01801 | ~40/30/20% KV reduction; 16.04%→55.0% e2e speedup as generation 512→16k |
| 93 | Zhang et al., *H2O: Heavy-Hitter Oracle*, arXiv:2306.14048 (NeurIPS 2023) | 20% heavy hitters; 29×/29×/3× throughput; 1.9× latency |
| 94 | Li et al., *SnapKV*, arXiv:2404.14469 (NeurIPS 2024) | 3.6× generation speed, 8.2× memory @16K; fine-tuning-free |
| 95 | Xiao et al., *Efficient Streaming Language Models with Attention Sinks*, arXiv:2309.17453 | sink + window; 4M tokens; 22.2× vs sliding-window recompute |
| 96 | Gu et al., *When Attention Sink Emerges in Language Models*, arXiv:2410.10781 (ICLR 2025) | sink is training-emergent and universal, not incidental |
| 97 | Xiao et al., *DuoAttention*, arXiv:2410.10819 | retrieval vs streaming heads; 2.55×/1.67× memory, 2.18×/1.50× decode, 1.73×/1.63× prefill |
| 98 | Qin et al., *Lightning Attention-2*, arXiv:2401.04658 | chunkwise tiled linear attention; O(n) is theoretical, wall-clock needs IO-aware tiling |
| 99 | MiniMax, *MiniMax-01: Scaling Foundation Models with Lightning Attention*, arXiv:2501.08313 | 32 experts, 456B total / 45.9B active; 1M training context, 4M inference |
| 100 | Nawrot et al., *The Sparse Frontier*, arXiv:2504.17768 (ACL Findings 2026) | 6 methods, ≤128K, sparsity ≤0.95, 9 tasks; prefill per-query estimation impractical; longer sequences tolerate more sparsity |
| 101 | *SinkRouter*, arXiv:2604.16883 (2026) | training-free sink-aware routing; 2.03× @512K |
| 102 | Ahmad et al., *SpotAttention: Plug-In Block-Sparse Routing for Pretrained Long-Context Transformers*, arXiv:2606.22874 (2026) | frozen backbone + KL-distilled selector; dense parity to 128K (8× train length); 3.9× decode vs FA, 1.8× vs Twilight; INT4/FP4 K-cache 3.5× shrink |
| 103 | *HiSA: Sparsifying the Hidden Dimension of RoPE Attention*, arXiv:2611.26363 (2026) | hidden-dimension sparsity, ~1/3 of QK products, trained from scratch, RoPE-dependent |
| 104 | *HISA: Efficient Hierarchical Indexing for Fine-Grained Sparse Attention*, arXiv:2603.28458 (2026) | flat indexer scan is the bottleneck; two-stage coarse-then-fine selection |
| 105 | *Learning how to Forget: Fine-tuning for Long-Context Sparse Attention*, arXiv:2608.19920 (2026) | policy co-adaptation on one 40 GB A100; often beats exact-attention training |
| 106 | *Block Sparse Flash Attention*, NeurIPS 2026 | drop-in block-sparse kernel for long-context inference |
| 107 | Bai et al., *LongBench v2*, arXiv:2412.15204 | 503 MCQ, 8k–2M words, reasoning-heavy second gate |
| 108 | Hsieh et al., *RULER: What's the Real Context Size of Your Long-Context Language Models?*, arXiv:2404.06654 | benchmark definition and its limits vs plain NIAH |
| 109 | *ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference*, arXiv:2410.21465 | low-rank key cache + offloaded value cache, per-query minimal sparse KV reconstruction; RULER/LongBench/NIAH |
