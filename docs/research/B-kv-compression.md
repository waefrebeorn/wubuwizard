# KV-cache compression, 2024-2026: what is adoptable post-hoc on a frozen checkpoint

Status: research note, 2026-11-26. Citation ids 41-70, ledger entries in `docs/research/LEDGER.md`.
Audience: the `inkling2` pure-C engine. Everything below is either quoted from a source or
arithmetic derived from the stated local constants (marked *derived*).

## 0. Test article and the actual question

Inkling-2-Mini, as implemented: 48 layers, d=2304, HD=128, NH=24, NKV=8, 72 attention visits
= 8 full + 64 sliding-window(1024). Weights 7.9 GB, IQ2_XXS. Host 16 GB RAM, RTX 4050 6 GB.

The consequence that drives everything below: **only 8 of 72 visits grow with context**. The 64 SWA
visits are capped at 1024 tokens each. So marginal KV is

    8 layers x 8 KV heads x 128 dim x 2 (K,V) = 16,384 values/token   (*derived*)

and the fixed SWA floor is 64 x 1024 x 2048 values = 134.2 M values (*derived*).

Measured baseline (ours): 12.06 KB/token at 6 bits, i.e. 12,288 B/token of payload +0.5% metadata.
That matches 16,384 x 6/8 exactly, so the accounting model is sound and the whole problem is a
single scalar: **bits per coordinate in the 8 full layers**.

### 0.1 The bit budget for "1M under 8 GB"

| bits/coord | B/token | KV at 1M | note |
|---|---|---|---|
| 16 (fp16) | 32,768 | 32.0 GiB | no compression |
| 8 | 16,384 | 16.0 GiB | |
| 6 | 12,288 | 12.0 GiB | our measured floor for reference-exact output |
| 4.75 | 9,728 | 9.5 GiB | Jet-QSparse packed row, d=128 [59] |
| 4 | 8,192 | 8.00 GiB | exactly the budget, zero metadata room |
| 3.5 | 7,168 | 7.00 GiB | TurboQuant "quality neutral" point [51] |
| 3.5 + f16 norms | 7,424 | 7.25 GiB | what we would actually store |
| 3.0 | 6,144 | 6.00 GiB | TurboQuant "marginal" territory [51] |
| 2 | 4,096 | 4.00 GiB | KIVI/QJL regime [46][48] |

(*derived*: bytes = 16,384 x bits / 8; GiB assumes binary units.)

Adding the 56-96 MiB SWA floor and 256 B/token of f16 vector norms, **3.5 bits/channel lands at
7.3 GiB (7.84 GB) for 1M tokens**. With the 7.9 GB model that is 15.74 GB of a 16 GB box: it
*meets the stated KV budget* and leaves ~260 MB of headroom, which is not a working configuration
(no room for activations, page cache, or the OS). The honest target is **3.0-3.25 bits (6.0-6.5 GiB
KV, ~14.4-14.9 GB total)**, or 3.5 bits at ~85% token density.

Two hard facts fall out immediately:

1. **Pure post-hoc quantization cannot give reference-exact 1M.** Reference-exactness costs 6 bits
   (12.0 GiB). 4 bits is the absolute ceiling of the 8 GB budget and carries zero metadata room.
   The gap to close is 1.5x, and no codec closes 1.5x without giving up either bits or tokens.
2. **A 1.5x structural lever is required.** Either accept benchmark-neutral (not reference-exact)
   fidelity at 3.0-3.5 bits, or drop ~1/3 of the tokens in the 8 full layers at 6 bits
   (6 bits x 2/3 density = 8.0 GiB). §7 quantifies both.

---

## 1. The distinction that the marketing numbers hide: bytes vs work

Every 2025 "sparse attention" headline (NSA, MoBA, DSA) is a **work** reduction, not a **capacity**
reduction. They all still store every K/V vector; they skip *reading* most of them per query. On a
16 GB box the binding constraint is capacity, so these methods cannot solve the stated problem even
if they were free. This is worth stating plainly because their published speedups are routinely
quoted as if they were compression ratios.

| family | cuts stored bytes? | cuts work per query? | needs training? |
|---|---|---|---|
| MLA [41] | yes, ~25x | no (same FLOPs) | yes, architecture |
| NSA [44] | no (adds compressed blocks) | yes, 11.6x decode traffic | yes, pretraining |
| MoBA [45] | no | yes, 6.5x prefill at 1M | yes, continued pretraining |
| DSA [42] | no (adds indexer keys) | yes, top-2048 | yes, 943.7B tokens |
| quantisation [46-51] | yes, 2-8x | only via bandwidth | no |
| low-rank [54][55] | yes, 2-11x | no | calibration, sometimes fine-tune |
| cross-layer [52][53] | yes, 2x | no | yes |
| eviction/sparsity [56][61] | yes (drops tokens) | yes | no |

---

## 2. Trained-in architectural compression

### 2.1 MLA — Multi-head Latent Attention (DeepSeek-V2) [41]

Actual mechanism: per token, cache a single latent `c_t^KV` of dimension d_c plus a decoupled RoPE key
of dimension d_h^R. DeepSeek-V2 sets d_c=512, d_h^R=64, 60 layers, n_h=128, d_h=128 [41].
Cache per token = (d_c + d_h^R) x layers = 576 x 60 = **34,560 elements** [41, Table 9].

Reported compression, three different numbers that get conflated:

- **vs DeepSeek 67B (GQA): 93.3% reduction in KV cache, 5.76x max generation throughput** [41, abstract].
- **vs MHA at the same depth/width: 24.9x** — Table 9 gives 860.2K elements (MHA) vs 34.6K (MLA) on
  the 250B-parameter ablation model [41].
- **vs a GQA-8 model of Inkling's shape: 3.56x.** 576 vs 2,048 values per layer per token (*derived*).
  This is the number that matters to us, and it is 6x less impressive than the headline.

The `Absorb` trick matters more than the ratio: W_UK is folded into W_Q and W_UV into W_O, so the
latent is *never* expanded to per-head K/V. Attention runs on the 512-dim latent directly. That is
the only architecture in this survey that genuinely decodes inside the kernel by construction [55].

Requires training from scratch? **Yes for real MLA.** It is a learned down-projection; random init
destroys the model. The salvage path is TransMLA §4.2.

### 2.2 NSA — Native Sparse Attention [44]

Three parallel branches: compressed (block-level), selected (token-level top-n), and a fixed sliding
window. Pretrained on a 27B backbone with 260B tokens [44]. Headline numbers, precisely:

- **11.6x decode speedup is an *expected* figure**, not a measured kernel result. Table 4 of [44]
  lists memory-access volume in equivalent tokens (65,536 vs 5,632 at 64k context) and states the
  expected speedup is "approximately linear with the volume of memory access" → 11.6x. It is an
  arithmetic projection from bytes-read.
- **Measured: 9.0x forward and 6.0x backward at 64k** context on A100/Triton [44, §5].
- The 11.6x/9.0x numbers **only exist for a model pretrained with NSA**. Applied to a frozen
  full-attention checkpoint, the selection branch picks the wrong tokens: the model was never
  trained to route through compression/selection branches, and NSA's own ablation shows a
  from-scratch-vs-retrofit gap. There is no post-hoc mode.
- NSA does **not** shrink the cache. It adds compressed block representations on top of the full KV.

### 2.3 MoBA — Mixture of Block Attention (Moonshot) [45]

Block-partitioned KV, gated top-k block routing per query, MoE-style. Reported: **6.5x prefill
speedup at 1M tokens, 16x at 10M** (block size scaled to hold sparsity constant) [45, §2.3]. Trained:
1.5B models on 30B tokens with B=2048, top-k=3 [45], and Llama-3.1-8B continually pretrained with
context extended 128K→256K→512K→1M at B=4096 to make Llama-8B-1M-MoBA.

MoBA's routing is a parameter-free gate over block means, so it is *more* retrofit-friendly than NSA
in principle — but the paper's own claim is that models must see the sparse pattern during training.
No post-hoc accuracy number is published, and the gate's top-k is applied to blocks the model never
learned to prefer. Verdict: needs continued pretraining; a post-hoc version is an unmeasured
approximation (this is exactly what Jet-QSparse [59] later measures for block-sparse selection, and
it finds 128 blocks/head safe, 64 blocks/head model-dependent).

### 2.4 DSA — DeepSeek Sparse Attention (V3.2-Exp / V3.2) [42][43]

A "lightning indexer" scores all preceding tokens per query with a few FP8 heads, ReLU-gated, then
attention runs on the **top-2048** K/V entries only [42, §2.1]. Cost of adoption, in tokens:

- Dense warm-up: indexer only, 1,000 steps x 16 seqs x 128K = **2.1B tokens**.
- Sparse stage: all parameters, 15,000 steps x 480 seqs x 128K = **943.7B tokens**, LR 7.3e-6 [42, §2.1].
- Delivered API price cut >50% and is the only architectural change vs V3.1-Terminus [42][43].

So DSA is "continued training" in name and ~1T tokens in practice. It is the strongest evidence that
learned sparse selection needs the model to co-adapt — and it is **inapplicable post-hoc**.
Capacity: DSA needs an extra indexer key per token per layer (FP8) [42], so stored bytes go *up*.

---

## 3. Post-hoc quantisation: the numbers that count

### 3.1 KIVI [46] — the per-channel/per-token asymmetry

Two findings, both load-bearing:

1. **Keys must be quantised per-channel** (group along the channel dim, i.e. across tokens); **values
   per-token** [46, §3]. K channels carry persistent outliers that per-token scaling cannot confine.
2. A full-precision **residual window** of the most recent R tokens, R ≤ 128, absorbs the transient
   part [46].

Reported: 2-bit KV, **2.6x lower peak memory including weights, up to 4x batch, 2.35-3.47x
throughput**, tuning-free, no calibration [46, abstract].

This is the single most actionable diagnostic for our 4-bit failure. Our TQ quantises a **per-token
row** (1024-dim, one scale per row) for *both* K and V. That is the exact configuration KIVI
identifies as wrong for K. A Hadamard rotation homogeneises the row, which mitigates but does not
remove the channel-outlier structure — and our measured failure at 4 bits while 6 bits is
reference-exact is consistent with K-specific error, not V-specific. **Cheap experiment: keep V at
6 bits, push K to 4 bits with per-channel (cross-token) grouping inside a 128-token page.** Page-local
per-channel scales cost 128 dims x 2 B per 128 tokens = 2 B/token of metadata, i.e. free.

### 3.2 KVQuant [47] — sub-4-bit with calibration

Adds (i) per-channel K quantisation, (ii) **pre-RoPE** K quantisation, (iii) per-layer
sensitivity-weighted non-uniform datatypes, (iv) per-vector dense-and-sparse outlier isolation.
Result: **< 0.1 perplexity degradation at 3 bits**, 1.7x speedup, 10M-token context on 8 GPUs [47].
Offline calibration required (not retraining). For us, (ii) is moot (Inkling has no RoPE:
`rope.dimension_count = 0`, relative-bias attention), but (iv) is directly applicable and is the
missing piece in a symmetric-codebook-only design: a handful of outlier coordinates per vector need
to be stored separately rather than widening the codebook for everyone.

### 3.3 QJL [48] — 1-bit JL sketch, zero per-block constants

JL transform then a sign bit; the inner product uses an **asymmetric unbiased estimator**: one vector
sketched, the other transformed exactly. Guarantees relative distortion 1±ε on attention scores, and
the bit count is *independent of embedding dimension* [48, §3, Lemma 3.2/3.5]. Applied to K only, with
V at 2 bits, total 3 bits; **>5x KV memory reduction with no accuracy loss**, plus a lightweight CUDA
kernel for the sketch and estimator [48, abstract].

Why it matters here: it eliminates the per-row scale entirely (no normalisation constants to store),
and the score is computed from sign bits — no dequantisation, no f32 tile. The estimator is unbiased
for *inner products*, which is the quantity attention consumes; MSE-optimal scalar quantisers are
biased for it. That is precisely the failure mode of a nearest-centroid codebook optimised for row
MSE. A 2026 paper formalises QJL as 1-bit compressive sensing and derives the consequences for KV
compression [65], and a 2026 protein-LM paper reports QJL residual correction being what makes 3-bit
usable on top of orthogonal rotation [70] — i.e. the recipe (rotate, MSE-quantise, 1-bit QJL
residual) is being re-derived independently in adjacent fields.

### 3.4 PolarQuant [49][50]

Random preconditioning + recursive polar transform + per-angle codebook; the angle distribution after
preconditioning is analytically Beta and tightly concentrated, so **no per-block scale/zero-point is
stored at all**. **>4.2x compression** with the best long-context quality among compared methods [49].
The NeurIPS version [50] specialises this to the post-RoPE key cache and adds decoding acceleration.

Two cautions. First, the method's premise is post-RoPE key geometry with a standard 1/sqrt(d) scale.
Second — and this is the finding that should worry us — Open-TQ-Metal's cross-architecture sweep [60]
shows angular/rotational schemes succeed or fail based on the **attention scale factor, not model
size**: Gemma-4's `attn_scale = 1.0` amplifies directional error 25-100x versus Llama's 0.0884, with
KL divergence differing by that factor [60, §4]. Inkling has no RoPE and a learned relative bias, so
its key geometry is *neither* of the two validated regimes. Any rotational scheme (including our
Hadamard TQ) needs its scale factor checked before its error numbers transfer.

### 3.5 TurboQuant [51] — the strongest post-hoc claim at the bit-width we need

Data-oblivious, online, no calibration, no training. Random rotation induces a concentrated Beta
distribution on coordinates; coordinates become near-independent in high dimension, so an optimal
**scalar** quantiser per coordinate is near-optimal for the vector. Then the key move: because
MSE-optimal quantisers are **biased for inner products**, they add a **1-bit QJL transform on the
residual**, yielding an unbiased inner-product quantiser. Theory: within a **≈2.7x constant factor of
the information-theoretic lower bound** at all bit-widths and dimensions [51, abstract].

Measured quality: **"absolute quality neutrality" at 3.5 bits per channel; marginal degradation at
2.5 bits** [51, abstract]. ICLR 2026.

This is the number that matters: 3.5 bits is simultaneously (a) the point where the best post-hoc
method claims zero quality loss, and (b) the point where 1M fits in 7.3 GiB. Those two facts
coincide, which is why §7 recommends it.

### 3.6 The 2026 systems papers — what "operating on compressed KV" now means

These are the papers that answer "fused dequant vs decompress-to-f32":

- **BitDecoding** [57]: tensor-core-friendly layouts + **warp-level dequantisation parallelism** +
  software-pipelined dequant kernel, mixed precision, Hopper wgmma and Blackwell NVFP4 paths.
  **7.5x average decode speedup over FP16 FlashDecoding-v2, up to 8.6x on Blackwell with NVFP4, 4.3x
  over prior SOTA, 3x lower single-batch latency at 128K on Llama-3.1-8B.** Dequantisation happens
  inside the GEMM, into registers, per warp — the f32 tile is never materialised.
- **FOSCAR** [58]: the *real* bottleneck in 2-bit fused kernels is the per-element scale+offset
  before tensor cores; prior 2-bit kernels reach <31% (H100) / <15% (B200) of peak HBM. **Affine
  folding** applies scale/offset to the *other* operand or to the matmul output instead of to each
  value, cutting scale-and-offset ops by up to two orders of magnitude **while preserving the
  dequantised attention output in exact arithmetic**, at 11 (H100) and 0.19 (B200) ALU instructions
  per element. 3.36x (INT2) / 2.83x (INT4) over FlashInfer BF16 on H100 [58].
- **Open-TQ-Metal** [60]: fused compressed-domain attention on Metal, int4, **no intermediate
  dequantisation matrices**, **48x attention speedup at 128K over dequantise-then-attend**, KV 40 GB
  → 12.5 GB (3.2x), and **identical top-1 token predictions to FP16 under greedy decode** across 330
  experiments on Llama-3.1-70B and Gemma-4-31B. Also the PolarQuant attn_scale finding above.
  This is the strongest *measured* 4-bit fidelity claim in the literature — and it directly
  contradicts "4 bits cannot work". The difference is implementation: fused kernel, per-group scale,
  4.0-4.1 effective bits, no materialised temporary.
- **Jet-QSparse** [59]: **training-free** NVFP4 storage + **native FP4 QK on Blackwell tensor cores**,
  dense FP4 scan → 128-token block selection → FP16 value aggregation with FP32 accumulate,
  calibrated key centring. Row cost **d/2 + d/16 + 4 bytes = 4.75 bits/element at d=128, 4.625 at
  d=256; 3.37-3.46x packed-row compression**; measured pool = 28.9-29.7% of BF16 bytes/token.
  **128 blocks/KV head: no detected accuracy loss across four models and five benchmarks**; 64 blocks
  loses on some model-benchmark pairs. 1.94x end-to-end decode speedup at 128K, batch 16, GB300.
  Caveat for us: native FP4 needs Blackwell; RTX 4050 is Ada (sm89), so FP4 would be emulated — the
  FP4-native decode path is BitDecoding-FP4 [66].
- **Self-Indexing KVCache** [61]: sign-based 1-bit VQ that doubles as the *index* — the compressed key
  representation is the retrieval structure, no auxiliary index, no learned predictor. Custom sparse
  FlashAttention kernel, **6-7x speedup over FlashAttention-2**. This is the cleanest published
  instance of "don't decompress; the code IS the index".
- **Runtime-Certified Bounded-Error Quantized Attention** [62]: INT8 K / INT4 V in VRAM with FP16
  originals in RAM, a two-term per-head/per-step error bound computed online, and a fallback ladder
  that recovers exact dense output. Matches dense FP16 within noise on PG-19/NIAH/RULER to 128K on
  Llama-3.1-8B, and *recovers catastrophic failures that naive INT8/INT4 baselines exhibit*.
  The important negative result: naive per-tensor INT8/INT4 **does** fail catastrophically on some
  prompts; average-case benchmarks hide it. This is the argument for a per-step error check in our
  engine, and it is cheap (one bound per head per step).
- **MeltKV** [64]: memory-aligned **variable-bit** index packing, bit-widths assigned per grouped key
  channel, head-wise codebook sharing; sub-4-bit down to 1 bit; **48.1% less memory and 52.9% lower
  latency than SOTA VQ at comparable perplexity** (ICCAD 2026). Relevant because it attacks the thing
  that actually bites us at 3.5 bits: metadata padding. A 3.5-bit code in a 4-bit container is a 14%
  waste; variable-bit packing recovers it.
- **LoomKV** [63]: **lossless** — a calibrated per-head two-level 3-bit code over the BF16 *high
  bytes* (sign + 7 exponent bits), storing BF16 KV at **0.71 of raw size** (FP8 at 0.85-0.87) across
  nine models, bit-exact in tier mode, with a fused decode-attention kernel in pool mode that beats
  FlashAttention-3 on BF16. Only 1.4x, but it is *exact*, which no other method here is. If our
  fidelity requirement is truly reference-exactness, 1.4x exact + 6-bit lossy is a better product
  than 4x approximate.

---

## 4. Low-rank and cross-layer: the two levers that also cut bytes

### 4.1 Palu [54]

Offline SVD of W_K/W_V with truncation-aware decomposition and Fisher-information rank search, then
cache the low-rank latent and reconstruct **inside** the attention kernel ("online reconstruction").
**50% KV compression at 1.89x attention speedup**; combined with 2-3-bit quantisation, **91.25%
compression (11.4x)** with perplexity 1.19 lower than KVQuant at the same budget, Llama-2-7B,
PPL 5.12 → 5.25 at 30% [54]. Adaptive-rank follow-up in 2026 (STAR-KV, soft thresholding) [68].

Retrofit: **calibration + weight surgery, then measure**. The weights are replaced by a factor pair;
nothing is retrained, but the model's function changes. Crucially the decomposition needs a
calibration set and, for RoPE models, pre-RoPE handling. No RoPE here, which removes one obstacle and
one crutch.

### 4.2 TransMLA [55] — GQA → MLA conversion, and what it actually costs

Directly on point for "MLA requires training from scratch?" — **no, but the cheap version costs
accuracy**:

- 68.75% KV compression on Llama-2-7B, **training-free, 1.65% average drop across 6 benchmarks** [55].
- 93% compression, training-free, produces "meaningful output" but not usable quality; **6B tokens of
  fine-tuning restores most of it** [55].
- 10.6x speedup at 8K context via DeepSeek-compatible kernels (vLLM/SGLang) [55].
- Mechanism: RoPE folded into a single head (RoRoPE + FreqFold), balanced K/V magnitudes (BKV), then
  joint low-rank approximation [55].

**This recipe does not transfer to Inkling.** It is built entirely around decoupled RoPE and the Absorb
operation; `rope.dimension_count = 0` means there is no positional head to fold into and no RoPE
obstacle to work around — but also no RoPE structure to exploit for the low-rank split. A 2026 note on
MLA draft models reports that naive GQA→MLA conversion "can severely degrade" functional agreement,
i.e. the 1.65% figure is best-case and architecture-specific [67].

### 4.3 Cross-layer attention and cross-layer KV sharing [52][53]

CLA shares K/V heads between **adjacent layers** on top of GQA/MQA: **another 2x** KV reduction with
"nearly the same accuracy as unmodified MQA", demonstrated on 1B and 3B models **trained from
scratch** [52]. The systematic follow-up [53] unifies CLA/LCKV/YOCO into a "pizza/lasagna/sandwich"
framework and reports: at **2x reduction most configurations beat standard transformers on throughput
with competitive quality**; below that, pairing all layers' queries with upper-layer KVs wins, at
additional training cost and prefill latency [53]. All configurations are trained; none is post-hoc.

A 2024 architecture proposal (CLLA) claims KV compression to <2% by stacking head/dimension reduction,
cross-layer sharing and quantisation [69] — treat as an architecture, not a codec.

Post-hoc relevance: adjacent-layer KV similarity is real, but every published number for sharing it is
with trained weights. Retrofit = *approximation-must-be-measured*; expected accuracy cost is not
characterised for a frozen model, and the failure mode (queries at layer i attending to layer i-1's
keys) is not a rounding error but a functional substitution. **Do not attempt without a reference-exact
harness** — which we have, so it is measurable, but it is the riskiest lever in this document.

---

## 5. Token-side sparsity: the lever that composes with quantisation

Quantisation cuts bits; eviction cuts N. They multiply. Our 1.5x gap is exactly one such factor.

- **Quest** [56]: per-page metadata of min/max key values; a query estimates page criticality from
  that metadata and loads only the top-K pages. **7.03x self-attention latency reduction vs FlashInfer
  at 32K with a 2048-token budget; 2.23x end-to-end**; "negligible accuracy loss" on LongBench,
  passkey retrieval, and 256-4K token budgets [56]. No training, no calibration, metadata is 2x d
  floats per page (for us: 2 x 128 x 2 B per 128-token page per KV head = negligible).
- **Self-Indexing** [61]: the 1-bit sign code serves as the index — free selection, 6-7x kernel speedup.

For Inkling the page-granular form is the right one, because 8 full layers x 8 heads = 64 independent
page-metadata sets, all tiny, and block-contiguous reads keep the kernel bandwidth-friendly (NSA's
own §2 argues token-granular selection destroys FlashAttention-style locality [44]).

Modelling the combination: 6 bits x 2/3 density = 8.0 GiB at 1M; 6 bits x 1/2 density = 6.0 GiB.
The fidelity question is not "is top-k lossless" (it is not) but "does dropping 1/3 of blocks in the
8 global layers change the answer". Jet-QSparse's measured answer on four hybrid models is that 128
blocks/head (16K tokens of 128K) shows no detected loss and 64 blocks/head is model-dependent [59] —
that is 8x and 16x sparsity respectively, far more aggressive than the 1.5x we need, on models with
hybrid attention that already lean on their linear layers.

---

## 6. Decode-in-kernel vs decompress-then-attend

The distinction that decides whether prefill is O(N^2) *decompressions* or O(N^2) *cheap MACs*:

**Compute on the code, no f32 tile** (fused dequant into registers / estimator on codes):
MLA absorb [41][55], QJL sign-dot estimator [48], TurboQuant MSE+QJL [51], Palu online
reconstruction [54], BitDecoding warp-level dequant [57], FOSCAR affine folding [58], Open-TQ-Metal
[60], Jet-QSparse native FP4 QK [59], Self-Indexing [61], LoomKV pool mode [63], and all three
trained sparse methods (NSA/MoBA/DSA) which skip blocks entirely [42][44][45].

**Materialise f16/f32 then attend** (the thing we do now): KIVI [46], KVQuant [47], PolarQuant [49],
MeltKV [64] (index unpack then dequant).

Correction to our own framing: a fused kernel does **not** remove the O(N^2) work of prefill. It
reduces bytes moved by the compression ratio and removes the intermediate write. The exponent is
untouched. Only block sparsity (skip work) or a linear/SSM layer changes the exponent. So "fused
dequant" buys 3-5x bandwidth, not asymptotics — and on a 6 GB laptop GPU with the model already
spilling to host RAM, bandwidth is the binding constraint during decode anyway.

---

## 7. RETROFIT VERDICT

Legend: **ADOPT** = post-hoc, no weight change; **MEASURE** = post-hoc but changes the function, must
be validated against the reference-exact harness; **RETRAIN** = headline requires trained weights;
**N/A** = does not fit 48-layer GQA-8 with no RoPE, or does not reduce capacity.

| method | verdict | justification (one sentence) |
|---|---|---|
| MLA [41] | **RETRAIN** | the 93.3%/5.76x numbers are a from-scratch 8.1T-token model; for our shape the real ratio is 3.56x and the projection is learned. |
| TransMLA GQA→MLA [55] | **RETRAIN** (68.75% variant: MEASURE) | 68.75% is training-free at a measured 1.65% drop, but 93% needs 6B tokens and the whole recipe assumes decoupled RoPE, which Inkling does not have. |
| NSA [44] | **RETRAIN** | 11.6x decode is an arithmetic projection, measured is 9.0x forward / 6.0x backward, and all of it is on a 27B/260B-token pretrained backbone with no post-hoc mode. |
| MoBA [45] | **RETRAIN** | 6.5x@1M / 16x@10M come from 30B-token-trained routers and a 128K→1M continued-pretrained Llama. |
| DSA [42][43] | **RETRAIN** | 2.1B warm-up + 943.7B sparse tokens; it is the definition of co-adaptation, not a codec. |
| KIVI 2-bit [46] | **ADOPT** (2-bit fidelity: MEASURE) | tuning-free, no calibration, 2.6x peak-memory and 2.35-3.47x throughput, but 2 bits is below our measured floor and its per-channel-K finding is the fix for our 4-bit failure. |
| KVQuant 3-bit [47] | **ADOPT** (needs calibration) | <0.1 PPL at 3 bits with per-channel + outlier isolation; pre-RoPE trick is moot here (no RoPE), the outlier split is not. |
| QJL [48] | **ADOPT** | 3 bits total, >5x reduction, unbiased inner-product estimator with zero per-block constants, and the score is computable from sign bits with no dequantisation. |
| PolarQuant [49][50] | **MEASURE** | 4.2x with no stored scales, but it is validated only on post-RoPE 1/sqrt(d) keys and angular schemes swing 25-100x on attention scale alone [60]. |
| TurboQuant 3.5-bit [51] | **ADOPT — best post-hoc option** | data-oblivious, no calibration, within 2.7x of the distortion lower bound, and "absolute quality neutrality" at exactly the 3.5 bits that puts 1M at 7.3 GiB. |
| Jet-QSparse [59] | **ADOPT** (NVFP4 path N/A on Ada) | training-free 4.75 bits/element with 128-block budget showing no detected loss, but native FP4 QK needs Blackwell; the block-selection half is portable. |
| CLA / cross-layer sharing [52][53] | **RETRAIN** (sharing a frozen model: MEASURE) | every 2x number is from-scratch or continued-trained; retrofitting is a functional substitution, not a rounding error. |
| Palu low-rank [54] | **MEASURE** | 50% at 1.89x with calibration-only SVD and in-kernel reconstruction, but it rewrites W_K/W_V, so it must be diffed against reference-exact output. |
| Quest page-sparsity [56] | **ADOPT** (density: MEASURE) | min/max page metadata plus top-k page loads, 7.03x kernel / 2.23x e2e, negligible loss, and it multiplies with any bit-width. |
| Self-Indexing 1-bit VQ [61] | **MEASURE** | the sign code doubles as the index and the kernel is 6-7x FA2, but 1 bit as storage is far below our fidelity floor — use it as an index, not as the value store. |
| BitDecoding [57] | **ADOPT** (systems) | warp-level dequant inside the GEMM, 7.5x average / 8.6x NVFP4 — this is how to make our decode-on-read stop being O(N^2) decodes. |
| FOSCAR [58] | **ADOPT** (systems) | affine folding keeps the dequantised result exact in arithmetic while removing two orders of magnitude of scale/offset ops; directly portable to our int2/int4 path. |
| Open-TQ-Metal [60] | **ADOPT** (systems) | fused compressed-domain int4 with identical top-1 vs FP16 and 48x over dequantise-then-attend; the only measured 4-bit claim that survives contact. |
| LoomKV [63] | **ADOPT** (systems, 1.4x) | bit-exact 0.71x of BF16 via a 3-bit code over BF16 high bytes — the only lossless option, and the right choice if exactness is non-negotiable. |
| MeltKV [64] | **ADOPT** (packing) | variable-bit index packing recovers the metadata waste that a 3.5-bit code otherwise pays in a 4-bit container. |
| Runtime-certified attention [62] | **ADOPT** (safety) | per-head/per-step error bounds with FP16 fallback, and it documents that naive INT8/INT4 fails catastrophically on individual prompts while benchmarks look fine. |

**Explicit "headline only holds with trained weights" flags:** MLA 93.3%/5.76x [41], NSA
11.6x/9.0x/6.0x [44], MoBA 6.5x/16x [45], DSA's 50% price cut [43], CLA's 2x [52][53]. None of
these is reachable on the frozen checkpoint.

---

## 8. The hard answer

**Best post-hoc representation for 1M under ~8 GB: TurboQuant-style two-stage quantisation at 3.5
bits/channel (rotated MSE scalar code + 1-bit QJL residual on K), with V allowed down to ~3 bits.**

- 7.25 GiB of KV at 1M (7.42 KB/token incl. f16 norms) + 56 MiB SWA floor = **7.3 GiB** (*derived*).
- Claimed fidelity: "absolute quality neutrality" at 3.5 bits, within ≈2.7x of the
  information-theoretic distortion bound [51]. Independent systems corroboration at 4 bits: identical
  top-1 vs FP16 across 330 experiments [60]; at 3.5 bits the corroboration is benchmark-level, not
  token-exact.
- **What it is not:** reference-exact. Every number above is benchmark-level (PPL, NIAH, LongBench,
  top-1 on an eval set). Our measured floor for reference-exactness is 6 bits, and 6 bits does not fit
  at 1M by 4 GiB. That gap is real and no paper in this survey closes it.

**So the honest statement is: 1M under 8 GB at reference-exact fidelity is not achievable post-hoc.
1M under 8 GB at benchmark-neutral fidelity is achievable today at 3.5 bits, and 1M at reference-exact
fidelity needs ~6 bits + ~2/3 block density (8.0 GiB) or 3.5 bits + 85% density (~7.3 GiB).**

### 8.1 Why our 4-bit point failed, and the four experiments that would settle it

Our TQ is: per-1024-dim row, random-sign Hadamard, per-coordinate nearest centroid from a symmetric
codebook, one scale per row. Against the literature that is three identifiable deviations:

1. **K is quantised per-token, not per-channel.** KIVI's central result is that this is the wrong axis
   for K [46]. Fix: group K across tokens within a 128-token page, one scale per channel per page
   (2 B/token of metadata at HD=128 — free). This alone may recover 4-bit for K.
2. **The codebook optimises row MSE, but attention consumes inner products.** MSE-optimal scalar
   quantisers are biased for ⟨q,k⟩; the fix is a 1-bit QJL residual on K [48][51]. Cheap: one sign bit
   per coordinate plus a per-vector norm, and the score becomes `⟨q,k̂⟩ + s·⟨q,sign(r)⟩`.
3. **Outliers are not isolated.** One symmetric codebook for all coordinates means a handful of
   outlier channels widen the range for everyone. KVQuant's per-vector dense-and-sparse split [47] is
   the standard fix, and at HD=128 storing 2-4 outliers as (index, f16) costs ~24 B per 128 values =
   1.5 bits/value — only worth it on K.

Ranked experiments, each one afternoon on the existing harness:

1. V at 6 bits / K at 4 bits, per-channel grouping inside pages. If reference-exactness holds, the
   problem is K-only and 4.75-5 effective bits fits 1M at 9.5-10 GiB (still over budget, but it tells
   us where the error lives).
2. Add the 1-bit QJL residual on K at 3.5 bits total. This is the TurboQuant recipe and the single
   highest-value test in this document: it is the only configuration at the fitting bit-width with a
   published neutrality claim.
3. If (2) is not reference-exact, add Quest-style min/max page metadata and sweep 6-bit density
   100% → 50%. Report the density where reference-exactness breaks; that number is the real product
   spec.
4. Independent of all of the above: implement a runtime error bound [62] — one bound per head per
   step, FP16 fallback for the affected head. On a 16 GB box where we are 260 MB from the edge, a
   per-step certificate is worth more than any further ratio.

### 8.2 What to do about prefill

Our prefill is O(N^2) decodes because we decompress on read. The fix is not a better codec, it is
FOSCAR-style affine folding [58] or BitDecoding-style warp-level dequant [57]: dequantise into
registers inside the score computation and never write an f32 tile. That converts prefill from
O(N^2) decompressions to O(N^2) int-FMA with a codebook gather — a 3-5x constant, and it makes the
3.5-bit path's in-kernel property real rather than nominal. It does not reduce the exponent; only
block sparsity does, and block sparsity on a frozen checkpoint is an unmeasured approximation
(Jet-QSparse's 128-blocks-safe result [59] is the closest evidence, on hybrid models).

---

## References (ids 41-70)

[41] DeepSeek-AI. *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model.* arXiv:2405.04434. https://arxiv.org/abs/2405.04434
[42] DeepSeek-AI. *DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models.* arXiv:2512.02556. https://arxiv.org/abs/2512.02556
[43] DeepSeek. *Introducing DeepSeek-V3.2-Exp.* 2025-09-29. https://www.deepseek.com/en/news/v3-2-exp/
[44] Yuan, J. et al. *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention.* arXiv:2502.11089. https://arxiv.org/abs/2502.11089
[45] Lu, E. et al. *MoBA: Mixture of Block Attention for Long-Context LLMs.* arXiv:2502.13189. https://arxiv.org/abs/2502.13189
[46] Liu, Z. et al. *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.* ICML 2024, arXiv:2402.02750. https://arxiv.org/abs/2402.02750
[47] Hooper, C. et al. *KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization.* arXiv:2401.18079. https://arxiv.org/abs/2401.18079
[48] Zandieh, A., Daliri, M., Han, I. *QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead.* arXiv:2406.03482. https://arxiv.org/abs/2406.03482
[49] Han, I. et al. *PolarQuant: Quantizing KV Caches with Polar Transformation.* arXiv:2502.02617. https://arxiv.org/abs/2502.02617
[50] Wu, S. et al. *PolarQuant: Leveraging Polar Transformation for Key Cache Quantization and Decoding Acceleration.* NeurIPS 2025, arXiv:2502.00527. https://arxiv.org/abs/2502.00527
[51] Zandieh, A., Daliri, M., Hadian, M., Mirrokni, V. *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.* ICLR 2026, arXiv:2504.19874. https://arxiv.org/abs/2504.19874
[52] Brandon, W. et al. *Reducing Transformer Key-Value Cache Size with Cross-Layer Attention.* arXiv:2405.12981. https://arxiv.org/abs/2405.12981
[53] Wu, Y., Wu, H., Tu, K. *A Systematic Study of Cross-Layer KV Sharing for Efficient LLM Inference.* arXiv:2410.14442. https://arxiv.org/abs/2410.14442
[54] Chang, C.-C. et al. *Palu: KV-Cache Compression with Low-Rank Projection.* ICLR 2025, arXiv:2407.21118. https://arxiv.org/abs/2407.21118
[55] Meng, F. et al. *TransMLA: MLA Is All You Need.* NeurIPS 2025, arXiv:2502.07864. https://arxiv.org/abs/2502.07864
[56] Tang, J. et al. *Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference.* ICML 2024, arXiv:2406.10774. https://arxiv.org/abs/2406.10774
[57] Du, D. et al. *BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache.* arXiv:2503.18773. https://arxiv.org/abs/2503.18773
[58] Zhuang, D. et al. *FOSCAR: Accelerating Decode Attention over 2- and 4-Bit KV Caches through Affine Folding.* arXiv:2611.09722. https://arxiv.org/abs/2611.09722
[59] Tang, H. et al. *Jet-QSparse: Accelerating Agentic LLM Inference with 4-Bit Sparse Attention.* arXiv:2611.01022. https://arxiv.org/abs/2611.01022
[60] Vegasena, S. *Open-TQ-Metal: Fused Compressed-Domain Attention for Long-Context LLM Inference on Apple Silicon.* arXiv:2604.16957. https://arxiv.org/abs/2604.16957
[61] Yang, X. et al. *Self-Indexing KVCache: Predicting Sparse Attention from Compressed Keys.* arXiv:2603.14224. https://arxiv.org/abs/2603.14224
[62] Calver, D. *Runtime-Certified Bounded-Error Quantized Attention: Per-Head, Per-Step Error Bounds for Compressed KV Caches with Dense FP16 Fallback.* arXiv:2605.20868. https://arxiv.org/abs/2605.20868
[63] Zhu, W. et al. *LoomKV: Lossless KV Cache Compression from GPU Memory to SSD for LLM Serving.* arXiv:2611.32954. https://arxiv.org/abs/2611.32954
[64] Han, J., Cho, S., Choi, S. *MeltKV: Memory-Aligned KV Cache Compression via Variable-Bit Packed Vector Quantization for Long-Context LLMs.* ICCAD 2026, arXiv:2611.18652. https://arxiv.org/abs/2611.18652
[65] *QJL is 1-bit Compressive Sensing: An Equivalence and Its Consequences for KV Cache Compression in LLMs.* ICML 2026. https://icml.cc/virtual/2026/75315
[66] Du, D. et al. *BitDecoding-FP4: Native FP4 KV Cache Decoding via Algorithm-System Co-Design.* arXiv:2611.09825. https://arxiv.org/abs/2611.09825
[67] *Functional Reconstruction for MLA Draft Models.* arXiv:2607.27269. https://arxiv.org/pdf/2607.27269
[68] *STAR-KV: Low-Rank KV Cache Compression via Soft Thresholding for Adaptive Rank Control.* arXiv:2606.08382. https://arxiv.org/pdf/2606.08382
[69] *Lossless KV Cache Compression to 2% (CLLA).* arXiv:2410.15252. https://arxiv.org/pdf/2410.15252
[70] *TurboESM: Ultra-Efficient 3-Bit KV Cache Quantization for Protein Language Models with Orthogonal Rotation and QJL Correction.* arXiv:2603.26110. https://arxiv.org/html/2603.26110
