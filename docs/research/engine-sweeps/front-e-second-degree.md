# Front E — second-degree traversal of the engine sweep outbound links

**Date:** 2026-11-22 · **Budget actually spent:** 10 `web_search`, 17 `web_extract` (one arXiv HTML 404 recovered via ar5iv).
**Target:** everyman inference, 6 GB VRAM / 16 GB RAM / SSD, reimplement in our own C11, never vendor.

## Method

Read `docs/research/engine-sweeps/front-d-recent-2026-11.md` first, then `front-a-serving.md`,
`front-b-edge.md`, `front-c-offload.md`, `local-audit.md`, `docs/ENGINE-GAP-REGISTER.md` and
`docs/ENGINE-SOURCE-INDEX.md`. Extracted all 119 distinct URLs from those five sweep reports and
followed the **outbound** ones that are hubs rather than endpoints: the awesome-speculative-decoding
list (a front-c source), the SGLang speculative-decoding documentation (a front-d/front-c source),
the TensorRT-LLM sparse-attention page and its "Further Reading" Blog 17 (a front-a/front-c source),
and the `awesome-pro/hiqcache` GitHub repo linked from front-a. From those pages followed the
paper references and repo symbols to primary papers.

**Diff rule.** Every finding was checked against all 46 EG rows and the already-implemented list.
Only mechanisms whose name/concept does not appear in `ENGINE-GAP-REGISTER.md` are reported.
Where a mechanism is adjacent to an existing gap, the "Why not EG" line says exactly which words
the gap does and does not cover. **A reported speedup is the author's number, not our measurement.**

---

## E1 — UNO: gated-LoRA diffusion drafter on the verifier's own backbone

- **Source / date:** arXiv 2609.04010, *Unlocking Lossless Speedups in LLMs via Discrete Diffusion*, 3 Sep 2026 — https://ar5iv.labs.arxiv.org/html/2609.04010 ; SGLang speculative-decoding docs (current) — https://docs.sglang.io/docs/advanced_features/speculative_decoding.md
- **Traversal:** front-d D3/D4 SGLang blogs → SGLang speculative-decoding doc → "UNO" section → paper.
- **Exact mechanism:** split weights into frozen AR θ_AR and lightweight diffusion θ_Δ implemented as rank-128 LoRA on every q/k/v/o/gate/up/down projection. One forward pass over `[clean context, noisy block]` with a block-causal mask produces **both** distributions: rows flagged clean skip the adapter and emit verifier p; rows flagged noisy add the LoRA delta and emit drafter q. A Ψ-Spec rejection sampler makes the output distribution identical to the base AR model. One KV cache, not two. θ_Δ is trained after the AR model is frozen by discrete consistency distillation (uniform-state discrete diffusion, one denoising step per block) with a KL term plus a total-variation term rewarding longer accepted prefixes; gated LoRA disables adapters at clean positions.
- **Upstream pointer:** paper §3 (diffusion-augmented model / gated LoRA), §4 (Ψ-Spec sampler), §5.1 throughput table; SGLang `--speculative-algorithm UNO`, `--uno-lora-path`, B/K/V width semantics ("first row of each request uses base weights, remaining B−1 rows use a trained UNO LoRA; target verification is entirely base-only"), `--speculative-use-block-verification`; repo `ifm-ai/uno`.
- **T1 fit:** HIGH — no second draft model and no second KV cache; the only extra residency is a LoRA delta. TP=PP=1 constraint matches a single 6 GB card.
- **Why not EG-01..46:** EG-17 is *multi-adapter* LoRA serving (Punica `bgmv`/`sgmv` grouped apply, `LoRALRUCache`) — here the LoRA is a fixed self-drafter, not a request-selected slot. EG-24 EAGLE-3 needs a separate draft model/feature head; EG-25 lists Sequoia/SpecTr/DAG/recycling/relaxed verification, not gated same-weight drafting; EG-43 quantizes a draft KV that UNO does not have. "Diffusion", "gated LoRA", "single KV cache self-draft" have zero hits in the register.

## E2 — HiQCache: INT8-encoded host L2 KV tier with a row-local record codec

- **Source / date:** https://github.com/awesome-pro/hiqcache (README, `docs/design.md`, `docs/sglang-integration.md`), commits Sep 25 / Oct 24 / Nov 10 2026.
- **Traversal:** front-a source list → `awesome-pro/hiqcache` `sglang-integration.md` already in sweep → repo README and `docs/design.md`.
- **Exact mechanism:** keep active KV BF16 in GPU L1; compress **only on eviction** into an INT8 host L2 representation; restore to BF16 on H2D before attention. Intercepts exactly the two methods through which 100% of L1↔L2 bytes flow (`backup_from_device_all_layer_physical` / `load_to_device_per_layer_physical`), so radix tree, eviction policy and scheduler are untouched. One aligned 1152-byte record per `(token, layer, K|V)` row: 1024 B INT8 payload (8 heads × 128 dims), 16 B BF16 per-head scales, 112 B padding so `ROW_BYTES = 9 × 128` matches the CUDA mover's 128-byte-group requirement (2048 B BF16 baseline → 43.75% reduction). Symmetric per-head absmax `s = max(|x|)/127`, `q = clamp(round(x/s), -127, 127)`, quotient formed in FP32, `AMAX_FLOOR = 2^-112` for zero heads; a fail-fast matrix rejects incompatible layouts (`page_size != 1`, layer sharding, MTP draft pools, device-side quantized KV, `head_num*head_dim != 1024`). Authors report 54,254 → 96,451 tokens at equal host budget and ~57% → ~73% hit rate on Qwen3-8B/A6000.
- **Upstream pointer:** `python/sglang/srt/mem_cache/pool_host/int8_codec.py`, `int8_staging.py`, `mha_int8.py`, one dispatch branch in `mha.py:get_mha_host_pool_cls`, two env vars in `environ.py`; interception at `l2_transfer.py:136/169` and `base.py:371/378`; `docs/design.md` §§1–4, 7, 11.
- **T1 fit:** HIGH — exactly the 16 GB host-RAM tier, no VRAM cost, and the record stays on the existing JIT mover.
- **Why not EG-01..46:** EG-18 (HiCache L1/L2/L3) is tiers, prefetch/write-back and deadlines but never changes the *representation* of a tier slot; EG-28 is a budget-scored lazy KV quantization in the device cache, not a host-tier record with 128-byte alignment; EG-36 is OCP FP8 KV for attention; EG-37 is one arena. "INT8" has zero hits in the register; no gap names a host L2 codec or per-head absmax row-local encoding.

## E3 — FR-Spec: frequency-ranked draft vocabulary compression

- **Source / date:** arXiv 2502.14856 v2, 11 Mar 2025 — https://arxiv.org/html/2502.14856
- **Traversal:** SGLang speculative-decoding doc "EAGLE-2 Decoding via Frequency-Ranked Speculative Sampling" → FR-Spec paper.
- **Exact mechanism:** build a high-frequency token subset V_high ⊂ V from corpus statistics; restrict only the **draft** LM-head projection to the subset (`W̃_LM`), keeping the full vocabulary for verification so output distribution is unchanged. Draft distribution `D_FR(x) = Softmax(H_D(x) W̃_LMᵀ)` (Eq. 4) cuts the projection and softmax from `O(nd|V|)` to `O(nd|V_high|)`. Training-free, plug-and-play with EAGLE-2 (authors' 1.12×) and Medusa (1.08×); authors report 75% draft LM-head reduction.
- **Upstream pointer:** paper §2 "Frequency-Ranked Speculative Sampling", Eq. (2)→(4), §§4.2–4.3; repo `thunlp/FR-Spec`; SGLang `--speculative-token-map` and `thunlp/LLaMA3-Instruct-8B-FR-Spec` vocabulary map.
- **T1 fit:** HIGH — the draft LM head is a large share of decode work on a 6 GB card with 128k vocab; no retraining, no extra memory beyond a token-ID map.
- **Why not EG-01..46:** "frequency" has zero hits in the register. EG-24 names EAGLE-3 direct token prediction + tri-layer fusion only; EG-25 lists Sequoia/SpecTr/DAG/recycling/relaxed verification only; EG-43 is draft-KV quantization, not vocabulary compression. Nothing in the 46 covers restricting the draft projection to a hot-token subset while verifying in full V.

## E4 — SpecPrefill: draft-attention prompt compression to cut TTFT

- **Source / date:** arXiv 2502.02789, Feb 2025 — https://ar5iv.labs.arxiv.org/html/2502.02789 ; cross-family follow-up arXiv 2603.02631 (2026).
- **Traversal:** awesome-speculative-decoding references → SpecPrefill paper → its citing cross-family paper.
- **Exact mechanism:** a lightweight speculator runs N lookahead steps on the prompt; from its attention tensor `[N, L, S, H]` take max over layers and heads then mean over lookahead to get one scalar importance per prompt token; smooth with 1D average pooling, chunk the prompt contiguously, average within each chunk, keep top-K chunks. Selected tokens are sent to the base model **with their original positions**; all other prompt tokens are dropped. Training-free; no change to the base model. Cross-family variant retokenizes the selected spans with the target tokenizer so the speculator need not share a tokenizer. Authors report up to 7× end-to-end QPS and 7.66× TTFT on Llama-3.1-405B-FP8; cross-family 128k→16k compression gives ~18× TTFT reduction.
- **Upstream pointer:** paper §3.2.1 lookahead, §3.2.2 max-mean aggregation, §3.2.3 chunk selection, §3.2.4 position handling, Appendix E FLOP-overhead table; repo `Jingyu6/speculative_prefill`.
- **T1 fit:** MEDIUM–HIGH — prefill is compute-bound on 6 GB; needs a small speculator already resident on SSD/disk, no fine-tune.
- **Why not EG-01..46:** no gap covers dropping prompt tokens before the base model prefill. EG-23 and EG-44 are sparse attention *inside* the kernel (block skipping / trained indexer) and still process the full prompt; EG-25 is decode-time speculation. The five "prefill" hits in the register are chunked prefill (implemented), the PD gate (EG-15), lazy KV quant (EG-28) and "no-regress" prose — none is token-importance prompt compression.

## E5 — Online Speculative Decoding: drafter retrained from live verification corrections

- **Source / date:** arXiv 2310.07177 v4, 10 Jun 2024 — https://arxiv.org/html/2310.07177
- **Traversal:** awesome-speculative-decoding list → OSD paper/repo.
- **Exact mechanism:** during normal speculation, record the token index where the draft was rejected plus both draft and target distributions; buffer them; every I iterations (I dynamically adjustable) fine-tune the draft model by online knowledge distillation with the target as teacher, using the target's corrections as supervision. Updates can run in background or on otherwise-idle serving capacity. Authors report acceptance-rate gains of 0.1–0.65 and 1.42–2.17× latency reduction.
- **Upstream pointer:** paper §4.1 "Knowledge Distillation for Speculative Decoding", §4.2.1 Algorithm 1 (`error_index`, target logits, update every I), §4.2.2 latency & FLOPs analysis; repo `LiuXiaoxuanPKU/OSD`.
- **T1 fit:** MEDIUM — a tiny drafter update fits the 16 GB host pool and can be overlapped; acceptance gains matter more than generation length for short local outputs.
- **Why not EG-01..46:** EG-42 (TVP) is a *training objective* (acceptance probability = 1 − total-variation distance), not post-deployment weight updates from rejected-token logits; EG-41 (SpecSearch) searches a static configuration, not drafter weights; EG-25 names no adaptation. "online", "distillation" and "correction" have zero hits in the register.

## E6 — LiLiCorr: single-pass candidate-lattice correlation for block drafts

- **Source / date:** arXiv 2608.20530 v2, 22 Sep 2026 — https://arxiv.org/html/2608.20530 ; SGLang docs and vLLM Ascend speculative-decoding guide.
- **Traversal:** SGLang speculative-decoding doc "DFlash Decoding / LiLiCorr" → paper; vLLM Ascend guide as second implementation.
- **Exact mechanism:** a block-diffusion drafter emits per-slot marginals, so per-slot argmax is locally plausible but jointly incoherent. Keep top-K candidates at each of B−1 slots; a small two-layer Transformer attends jointly over the whole `(B−1)×K` lattice (competitors within a slot and candidates at neighbouring slots), fuses each node with a projection of the target's last verified hidden state (concatenate candidate state, projection and their elementwise product), and emits unit-normalized `in` and `out` vectors. Adjacent compatibility is `cos(out_i, in_{i+1})`; all pairs are batched matrix products, and only a greedy argmax walk (no network weights) commits the path. The exponential joint distribution is never materialized. Optional sampling draws each slot from `softmax(scores/T)` and publishes that proposal so rejection sampling stays exact. Co-training the drafter with the head raises accepted length 7–19% at ~3% of per-block latency.
- **Upstream pointer:** paper §1 Fig. 1 and §3 (lattice, in/out vectors, greedy commit), training coverage clause; SGLang `LiLiCorrDraftModel`, `lilicorr_*` fields in `dflash_config`, `SGLANG_ENABLE_LILICORR_SAMPLING`; vLLM Ascend `LiLiCorr` guide.
- **T1 fit:** MEDIUM — one extra small head folded into the draft CUDA graph, no extra target pass; helps acceptance, which is what a small local drafter needs.
- **Why not EG-01..46:** EG-25 does not name lattice reranking; EG-21 is full *Medusa tree verification* at the target, not draft-time candidate correlation; EG-42 is a training objective. "lattice" and "rerank" have zero hits in the register.

## E7 — DFlash: block-diffusion parallel drafter with fused target-feature KV injection

- **Source / date:** arXiv 2602.06036 v1, 5 Feb 2026 — https://arxiv.org/html/2602.06036v1 ; follow-up DynaDFlash arXiv 2611.19085.
- **Traversal:** LiLiCorr references and SGLang spec doc → DFlash paper → DynaDFlash.
- **Exact mechanism:** a lightweight block diffusion model drafts a whole masked block in one non-causal forward pass. Target hidden states from a fixed set of layers (paper example: 2, 10, 18, 26, 34) are fused by an FC layer into a context feature `c_t`, injected into the **K and V projections of every draft layer** and stored in the draft KV cache for reuse across iterations. Training samples anchor tokens and masks the rest so training matches the bonus-token-at-inference condition; a position-dependent loss decay weights early tokens more. Authors report 6.1× lossless acceleration and ~2.5× over EAGLE-3. DynaDFlash adds a per-draft-layer linear gate on the fused projection instead of one shared representation.
- **Upstream pointer:** paper §4 Method, §4.2 Training, Appendix A.3.1 (loss decay) and A.3.2 (random anchor sampling); repo `z-lab/dflash`.
- **T1 fit:** MEDIUM — a second small model must still fit 6 GB, but drafting cost collapses to one pass; no change to target kernels.
- **Why not EG-01..46:** front-d §D6 mentions DFlash only as a SpecSearch *benchmark datum*, not as a mechanism. EG-24 is autoregressive EAGLE-3 feature prediction; EG-43 quantizes a draft KV, it does not create a diffusion drafter. "diffusion" has zero hits in the register.

## E8 — Gumiho: asymmetric serial/parallel draft heads + Full Tree Attention

- **Source / date:** arXiv 2503.10135, ICML 2025 — https://arxiv.org/html/2503.10135 ; AMD ROCm blog and repo `AMD-AIG-AIMA/Gumiho`.
- **Traversal:** awesome-speculative-decoding list → Gumiho paper/repo/blog.
- **Exact mechanism:** Theorem 3.1 proves that, for a fixed parameter/time budget, accuracy on *early* draft positions dominates mean accepted length because the first mismatch discards the rest. The head architecture is a two-layer serial Transformer for the first two draft tokens plus five independent parallel MLP heads for the remaining five. Full Tree Attention (FTA) fixes short candidate paths by borrowing tokens from longer paths at the corresponding positions, reusing already-computed Q/K/V at no extra compute; top-8 paths are verified. Authors report 4.5–15.8% over EAGLE-2.
- **Upstream pointer:** Theorem 3.1, §3.3 "Gumiho" (heads and FTA), §4.1 setup; repo `AMD-AIG-AIMA/Gumiho`.
- **T1 fit:** MEDIUM — cheap asymmetric heads, no second model, but a trained draft checkpoint is needed.
- **Why not EG-01..46:** EG-21 is *full Medusa tree verification* inside wubuwizard, not asymmetric draft-head allocation or FTA path supplementation; EG-25 does not list Gumiho; EG-24 is EAGLE-3's feature fusion. No gap states that early draft tokens deserve more compute than late ones.

## E9 — LayerSkip: early-exit self-speculation with exit-layer query-cache reuse

- **Source / date:** arXiv 2404.16710 v4, 18 Oct 2024 — https://arxiv.org/html/2404.16710
- **Traversal:** awesome-speculative-decoding list → LayerSkip paper; Hugging Face blog.
- **Exact mechanism:** training recipe = layer dropout with rates increasing by depth plus a shared LM-head early-exit loss at every layer, so intermediate layers can unembed directly without auxiliary modules. Inference `forward_early` runs the first E layers to draft d tokens and saves both KV and the **exit-layer query cache (`kvq`)**; `forward_remainder` verifies all d draft tokens in one pass over the last L−E layers reusing that `kvq` cache; after acceptance the cache is cropped to remove the last token. One model, no separate drafter. Authors report 1.34–2.16×.
- **Upstream pointer:** §1 contributions, self-speculation algorithm / `self_speculate()` pseudocode (`forward_early`, `forward_remainder`, `kvq_cache`, `crop`), Figure 7; repo `facebookresearch/LayerSkip`.
- **T1 fit:** MEDIUM–HIGH — smallest possible memory footprint for speculation; requires a model trained or fine-tuned with the recipe.
- **Why not EG-01..46:** EG-25's local-evidence row mentions an existing generic "early-exit" variant in `wubu_more_spec.c`, but no gap covers the LayerSkip training recipe (increasing layer dropout + shared exit loss) or the exit-layer **query** cache reused by the verification pass. EG-24 is a separate EAGLE-3 draft head; "kvq" does not occur anywhere in the register.

## E10 — RocketKV: training-free prompt-KV eviction then page-level Top-K decode selection

- **Source / date:** TensorRT-LLM docs "Sparse Attention" and Blog 17, current 1.4.0rc4 docs (2026) — https://nvidia.github.io/TensorRT-LLM/latest/features/sparse-attention.html and https://nvidia.github.io/TensorRT-LLM/latest/blogs/tech_blog/blog17_Sparse_Attention_in_TensorRT-LLM.html
- **Traversal:** front-c §3c → sparse-attention page → its "Further Reading" Blog 17.
- **Exact mechanism:** framework-level prediction workflow `sparse_kv_predict` / `sparse_attn_predict` feeding a unified `AttentionOp`. `RocketSparseAttentionConfig` (algorithm `rocket`) evicts **prompt KV first**, then selects Top-K KV **pages** during decode, for MHA/MQA/GQA, training-free; it manages a KT-cache auxiliary memory through the prediction module. The sibling `dsa` algorithm applies the same prediction workflow to MLA with token-level selection and an indexer-K cache. Blog 17 lists unified auxiliary-memory management (reuse and offload of these pools) as future work.
- **Upstream pointer:** docs "Supported Algorithms" table (`RocketSparseAttentionConfig`, `DeepSeekSparseAttentionConfig`), "Framework-level" paragraph in Blog 17, tests `tests/unittest/_torch/attention/sparse/test_sparse_mha.py`.
- **T1 fit:** MEDIUM–HIGH — reduces KV footprint at long prompt with no training; page granularity suits paged KV already implemented here.
- **Why not EG-01..46:** EG-23 skip-softmax prunes attention *tiles* inside the kernel using a calibrated threshold and never evicts KV; EG-44 is a trained indexer; neither performs training-free prompt-KV eviction with page-level Top-K decode selection. front-c E3c named "rocket", but the gap register never does.

## E11 — Block-sparse proxy routes: unselected KV blocks contribute a summary instead of being dropped

- **Source / date:** same TensorRT-LLM sparse-attention page (2026).
- **Traversal:** as E10.
- **Exact mechanism:** generic block-sparse path consumes routes from `block_sparse_attn_predict` as `BlockSparseForwardInputs`: canonical BSR (`block_indptr` + `block_indices`) or a packed block bitmask, **optionally with K/V block summaries so unselected blocks contribute a proxy instead of being dropped**, plus packed `kv_valid_bits` for ragged KV tails. Works for separate-Q/K/V dense prefill and paged generation (page 64/128); a request with unsupported routes raises rather than silently degrading to dense.
- **Upstream pointer:** docs section "Block-sparse MHA/MQA/GQA", hook `block_sparse_attn_predict`, type `BlockSparseForwardInputs`, `supports_block_sparse_inputs`; test `test_prims_ts_block_sparse.py`.
- **T1 fit:** MEDIUM — an accuracy-preserving sparse-attention contract (proxy rather than zero) that can be applied to our flash kernels without extra memory beyond block summaries.
- **Why not EG-01..46:** EG-23 explicitly *skips* blocks (no exp, no BMM2, no V load); EG-44 selects blocks with a trained indexer. No gap describes a sparse route carrying a K/V summary so an unselected block still contributes mass — "proxy" has zero hits in the register.

## E12 — MTP: native multi-token prediction heads as a draft path

- **Source / date:** arXiv 2404.19737, ICML 2024 — https://arxiv.org/abs/2404.19737 ; SGLang speculative-decoding doc, "Multi-Token Prediction" section.
- **Traversal:** SGLang doc (also awesome list) → MTP paper.
- **Exact mechanism:** train n output heads to predict n future tokens from one hidden state; SGLang runs those built-in heads as the draft path of speculative decoding ("often no separate draft model"), with DSA-indexer top-k for draft workers where applicable; DeepSeek-V3.2 uses the MTP+EAGLE combination. The draft is trained with the base model rather than added afterwards.
- **Upstream pointer:** paper "Better & Faster Large Language Models via Multi-token Prediction"; SGLang doc "We support MTP (Multi-Token Prediction) in SGLang by using speculative decoding"; DeepSeek-V3.2 cookbook §4.2.3.
- **T1 fit:** HIGH — no second model on a 6 GB box; heads are small and can be trained once and shipped with the checkpoint.
- **Why not EG-01..46:** EG-24 is EAGLE-3's feature-regression head added to a target; EG-21 is full Medusa tree verification; EG-25 does not list MTP. "MTP" has zero hits in the register (the only MTP string in the repo's sweep is a HiQCache fail-fast condition, not a gap). Native jointly-trained prediction heads are a different mechanism from either.

## E13 — Speculative Actions: speculation moved outside the model, to agent API calls

- **Source / date:** arXiv 2510.04371 v2, 23 Apr 2026 — https://arxiv.org/html/2510.04371
- **Traversal:** awesome-speculative-decoding list → paper/repo.
- **Exact mechanism:** in an agent loop every step is an API call (LLM call, tool, MCP request, human response). A cheap *Speculator* predicts the next action, arguments and expected observation/state delta while the authoritative but slow *Actor* computes the real one; k-way parallel next calls are launched from cached API intents, so validation — not waiting — is the critical path. Losslessness comes from semantic guards (Actor confirms state-transition equivalence before commit), safety envelopes (only idempotent/reversible/sandboxed side effects) and repair paths (rollback/compensating action); one OS variant uses last-write-wins instead. A cost–latency analysis gives the asymptotic reduction factor `p/(1+p)` and guides selective branch launching. Authors report up to 55% next-action prediction accuracy and up to 20% latency reduction.
- **Upstream pointer:** paper §1 (Actor/Speculator roles, Fig. 1), Algorithm 1 "speculative actions with k-way parallel next calls", cost–latency appendix; repo `naimengye/speculative-action`.
- **T1 fit:** MEDIUM — zero VRAM cost and aligns with the agentic-OS direction; pays off when tool latency dominates decode latency.
- **Why not EG-01..46:** EG-39 Hibana preserves per-agent KV state across turns (memory), EG-40 Callo reuses invalidated KV as drafts (KV cache); neither predicts the *environment's* next API response before the authoritative executor returns. "actions" hits in the register are the Word "action" in prose about agents, not a mechanism.

## E14 — Block verification keeping the longest accepted prefix with a correction distribution

- **Source / date:** SGLang speculative-decoding docs, current 2026 — https://docs.sglang.io/docs/advanced_features/speculative_decoding.md
- **Traversal:** front-d → SGLang docs (same page as E1/E3/E6).
- **Exact mechanism:** instead of stopping at the first mismatch, block verification checks *every* draft prefix and keeps the longest accepted prefix even if an earlier prefix was rejected; a corresponding correction distribution preserves target sampling. Enabled with `--speculative-use-block-verification` at `--speculative-eagle-topk 1`; the verifier reads the draft's proposal distribution in the target vocabulary, which also means a draft with a reduced hot vocabulary does not need a separate `--speculative-use-rejection-sampling` pass.
- **Upstream pointer:** SGLang doc section "Block verification"; parameter `--speculative-use-block-verification`, interaction with `--speculative-use-rejection-sampling` and reduced-vocabulary drafts.
- **T1 fit:** MEDIUM — pure verification-policy change, no kernel or memory cost; raises accepted tokens per target pass.
- **Why not EG-01..46:** EG-25's "relaxed verification" trades exactness for acceptance; EG-42 (TVP) is a drafter training objective; EG-24 is EAGLE-3 drafting. No gap describes a verifier that rescues a later prefix after an earlier rejection while applying a correction distribution to stay distribution-exact. "correction" has zero hits in the register.

---

## Rejected / not reported (already named or non-fits)

- FR-Spec cross-family variant, DynaDFlash, LiLiCorr grouped dynamic convolutions and cross-family SpecPrefill are refinements of E3/E7/E6/E4 respectively, not separate mechanisms.
- MindGPT-4ov modality-split prefill/session-keyed vision cache, ApCM, DirectKV and compute-in-flash were already recorded as conditional/non-fits in front-c E17/E18 and are not re-reported.
- "Adaptive speculative decoding" in the SGLang docs is close to EG-41 SpecSearch and was not registered again; Mamba drafters, DistillSpec, BiLD, Clover, PaSS, HASS, BiTA, SWIFT, Kangaroo, Amphista and Jakiro appear in the awesome list but are single-model/head variants of EG-21/EG-24/EG-25 families without a distinct systems mechanism, so they are not claimed as new gaps here.

## What not to break

All 46 gap boundaries still hold: one mechanism → one gate → one measured evidence row; no vendored source (LoRA drafts, codecs, drafters and sparse selectors must be reimplemented in C11); keep the mmap zero-copy weight law, the single-process no-dependency rule, the engine-ignorant-of-domain boundary, and the "no gate, no ship" discipline. A speedup quoted above is a reason to write a gate, never evidence that wubuwizard has the feature.

## Sources

1. https://docs.sglang.io/docs/advanced_features/speculative_decoding.md
2. https://github.com/NeuraLiying/awesome-speculative-decoding
3. https://github.com/awesome-pro/hiqcache — README
4. https://github.com/awesome-pro/hiqcache/blob/main/docs/design.md
5. https://github.com/awesome-pro/hiqcache/blob/main/docs/sglang-integration.md
6. https://arxiv.org/html/2502.14856 — FR-Spec
7. https://ar5iv.labs.arxiv.org/html/2502.02789 — SpecPrefill
8. https://arxiv.org/html/2603.02631 — cross-family SpecPrefill
9. https://arxiv.org/html/2310.07177 — Online Speculative Decoding
10. https://arxiv.org/html/2608.20530 — LiLiCorr
11. https://arxiv.org/html/2602.06036v1 — DFlash
12. https://arxiv.org/pdf/2611.19085 — DynaDFlash
13. https://arxiv.org/html/2503.10135 — Gumiho
14. https://arxiv.org/html/2404.16710 — LayerSkip
15. https://nvidia.github.io/TensorRT-LLM/latest/features/sparse-attention.html
16. https://nvidia.github.io/TensorRT-LLM/latest/blogs/tech_blog/blog17_Sparse_Attention_in_TensorRT-LLM.html
17. https://arxiv.org/abs/2404.19737 — Multi-token prediction
18. https://arxiv.org/html/2510.04371 — Speculative Actions
19. https://ar5iv.labs.arxiv.org/html/2609.04010 — Uno / Ψ-Spec
20. https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/speculative_decoding.html — DFlash2/LiLiCorr serving
21. https://arxiv.org/html/2502.19732v3 — speculative decoding survey (front-c source hub)
