# Front J — second-degree code mining of the mainstream serving clones

**Date:** 2026-11-22 · **Clones:** llama.cpp `94f3af3`, vLLM `ef687f6`, SGLang `1918da6`, FlashInfer `3adb5ac`
**Root:** `C:/Users/eman5/WuBu/research/engine-sources/`
**Target:** 6 GB VRAM / 16 GB RAM / SSD, single machine, model- and backend-agnostic.
**Method:** read `docs/ENGINE-GAP-REGISTER.md` (EG-01..EG-70) and `docs/ENGINE-SOURCE-INDEX.md` first, then followed *outbound references inside the clones* — arXiv citations in source comments and docs, `References:` docstrings, and cross-links between docs — to adjacent techniques that are not named in the gap register. Every finding below cites the exact repo path and line. Nothing here is implemented locally; this is a reading map, not evidence of absorption.

**Already absorbed — do not re-report:** EG-01 vec_dot dispatch, EG-04 attention sinks, EG-07 SSD row reuse, EG-08 radix cache, EG-09 overlap sched, EG-11 grammar FSM, EG-12 sampler chain (incl. min-p/top-n-sigma/mirostat), EG-14 Jinja, EG-16 sleep/wake allocator, EG-17 LoRA slots, EG-18 HiCache, EG-19 recompute-vs-restore, EG-20/EG-50/EG-67 MoE offload, EG-21/24/25/56/62 speculate families, EG-23/44 sparse attn, EG-28/47/49 KV quant, EG-30/36 FP8, EG-35/37/38 arena+index, EG-53/54 prompt-KV eviction, EG-66/68 disk snapshots/split-KV, EG-70 exactness certificates.

---

## Summary table — 16 second-degree mechanisms

| # | Repo | Exact file path : symbol | Mechanism (one line) | T1 fit | Local landing module | Gap id / why new |
|---|------|--------------------------|----------------------|--------|----------------------|------------------|
| F1 | llama.cpp | `common/ngram-mod.h:13` `common_ngram_mod`; `common/ngram-mod.cpp:15,27,37` `idx/add/get`; `common/speculative.cpp:2447` `common_speculative_impl_ngram_mod` | Rolling-hash n-gram→next-token pool (4 M entries ≈16 MB) shared across all slots; occupancy/low-acceptance resets | **HIGH** | new `src/wubu_ngram_spec.c` | **NEW EG-71** — EG-25 only says "token recycling", no hash pool, cross-slot sharing or reset policy |
| F2 | llama.cpp | `common/ngram-map.h:24,45,52,61`; `common/speculative.cpp:2332,2375,2615`; `common/ngram-cache.cpp` | key n-gram → up to 4 m-gram values with hit counts + persisted static/dynamic lookup files | **HIGH** | same as F1 | **NEW EG-72** — distinct from EG-08 (prefix reuse) and F1 (single next token) |
| F3 | vLLM | `docs/features/speculative_decoding/suffix.md:1-60`; `vllm/v1/worker/gpu/spec_decode/suffix/speculator.py` | Longest-suffix match against own prompt/output **plus a GPU corpus of finished requests**, frequency-proposed continuation, online-learned per-slot acceptance | **HIGH** | new `src/wubu_suffix_spec.c` | **NEW EG-73** — draftless, no trained head; EG-25 token recycling has no corpus or adaptive verify |
| F4 | llama.cpp | `docs/speculative.md:86-164` (`draft-dflash2`, `draft-dspark`, `draft-lilicorr`); `--spec-draft-p-min` | DFlash2 local conv + candidate selector; LiliCorr top-k candidate scorer conditioned on target hidden state; confidence truncation | MED | `src/wubu_eagle.c`, `src/wubu_more_spec.c` | **NEW EG-74** — extends EG-56/EG-62 with selector/scorer heads and p-min cut |
| F5 | SGLang | `python/sglang/srt/mem_cache/sparsity/algorithms/base_algorithm.py:10-24,32,55,79,106` `BaseSparseAlgorithm`; `quest_algorithm.py:21,30,55,120` `QuestAlgorithm` | Retrievable KV compression lifecycle: representation pool → construct → incremental update → query-conditioned top-k retrieve; token-wise = page_size 1 | **HIGH** | new `src/wubu_sparse_kv.c` | **NEW EG-75** — EG-44 is a *trained* indexer; EG-53/54 evict/summarize. This is training-free retrieve with an explicit repr lifecycle (needs accuracy gate) |
| F6 | vLLM | `docs/features/index_cache.md:1-45`; `vllm/model_executor/layers/mla.py:109`; `models/deepseek_v2.py:1080` | Reuse DSA top-k indices across layers (`index_topk_freq`, F/S pattern); reusing layers skip index-key cache allocation | **HIGH** | `src/wubu_attn_gate.c` | **NEW EG-76** — EG-44 builds an indexer; this reuses indices across layers and saves a KV allocation |
| F7 | FlashInfer | `flashinfer/attention/prims_ts/README.md:674-690` `vc_config=VCAttentionConfig`, `vc_quantize_fp8/repair`, `vc_quantize_int8`, `VCAttentionPreprocessor` | E4M3 V stored as residuals around per-128-token-tile means restored *inside* online softmax + direct probability cast; QK8 = per-head channel centering + normalized 128-pt Hadamard + per-128-token scales | **HIGH** | `src/wubu_fp8.c`, `src/wubu_attn_gate.c` | **NEW EG-77** — EG-30/36 have FP8 KV but no residual-V-in-softmax or rotated low-bit Q/K |
| F8 | vLLM | `docs/design/expert_substitution.md:1-70`; `vllm/model_executor/layers/fused_moe/expert_substitution.py:1-70` `ExpertSubstitutionSpec`, `APPROX_VALUE_SUFFIX`, `_parse_approximate_experts`, `decoder_layer_index` | Replace redundant MoE experts with "novices" returning a constant `[hidden_size]` vector; compact retained rows; preserve logical ids, no renormalize | **HIGH** | `src/wubu_moe.c`, `src/wubu_ssd_moe.c` | **NEW EG-78** — EG-51 uses a low-rank surrogate; this is load-time constant-vector pruning with zero MLP bytes |
| F9 | llama.cpp | `ggml/src/ggml-cuda/mmq.cuh:1011-1024` `mul_mat_q`, `tmp_fixup`, `write_back`, `force_tiling` | Stream-K work partitioning for quantized GEMM with partial-sum fixup; host escape hatch to native tiling | MED | `src/wubu_vecdot_dispatch.c` or new `src/wubu_gemm_stream.c` | **NEW EG-79** — EG-01 is vec_dot/GEMV, no CTA work partitioning or fixup path |
| F10 | vLLM | `docs/features/speculative_decoding/fly.md:1-30` `rejection_sample_method="fly"`, `fly_window_size`, `fly_entropy_threshold`, `fly_entropy_top_k` | Defer a native rejection at a high-entropy position when the next `fly_window_size` decisions align; entropy normalized by `log(top_k)`; **intentionally lossy** | **HIGH** (zero memory) | `src/wubu_spec_decode.c` | **NEW EG-80** — EG-25 "relaxed verification"/EG-61 block-verify are distribution-exact; this is an entropy-gated deferral, opt-in behind an accuracy gate |
| F11 | vLLM | `docs/features/speculative_decoding/dfly.md:1-30` | Checkpoint-native drafting from target hidden states + sparse indices + **shared KV storage**; online MoE-only block FP8 | MED (8-GPU example) | `src/wubu_more_spec.c` | **NEW EG-81** — distinct from EG-24/55/56 heads: indices + shared KV, no separate draft KV |
| F12 | vLLM | `docs/features/engram.md:1-45` | Engram conditional memory: NFKC tokenizer compression; multiplicative-XOR multi-head n-gram hashing into prime-sized disjoint buckets; gated fusion; O(1) lookup, no FLOPs | MED-HIGH (tables SSD-offloadable) | new `src/wubu_cond_mem.c` | **NEW EG-82** — no conditional-memory concept in register; EG-05 is neuron-sparsity prediction |
| F13 | FlashInfer | `flashinfer/flashinfer/pod.py:49-63,127-145` `gen_pod_module`, `gen_batch_pod_module`, `PODWithPagedKVCacheWrapper` | POD-Attention: split-K decode with a float workspace for intermediate results, auxiliary plan structures reused across layers, PDL support | MED | `src/wubu_attn_gate.c` | **Refines EG-68** (Strata 4-way split-KV) — new exact mechanism: reusable per-plan aux + split-K workspace |
| F14 | vLLM | `vllm/model_executor/layers/quantization/auto_awq.py:176-190` `AutoAWQConfig`, `TYPE_MAP` (uint4) | Activation-aware 4-bit weight quantization with per-channel/group scales, group_size, zero-points; Triton/Marlin/XPU backends | **HIGH** | `src/wubu_quant_selector.c` | **NEW EG-84** — EG-26 is per-token *activation* quant, EG-27 is quantize-on-load; no activation-magnitude-derived weight scale scheme |
| F15 | vLLM | `vllm/vllm/_custom_ops.py:3547-3557` `hadacore_transform` | Sylvester-Hadamard transform exploiting recursive structure, no transform weight data; symmetric = its own transpose/inverse | MED | `src/wubu_fp8.c` | **NEW EG-85** — pairs with F7; EG-26/27 have no rotation-based outlier suppression |
| F16 | llama.cpp | `ggml/include/ggml.h:2461-2470` (DeepSeek V4 hyper-connections `hc_comb`) | Multi-stream residual replacement: mixes stream pairs, softmax over dst/src, iterative normalization | CONDITIONAL | `src/wubu_model_adapter.c` | **NEW EG-86**, checkpoint-specific, low engine fit — record, do not prioritize |

---

## Detail and exact citations

### A. Draftless speculation — best 6 GB fit (F1–F4)

**F1 / F2 — llama.cpp n-gram family.** `docs/speculative.md:166-252` is the outbound index: `ngram-cache`, `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`. The concrete code is in `common/`, not `src/`:
- `common/ngram-mod.h:13` `struct common_ngram_mod`, `EMPTY=-1` (`:16`), `n` (`:33`); `common/ngram-mod.cpp:15 idx()` (rolling hash), `:27 add()`, `:37 get()`, `:43 reset()`.
- `common/speculative.cpp:2447` `common_speculative_impl_ngram_mod`; the pool is constructed with `4*1024*1024` entries at `:2452`; occupancy-triggered reset at `:2492`; low-acceptance-streak reset at `:2586`. Docs state "~16 MB, constant memory and complexity, variable draft lengths, single hash pool shared across all server slots" (`docs/speculative.md:219-225`).
- `common/ngram-map.h:24 common_ngram_simple_config`, `:45 common_ngram_map_value`, `:52 common_ngram_map_key`, `:61 common_ngram_map`; `common/speculative.cpp:2332 _ngram_simple`, `:2375 _ngram_map_k`, `:2615 _ngram_cache`, with `common_ngram_map_draft/begin/accept` at `:2410/:2393/:2421` and dynamic-cache merge/save at `:2675/:2679`.

Why it matters here: zero draft model, zero draft KV, fixed ~16 MB — the only speculative family in these clones that is strictly *cheaper* in VRAM than not speculating. The two reset policies are the non-obvious part and are absent from EG-25.

**F3 — vLLM Suffix Decoding.** `docs/features/speculative_decoding/suffix.md:3` cites arXiv 2411.04975; `:5` distinguishes it from n-gram: matches the request's own prompt+output *and* previous requests' responses, using frequency counts. `:33-38` gives the mechanism — GPU corpus (`suffix_decoding_corpus_tokens`, 1 M tokens default, oldest-first-out), lookup by last 2 and last 4 tokens, extend to longest suffix match, unmatched slots filled with last sampled token; `:42-43` warns the corpus is shared across adapters/salts; `:45-60` adds adaptive verification with per-slot acceptance learned online by draft source, match length and position. Symbol: `vllm/v1/worker/gpu/spec_decode/suffix/speculator.py`. This is a natural C11 fit next to F1: a token-history corpus plus a frequency table, no tensors beyond token ids.

**F4 — DFlash heads.** `llama.cpp/docs/speculative.md:77-80` names the head/driver split; `:88-103` DFlash2 = DFlash + local convolution + candidate selector with `--spec-draft-p-min` truncation and probabilistic path sampling; `:111-133` DSpark (EG-62, already registered); `:148-160` LiliCorr = small transformer over top-k candidates of each block position conditioned on the target hidden state at the last position, truncated by scorer probability. These are *second-degree*: llama.cpp's doc is an outbound index to the 2607/2608 papers, and vLLM separately cites DFlash (EG-56) — two clones converge on the same block-diffusion family.

### B. KV compression / sparse attention (F5–F7)

**F5 — SGLang retrievable KV compression.** Accuracy warning: only `quest_algorithm.py` is concretely implemented in this clone; ChunkKV/PQCache/SnapKV/Look-ahead QCache are named with concrete per-algorithm representation and retrieval behaviour in the `base_algorithm.py` docstring, i.e. a design index rather than shippable code. Mechanics worth taking (`:46-51`, `:70-75`, `:96-101`, `:136-141`): ChunkKV = aggregated key L2 norms per semantic chunk + layer-wise index reuse; Quest = mean-pooled key repr per page, query–page dot → top-k; PQCache = K-means centroids `[n_subvec,n_centroids,subvec_dim]` + token codes, rank tokens by query–centroid similarity; SnapKV = voting + 1-D pooling of important prefix positions + observation window; Look-ahead QCache = pseudo query (mean of last k queries) marks low-importance KVs for eviction. `BaseSparseAlgorithmImpl` (`:146-160`) enforces the construct/update/retrieve lifecycle and "recent page retention". This is the clearest missing *abstraction* relative to the register: wubuwizard has `wubu_attn_gate.c` for skipping tiles (EG-23) but no representation pool or retrieve step.

**F6 — vLLM IndexCache.** arXiv 2603.12201 cited at `docs/features/index_cache.md:9`; `:3-7` states the mechanism (DSA top-k computed per layer, reused across layers); `:29` is the memory consequence — layers that reuse indices do not allocate a separate index-key cache. Concrete pointers `vllm/model_executor/layers/mla.py:109` and `vllm/model_executor/models/deepseek_v2.py:1080`.

**F7 — FlashInfer VC-Attention.** `flashinfer/attention/prims_ts/README.md:674-690`. The two ideas to reimplement in C11: (1) store E4M3 V as residuals around a per-128-token-tile mean and restore the mean inside the online softmax, with a direct probability cast (ExpCast) — no V dequantization pass; (2) QK8 path: centre K by its per-head channel mean, rotate Q and K by the normalized 128-point Hadamard matrix, one scale per 128-token Q block and K tile, with `vc_quantize_fp8`/`vc_quantize_repair`, `vc_quantize_int8`/`vc_quantize_repair_int8`, and `vc_config=VCAttentionConfig(...)` selecting the recipe at plan time. `docs/api/attention.rst:227-228` ties this to the MLA/DSA family. This is the most concrete low-bit-attention mechanism found in any of the four clones.

### C. MoE memory (F8)

**F8 — MoNE constant expert substitution.** arXiv 2507.00390 at `expert_substitution.py:5`. `config.approximate_experts` maps layer ids to replaced expert ids (`docs/design/expert_substitution.md:14-23`); each replaced expert keeps one `[hidden_size]` tensor `.approx_value` instead of its MLP (`:26-32`, `APPROX_VALUE_SUFFIX` at `expert_substitution.py:30`); `_ROUTER_SEMANTICS` at `:48-53` fixes router semantics to preserve logical ids and weights, no renormalization; `ExpertSubstitutionSpec` at `:39-45`. The payoff for a 6 GB box: pruned experts use zero MLP weight memory and no MoE backend changes. Clean C11 landing: a GGUF/buffer-type substitution at load plus a constant add in the expert loop — no new kernel.

### D. Kernels and quant (F9, F14, F15)

**F9 — stream-K.** `mmq.cuh:1019` is the outbound cite (arXiv 2301.03598); `:1020` documents `force_tiling` as the host escape hatch; `:1011-1015` shows the `tmp_fixup`/`write_back` partial-sum path; `:1022-1024` the templated `mul_mat_q<type,J,fallback,prec_src1,has_fusion,force_tiling>` kernel. FlashInfer independently cites the same paper at its `README.md:265` — strong evidence it is a transferable, not model-specific, technique. EG-01's dispatch table is the natural owner, but the fixup buffer is a new integration surface.

**F14 — AWQ.** `auto_awq.py:176-190`: `AutoAWQConfig` (arXiv 2306.00978 at `:180`), `TYPE_MAP = {4: uint4}` (`:184-186`), `weight_bits`/`group_size` (`:188-191`), zero-point and pack-factor handling at `:168-173`. Gap register has W4A16 (EG-26) and quantize-on-load (EG-27) but not activation-magnitude-derived per-channel scales — a distinct scheme worth a codec gate.

**F15 — Hadacore.** `_custom_ops.py:3547-3557` `hadacore_transform(x, inplace=True)` (arXiv 2412.08832): exploits Sylvester-Hadamard recursion, needs no transform weight data, and is its own transpose/inverse. Pairs with F7's 128-point Hadamard rotation; include in the same gate rather than a separate one.

### E. Verification, memory and serving policy (F10–F13, F16)

**F10 — FLy.** `fly.md:3-6` (arXiv 2511.22972, intentionally lossy); `:27-29` defines `fly_window_size` (number of subsequent native decisions checked, default `min(6, n_spec-1)`) and the entropy gate: renormalize the top-k processed target scores and divide entropy by `log(k)`, giving a `[0,1]` gate comparable across `k`; a native rejection at a high-entropy position is deferred if the following decisions stay aligned. Gate value must be distribution-loss measured, not speed.

**F11 — DFly.** `dfly.md:3-10` (AngelSpec, arXiv 2607.25852): drafts from target hidden states, sparse attention indices and shared KV storage; checkpoint-native; the shipped example is 8×GPU HY4 Flash FP8 and MXFP4 group-32 weights, so only the shared-KV + sparse-index drafting idea is T1-relevant. EG-81 should be scoped to shared KV, not the FP8/MXFP4 path.

**F12 — Engram.** `engram.md:3-11` (arXiv 2601.07372): conditional *memory* as the axis complementary to MoE's conditional compute — parameters looked up, never computed over, O(1) per token; `:31-45` NFKC/accent/case/whitespace tokenizer compression into a smaller id space, then multi-head multiplicative-XOR hashing over suffix n-grams into prime-sized disjoint bucket ranges so collisions cannot collide twice. Only deployable with checkpoints declaring `engram_layer_ids`/`ple_layer_ids` (`:19-25`); tables are static and SSD-offloadable, which suits the target.

**F13 — POD.** `flashinfer/pod.py:61-63` (arXiv 2410.18038); `:49-58` JIT modules `gen_pod_module`/`gen_batch_pod_module`; `:142-145` float workspace (128 MB recommended) holds split-k intermediate results; `:119-124` auxiliary structures reused across layers. Refines EG-68 rather than replacing it.

**F16 — hyper-connections.** `ggml/include/ggml.h:2461-2470` (arXiv 2512.24880): multi-stream replacement of the residual connection, `hc_comb` mixes `[(2+hc)*hc, n_tokens]`, softmax over dst, normalize over src, repeat for `n_iter`. Checkpoint-specific; record as conditional, do not reserve a near-term gate.

### Outbound-reference graph (why "second-degree" found these)

- `llama.cpp/docs/speculative.md` → DFlash2 (inco.ai blog), DSpark 2607.05147, LiliCorr 2608.20530, speculators repo, llama.vim. Two new heads (F4) came from following these, not from file names.
- `vllm/docs/features/speculative_decoding/` → Suffix 2411.04975, FLy 2511.22972, DFly 2607.25852, plus `n_gram.md` and `adaptive_verification.md` (F3/F10/F11).
- `sglang/.../sparsity/algorithms/base_algorithm.py` `References:` → ChunkKV 2502.00299, Quest 2406.10774, PQCache 2407.12820, SnapKV 2404.14469, Look-ahead QCache 2505.20334 (F5).
- `flashinfer/.../prims_ts/README.md` → VC-Attention 2609.15810 and SAGE block summaries (F7); `pod.py` → 2410.18038 (F13).
- Convergent citations across clones: stream-K 2301.03598 appears in both llama.cpp `mmq.cuh:1019` and FlashInfer `README.md:265`; `merge_attn_states` (arXiv 2501.01005 §2.2) appears in vLLM `csrc/attention/merge_attn_states.cu:14` and SGLang `kernels/aot/csrc/attention/merge_attn_states.cu:30` — the latter is already close to EG-68's split-KV reduction and is the best-supported *shared* kernel target.

### Rejected as low-fit (read, but not proposed as gates)

- AdamW optimizer step `ggml.h:2631` / `ops.cpp:12366-12372` (decoupled weight decay, arXiv 1711.05101) — training-side, no training stack in scope.
- Online softmax ref `ops.cpp:9088-9090` (arXiv 2112.05682) — already the basis of in-tree flash attention; no new mechanism.
- Repo-level FIM pattern `tools/server/README.md:931`, `server-common.cpp:2020` (arXiv 2409.12186) — prompt construction, not engine core.
- Ring attention `docs/serving/context_parallel_deployment.md:15` (arXiv 2310.01889), DistServe goodput `benchmark_serving_structured_output.py:1063`, Engram/DFly multi-GPU topology — multi-GPU, tier-gated T2/T3.
- LayerDrop 1909.11556, rsLoRA 2312.03732, DoRA 2402.09353, YaRN 2309.00071 — model-architecture/training variants; tracked only if a matching checkpoint enters scope.

## Priority shortlist (start here)

1. **EG-71** n-gram-mod shared hash pool — F1, HIGH fit, small C module, no dependency.
2. **EG-75** retrievable sparse-KV abstraction + Quest — F5, HIGH fit, accuracy gate first.
3. **EG-77** VC-Attention residual-V + rotated low-bit Q/K — F7, HIGH fit, reuses `wubu_fp8.c`.
4. **EG-73** Suffix Decoding corpus — F3, HIGH fit, builds on F1's token-history structures.
5. **EG-78** MoNE constant expert substitution — F8, HIGH fit, load-time only, no new kernel.
6. **EG-80** FLy entropy-gated deferral — F10, zero memory, must gate distribution loss.
7. **EG-84** AWQ activation-aware uint4 scales — F14, HIGH fit for a 6 GB model footprint.

## Verification status

Read-only research: four clones inspected at the commits above; 136 unique arXiv IDs and 4,801 "based on/inspired by" lines were collected, then narrowed to the 16 findings with exact file:line citations. No local source, gate, Makefile or gap register was modified. Per the standing rule, none of these are DONE until a `test_egNN_*` target exists in both Makefiles and its measured evidence is quoted in `docs/WORKLIST.md`.
