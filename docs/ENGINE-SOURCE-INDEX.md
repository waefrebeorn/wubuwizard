# Engine Source Index — where to read, what to take

Navigation for the "acquire and learn all inference engines" work. This is a **reading
map**, not a dependency list: every entry names the upstream mechanism and the exact
place it lives so a later session can go straight to code. The project rule is unchanged
— read the reference, reimplement from scratch in C11, never vendor, never link.

Standing target: **6 GB VRAM / 16 GB RAM / SSD, model- and backend-agnostic**. "T1 fit"
means the mechanism helps that machine; "tier-gated" means it waits for a higher tier.

## Navigation

| Document | Use it for |
|---|---|
| `.hermes/GOAL.md` | Authoritative direction and phase plane P0–P7 |
| `docs/ENGINE-AGGREGATION.md` | Ownership map, layering, what is already aggregated |
| `docs/WORKLIST.md` | Shipped pads and their measured evidence |
| **`docs/ENGINE-GAP-REGISTER.md`** | **Ordered TODO: each gap, gate name, dependency, next action** |
| **`docs/ENGINE-SOURCE-INDEX.md`** | This file: engine catalog and exact upstream pointers |
| `docs/research/engine-sweeps/` | Full cited reports (front A/B/C + local audit) |
| `llama/README.md` | Existing local reference files (ggml-cpu, ggml-cuda, llama-model) |

Build reality on Windows: `make` is not on PATH. Use
`mingw32-make -f Makefile.win <gate>` from the repo root.

## Sweep provenance

Three parallel fronts, 2026-11-22: serving engines (6), local/edge engines (7),
offload and decode acceleration (18). Full reports:

- `docs/research/engine-sweeps/front-a-serving.md`
- `docs/research/engine-sweeps/front-b-edge.md`
- `docs/research/engine-sweeps/front-c-offload.md`
- `docs/research/engine-sweeps/local-audit.md`

`research/master-engine-compendium/` is a TV/writing-room archive, **not** an engine
corpus. Do not mine it for this work. CM4/CM5 distiller hardware was a time snipe;
it is not a work order.

## Serving engines

| Engine | Upstream | Language / license | Signature mechanism | Exact pointer | T1 fit | Local landing zone |
|---|---|---|---|---|---|---|
| vLLM | github.com/vllm-project/vllm | Python / Apache-2.0 | Tagged CUDA sleep/wake with partial restore | `vllm/v1/worker/gpu_worker.py`: `Worker.sleep`, `wake_up`, `CuMemAllocator.sleep(offload_tags=)` | HIGH | `src/wubu_mem_budget.c` |
| vLLM | same | same | Multi-adapter LoRA serving | Punica `bgmv`/`sgmv` grouped apply; `LoRALRUCache`, `LoRAModelManager` slots; per-request adapter map | HIGH | `src/wubu_lora.c`, `src/wubu_model_adapter.c` |
| vLLM | same | same | Structured output via pushdown automaton + batched grammar bitmasks | `vllm/v1/structured_output/StructuredOutputManager` | HIGH | `src/wubu_spec_constrained.c` |
| SGLang | github.com/sgl-project/sglang | Python / Apache-2.0 | Compressed FSM + jump-forward constrained decode | `python/sglang/srt/constrained/xgrammar_backend.py`; paper appendix B.1 | HIGH | `src/wubu_spec_constrained.c` |
| SGLang | same | same | Overlapping scheduler (deferred result queue) | `srt/managers/scheduler.py`: `event_loop_overlap`; `mini-sglang`: `overlap_loop` | HIGH | `src/wubu_scheduler.c`, `src/wubu_continuous_batching.c` |
| SGLang / Mooncake | kvcache-ai.github.io/Mooncake | C++/Python | HiCache L1/L2/L3, async prefetch/write-back, deadline policy, layer-N+1 overlap | Mooncake `hicache-design.md`; `sglang.io/blog/hicache` | HIGH | `src/wubu_lmcache.c`, `src/wubu_kv_tier.c` |
| TensorRT-LLM | github.com/NVIDIA/TensorRT-LLM | C++/Python / Apache-2.0 | Capacity + micro-batch two-level scheduling, kGUARANTEED_NO_EVICT | `www.stanleyjacob.dev/oss/tensorrt-llm` walkthrough | MEDIUM | `src/wubu_scheduler.c` |
| TensorRT-LLM | same | same | Skip-Softmax / BLASST dynamic attention-block pruning with calibrated threshold | `features/sparse-attention.html`; NVIDIA Skip Softmax blog | HIGH (needs accuracy gate) | `src/wubu_flash_prefill.c`, `src/wubu_attn_gate.c` |
| TensorRT Edge-LLM | github.com/NVIDIA/TensorRT-Edge-LLM | C++ / Apache-2.0 (verify per repo) | Persistent `RuntimeSession`, actor `RequestEngine`, padded-free `executeCompact` | `developer_guide/software-design/in-flight-batching.html`; `compactCapability.cpp` | HIGH as design reference | `src/wubu_serve.c`, `src/wubu_pd_serve.c` |
| LMDeploy | github.com/InternLM/lmdeploy | C++/Python | Persistent batch, LRU KV pool, dynamic split-and-fuse | `docs/inference/turbomind.html` | LOW novelty | `src/wubu_continuous_batching.c` |
| HF TGI | github.com/huggingface/text-generation-inference | Rust | Single shared request validator, tool grammar, Jinja templates | `pyshine.com/tgi/` source walkthrough | HIGH as pattern | `src/wubu_ctx_manage.c` |

## Local / edge engines

| Engine | Upstream | Language / license | Signature mechanism | Exact pointer | T1 fit | Local landing zone |
|---|---|---|---|---|---|---|
| llama.cpp | github.com/ggml-org/llama.cpp | C/C++ / MIT | Packed per-block-scale quantized dot kernels | `ggml/src/ggml-cpu/quants.c` | HIGH | `src/quantized_dot_generic.c`, `src/wubu_simd_quant.c` |
| llama.cpp | same | same | Runtime per-type `vec_dot` table + multi-row GEMV | `ggml/src/ggml-cpu/ggml-cpu.c`: `type_traits_cpu[].vec_dot/nrows/ncols`, `num_rows_per_vec_dot` | HIGH (biggest decode win) | `src/quantized_dot_generic.c` |
| llama.cpp | same | same | GGUF metadata-first sizing, alignment, shard validation | `src/llama-mmap.cpp` | HIGH | `src/gguf_reader.c` |
| llama.cpp | same | same | mmap prefetch policy — **do not add `MADV_SEQUENTIAL`** | github issue #91 | HIGH | `src/gguf_reader.c`, weight streaming |
| llama.cpp | same | same | Tokenizer families + per-model regex pre-tokenizer registry | `src/llama-vocab.cpp/.h`, `src/unicode.cpp` | HIGH | `src/wubu_gguf_tokenizer.c` |
| llama.cpp | same | same | GBNF stack machine, JSON-Schema→GBNF, lazy patterns | `src/llama-grammar.cpp/.h`, `common/json-schema-to-grammar.cpp` | HIGH | `src/wubu_spec_constrained.c` |
| llama.cpp | same | same | minja Jinja-subset chat templates | `common/jinja/`, `src/llama-chat.cpp` | MEDIUM | `src/wubu_tokenizer.c` |
| mistral.rs | github.com/EricLBuehler/mistral.rs | Rust (license verify) | Hardware-aware auto tuning; in-situ quantization before GPU load to avoid VRAM spikes | repo README, `CLAUDE.md` crate layout | HIGH | `src/wubu_mem_budget.c`, safetensors bridge |
| candle | github.com/huggingface/candle | Rust (verify) | Packed-weight `QMatMul` with explicit per-process dequantize-all escape hatch | `candle-core/src/quantized/mod.rs` | MEDIUM | `src/quantized_matmul.c` |
| llama.swift | github.com/profclaw/swift-llama | Swift (verify) | Thin wrapper around llama.cpp; no new engine mechanism | repo README | LOW | reference only |
| WebLLM / MLC LLM | arxiv 2412.15803, blog.mlc.ai | TS / Python | Browser WebGPU scheduling, ML compilation for universal deployment | WebLLM paper; MLC universal deployment blog | LOW on this host | reference only |
| MLX / mlx-lm | github.com/ml-explore/mlx-lm | Python / MIT | Composable sampler chain; lazy per-layer KV quantization with score-matrix budget check | mlx-lm README; generation/caching guide | HIGH (samplers), MEDIUM (KV) | `src/wubu_token.c`, `src/wubu_kv_tier.c` |
| Ollama | github.com/ollama/ollama | Go (verify) | Metadata-only VRAM prediction, 80%-of-free rule, OOM ctx shrink, recovery wait | Ollama Internals Handbook, codebase unpack | HIGH | `src/wubu_mem_budget.c`, `src/wubu_resource.c` |
| Ollama | same | same | Content-addressed sha256 blob store + manifests, dedup, resumable writes | same | MEDIUM | `src/wubu_sha1.c` lineage |

## Offload and decode acceleration

| Engine / paper | Upstream | Concrete mechanism | Exact pointer | T1 fit | Local landing zone |
|---|---|---|---|---|---|
| PowerInfer | github.com/Tiiny-AI/PowerInfer | Activation locality: hot neurons GPU, cold neurons CPU, adaptive predictors | SOSP 2024 paper | HIGH | new `src/wubu_neuron_cluster.c` |
| PowerInfer-2 | arxiv.org/html/2406.06282 | Neuron cluster as unit, runtime sizing, sparsity-aware adaptation, segmented neuron cache, cluster-level I/O-compute pipeline | §1, §3 | HIGH | same |
| FlexGen | proceedings.mlr.press/v202/sheng23a | Offload-placement LP solver, zigzag overlapping block schedule, 4-bit grouped weights/KV | §4 Algorithm 1 | HIGH | `src/wubu_mem_budget.c` |
| LLM in a Flash | ar5iv.labs.arxiv.org/html/2312.11514 | Sliding-window weight-row reuse `s_agg(k+1)-s_agg(k)`, >=32 KiB coalesced reads | §2–3 | HIGH | `src/wubu_ssd_moe.c`, `docs/ssd_moe.md` |
| MoE-Infinity | arxiv.org/html/2401.14361 | Expert activation traces (iEAM/pEAM), layer-decay eviction, EAMC | paper | HIGH | `src/wubu_ssd_moe.c` |
| Fiddler | arxiv.org/html/2402.07033 | Compute-centric MoE offload: move activations to CPU, AVX512_BF16 kernel, per-expert latency model | paper + github.com/efeslab/fiddler | HIGH | `src/wubu_expert_allreduce.c` |
| PIPO | arxiv.org/html/2504.03664 | Auto-configured pipelined offload proven on 6 GB RTX 3060 (<40% -> >90% GPU util, 3.1x) | paper | HIGH | `src/wubu_mem_budget.c` |
| TightLLM | ieeeexplore.ieee.org/document/10949701 | Per-iteration adaptive policy, weight slicing across batches | paper | HIGH | new policy module |
| ALISA | arxiv.org/html/2403.17312 | Three-phase token scheduling with Phase-III recomputation caching | paper | HIGH | `src/wubu_kv_tier.c` |
| KVPR | aclanthology.org/2025.findings-acl.997 | Partial-activation-first recompute overlapped with transfer | paper | HIGH | same |
| mzCache | arxiv.org/html/2609.01338 | Fine-grained shared buffers, compressed-KV restore path | paper | MEDIUM | same |
| EAGLE-3 | arxiv.org/html/2503.01840 | Direct token prediction, tri-layer feature fusion, dynamic tree, high-frequency draft vocab | paper; SpecForge concepts/EAGLE3 | HIGH | `src/wubu_eagle.c` |
| Lookahead / Sequoia / SpecTr | arxiv 2402.02057; SGLang spec docs | DP tree sizing, optimal-transport multi-draft, DAG verification, token recycling, relaxed verification | survey 2502.19732v3; awesome-speculative-decoding | MEDIUM | `src/wubu_lookahead.c`, `src/wubu_more_spec.c` |
| Yggdrasil / Hydra / TETRIS / PEARL | survey 2502.19732v3 | Static runtime, multi-head draft, draft-window scheduling, parallel verification | same survey | MEDIUM | same |
| ZeRO-Inference | arxiv.org/html/2207.00032 | N-layer prefetch window, activation-offload threshold, PCIe contention scheduling | paper + DeepSpeed blog | HIGH | weight streaming |
| ZeRO-Infinity | ar5iv.labs.arxiv.org/html/2104.07857 | Memory-centric tiling for layers larger than device memory | paper | MEDIUM | `src/tile_manager.c` |
| DeepNVMe | pytorch.org/blog/deepnvme-affordable-i-o-scaling... | Pinned async bulk reads | blog | HIGH | `src/wubu_ssd_moe.c` |
| MindGPT-4ov | arxiv.org/html/2512.02895v2 | Modality-split chunked prefill, session-keyed vision-embedding cache | §5.2 | CONDITIONAL | multimodal only |
| DirectKV / compute-in-flash | dl.acm.org/doi/10.5555/3843968.3843971; arxiv 2609.16161 | Hardware-specialized KV / in-flash compute | papers | LOW | record as hardware non-fit, not a port target |

## Code to gather and compare next

Highest-yield exact reads, in order. Each has a matching row in the gap register.

1. `ggml/src/ggml-cpu/ggml-cpu.c` — `type_traits_cpu` vec_dot dispatch and multi-row GEMV → compare `src/quantized_dot_generic.c`.
2. `src/llama-vocab.cpp` + `src/unicode.cpp` — pre-tokenizer registry by `tokenizer.ggml.pre` → compare `src/wubu_gguf_tokenizer.c`.
3. `src/llama-grammar.cpp` — GBNF stack machine and jump-forward → compare `src/wubu_spec_constrained.c`.
4. `vllm/v1/worker/gpu_worker.py` — tagged sleep/wake partial restore → compare `src/wubu_mem_budget.c`.
5. Mooncake `hicache-design.md` — async prefetch/write-back tiers → compare `src/wubu_lmcache.c`.
6. PowerInfer-2 §3 — neuron cluster descriptor and segmented cache → no local equivalent.
7. FlexGen §4 Algorithm 1 — placement solver and zigzag schedule → compare `docs/ssd_moe.md`.
8. EAGLE-3 §3 — direct token prediction and tri-layer fusion → compare `src/wubu_eagle.c`.
9. NVIDIA Skip Softmax blog — calibrated block pruning → compare `src/wubu_attn_gate.c`.
10. Ollama VRAM prediction — 80% rule and OOM context shrink → compare `src/wubu_resource.c`.
