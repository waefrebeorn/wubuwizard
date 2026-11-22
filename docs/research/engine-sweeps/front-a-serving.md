# Front A — serving-engine sweep: mechanisms wubuwizard has NOT absorbed

**Date:** 2026-11-22 · **Retrieval budget used:** 14 searches, 21 extractions · **Target:** everyman local inference, 6 GB VRAM / 16 GB RAM / SSD, C11 reimplementation only.

## Scope and method

This front covers six serving engines: vLLM, SGLang, TensorRT-LLM, TensorRT Edge-LLM, LMDeploy, and HuggingFace TGI. Each was mined for *concrete scheduling / memory / decode mechanisms with named source files and symbols*, not feature-marketing claims. Every candidate was then grepped against the actual tree at `C:/Users/eman5/WuBu/repos/wubuwizard` (460 C/CU sources, 462 test sources, 95 named make gates) to distinguish "not documented" from "not implemented". Absence was checked by symbol, not by word: zero hits for `jump.forward`, `xgrammar`, `compressed_fsm`, `grammar_bitmask`, `constrained_decode`, `event_loop_overlap`, `overlap_schedule`, `result_queue`, `capacityScheduler`, `microBatchScheduler`, `LookupBuffer`, `drop_select`, `CuMemAllocator`, `sleep_mode`, `wake_up`, `hicache`, `skip_softmax`, `BLASST`, `punica`, `bgmv`, `sgmv`, `max_loras`, `lora_slots`, `set_adapter`.

**Standing rule:** nothing here recommends vendoring. Each item names the upstream reference to *read and reimplement from scratch in C11*. No copied source, no linked dependency.

### Fit rubric
- **HIGH** — runs on 6 GB today, buys VRAM/RAM/SSD headroom or correctness with no new hardware requirement.
- **MEDIUM** — useful but needs careful integration or only pays off at scale/long context.
- **LOW** — needs multi-node, RDMA, a compiler toolchain for a different GPU, or solves a problem a single-user local engine does not have.

### Already in wubuwizard — explicitly excluded from the gap list
Verified present before writing: continuous/iteration-level batching; cross-request prefix KV reuse; paged KV/blocks; chunked prefill; flash attention/prefill/decode; KV cache filesystem (`src/wubu_lmcache.c`, LMCache-style file-backed prefix + PD KV persistence); KV compression (TurboQuant/RotorQuant/PolarQuant); IQ2/FP4/MXFP4/NVFP4/AWQ/GPTQ dequant; MoE grouped/expert-choice/hyperbolic/expert-allreduce; speculative decoding (Medusa, EAGLE, lookahead, n-gram, cascade, parallel); GGUF+safetensors readers; Gemma4/Qwen/MoE/SSM/LFM2 kernels; CUDA graphs; weight streaming; LM-head cache; 16 LP pads; 5 reference ports. Also present but deliberately *not* counted as the mechanisms below: single-adapter LoRA **load-time merge** (`wubu_lora.c`, `make test_btl3_lora`) and two **policy-only** helpers — `wubu_tier_advice()` in `src/wubu_ctx_manage.c` (HOT/WARM/COLD recommendation) and `wubu_offload_decision()` in `src/wubu_spec_variants.c` (L14 activation-beam offload decision). Those are decision functions, not cache engines.

---

## 1. vLLM

- **Repo:** https://github.com/vllm-project/vllm · **Language:** Python + C++/CUDA custom ops · **License:** Apache-2.0.

### 1a. Tagged CUDA allocator sleep/wake with partial restore — HIGH fit
The mechanism is not "offload the model". A single tagged memory pool tags allocations `weights` and `kv_cache`; `sleep(level=1)` hands the tagged bytes to CPU RAM and frees the VRAM, `sleep(level=2)` additionally clones every `model.named_buffers()` tensor to CPU and discards parameters, and `wake_up(tags=[...])` restores only the named tag so a weight update can be staged before KV is reallocated. Concretely: `vllm/v1/worker/gpu_worker.py::Worker.sleep()` and `Worker.wake_up()` (also `discard(tags)`, `_restore_sleep_parameters`), which call `vllm/device_allocator/cumem.py::CuMemAllocator.get_instance()`, `.sleep(offload_tags=("weights",) if level == 1 else tuple())`, `.wake_up(tags)`, `.use_memory_pool(tag=tag)`; `sleep_mode_offload_cudagraph` captures CUDA graphs into a cuMem pool that is backed up to CPU and restored in place, so graphs are replayed rather than recaptured [7][37][38].

**Why it fits 6 GB:** it is the only mechanism in this sweep that directly trades 16 GB host RAM for VRAM on demand — load a 4-bit model, run, sleep to free ~90% VRAM, load the next model without process teardown.

**NOT in wubuwizard:** no allocator tagging, no sleep/wake lifecycle, no partial restore, no saved-buffer table. `src/wubu_ctx_manage.c` only *recommends* a tier; it performs no allocation or copy. Reimplement as a C11 tagged-arena overlay over the existing weight/KV allocators.

### 1b. Multi-adapter LoRA serving (Punica batched grouped GEMM) — HIGH fit
Many LoRA adapters against one frozen base, loaded on demand into a bounded LRU, selected per request, and applied as a *batched* grouped matrix-vector product rather than one matrix multiply per request. Exact chain: `vllm/lora/lora_weights.py::LoRALayerWeights` and `PackedLoRALayerWeights` (`pack`, per-slice `optimize`); `vllm/lora/lora_model.py::LoRAModel`; `vllm/lora/models.py::LoRAModelManager`, `LRUCacheLoRAModelManager`, `LoRALRUCache` (activate/deactivate slots, `set_adapter_mapping`); `vllm/lora/worker_manager.py::WorkerLoRAManager`; `vllm/lora/punica_wrapper/punica_base.py::PunicaWrapperBase` and `punica_wrapper/utils.py::compute_meta`; `vllm/lora/ops/` for bgmv (batched grouped matrix-vector, many small adapters) and sgmv (sequence-major grouped matrix-vector, few adapters across many tokens); `vllm/v1/worker/lora_model_runner_mixin.py` for batch→adapter mapping; `vllm/lora/resolver.py`, `request.py`, `entrypoints/serve/lora/api_router.py` for the load/unload endpoints [16][17][18]. Punica is attributed to Chen et al., arXiv:2310.18547 [18].

**NOT in wubuwizard:** only a single adapter merged at load time. There is no `lora_slots`, no LRU adapter cache, no per-request adapter id, and no grouped shrink/expand kernel. The unabsorbed unit is *batched multi-adapter apply + slot manager*, not LoRA arithmetic itself. On 6 GB, adapters live in the 16 GB host pool and only the hot slots are paged to VRAM — a good fit.

### 1c. Structured output via pushdown automaton + batched grammar bitmasks — HIGH fit
Compile a JSON schema / regex / CFG to a grammar, then to a pushdown automaton (a recursive collection of FSMs, enabling multiple state transitions per step); at each decode step fill an int32 bitmask of legal next tokens and AND it into logits [12]. The batching detail matters: `vllm/v1/structured_output/__init__.py::StructuredOutputManager` owns `grammar_init()`, `_async_create_grammar()` (compiled on a CPU `ThreadPoolExecutor`), `_fill_bitmasks()`, `_async_submit_fill_bitmask()`, `grammar_bitmask()`, `should_advance()`, `should_fill_bitmask()`, plus a `fill_bitmask_parallel_threshold` (128) and a separate fillmask executor; backends live in `vllm/v1/structured_output/backend_xgrammar.py::XgrammarBackend` (`accept_tokens`, `validate_tokens`), `backend_guidance.py`, `backend_outlines.py` [10][11]. With speculative decoding, one mask per bonus-token position is packed inline for the GPU runner [10].

**NOT in wubuwizard:** no grammar representation, no logit mask, no constrained sampler. This is pure CPU work plus a bitmask AND — zero VRAM cost and a large correctness win for tool calling / JSON. Reimplement the PDA-to-bitmask path independently; do not link XGrammar.

### 1d. Disaggregated prefill/decode connector protocol — MEDIUM fit, mostly LOW for target
Two instances with a connector moving KV: `vllm/distributed/kv_transfer/` defines `Connector` (consumer pulls from producer), `LookupBuffer` (`insert` / `drop_select`, SQL-like), and `Pipe` (`send_tensor` / `recv_tensor`, single-direction FIFO), split into a **scheduler connector** that schedules transfers and **worker connectors** that execute them layer-by-layer; nine connector types ship, including `OffloadingConnector` for CPU-memory KV offload (configurable `block_size`, `cpu_bytes_to_use`) [34][35][33].

**NOT in wubuwizard:** `src/wubu_lmcache.c` already does a *single-process, file-backed* prefill→decode KV handoff, so the basic idea is covered. Genuinely missing is the scheduler/worker connector split, `insert`/`drop_select` lookup semantics, and async pipelined layer transfer. For one 6 GB GPU, only the CPU-offload connector variant is worth studying; RDMA/Mooncake multi-node transfer is LOW fit [33].

---

## 2. SGLang

- **Repo:** https://github.com/sgl-project/sglang · **Language:** Python runtime + C++/CUDA (`sgl-kernel`) · **License:** Apache-2.0.

### 2a. Compressed FSM + jump-forward constrained decoding — HIGH fit
Build the FSM over *characters/strings*, merge consecutive singular-transition edges (source has one successor and one acceptable string) into a single compressed edge, then retokenize the compressed edge text against the prompt prefix so token boundaries stay valid. At a compressed edge the engine emits many tokens in one step instead of sampling them one at a time — it terminates the request and enqueues a new one, letting the prefix cache absorb the recomputation. Upstream sites: `python/sglang/srt/constrained/xgrammar_backend.py` (default), `base_grammar_backend.py` (Outlines / llguidance alternatives), `outlines_jump_forward.py` (jump-forward map + retokenization for the regex/FSM backend). The paper appendix §B.1 defines singular edge / compressed edge precisely [1][2][3].

Crucially, the CPU mask work is **overlapped** with GPU forward: `update_regex_vocab_mask()` / `fill_vocab_mask()` build the bitmask on CPU before the step, and `move_vocab_mask(device)` transfers it with `non_blocking=True` while the model computes; the scheduler coordinates this through `SamplingBatchInfo` [14].

**NOT in wubuwizard:** nothing under constrained decoding (zero hits for grammar/bitmask/jump-forward). HIGH fit — no VRAM cost, fewer forward passes, and it reuses the prefix cache already implemented.

### 2b. Zero-overhead overlapping scheduler — HIGH fit
The scheduler runs CPU batch preparation for step N+1 while the GPU executes step N. The loop is `python/sglang/srt/managers/scheduler.py::event_loop_overlap()`, which keeps a `result_queue: Deque` and defers `process_batch_result(tmp_batch, tmp_result)` for the *previous* batch (sampling, stop checking, ZMQ dispatch to the detokenizer) until the current GPU batch is launched. Enabled by default (`disable_overlap_schedule=False`). The minimal reference is `python/sgl-project/mini-sglang/python/minisgl/scheduler/scheduler.py::overlap_loop()` — same idea, far fewer lines. The process split is documented in `entrypoints/engine.py`: HTTP server (`entrypoints/http_server.py`), `TokenizerManager` (`managers/tokenizer_manager.py`), per-GPU `Scheduler` (`managers/scheduler.py`), `TpModelWorker` (`managers/tp_worker.py`), `DetokenizerManager` (`managers/detokenizer_manager.py`), with a three-stage batch handoff `ScheduleBatch` → `ModelWorkerBatch` → `ForwardBatch` [13][15][21].

**NOT in wubuwizard:** no deferred result queue, no CPU/GPU pipeline across steps. This is the highest-leverage *pure-CPU* change in the sweep and does not touch any kernel.

### 2c. HiCache: L1 GPU / L2 host / L3 storage with async prefetch and write-back — HIGH fit
Extends the radix tree to `HiRadixTree`, where nodes track which tier holds the KV. Three operations: **local match** against L1/L2; **prefetch** from L3 when the consecutive hit exceeds a threshold (default 256 tokens) using a two-thread pipeline (`prefetch_thread_func`), with a dynamic deadline `prefetch_timeout_base + prefetch_timeout_per_ki_token * n_tokens/1024` and policies `wait_complete` / `best_effort` / `timeout`; **write-back** at node granularity with three policies (`write_back`, `write_through`, `write_on_evict`), `write_backup()` for L1→L2 and `backup_queue` + `backup_thread_func` for L2→L3. Two more mechanisms make it more than a tiered map: **compute-transfer overlap** (load layer N+1 from host while computing layer N during prefill) and **GPU-assisted I/O kernels** on top of `cudaMemcpyAsync` (reported up to 3× transfer speed). `all_reduce(op=min)` synchronizes TP ranks on hit counts; MHA stores `1/tp_size` of a token's KV per rank while MLA stores the full KV, so only one rank writes back [4][5]. Transfer funnels through `python/sglang/srt/mem_cache/l2_transfer.py::L2TransferEngine.submit_device_to_host` and `submit_host_to_device`, calling `host_pool.backup_from_device_all_layer_physical` / `load_to_device_per_layer_physical`; other sites are `srt/mem_cache/hiradix_cache.py`, `hicache_storage.py`, `pool_host/mha.py` [6].

**NOT in wubuwizard:** the KV *filesystem* exists (`src/wubu_lmcache.c`), and `wubu_tier_advice()` recommends a tier, but there is no actual L1/L2/L3 engine, no prefetch thread, no backup queue/thread, no layer-N+1 overlap, and no page-first transfer layout. This is the single best match to the 16 GB RAM + SSD target: host RAM as a real L2 and the SSD as L3, with recompute-vs-wait decided by a deadline.

---

## 3. TensorRT-LLM

- **Repo:** https://github.com/NVIDIA/TensorRT-LLM · **Language:** C++/CUDA core + Python API · **License:** Apache-2.0.

### 3a. Two-level scheduler: capacity admission + micro-batch packing — MEDIUM fit
Admission and compute are deliberately separate. `cpp/tensorrt_llm/batch_manager/capacityScheduler.cpp` answers "which requests may be resident given KV blocks" under an enum in `cpp/include/tensorrt_llm/executor/types.h`: `kGUARANTEED_NO_EVICT` (default — admit only if the request can finish without eviction), `kMAX_UTILIZATION` (pack harder, accept pausing; `pauseRequests.cpp` exists for exactly this), `kSTATIC_BATCH`. Then `microBatchScheduler.cpp` packs the admitted requests' prefill chunks and decode steps into the iteration under a token budget (`ContextChunkingPolicy`, FCFS by default). Supporting cast, all in `cpp/tensorrt_llm/batch_manager/`: `llmRequest.cpp` (state machine), `kvCacheManager.cpp` + `evictionPolicy.cpp` (32-token blocks, prefix reuse), `sequenceSlotManager.cpp`, plus cache-transceiver files for disaggregated serving. Python side: `tensorrt_llm/_torch/pyexecutor/scheduler/scheduler.py::BindCapacityScheduler` / `BindMicroBatchScheduler` are thin nanobind wrappers. `tensorrt_llm/_torch/autotuner.py` picks the best kernel variant at runtime [19][27].

**NOT in wubuwizard:** chunked prefill and paged KV exist, but not the *split into a memory-admission policy with a no-eviction guarantee and a separate token-budget packer*, nor request pausing/resumption. MEDIUM fit: the no-eviction policy is a latency guarantee that matters when 6 GB is tight; kernel autotuning is less relevant to a C11 engine already tuned per shape.

### 3b. Skip-Softmax / BLASST dynamic attention-block pruning — HIGH fit
Inside the FlashAttention loop: compute the local max logit of the current QKᵀ tile, compare with the running global max, and if the gap exceeds a calibrated threshold λ, skip the `exp`, skip BMM2, and — the bandwidth win — **do not load the V block from HBM**. λ scales as `threshold_scale_factor / context_length`, so it is self-adjusting with L; separate prefill and decode values are supported. Training-free, works with MHA/MQA/GQA/MLA. Config: `tensorrt_llm.llmapi.SkipSoftmaxAttentionConfig(threshold_scale_factor=...)`, `sparse_attention_config.algorithm = "skip_softmax"`, docs at `docs/source/blogs/tech_blog/blog16_Accelerating_Long_Context_Inference_with_Skip_Softmax_Attention.md`; accuracy parity reported on MATH-500 vs dense attention, up to 1.4× faster prefill at 128K [30][31].

Edge-LLM deploys the same idea with two extra tricks worth studying: a **discretized per-length scale table** replaces the S/L formula (`skip_softmax_prefill_scale` / `skip_softmax_decode_scale` plugin inputs), and `kernelSrcs/fmha_cutedsl_blackwell/calibrate_skip_softmax.py` calibrates with ModelOpt's `attention_sparsity` calibrator, weighting the *share of softmax mass* skipped rather than just counting tiles; a `skip_softmax_<phase>_target_sparsity` ladder selects a tier, and `none` disables a phase [32].

**NOT in wubuwizard:** flash attention exists; block-level dynamic pruning inside it does not (zero hits for skip_softmax/BLASST). HIGH fit at long context with no extra memory, but it needs a calibration gate — ship it only with an accuracy check against dense attention.

### 3c. `rocket` prompt-KV eviction + page-level Top-K decode — MEDIUM fit
`RocketSparseAttentionConfig` (`algorithm: rocket`): evict prompt KV first, then select the top-K *pages* during decode; training-free, for MHA/MQA/GQA. Related configs in the same table: `dsa` (learned token-level indexer + sparse MLA), `deepseek_v4` / `csa2` (sliding-window + compressed history), `minimax_m3` (learned block selection + sparse GQA) [31].

**NOT in wubuwizard:** the model-specific sparse families are largely covered by the five existing reference ports, but generic training-free prompt-KV eviction with page-level Top-K selection is absent. MEDIUM fit — complementary to skip-softmax, but validate accuracy.

---

## 4. TensorRT Edge-LLM

- **Repo:** https://github.com/NVIDIA/TensorRT-Edge-LLM · **Language:** Python 45.5% / C++ 42.6% / CUDA 11.1%; the *inference path* is pure C++ with no Python interpreter · **License:** Apache-2.0 [23][43].

### 4a. Single long-lived session + actor-style request engine with compact execution — HIGH fit as a design reference
One `RuntimeSession` in `cpp/runtime/`, opened once by `LLMInferenceRuntime::openSession`, lives as long as the engine and holds the command queue. `RequestEngine` in `cpp/scheduler/` owns the loop: at each step boundary it folds in the command queue, admits queued requests into the session, schedules one step, commits the result. Under **compact execution** (phase 8, `RankRuntimeSession::executeCompact`) a step is any subset of residents — prompt chunks first, then decode rows — packed without padding; `cpp/runtime/config/compactCapability.cpp` sorts layers into kinds to decide whether this is legal, `SessionOptions::allowCompactExecution` gates it, `RequestEngine::resolvedPolicy()` reports the effective policy, `RequestEngine::mayAdmit` holds a head when residents need uniform sampling, `DecodingStrategy::seatRow` places the draft prefill, and `RuntimeSession::capabilities()` publishes `(compact, reason)` for `/health` [20][22][28].

**NOT in wubuwizard:** there is no persistent session abstraction; batching is rebuilt around per-step request sets. The unabsorbed mechanism is *modeling the batch as a long-lived session with a command queue and a padded-free compact step*. Fit is HIGH for the **architecture** (C++-first, no Python in the hot path — close to the C11 goal); LOW for the artifact, because TensorRT engines are compiled per exact GPU and are not portable to an arbitrary 6 GB card. Read the design, not the build flow.

### 4b. Edge skip-softmax — see §3b
Same mechanism, deployment-specific calibration path: `calibrate_skip_softmax.py`, per-length table, `skip_softmax_prefill_shape_advice` [32].

**NOT in wubuwizard:** as §3b.

---

## 5. LMDeploy

- **Repo:** https://github.com/InternLM/lmdeploy · **Language:** Python + C++/CUDA (TurboMind core, descended from FasterTransformer) · **License:** Apache-2.0 [25][41].

### 5a. Persistent batch + LRU KV pool + dynamic split-and-fuse — LOW novelty, one MEDIUM nugget
TurboMind models inference as a batch whose lifetime spans the entire serving process, with N pre-configured slots; a slot is released when generation finishes. Its `src/turbomind/models/llama/SequenceManager.h` is a memory-pool-like object that also implements LRU — explicitly described as a "cache of KV caches". Dynamic split-and-fuse breaks large prefills into chunks for interleaving with decode; tensor-parallel worker threads need a dedicated synchronization mechanism; the context decoder uses an FMHA implementation supporting **mismatched Q/K lengths**, and split-K decoding (Flash Decoding) is present [24][42][41].

**NOT in wubuwizard:** almost nothing — continuous batching, paged KV, chunked prefill and flash decode are already implemented. The one genuinely unabsorbed detail is the **mismatched-Q/K FMHA context path** (prefill where Q and K lengths differ without padding), plus having the scheduler and LRU KV pool resident in the native runtime rather than at the Python layer. MEDIUM fit, low priority.

### 5b. Composable quantizations — already covered
Online INT4/INT8 KV quantization, AWQ, SmoothQuant W8A8, MXFP4, W4A16 with CUTLASS GEMMs [24][41]. **NOT in wubuwizard:** no new mechanism — the dequant/KV-compression surface is already implemented and is explicitly excluded from this report.

---

## 6. HuggingFace TGI

- **Repo:** https://github.com/huggingface/text-generation-inference · **Language:** Rust 16.3% + Python 78.6% + C++ 0.9%, gRPC between halves · **License:** Apache-2.0 · **Status:** maintenance mode since 2025-12-11 [26][27][29].

### 6a. Request admission control as a single shared validator — HIGH fit as a pattern
`router/src/validation.rs` is one shared object that knows max sequence length, max batch-total tokens and stop-sequence rules, so every route gets identical guardrails before a request can poison a batch. `router/src/infer/mod.rs` is the continuous-batching queue: it pairs each caller with a oneshot channel, applies queue timeouts, forms the next batch, and drops slow consumers rather than stalling the fleet [26][28].

**NOT in wubuwizard:** the batching loop exists but not a single admission object enforcing per-batch token budgets plus queue-timeout/slow-consumer eviction. HIGH fit — small C11 change, protects a 6 GB engine from a single oversized request.

### 6b. Tool-call grammar and Jinja chat templating — MEDIUM fit
`router/src/infer/tool_grammar.rs` compiles function definitions into constrained grammars so the model can only emit valid JSON arguments; `router/src/infer/chat_template.rs` renders Jinja templates with tool schemas; `router/src/chat.rs` parses tool calls back out of the stream [26]. This is the *application-level* counterpart to SGLang's FSM and vLLM's PDA — same family, different entry point.

**NOT in wubuwizard:** tool-call parsing and schema-to-grammar compilation. MEDIUM fit; overlaps §1c/§2a, so implement once.

### 6c. Watermarking and composable logits processors — LOW priority
`server/text_generation_server/utils/watermark.py` implements the Sotoudeh–Gerard green-list watermark as one more logits bias; `server/text_generation_server/utils/logits_process.py` stacks temperature, top-k, top-p and typical-p as composable processors [26]. Pluggable backends speak the `proto/generate.proto` Shard service; `server/text_generation_server/utils/weights.py` memory-maps sharded safetensors [26].

**NOT in wubuwizard:** no watermark processor and no typical-p sampler. LOW priority — this is a provenance feature, not a memory/throughput gap. The protobuf/backend-plug architecture is not relevant to a single C11 binary.

---

## Cross-engine gap path (dependency order, not importance)

1. **LP-A1 — Overlap scheduler** (§2b): deferred result queue; no kernel changes; everything below is easier to hide behind it. *HIGH fit.*
2. **LP-A2 — Admission + token budget** (§6a): one validator + queue timeout; protects the small-VRAM path. *HIGH fit.*
3. **LP-A3 — Structured decoding, one implementation** (§1c, §2a, §6b): grammar → automaton → bitmask; add jump-forward only after the bitmask path is gated. *HIGH fit.*
4. **LP-A4 — Tagged sleep/wake allocator** (§1a): needs the allocators to be tag-aware; enables model swapping on 6 GB. *HIGH fit.*
5. **LP-A5 — Multi-adapter LoRA slots + grouped apply** (§1b): depends on LP-A4's host paging for the working set. *HIGH fit.*
6. **LP-A6 — HiCache tiers with async prefetch/write-back** (§2c): study next to `wubu_lmcache.c`; add the deadline policy and layer-N+1 overlap before adding tiers. *HIGH fit.*
7. **LP-A7 — Skip-softmax pruning** (§3b/§4b): needs a calibration + accuracy gate. *HIGH fit with gate.*
8. **LP-A8 — Capacity/micro-batch split** (§3a) and **LP-A9 — persistent compact session** (§4a): architectural, medium fit.
9. **LP-A10 — Disaggregated connector protocol** (§1d) and **LP-A11 — watermark/typical-p** (§6c): only if a use case appears. *LOW fit for a single 6 GB box.*

## What not to break

- Keep the engine ignorant of domain concepts. Grammar/tool schemas are request metadata supplied by the caller; do not teach the core an application schema.
- Do not turn `wubu_ctx_manage.c`'s `wubu_tier_advice()` or `wubu_spec_variants.c`'s `wubu_offload_decision()` into a cache engine — they are policy functions the operator calls per step; HiCache-style threading belongs in a new module beside `wubu_lmcache.c`.
- Preserve the single-process, no-third-party-dependency property. Every pad above must be a from-scratch C11 implementation with a real make gate.
- Do not regress the existing file-backed PD handoff (`wubu_lmcache.c`) when adding tiers; it is the compatibility path for low-RAM hosts.
- Never vendor Punica/XGrammar/sgl-kernel/TensorRT code; read the named files and reimplement the algorithm.

## Sources

[1] https://www.stanleyjacob.dev/oss/sglang
[2] https://arxiv.org/html/2312.07104v2
[3] https://www.lmsys.org/blog/2024-02-05-compressed-fsm
[4] https://kvcache-ai.github.io/Mooncake/_sources/design/hicache-design.md
[5] https://www.sglang.io/blog/hicache
[6] https://github.com/awesome-pro/hiqcache/blob/main/docs/sglang-integration.md
[7] https://docs.vllm.ai/en/latest/features/sleep_mode
[10] https://docs.vllm.ai/en/v0.10.1.1/api/vllm/v1/structured_output/index.html
[11] https://developers.redhat.com/articles/2025/06/03/structured-outputs-vllm-guiding-ai-responses
[12] https://docs.vllm.ai/projects/ascend/zh-cn/v0.9.1/_sources/user_guide/feature_guide/structured_output.md
[13] https://hungchun0201.github.io/agentic-ai-survey/ai-infra-basics/sglang/index.html
[14] https://hungchun0201.github.io/agentic-ai-survey/ai-infra-basics/sglang/frontend.html
[15] https://github.com/sgl-project/mini-sglang/blob/main/python/minisgl/scheduler/scheduler.py
[16] https://github.com/localai-org/vllm.cpp/blob/main/.agents/specs/lora-adapter.md
[17] https://factory.com/open-source-wikis/vllm?page=features/lora.md
[18] https://docs.vllm.ai/en/v0.29.0/api/vllm/lora/punica_wrapper/punica_base
[19] https://www.stanleyjacob.dev/oss/tensorrt-llm
[20] https://nvidia.github.io/TensorRT-Edge-LLM/developer_guide/software-design/in-flight-batching.html
[21] https://nvidia.github.io/TensorRT-Edge-LLM/overview.html
[22] https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/software-design/cpp-runtime-overview.html
[23] https://github.com/nvidia/tensorrt-edge-llm
[24] https://www.madebyagents.com/inference-engines/lmdeploy
[25] https://github.com/internlm/lmdeploy
[26] https://pyshine.com/tgi
[27] https://github.com/huggingface/text-generation-inference
[28] https://huggingface.co/docs/text-generation-inference/architecture
[29] https://huggingface.co/docs/inference-endpoints/en/engines/tgi
[30] https://developer.nvidia.com/blog/accelerating-long-context-inference-with-skip-softmax-in-nvidia-tensorrt-llm
[31] https://nvidia.github.io/TensorRT-LLM/latest/features/sparse-attention.html
[32] https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/features/skip-softmax.html
[33] https://kvcache-ai.github.io/Mooncake/_sources/deployment/integrations/vllm/disagg-prefill-decode.md
[34] https://docs.vllm.ai/en/v0.34.0/features/disagg_prefill
[35] https://docs.vllm.ai/en/stable/examples/disaggregated/disaggregated_serving
[37] https://docs.vllm.ai/en/v0.34.0/api/vllm/v1/worker/gpu_worker
[38] https://github.com/vllm-project/vllm/blob/352c0c8a285414b11373e65fef095af7b07b94d8/vllm/v1/worker/gpu_worker.py
[41] https://aiwiki.ai/wiki/lmdeploy
[42] https://lmdeploy.readthedocs.io/en/v0.5.0/inference/turbomind.html
[43] https://raw.githubusercontent.com/NVIDIA/TensorRT-Edge-LLM/main/LICENSE — TensorRT Edge-LLM LICENSE (Apache-2.0)
