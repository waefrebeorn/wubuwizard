# Front K — Unacquired inference engines (newer serving / edge / offload, 2026 releases)

**Scope.** Study-only sweep of inference engines **absent from the 32-engine study tree at
`~/WuBu/research/engine-sources/`** and absent as a *mechanism gate* from the 105 EG rows in
`docs/ENGINE-GAP-REGISTER.md`. Fronts A–J already cover vLLM, SGLang, TensorRT-LLM, Triton
Edge-LLM, LMDeploy, HF TGI, Mooncake/HiCache, Ollama, LocalAI-adjacent serving shells, the
PowerInfer/FlexGen/LLM-in-a-Flash offload canon, the EAGLE/speculative families, and the
2026-11 releases (Hibana, Callo, Hyperflare, titan-engine, Strata, TensorFold, llm-d 1.0).
None of the 18 engines below is in that list. A name that merely appears as an inspiration
(LMCache in `ADR-003-kv-cache-is-a-filesystem.md`, or a bare `vllm.cpp` URL in
`front-a-serving.md`) is reported only where a *concrete, ungated* mechanism exists.

**Standing target.** 6 GB VRAM / 16 GB RAM / SSD, model- and backend-agnostic, Windows 11
host, C11 core. "T1 fit" means the mechanism helps *that* machine; "tier-gated" means it
waits for a larger tier.

**Rule, unchanged.** Read upstream, extract the idea, reimplement in C11 behind a named gate.
**Never vendor, copy verbatim, link, or add a third-party object to `CORE_OBJ`.** Every
"clone" recommendation below means a *shallow study clone into `engine-sources/` only*.

**Ledger note.** All URLs were registered at retrieval time in the shared
`bench-ledger.json` via `HERMES_CITATION_LEDGER` + `sources.py add`; no `reset` was run.

---

## A. Heterogeneous CPU/GPU and offload engines

### 1. KTransformers — `kvcache-ai/ktransformers`

- **URL / license:** https://github.com/kvcache-ai/ktransformers — Apache-2.0 [28][78].
- **Concrete mechanism:** CPU-optimised MoE kernels with NUMA-aware memory management and
  heterogeneous expert placement — hot experts resident on GPU, cold experts on host CPU —
  exposed as an *injection* framework that swaps selected operators inside a serving stack;
  the project documents an AVX2-only CPU backend as of 26 Mar 2026 [28].
- **Exact upstream path / symbol:** `kt-kernel/` tree; symbol `fp8x8_to_fp32x8()` in the
  `[perf](kt-kernel): AVX2 FP8 to FP32 by shifts instead of a table gather` change
  (commit `3a78b9a`, 22 Nov 2026), which replaces `_mm256_i32gather_ps` over a 256-entry
  table with exponent rebias and subnormal blend for E4M3 weights [28]. Release v0.7.2
  (15 Oct 2026) [28].
- **Why missing locally:** EG-05 added a neuron-cluster abstraction and EG-06 an offload
  placement solver, but neither a NUMA-aware hot/cold *expert* placement policy nor an
  AVX2 FP8 expert kernel exists; no injection/operator-swap surface exists in the C11 core.
- **T1 fit:** HIGH — 16 GB host RAM holds cold experts while 6 GB VRAM holds hot ones; AVX2
  backend matches a laptop CPU.
- **Clone:** **YES** (study-only): read `kt-kernel/` expert kernels and the injection spec.

### 2. vAttention — `microsoft/vattention`

- **URL / license:** https://github.com/microsoft/vattention — MIT [30].
- **Concrete mechanism:** decouples *virtual* from *physical* KV allocation with CUDA VMM
  APIs: reserve one large contiguous virtual KV buffer up front, map physical pages on
  demand, so attention kernels need no paged-kernel rewrite; up to 1.23× throughput over
  PagedAttention kernels of FlashAttention-2 and FlashInfer (ASPLOS '25, DOI
  10.1145/3669940.3707256) [29][30].
- **Exact upstream path / symbol:** `vattention/` allocator; `sarathi-lean/.../vATTN_cache_engine.py`
  with class `vATTNCacheEngine`, calls `vattention.init_kvcache(...)` and
  `alloc_new_batch_idx()` to map a request to its KV offset [30]. Paper §3.3 compares
  PagedAttention kernel overheads [29].
- **Why missing locally:** the local KV allocator assumes a fixed block table and paged
  layout; there is no demand-paging path that keeps a contiguous virtual KV region and
  leaves the kernel unchanged.
- **T1 fit:** MEDIUM–HIGH — the "reserve virtual, commit physical on demand" rule maps
  cleanly onto a 6 GB device where peak KV is budgeted but rarely materialised.
- **Clone:** **YES** (study-only): read the VMM allocator and cache-engine API shape.

### 3. LMCache — `LMCache/LMCache`

- **URL / license:** https://github.com/LMCache/LMCache — Apache-2.0 [32][79]. Paper
  https://arxiv.org/html/2510.09665 [31].
- **Concrete mechanism:** chunk-granularity KV movement (chunk size configuration independent
  of the engine's small page size), batched transfers with compute/I/O pipelining and
  zero-copy, plus a **modular KV connector** that decouples the cache layer from the engine
  and a first-class control API (pin, lookup, cleanup, move, compress); hooks sit before and
  after model execution and attention so both bulky and layer-wise offload work [31].
- **Exact upstream path / symbol:** paper §3.1.1 "I/O inefficiency under paged memory" and
  §5 "Performance Optimizations" describe the contiguous streaming buffer filled by a
  custom CUDA kernel; repo advertises multiprocess architecture (Apr 2026) [32]; the
  tier layout is GPU / pinned CPU DRAM / local NVMe / remote (Redis, Mooncake, InfiniStore)
  [33].
- **Why missing locally:** `ADR-003` names LMCache as inspiration and EG-18 gates Mooncake
  HiCache tiers, but nothing gates LMCache's *chunk-versus-page* transfer granularity, its
  engine-agnostic connector boundary, or its pin/lookup/compress control API.
- **T1 fit:** HIGH — CPU DRAM and NVMe tiers are exactly the 16 GB RAM + SSD envelope.
- **Clone:** **YES** (study-only): read the connector interface and chunk codec, not the
  vLLM integration glue.

### 4. MNN / MNN-LLM — `alibaba/MNN`

- **URL / license:** https://github.com/alibaba/MNN — Apache-2.0 [43][82]. Paper
  https://arxiv.org/html/2506.10443v1 [44].
- **Concrete mechanism:** MNN-LLM combines model quantisation with **DRAM–Flash hybrid
  storage** to cut residency, rearranges weights/inputs for mobile CPU instruction sets and
  GPU characteristics, and applies multicore load balancing, mixed-precision FP ops and
  geometric computations; runtime activation and KV quantisation, W4A16 asymmetric INT4 with
  group size 64 and shared quantised embedding/LM-head weights [44].
- **Exact upstream path / symbol:** `transformers/` (MNN-LLM runtime) in the repo [43];
  paper abstract and §1; MNN 3.6.1 added a Hexagon backend (22 Jul 2026) and MNN-Pocket-0.3B
  shipped 4 Nov 2026 [43].
- **Why missing locally:** EG-07 handles sliding-window *expert row* reuse on SSD, but there
  is no general DRAM–Flash tensor placement layer, no runtime KV quantisation tied to that
  placement, and no combined embedding/LM-head weight sharing after quantisation.
- **T1 fit:** HIGH — the SSD tier and INT4 group-64 path transfer directly to a 6 GB laptop.
- **Clone:** **YES** (study-only): read `transformers/` and the quantisation converter.

### 5. flyweight — `yairpatch/flyweight`

- **URL / license:** https://github.com/yairpatch/flyweight — Apache-2.0 [46].
- **Concrete mechanism:** native C++/CUDA GGUF runtime with CUDA-graph decode, MoE expert
  offload, MTP, quantised KV caches and **per-slot snapshot prefix reuse**; a flat Q8_0
  tensor-core kernel `q80_q8_mmq_k32` treats Q8_0 as 32-value blocks (rather than the MMQ's
  256-element super-blocks), and a tiled fp32 router GEMM `f32_matmul_tiled64` cuts a
  16K-token streamed prefill from 22.9 s to 16.2 s of GPU kernels [46].
- **Exact upstream path / symbol:** `native/` tree; env switches `FLYWEIGHT_Q80_FLAT_MMQ`,
  `FLYWEIGHT_Q8_MMQ_ONE_SHAPED`, `FLYWEIGHT_F32_TILED`; test `Q80FlatMmqTests`; release
  v0.28.1 (5 Nov 2026) [46].
- **Why missing locally:** EG-08's radix cache shares token prefixes but has no per-slot
  snapshot; the quantised GEMV kernels have no 32-value-block tensor-core arm and no
  environment-gated kernel fallback with a quality check.
- **T1 fit:** HIGH — GGUF + CPU/CUDA hybrid, and the flat-Q8 trick helps a 6 GB card.
- **Clone:** **YES** (study-only): small repo, 810 commits; read `native/` kernels.

### 6. ik_llama.cpp — `ikawrakow/ik_llama.cpp`

- **URL / license:** https://github.com/ikawrakow/ik_llama.cpp — MIT [42][81].
- **Concrete mechanism:** IQK quantisation with AVX-512 repacking plus trellis and ternary
  types; a **dynamic MoE expert cache with on-demand tensor reload** exposed as
  `--moe-cache-mib` (PR 2915, 16 Nov 2026); Hadamard transforms for K/V cache (PR 1033/1034,
  CPU and CUDA), and a `Q8_KV` 8-bit KV type (PR 208) [42].
- **Exact upstream path / symbol:** flag `--moe-cache-mib` and PR 2915; `Q8_KV`; PR 2887
  fixes the quantised K-cache `vec_dot` in the generic CPU flash-attention path [42].
  5,299 commits, C++ 61% / CUDA 14.6% / C 11.8% [42].
- **Why missing locally:** EG-01's dispatch table has seven types but no IQK trellis/ternary
  or Hadamard K/V transform; there is no expert-cache residency policy and no `vec_dot`
  correctness case for a quantised K-cache under ALiBi.
- **T1 fit:** HIGH — AVX2/AVX-512 laptop CPU is precisely the target of the IQK repacking.
- **Clone:** **YES** (study-only): read the IQK repack and MoE-cache flag implementation.

---

## B. Serve / runtime shells with transferable mechanisms

### 7. LMCache-style serving shell: LocalAI — `mudler/LocalAI`

- **URL / license:** https://github.com/mudler/LocalAI and
  https://localai.io/docs/concepts/architecture/index.html — MIT [40][41].
- **Concrete mechanism:** a small Go core serving OpenAI/Anthropic APIs with each model
  placed in its **own gRPC backend process**; backends are OCI images pulled on demand, and
  external backends can be pointed at via `LOCALAI_EXTERNAL_GRPC_BACKENDS` [40]. The project
  also writes native C/C++/GGML backends, including `vllm.cpp` [41].
- **Exact upstream path / symbol:** `backend/` gRPC protocol and `LOCALAI_EXTERNAL_GRPC_BACKENDS`
  env var [40]; repo `backend/` directory docs describe the Go/Python/C++ backend split [41].
- **Why missing locally:** wubuwizard is a single in-process C11 engine; there is no
  out-of-process backend boundary, so an optional accelerator or a failing backend cannot be
  isolated from the core.
- **T1 fit:** LOW–MEDIUM — the isolation pattern is useful, but a gRPC hop is a poor trade on
  a single 6 GB laptop.
- **Clone:** **NO** — read the architecture page only.

### 8. ExecuTorch — `pytorch/executorch`

- **URL / license:** https://executorch.ai/ and https://arxiv.org/html/2605.08195 —
  GitHub API reports "Other" / NOASSERTION, so verify the LICENSE text before relying on it
  [38][39][84].
- **Concrete mechanism:** AOT graph compilation to a `.pte` binary with **selective backend
  delegation** — subgraphs run on XNNPACK, Vulkan, QNN, CoreML or CPU with fallback — plus
  static/dynamic quantisation and group-wise int4 weights; 12 hardware backends, MLSys 2026
  Industry Track [38][39].
- **Exact upstream path / symbol:** `torch.export` → `.pte` → lightweight C++ runtime;
  "backend delegation" is the named mechanism in the paper abstract and §1 [38].
- **Why missing locally:** the C11 core selects SIMD kernels at compile time with `#if
  __AVX2__`; there is no portable bytecode, no runtime delegate partitioning, and no
  CPU-fallback contract for an unsupported operator.
- **T1 fit:** MEDIUM — the Windows Vulkan and XNNPACK paths are relevant to a laptop without
  CUDA; the AOT artifact format is not.
- **Clone:** **NO** — read the paper for the delegation contract only.

### 9. nano-vllm — `GeeeekExplorer/nano-vllm`

- **URL / license:** https://github.com/GeeeekExplorer/nano-vllm — MIT [36].
- **Concrete mechanism:** a ~1,200-line from-scratch vLLM reimplementation carrying prefix
  caching, tensor parallelism, torch compilation and CUDA graphs, with a chunked-prefill
  refactor merged 26 Apr 2026 [36].
- **Exact upstream path / symbol:** package `nanovllm/`, API `LLM.generate` in `example.py`,
  harness `bench.py`; PR #218 `chunked-prefill-refactor` [36].
- **Why missing locally:** EG-08/09/10 gate radix caching, overlap scheduling and admission,
  but there is no single readable reference tying chunked prefill to CUDA-graph capture in
  one scheduling loop.
- **T1 fit:** LOW–MEDIUM — Python/PyTorch and CUDA graphs, so it is a design read, not a port.
- **Clone:** **MAYBE** (study-only): one afternoon to read the scheduler; no gate.

### 10. nanovllm-omni — `MciG-ggg/nanovllm-omni`

- **URL / license:** https://github.com/MciG-ggg/nanovllm-omni — license not verified in this
  sweep [37].
- **Concrete mechanism:** a single-card reference implementation of vLLM-Omni's stage-based
  serving architecture, explicitly targeting one **4 GB card** with per-family "stage
  factories" [37].
- **Exact upstream path / symbol:** repo README project table and per-family stage factories
  [37].
- **Why missing locally:** the C11 engine is single-stream; no stage graph, stage factory or
  inter-stage payload connector exists.
- **T1 fit:** MEDIUM — 4 GB target is tighter than 6 GB, so its memory discipline is
  instructive, but the workload is multimodal.
- **Clone:** **NO** — covered by the vLLM-Omni entry below; keep as a pointer.

### 11. vLLM-Omni — `vllm-project/vllm-omni`

- **URL / license:** https://github.com/vllm-project/vllm-omni — Apache-2.0 [51][86]; paper
  https://arxiv.org/html/2602.02204 [50].
- **Concrete mechanism:** a **stage graph** (nodes = model stages, edges = transfer functions)
  run by a single orchestrator; each stage owns one engine and an explicit resource budget,
  exchanges typed intermediate state (hidden states, codec codes, latents, packed KV blocks)
  through `OmniConnector`, and scales through independent stage replica pools [50].
- **Exact upstream path / symbol:** `AsyncOmniEngine`, `Orchestrator`, `StageRuntime`,
  `OmniConnector` in `docs/design/architecture_overview.md` [51]; v0.32.0 released Oct 2026
  [51]. Python 99.6%, 4,649 commits [51].
- **Why missing locally:** no stage abstraction, no cross-stage KV block transfer, no
  request-admission boundary separate from the engine.
- **T1 fit:** LOW — multimodal serving, Python, cluster-scale; not a 6 GB laptop workload.
- **Clone:** **NO** — read the architecture doc for the stage/KV naming only.

### 12. Triton Inference Server — NVIDIA

- **URL / license:**
  https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/vllm_backend/README.html
  — BSD-3-Clause [52].
- **Concrete mechanism:** a serving envelope whose LLM backends run in **decoupled mode**;
  the vLLM backend places every request on the `vLLM AsyncEngine` immediately and lets the
  backend's in-flight batcher, not Triton's generic request-level dynamic batcher, own
  token-level scheduling [52].
- **Exact upstream path / symbol:** `vllm_backend` Python backend; `model_transaction_policy
  { decoupled: true }` in `config.pbtxt`; release 26.10 added vLLM ≥ 0.30 support [52].
- **Why missing locally:** the local scheduler has continuous batching (EG-09) but no
  server-side envelope separating request admission from token-level scheduling.
- **T1 fit:** LOW — deployment infrastructure, not a laptop engine.
- **Clone:** **NO** — read the decoupled-batching note only.

---

## C. Edge / quantisation engines

### 13. Cactus Engine — `cactus-compute/engine`

- **URL / license:** https://github.com/cactus-compute/engine — GitHub API reports "Other" /
  NOASSERTION, so verify the LICENSE before relying on it [34][80].
- **Concrete mechanism:** **rotation-and-codebook quantisation from 4-bit down to 1-bit for
  every weight tensor**, paired with hand-written ARM NEON kernels and a confidence-based
  hybrid handoff that routes hard queries to the cloud [34]. The umbrella project advertises
  a sub-50 MB footprint [35].
- **Exact upstream path / symbol:** repo components "Cactus Quants" (rotation-and-codebook
  W4→W1) and "Cactus Kernels" (NEON matmul/attention/quantisation), C++ 32.3% / C 7% [34].
- **Why missing locally:** local quant path tops out at block-scaled (Q4_0/IQ-style) formats;
  there is no rotation+codebook 1–2 bit representation and no local-confidence handoff.
- **T1 fit:** MEDIUM — ARM/phone first; only the 1–2 bit codebook idea transfers to an x86
  laptop.
- **Clone:** **MAYBE** (study-only): read the quantiser design, skip the mobile runtime.

### 14. ONNX Runtime / GenAI — Microsoft

- **URL / license:** https://onnxruntime.ai/ — MIT by project convention [unverified]; release details below are from a secondary tracker [49].
- **Concrete mechanism:** v1.30.0 added **INT4 paged KV caches**, speculative decoding in
  paged XQA, QMoE and WebGPU quantised-LLM fusions; the GenAI package is a first-class pip
  install [49][54].
- **Exact upstream path / symbol:** `pip install onnxruntime-genai` on the project page [49];
  release-feature list in the 2026-W37 tracker [54].
- **Why missing locally:** EG-37 unified-memory arena and the KV tier have no INT4 paged-KV
  representation and no QMoE path; speculative decode is token-level, not paged-XQA.
- **T1 fit:** MEDIUM–HIGH — INT4 paged KV is directly relevant to a 6 GB VRAM ceiling.
- **Clone:** **NO** — too large; read the release notes and reproduce the KV layout idea.

### 15. ge — `Ad74751/ge`

- **URL / license:** https://github.com/Ad74751/ge — GitHub API reports "Other" /
  NOASSERTION, verify LICENSE [48][85].
- **Concrete mechanism:** one ~4 MB Rust executable carrying **its own CPU kernels and its
  own Vulkan GPU kernels**, with CPU+GPU hybrid execution and an OpenAI/Anthropic-compatible
  server; the GPU matmul design follows llama.cpp's Vulkan backend [48].
- **Exact upstream path / symbol:** repo README Vulkan-backend design note; v0.1.0-alpha.1
  released 11 Nov 2026, 3 commits at sweep time [48].
- **Why missing locally:** the C11 engine has no Vulkan backend at all; all GPU paths assume
  CUDA, which is a poor fit for many 6 GB laptops.
- **T1 fit:** MEDIUM — Vulkan is the relevant alternative backend; the project is alpha.
- **Clone:** **NO** yet — re-check after a release; read the Vulkan matmul notes now.

### 16. `llm-inference-cpp` — `fernandojcardenas/llm-inference-cpp`

- **URL / license:** https://github.com/fernandojcardenas/llm-inference-cpp — MIT [47].
- **Concrete mechanism:** a hardened, fuzzed GGUF loader plus a **bounded fail-closed GBNF
  parser, a byte-level pushdown matcher that handles UTF-8 split across tokens, a vocabulary
  prefix tree for masking, and a JSON-Schema converter that errors on any keyword it cannot
  enforce**; 157 unit tests, seven libFuzzer targets, ASan+UBSan, and a 4.5-million-input
  tokenizer cross-check against Hugging Face [47].
- **Exact upstream path / symbol:** `gguf::GGUFFile::parse()` / `open()` in
  `src/model/gguf.cpp`; `src/grammar/`; `fuzz/`; cross-check tools such as
  `tools/crosscheck_tokenizer.py` and `tools/crosscheck_grammar.py` [47].
- **Why missing locally:** EG-02 added GGUF tokenizer metadata and EG-11 a grammar trie, but
  there is no fuzz/ASan gate, no fail-closed loader discipline, and no differential oracle
  at this depth.
- **T1 fit:** HIGH as a *test-discipline* template — small models, laptop CPU, no GPU at all.
- **Clone:** **YES** (study-only): 24 commits; read `src/model/gguf.cpp`, `src/grammar/`, `fuzz/`.

### 17. cosmo-edge — `cosmo-wander-ai/cosmo-edge`

- **URL / license:** https://github.com/cosmo-wander-ai/cosmo-edge — licence not verified in
  this sweep [53].
- **Concrete mechanism:** a C++ edge AI engine for video analytics and on-device VLM across
  Sophon, Rockchip RKNN and x86, with visual orchestration, real-time OSD and reproducible
  benchmarks [53].
- **Exact upstream path / symbol:** repo description in the GitHub topic listing [53]
  (no file-level symbol verified).
- **Why missing locally:** no NPU vendor backend (RKNN/Sophon) and no video pipeline exist.
- **T1 fit:** LOW — vendor-specific accelerators; x86 path is only a design read.
- **Clone:** **NO**.

### 18. exllamav3 — `turboderp-org/exllamav3`

- **URL / license:** release v1.5.0 noted in the 2026-W37 tracker [54]; licence not verified
  in this sweep.
- **Concrete mechanism:** 1.5.0 added **AVX512BW CPU offload, zero-copy pinned-arena CPU
  offload, MoE prefill and decode**, plus autosplit accounting and nondeterminism fixes [54].
- **Exact upstream path / symbol:** version and feature list in the tracker's release section
  [54]; no file-level symbol verified.
- **Why missing locally:** EG-06 solves placement but has no pinned zero-copy arena and no
  AVX512BW expert-offload kernel; SSD streaming uses staged reads, not a pinned arena.
- **T1 fit:** HIGH for the pinned-arena idea on a 16 GB RAM host; AVX512BW is optional.
- **Clone:** **MAYBE** (study-only): confirm licence and read the offload allocator first.

---

## Cross-cutting note

Two mechanisms recur and would each be a single well-scoped gate: **(a) chunk-versus-page KV
transfer granularity** (LMCache [31], vLLM-Omni packed-KV blocks [50], INT4 paged KV [54]);
and **(b) compressed-block residency** — keep weights in their quantised block form and
dequantise a row on demand with a bounded thread-local cache (vllm.cpp [45], flyweight [46],
MNN [44]). Neither is covered by the existing seven-type `vec_dot` dispatch or the file-backed
`wubu_lmcache.c`.

## Top 8 clone shortlist (study-only, ordered by value)

1. **vllm.cpp** — Apache-2.0 C++20 keep-quant decode on Q8_0/Q5_K/Q6_K/IQ blocks, bounded row
   cache, external-KV client, token-exact oracle gating [45][83]. Highest signal per line for
   a C11 engine.
2. **ik_llama.cpp** — MIT; IQK repacking, Hadamard K/V cache, `--moe-cache-mib`, `Q8_KV`
   [42][81]. Directly matches the AVX2/AVX-512 laptop CPU.
3. **KTransformers** — Apache-2.0; NUMA-aware hot/cold expert placement and AVX2 FP8 expert
   kernels [28][78]. Fills the biggest hole in the offload solver.
4. **MNN** — Apache-2.0; DRAM–Flash hybrid residency with runtime W4A16 KV/weight quantisation
   [43][44][82]. Matches the SSD tier.
5. **flyweight** — Apache-2.0; per-slot snapshot prefix reuse and flat-Q8 tensor-core kernel
   with an explicit quality gate [46]. Small repo, fast read.
6. **`llm-inference-cpp`** — MIT; hardened GGUF loader, fail-closed grammar and a fuzz/ASan
   differential-oracle harness [47]. Raises gate quality, not throughput.
7. **LMCache** — Apache-2.0; chunked tiered KV connector and pin/lookup/compress control API
   [31][32][79]. Read the connector boundary, not the vLLM glue.
8. **vAttention** — MIT; contiguous-virtual/on-demand-physical KV with CUDA VMM [29][30].
   Read for the allocator contract behind a 6 GB budget.

No vendoring: each clone goes to `~/WuBu/research/engine-sources/` as study material, is read
for one mechanism, and produces a C11 gate plus evidence in `docs/WORKLIST.md`.

## Sources

[28] https://github.com/kvcache-ai/ktransformers
[29] https://arxiv.org/html/2405.04437
[30] https://github.com/microsoft/vattention
[31] https://arxiv.org/html/2510.09665
[32] https://github.com/LMCache/LMCache
[33] https://docs.lmcache.ai/v0.3.7/developer_guide/architecture.html
[34] https://github.com/cactus-compute/engine
[35] https://github.com/cactus-compute/cactus
[36] https://github.com/GeeeekExplorer/nano-vllm
[37] https://github.com/MciG-ggg/nanovllm-omni
[38] https://arxiv.org/html/2605.08195
[39] https://executorch.ai
[40] https://localai.io/docs/concepts/architecture/index.html
[41] https://github.com/mudler/LocalAI
[42] https://github.com/ikawrakow/ik_llama.cpp
[43] https://github.com/alibaba/MNN
[44] https://arxiv.org/html/2506.10443v1
[45] https://github.com/localai-org/vllm.cpp
[46] https://github.com/yairpatch/flyweight
[47] https://github.com/fernandojcardenas/llm-inference-cpp
[48] https://github.com/Ad74751/ge
[49] https://onnxruntime.ai
[50] https://arxiv.org/html/2602.02204
[51] https://github.com/vllm-project/vllm-omni
[52] https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/vllm_backend/README.html
[53] https://github.com/cosmo-wander-ai/cosmo-edge
[54] https://www.inference-radar.com/newsletter/2026-W37
[78] https://api.github.com/repos/kvcache-ai/ktransformers/license
[79] https://api.github.com/repos/LMCache/LMCache/license
[80] https://api.github.com/repos/cactus-compute/engine/license
[81] https://api.github.com/repos/ikawrakow/ik_llama.cpp/license
[82] https://api.github.com/repos/alibaba/MNN/license
[83] https://api.github.com/repos/localai-org/vllm.cpp/license
[84] https://api.github.com/repos/pytorch/executorch/license
[85] https://api.github.com/repos/Ad74751/ge/license
[86] https://api.github.com/repos/vllm-project/vllm-omni/license
