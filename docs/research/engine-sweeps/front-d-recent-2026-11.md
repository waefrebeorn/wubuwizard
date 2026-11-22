# Front D — time-bounded recent-gains sweep (2026-10-31 → 2026-11-22)

**Method.** Time-bounded `web_search` queries for November 2026 releases, arXiv
submissions and primary engineering blogs, followed by `web_extract` of the primary
pages. Search snippets were never cited as evidence. AI-news aggregator claims were
discarded unless a primary source confirmed them; where no primary was found, that is
stated explicitly rather than smoothed over. Registered sources: 18; extracted and
cited: 12. All URLs are in the Sources block generated from the citation ledger.

**Diff rule.** Every finding below was checked against
`docs/ENGINE-GAP-REGISTER.md` and `docs/ENGINE-SOURCE-INDEX.md`. Only genuinely new
mechanisms get a new EG id; refinements of an existing gap say so explicitly.

## D1 — vLLM v0.34.0: extensible KV cache becomes the default (16 Nov 2026)

vLLM now sizes the KV cache from memory actually free **after** warmup and CUDA graph
capture instead of from an estimate, and recommends removing `gpu_memory_utilization` in
most configurations.[1] The default block size changed from 16 to 64, and the release
reports the KV cache growing from 433k to 672k tokens on one H200 example, with a
documented TPOT regression of +6–18% on some models at concurrency 128.[1]

The same mechanism has a concrete integrated-GPU failure mode that matters at T1: on
UMA parts, raw `cudaMemGetInfo` does not count reclaimable page cache as free, so the
extensible cache undersized by up to 84% on a DGX Spark (GB10) and reached zero blocks
in P/D serving.[18] The fix reads free memory through a helper with the `psutil`
correction already used by the classic path; on GB10 that changed Qwen3.8-27B NVFP4 from
216 blocks / 11.2 GiB to 1302 blocks / 67.6 GiB, while discrete GPUs are unchanged.[18]
Device allocation through `cuMemCreate` + `cuMemMap` reclaimed the page cache, reaching
1.8× the free memory `cudaMemGetInfo` reported.[18]

**Diff:** refines **EG-03** (VRAM planner). New sub-mechanism: measure-after-warmup plus
UMA page-cache-corrected free memory. Reserve as **EG-35**.

## D2 — llama.cpp v1.0.0: OCP FP8 KV and MoE expert cache (13 Nov 2026)

llama.cpp v1.0.0 added OCP FP8 KV caches (`-ctk/-ctv f8_e4m3`) with optional static
per-tensor scales, and promoted the NextN/speculative API into the stable `llama.h`
surface.[2] It also added a host-memory MoE expert cache with LRFU/ClassLRU replacement
and no CPU wait on cached experts, plus loading of external PLE shards.[2]

**Diff:** FP8 KV is a concrete instance of **EG-30**; the replacement policies and
"no CPU wait on cached expert" behavior refine **EG-20**. Reserve as **EG-36**.

## D3 — SGLang unified memory: one arena, one entry per token (11 Nov 2026)

SGLang's unified-memory mode puts full-attention KV, sliding-window KV and recurrent
state in one arena so capacity follows demand instead of a startup partition.[14] Each
token's KV rows across all layers are packed into a single entry, so mixed shapes and
dtypes — including speculative draft KV — share one allocation and lifecycle.[14]
Freed pages are reclaimed with compaction that relocates live allocations, and request
admission checks whether the whole request fits.[14] The design composes with overlap
scheduling, Radix cache, HiCache, P/D disaggregation, speculative decoding, pipeline
parallelism and decode context parallelism.[14]

**Diff:** no existing gap covers a unified arena or the one-entry-per-token layout;
**EG-18** covers tiers, not a single allocator. Reserve as **EG-37** — the closest fit
to the 6 GB target in this sweep.

## D4 — HiCache across parallelism and disaggregation (2 Nov 2026)

The HiCache extension keeps a shared logical prefix while storing and restoring KV pages
on their owner ranks, using a logical radix tree with explicit page ownership.[13]
Reading sharded KV gathers one layer at a time, and each shard is restored before the
gather.[13] Speculative decoding restores the draft model's required state alongside
the target cache, and P/D disaggregation reuses decode-side prefixes so prefill transfers
only the missing KV.[13] A unified KV layout lets stored KV be reused across compatible
parallel topologies.[13]

**Diff:** extends **EG-08** (radix tree) and **EG-18** (HiCache tiers) with page
ownership and draft-state restore; multi-rank sharding is tier-gated. Reserve as
**EG-38** with T1/T2 split: keep logical radix + decode-side reuse for T1, defer sharding.

## D5 — New engines: Hibana and Callo (7 and 13 Nov 2026)

Hibana targets unified-memory devices and co-optimizes model representation, memory
management and decoding strategy: model surgery shrinks memory footprint and data
movement, unified-memory management preserves each agent's state across turns, and
adaptive speculation generates multiple tokens per step.[15] The paper reports up to
3.4× more agentic tasks per hour and 3.9× higher decode throughput than vLLM, NVIDIA
Flare and llama.cpp on a single DGX Spark with 128 GB memory.[15] The device class is
larger than T1, but the three mechanisms are model-agnostic.

Callo addresses agent sessions where context edits invalidate cached KV and force a
re-prefill before decoding.[16] It uses the invalidated KV as speculative drafts, then
verifies them against the exact KV after prefill before committing tokens; it schedules
speculation during decode underutilization and preempts when drafts are unlikely to be
accepted.[16] The reported all-request speedup is 1.05–1.19× over vLLM on a two-GPU
testbed with SWE-agent sessions.[16]

**Diff:** neither engine is in the source index. Hibana's persistent per-agent state
maps directly onto the agentic-OS direction; Callo is the first concrete handling of
cache invalidation from context edits. Reserve as **EG-39** and **EG-40**.

## D6 — Speculative decoding: three new mechanisms (31 Oct – 17 Nov 2026)

SpecSearch argues that speculative configuration depends jointly on workload, GPU and
available memory, reporting that DFlash reaches 3.50× autoregressive throughput at low
concurrency but only 0.76× at high concurrency with short context on an RTX 5090.[3]
TVP asserts that lossless speculative sampling accepts a draft token with probability
one minus total variation distance, and that drafters are usually trained with CE or KL
instead.[4] A separate study quantizes only the draft model's KV cache — K4V4 with
per-channel keys and per-token values, target cache left in BF16 — reducing draft cache
memory 66.20% and combined cache memory 23.95% while mean acceptance moves from 57.93%
to 57.14%.[17]

**Diff:** **EG-25** listed Sequoia/SpecTr/DAG/recycling/relaxed verification but not
hardware-aware configuration search, TV-objective drafter training, or draft-cache
quantization. Reserve as **EG-41**, **EG-42**, **EG-43**.

## D7 — Sparse attention: trained indexer vs post-training pruning (17 Nov 2026)

NVIDIA's sparse-attention co-design post distinguishes model-level sparsity, where
sparsity is trained in, from drop-in post-training methods such as BLASST, MInference
and Quest.[8] Its indexer runs as a batched matmul with a fused epilogue; arithmetic
intensity scales with group size `G = QH/KH` and speculative draft length `D+1`, which
must fit one Tensor Core tile of about 200 rows.[8] Head dimensions of 128 or 256 and an
FP16 epilogue/accumulation further improve throughput, while block-level selection
lets core attention fetch keys in bulk and share them across queries.[8]

**Diff:** **EG-23** covers Skip-Softmax/BLASST as a drop-in. The trained-indexer path is
a different mechanism and needs its own gate. Reserve as **EG-44**.

## D8 — KV compression and MoE offload papers (Oct–Nov 2026)

LowRAM identifies low-rank keys with amortised values as a compression axis
complementary to token/layer/dimension redundancy and lower precision.[5] MeltKV uses
memory-aligned variable-bit packed vector quantization with adaptive bit-widths per
grouped key channel, aimed at sub-4-bit KV with fast decode.[6] MoE-CORE coordinates
expert offloading and residency, staging complete expert layers in alternate memory to
reduce host-to-device transfers on memory-constrained devices.[7]

**Diff:** existing KV work uses TurboQuant/RotorQuant/PolarQuant; these three are new
families. **EG-20** covers expert traces but not coordinated layer staging. Reserve as
**EG-45** (KV) and refine **EG-20** with the MoE-CORE staging policy.

## D9 — Tier-gated: SiFAR (MICRO 2026)

SiFAR removes synchronization overhead from all-reduce in low-batch serving using
redundant pull through in-switch reduction and speculative reduction that starts data
transfer before the top barrier with lightweight validation.[10] It reports up to 52%
all-reduce latency reduction and 18.6% end-to-end throughput improvement for
Llama-3.1-8B at TP=8.[10] Multi-GPU only: record as **EG-46**, tier-gated T4.

## Rejected / unconfirmed

Claims about a "SGLang v1.0 consortium" and a "DeepSeek engine 3× GPU value" appeared
only in AI-news aggregators; no primary release or paper was retrieved for them in this
sweep, so no gap is based on them. The TensorRT-LLM release-notes page extracted to an
older changelog section, so the recent v1.3/v1.4 feature claims from search snippets are
not used here. SiFAR and Hibana report results on hardware larger than T1; the numbers
are the authors' reported figures, not local measurements.

## What not to break

Keep the mmap zero-copy weight law, the single-process no-vendoring rule, the
engine-ignorant-of-domain-concepts boundary, and the "no gate, no ship" discipline.
A reported speedup from a paper is a reason to write a gate, not evidence that our
engine has the feature.

## Sources

[1] https://github.com/vllm-project/vllm/releases/tag/v0.34.0 — v0.34.0
[2] https://github.com/ggml-org/llama.cpp/releases/tag/v1.0.0 — v1.0.0
[3] https://arxiv.org/abs/2611.22296 — 2611.22296
[4] https://arxiv.org/abs/2611.00588 — 2611.00588
[5] https://arxiv.org/abs/2611.07025 — 2611.07025
[6] https://arxiv.org/abs/2611.18652 — 2611.18652
[7] https://arxiv.org/abs/2610.01950 — 2610.01950
[8] https://developer.nvidia.com/blog/co-designing-sparse-attention-for-efficient-long-context-inference — co-designing-sparse-attention-for-efficient-long-context-inference
[10] https://ar5iv.labs.arxiv.org/html/2607.08973 — 2607.08973
[13] https://www.lmsys.org/blog/2026-11-02-hicache-distributed-serving — 2026-11-02-hicache-distributed-serving
[14] https://www.lmsys.org/blog/2026-11-11-unified-memory — 2026-11-11-unified-memory
[15] https://arxiv.org/html/2611.10469v1 — 2611.10469v1
[16] https://arxiv.org/html/2611.18459v1 — 2611.18459v1
[17] https://arxiv.org/html/2611.20565 — 2611.20565
[18] https://github.com/vllm-project/vllm/pull/65413 — 65413
