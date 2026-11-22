# Front-C — OFFLOAD and DECODE-ACCELERATION research sweep

**Target:** wubuwizard — everyman local inference on 6 GB VRAM / 16 GB RAM / SSD, reimplemented in our own C11, never vendored.
**Retrieval budget actually spent:** 13 `web_search` calls, 14 `web_extract` calls, primary sources preferred (papers, official repos, project docs).
**Local audit:** grepped `src/` and `docs/` for mechanism symbols (not just words) and read `src/wubu_eagle.c`, `src/wubu_more_spec.c`, `src/wubu_ecosystem.c`, `docs/ssd_moe.md`. Local truth per caller: 460 C/CU sources, 462 test sources, 95 named make gates.
**Standing rule:** every gap below names a *reference to study and reimplement in C11*. Nothing here recommends linking or vendoring third-party code.

## Already absorbed — NOT gaps, do not re-report

Caller-confirmed implemented: continuous batching, prefix KV reuse, paged KV, chunked prefill, flash attention/prefill/decode, KV filesystem, TurboQuant/RotorQuant/PolarQuant, IQ2/FP4/MXFP4/NVFP4/AWQ/GPTQ dequant, MoE variants, spec decode (Medusa/EAGLE/lookahead/n-gram/cascade/parallel), GGUF+safetensors readers, Gemma4/Qwen/MoE/SSM/LFM2 kernels, CUDA graphs, weight streaming, LM-head cache, 16 LP pads, 5 reference ports. Engine findings that only re-state these are marked ABSORBED below and are excluded from the gap list.

---

## Engine findings

### E1 — PowerInfer (SOSP 2024)
**URL:** [2]. **Mechanism:** exploits power-law *activation locality*: a small set of "hot" neurons is consistently activated and is preloaded to GPU; input-dependent "cold" neurons are computed on CPU. It adds adaptive predictors and neuron-aware sparse operators; under CPU-GPU hybrid mode dense activation blocks go to GPU and FFN is split by residency [2].
**Delivers:** a consumer-grade GPU can serve a much larger model with reduced PCIe traffic [2].
**Fit (6/16/SSD):** strong — the hot-in-VRAM/cold-in-RAM split is exactly the target hierarchy.
**wubu status:** GAP. Search of `src/` for `activation.predictor|hot.neuron|neuron.cluster|activation.locality` returned no inference-path hits (only unrelated "spectral"/"mel" words).

### E2 — PowerInfer-2 (arXiv 2406.06282)
**URL:** [1]. **Mechanism (paper §1, §3):** the *neuron cluster* is the basic processing unit; a cluster is a set of neurons with the same activation pattern and its size is decided at runtime per compute unit [1]. *Sparsity-Aware Adaptation* sends dense clusters to the NPU and sparse clusters to CPU and re-balances the cluster ratio as batch size changes [1]. *I/O-Aware Orchestration* breaks the barrier between matrix computations: as soon as one cluster finishes, the next in-memory cluster (possibly from the next matrix) starts, and a big core asynchronously preloads next-layer weights into a segmented neuron cache; cold clusters use on-demand small random reads while hot ones are prefetched [1].
**Delivers:** 47B on a 16–24 GB phone at 11.68 tok/s, 24.6× (up to 27.8×) llama.cpp, 40% less memory at 7B with matched speed [1]; with 7 GB and only 1.8% of FFN cached it still reaches 2.13 tok/s [1].
**Fit:** excellent — UFS random I/O is ~12× worse than our SSD, so its I/O orchestration is conservative for our hardware [1].
**wubu status:** GAP (neuron-cluster abstraction, segmented cache, cluster-level pipeline, runtime hot/cold rebalancing).

### E3 — FlexGen (ICML 2023)
**URL:** [3][29]. **Mechanism (paper §4):** generative inference with offloading is a graph traversal; a **zigzag block schedule with overlapping** (Algorithm 1) loads next-layer weights, stores previous-batch activations/KV, and loads next-batch activations/KV around each compute step [3]. A linear program searches tensor placement over weights/activations/KV across GPU, CPU and disk (policy tuple `wg,wc,cg,cc,hg,hc` × GPU-batch × #GPU-batches); the zigzag schedule is proven within 2× of optimal I/O [3]. Groupwise 4-bit quantization along weight output dim and KV hidden dim cuts I/O, and only the top-10% highest-score K are loaded for sparse attention [3].
**Delivers:** OPT-175B at 1 tok/s on a single 16 GB T4 with effective batch 144; ~40–100× throughput over ZeRO-Inference/Accelerate [3].
**Fit:** strong, directly models GPU/CPU/SSD tiers [3].
**wubu status:** PARTIAL → GAP. Weight streaming and KV filesystem exist, but no placement LP and no zigzag overlapping schedule were found.

### E4 — LLM in a Flash (Apple, arXiv 2312.11514)
**URL:** [4]. **Mechanism (§2–§3):** a flash-aware inference cost model drives two techniques — *windowing*, a DRAM cache of FFN weight rows predicted active for the recent token window with only the incremental set `s_agg(k+1) − s_agg(k)` loaded from flash; and *row-column bundling*, coalescing FFN up/down rows and columns into one large sequential read so the latency-to-first-byte is paid once [4]. A low-rank predictor marks active neurons; embeddings and attention matrices stay resident, only non-sparse FFN segments are fetched [4]. Throughput-appropriate sparse inference needs ≥32 KiB random reads across threads [4].
**Delivers:** models up to 2× DRAM size, 4× CPU and 20× GPU speedup over naive loading [4].
**Fit:** the closest paper to our SSD tier [4].
**wubu status:** PARTIAL → GAP. `docs/ssd_moe.md` pages experts into an LRU slot bank; there is no neuron-row sliding window, no incremental `s_agg` diff, and no row-column bundling.

### E5 — ZeRO-Inference / DeepSpeed Inference
**URL:** [9][10][28]. **Mechanism:** pin the entire model in CPU DRAM or NVMe, stream each layer into GPU for compute, release its weight memory after use, keep activations for the next layer [10]. A configurable *N-layer prefetch* overlaps fetching layer *i+N* with compute of earlier layers [9][10]. Activations are offloaded past a memory threshold, and odd/even layers offload on alternating GPUs to avoid PCIe contention [9]. DeepNVMe supplies bulk asynchronous read/write requests [28], and later versions add 4-bit weights and KV-cache offload to CPU [10].
**Delivers:** full offload at 43 tok/s (CPU) and 30 tok/s (NVMe) [10]; orders-of-magnitude GPU memory reduction.
**Fit:** strong; NVMe tier maps to our SSD [9].
**wubu status:** PARTIAL → GAP. Streams weights already, but no explicit N-layer prefetch window, activation-offload threshold, or PCIe-contention scheduling found.

### E6 — ZeRO-Infinity (SC 2021)
**URL:** [11]. **Mechanism:** the *infinity offload engine* places partitioned states across GPU/CPU/NVMe; *memory-centric tiling* splits one very large operator into sequential tiles so a layer too big for VRAM can still execute; *bandwidth-centric partitioning* and an *overlap engine* overlap NVMe→CPU, CPU→GPU and GPU-GPU traffic with compute [11].
**Delivers:** training-scale result but the tiling and overlap ideas are architecture-agnostic [11].
**Fit:** memory-centric tiling is directly useful at 6 GB; optimizer-state machinery is not [11].
**wubu status:** GAP (tiling; overlap partially covered by existing weight-streaming paths).

### E7 — Fiddler (ICLR 2025)
**URL:** [7][8]. **Mechanism (§3, Fig. 1):** *compute-centric offloading* — when an expert is absent from GPU, copy the small **activations** (`batch × 4096`) to CPU, run the expert there, copy the result back, instead of copying the huge weights (`3 × 4096 × 14336` per Mixtral expert) [8]. A latency model using per-expert token count from the gate and CPU/GPU batching behaviour picks the execution plan per layer [7]; frequently used experts are pinned to GPU by offline popularity profiling, and a custom AVX512_BF16 CPU expert kernel overlaps with GPU non-expert work [7].
**Delivers:** 1.26× single-batch, 1.30× long-prefill, 11.57× beam search [7]; 8.2–22.5× over DeepSpeed-MII/Mixtral-offloading, unquantized Mixtral-8x7B >3 tok/s on one 24 GB GPU [8].
**Fit:** ideal — it trades abundant 16 GB RAM and CPU compute for scarce 6 GB VRAM [7].
**wubu status:** GAP. SSD MoE pages weights in; no activation-to-CPU execution path or latency model found.

### E8 — MoE-Infinity (arXiv 2401.14361)
**URL:** [19]. **Mechanism (§4, Algorithm 1):** a *sparsity-aware expert cache* at batch size 1. `iEAM` traces tokens routed to each expert per iteration; `pEAM` predicts the next layer's experts from the current routing; eviction computes a reuse-likelihood priority by matching historical EAMs with **layer decay** giving earlier layers higher priority; a small expert-activation-matrix cache (EAMC, ~3% of requests) captures recurring patterns [19]. One pinned-memory/DMA I/O thread per GPU saturates PCIe 4.0 [19].
**Delivers:** 3.1–16.7× lower TPOT than vLLM/Ollama/DeepSpeed/BrainStorm on one A5000-24GB [19].
**Fit:** excellent; designed for the personal-machine batch-1 case [19].
**wubu status:** GAP. `docs/ssd_moe.md` is plain per-layer LRU — no activation trace, predictor, layer-decay eviction, or cross-layer prefetch.

### E9 — PIPO (arXiv 2504.03664)
**URL:** [23]. **Mechanism:** fine-grained pipelined offloading: weights placed in GPU/CPU/NVMe by capacity, KV cache in CPU with load-before-MHA and store-after; the runtime auto-selects a performance-optimized or memory-efficient pipeline from model type, precision, batch size, prompt/generation length, GPU/CPU memory, PCIe bandwidth and current load [23]. DP/TP/PP and expert parallelism are expressed by changing only what a layer-load fetches [23].
**Delivers:** on a laptop with an **RTX 3060 6 GB**, GPU utilization goes from <40% to >90% and throughput rises up to 3.1×; 10 GB less DRAM than FlexGen and 66.4% less VRAM than non-offload with 11.2% slowdown [23].
**Fit:** exact hardware match [23].
**wubu status:** GAP as a named pipeline + auto-configuration policy.

### E10 — TightLLM (IEEE Trans. Computers 2025)
**URL:** [24]. **Mechanism:** offloading policy is **adaptive per iteration**, not static. Its KV distributor uses a *trade-compute-for-transfer* strategy that recomputes portions of the KV cache on GPU while other portions transfer; its weight loader slices weights and spreads loading across batches so per-batch compute exceeds load latency [24].
**Delivers:** 1.3–23× decode and 1.2–22× prefill throughput over state-of-the-art offloaders; 59.6–94.9% lower completion time [24].
**Fit:** strong for variable-length local workloads [24].
**wubu status:** GAP.

### E11 — ALISA (arXiv 2403.17312)
**URL:** [12]. **Mechanism:** Sparse Window Attention keeps KV for the globally important tokens; a three-phase token-level scheduler moves between Phase I (all KV in GPU), Phase II (token-level GPU/CPU split, static local tokens kept on GPU), and Phase III *recomputation-caching* (delete oldest CPU KV and recompute on GPU when needed, since access cost eventually exceeds recompute) [12].
**Delivers:** up to 3× over FlexGen and 1.9× over vLLM on one GPU-CPU system [12].
**Fit:** strong — the crossover decision is exactly a 16 GB-RAM/SSD problem [12].
**wubu status:** PARTIAL → GAP. KV filesystem and paged KV exist; no three-phase or Phase-III recompute decision found.

### E12 — KVPR (Findings of ACL 2025)
**URL:** [13]. **Mechanism:** CPU first transfers a *partial set of activations* so the GPU can begin recomputing KV values; the remaining KV transfers concurrently while recompute runs. Profiler, scheduler and runtime choose the compute/communication split from input characteristics and hardware [13].
**Delivers:** up to 35.8% lower decode latency and 46.2% higher decode throughput [13].
**Fit:** strong; avoids depending on CPU compute power [13].
**wubu status:** GAP (partial-recompute-over-transfer overlap).

### E13 — mzCache (MobiCom 2026)
**URL:** [14]. **Mechanism:** *restoration-oriented memory management* for OS-level memory pressure. Weights and KV are partitioned into fine-grained shared buffers for partial eviction/restoration with concurrent CPU and GPU access; hybrid swap and backward-out eviction policies minimize restoration latency; an in-memory compressed-KV path (zRAM-inspired) complements storage reads; on resume the GPU computes preserved front layers while the CPU restores evicted layers in forward order [14].
**Delivers:** 2.1–5.5× lower TTFT than storage-backed partial offload [14].
**Fit:** strong for a 16 GB machine that also runs other apps [14].
**wubu status:** GAP.

### E14 — EAGLE-3 (NeurIPS 2025)
**URL:** [5][6][18]. **Mechanism (§1–§2):** removes the feature-regression constraint of EAGLE/EAGLE-2 and predicts tokens directly; *training-time test* simulates multi-step drafting during training (target features at early positions, the draft's own outputs at later positions) to cure distribution mismatch [5]. It fuses **low-, mid- and high-layer** target features instead of only the pre-LM-head layer, inherits EAGLE-2's context-aware dynamic draft tree, and can truncate the draft vocabulary to high-frequency tokens to cut LM-head cost [5][18].
**Delivers:** 3.0–6.5× over autoregressive, ~1.4× (20–40%) over EAGLE-2 [5][18].
**Fit:** strong — a small draft head is cheaper than a second model on 6 GB [5].
**wubu status:** GAP. `src/wubu_eagle.c:2-11` is explicitly EAGLE-1 (Zhang et al. 2402.00366) implemented as a **truncated target model** with `draft_layers`, not a feature-fusion head; no training-time-test, tri-layer fusion, dynamic tree, or high-frequency vocab truncation exists in `src/`.

### E15 — Lookahead / Jacobi family
**URL:** [20][27]. **Mechanism:** autoregressive decoding as Jacobi fixed-point iteration; a fixed 2D window (sequence × time) mines disjoint n-grams from the Jacobi trajectory; a verification branch accepts promising n-grams in one pass while preserving the output distribution; compatible with FlashAttention [20].
**Delivers:** up to 1.8× MT-bench, 4× with multi-GPU strong scaling [20].
**Fit:** strong [20].
**wubu status:** ABSORBED — `src/wubu_lookahead.c` and `src/wubu_ngram.c` exist and the caller confirms lookahead + n-gram. Only worth a regression check that the 2D Jacobi-window trajectory (not just generic n-grams) is what was ported; not a new gap.

### E16 — Speculative families beyond the current set
**URL:** [21][22][27]. **Concrete mechanisms:** SpecInfer merges candidates into a shared-prefix token tree with tree attention and DFS KV reuse; Sequoia uses a DP-optimal tree with a hardware-aware size/depth optimizer; SpecTr frames multi-draft selection as optimal transport; GSD generalizes the tree to a DAG; Ouroboros recycles rejected draft tokens; Judge Decoding uses a lightweight relaxed verifier; Yggdrasil unifies dynamic speculation with a static runtime; Hydra couples Medusa heads autoregressively; TETRIS selects draft tokens for batched verification; PEARL pipelines draft and verify stages [21][22][27].
**Fit:** most are cheap policy/tree changes that fit 6 GB [21].
**wubu status:** PARTIAL → GAP. `src/wubu_more_spec.c:5-19` implements REST, tree restructure, contrastive, distil, spec-MoE skip, cascade, swap, layer-stream resume and cascade+early-exit; no Sequoia/SpecTr/DAG/recycling/judge/runtime symbols found.

### E17 — MindGPT-4ov (Li Auto, arXiv 2512.02895)
**URL:** [16]. **Mechanism (§5.2):** FP8 inference, FlashInfer MHA with PagedAttention, DeepGEMM MoE; *modality-split chunked prefill* splits image and text chunks so vision and text encoding run in parallel during the user's speaking phase; vision-encoder outputs are cached and reused on later turns keyed by a session id; inference is disaggregated into vision / prefill / decode stages with static, chunked and dynamic batching respectively [16].
**Delivers:** lower streaming latency and multi-turn reuse in high concurrency [16].
**Fit:** the disaggregation is datacenter-scale; the embedding cache and modality split are transferable only if a multimodal path is in scope [16].
**wubu status:** LOW PRIORITY / conditional, not a core 6 GB gap. Adjacent runtime-memory work such as ApCM's invertible dimensionality reduction with a learned auxiliary predictor [15] targets persistent memory storage rather than weight/KV residency, so it is out of scope here.

### E18 — Recorded non-fits (do not port as-is)
DirectKV's zero-copy KV offload relies on NVLink-C2C between GPU and CPU on GH200/GB200 and CPU-memory-aware CUDA kernels [25] — no coherent high-bandwidth fabric on a 6 GB consumer card, so only its fused KV-generation+attention kernel and warp-level pipelining are transferable concepts. The compute-in-flash work requires Compute-in-Flash hardware and integer-only arithmetic [17]; its sparse-dictionary KV compression with a full-precision local window is an algorithmic idea, not a port target.

---

## Gap roadmap (dependency order — lilypads)

### LP-A — Neuron-cluster residency engine
Define a C11 `neuron_cluster` descriptor (activation-pattern bitmap, runtime cluster size), pin hot clusters in VRAM/RAM and cold clusters on SSD, add a low-rank or ReLU activation predictor, and implement the sliding-window incremental weight-row load `s_agg(k+1) − s_agg(k)` plus row-column bundling into ≥32 KiB reads [1][4]. Add a cluster-level pipeline that crosses matrix boundaries and asynchronously prefetches next-layer clusters [1]. Unlocks: >2× model footprint at fixed VRAM; depends on nothing. **Size XL.**

### LP-B — Offload placement LP + zigzag/adaptive pipeline
Solve a small LP over weight/activation/KV residency percentages and GPU-batch count with a measured cost model, execute a zigzag block schedule with overlapping load/store/compute, then make the split adaptive per iteration (recompute KV vs transfer, slice weight loads across batches) and auto-configure from a hardware probe [3][23][24]. Unlocks: high GPU utilisation at 6 GB; depends on LP-A's residency tiers. **Size L.**

### LP-C — Expert residency from activation traces
Replace the LRU slot bank with an expert-activation-matrix ring (`iEAM`), a next-layer predictor (`pEAM`), layer-decay eviction priority and popularity pinning; add the alternative *compute-on-CPU* path that moves activations instead of weights with an AVX-512 BF16 kernel and an explicit latency model [7][19]. Unlocks: fewer expert page-ins than LRU on the same SSD; depends on LP-A/B's tier plumbing. **Size L.**

### LP-D — KV restore-vs-recompute scheduler
Implement three-phase token-level KV placement (GPU → GPU/CPU → recompute-caching) [12]. Add partial-activation-first recompute overlapped with streaming transfer and a profiler-derived offload interval [13][26]. Include fine-grained shared buffers with a compressed-KV second restore path for memory pressure [14]. Unlocks: long contexts inside 16 GB; depends on LP-B's pipeline. **Size L.**

### LP-E — EAGLE-3 draft head + optimal tree sizing
Reimplement direct token prediction with training-time test and low/mid/high feature fusion on top of the existing `wubu_eagle.c`, inherit the context-aware dynamic tree, add Sequoia's DP/hardware-aware tree-size optimizer and optionally SpecTr multi-draft selection [5][21][22]. Unlocks: acceptance-rate gains without a second resident model; depends on a stable attention/tree-attention path. **Size M.**

### LP-F — I/O primitives for oversized layers
Add memory-centric tiling for a layer too large for 6 GB, pinned-buffer async bulk reads, and odd/even bandwidth-contention-aware scheduling [9][11][28]. Unlocks: layers that cannot fit whole; depends on LP-B. **Size M.**

## Explicitly NOT in wubuwizard already (verified gaps)

1. Activation-sparsity predictor, hot/cold neuron split, neuron-cluster abstraction and segmented neuron cache (E1, E2).
2. Cluster-level I/O-compute pipelining across matrix boundaries and runtime NPU/CPU-role rebalancing (E2).
3. Flash/SSD sliding-window weight-row reuse with `s_agg(k+1) − s_agg(k)` incremental loading and row-column bundling (E4).
4. Linear-programming offload-placement solver and zigzag overlapping block schedule with its 2× I/O bound (E3).
5. Configurable N-layer weight prefetch window, activation-offload threshold and PCIe-contention scheduling (E5).
6. Memory-centric tiling for oversized layers (E6).
7. Compute-centric MoE offload (move activations to CPU, run expert there) with a per-expert latency model and AVX512_BF16 CPU expert kernel (E7).
8. Expert activation traces (`iEAM`/`pEAM`), layer-decay eviction priority, EAMC pattern cache and cross-layer expert prefetch — current `docs/ssd_moe.md` is per-layer LRU only (E8).
9. Auto-configured fine-grained pipelined offload with performance/memory pipeline choice, validated on 6 GB GPUs (E9).
10. Per-iteration adaptive offloading: KV trade-compute-for-transfer recomputation and weight-slice spreading across batches (E10, E12).
11. ALISA three-phase token-level KV scheduling including Phase-III recomputation-caching (E11).
12. Restoration-oriented eviction/restore with fine-grained shared buffers and compressed-KV second restore path (E13).
13. EAGLE-3 mechanisms: direct token prediction, training-time test, tri-layer feature fusion, EAGLE-2 dynamic draft tree, high-frequency-only draft vocabulary (E14). Existing `wubu_eagle.c` is EAGLE-1 truncated-layer drafting.
14. Sequoia DP/hardware-aware tree sizing, SpecTr optimal-transport multi-draft selection, DAG verification, token recycling, relaxed verification, static-runtime speculation, Hydra coupling, TETRIS batched selection, PEARL pipelining (E16).
15. Modality-split chunked prefill and session-keyed encoder-output cache (E17, conditional on a multimodal path).

## What not to break

Keep continuous batching, paged KV, prefix reuse, chunked prefill, flash kernels, existing quant dequant formats, weight streaming and the 95 named gates. One technique → one gate → one measured evidence row; a pad is not shipped without a gate. The core stays ignorant of any hosted domain concept. No third-party source is copied: each pad names a paper/reference implementation to read and reimplement in C11.

## Sources

[1] https://arxiv.org/html/2406.06282 — PowerInfer-2
[2] https://github.com/Tiiny-AI/PowerInfer — PowerInfer README
[3] https://proceedings.mlr.press/v202/sheng23a/sheng23a.pdf — FlexGen ICML 2023
[4] https://ar5iv.labs.arxiv.org/html/2312.11514 — LLM in a Flash
[5] https://arxiv.org/html/2503.01840v1 — EAGLE-3
[6] https://sgl-project.github.io/SpecForge/concepts/EAGLE3.html — SpecForge EAGLE3
[7] https://arxiv.org/html/2402.07033 — Fiddler ICLR 2025
[8] https://github.com/efeslab/fiddler — Fiddler repo
[9] https://arxiv.org/html/2207.00032 — DeepSpeed Inference / ZeRO-Inference
[10] https://www.deepspeed.ai/2022/09/09/zero-inference.html — ZeRO-Inference DeepSpeed blog
[11] https://ar5iv.labs.arxiv.org/html/2104.07857 — ZeRO-Infinity
[12] https://arxiv.org/html/2403.17312 — ALISA
[13] https://aclanthology.org/2025.findings-acl.997 — KVPR
[14] https://arxiv.org/html/2609.01338 — mzCache
[15] https://arxiv.org/html/2601.11609 — ApCM model
[16] https://arxiv.org/html/2512.02895v2 — MindGPT-4ov
[17] https://arxiv.org/html/2609.16161 — LLM Inference in a Flash (compute-in-flash)
[18] https://docs.sglang.io/docs/advanced_features/speculative_decoding.md — SGLang speculative decoding docs
[19] https://arxiv.org/html/2401.14361 — MoE-Infinity personal machines
[20] https://arxiv.org/html/2402.02057 — Lookahead Decoding
[21] https://arxiv.org/html/2502.19732v3 — Speculative Decoding and Beyond survey
[22] https://github.com/NeuraLiying/awesome-speculative-decoding — Awesome Speculative Decoding
[23] https://arxiv.org/html/2504.03664v2 — PIPO pipelined offloading
[24] https://ieeexplore.ieee.org/document/10949701 — TightLLM IEEE TC 2025
[25] https://dl.acm.org/doi/10.5555/3843968.3843971 — DirectKV OSDI 2025
[26] https://arxiv.org/html/2502.08182 — Memory Offloading with Latency SLO Guarantees
[27] https://aclanthology.org/2024.findings-acl.456.pdf — Comprehensive Survey of Speculative Decoding
[28] https://pytorch.org/blog/deepnvme-affordable-i-o-scaling-for-deep-learning-applications — DeepNVMe blog
[29] https://arxiv.org/abs/2303.06865 — FlexGen arXiv
