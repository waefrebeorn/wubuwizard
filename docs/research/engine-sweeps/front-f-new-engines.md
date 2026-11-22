# Front F — new engine families not in the current catalog

Scope: sweep of families explicitly absent from `docs/ENGINE-SOURCE-INDEX.md` and
`docs/ENGINE-GAP-REGISTER.md` (gaps EG-01..EG-46 are **not** re-reported here).
Standing target: **everyman inference — 6 GB VRAM / 16 GB RAM / SSD, model- and
backend-agnostic**. Evidence gathered 2026-11-22 via 10 web searches and 17
primary-source extractions; every claim below is tied to a source in **Sources**.

**Rule (unchanged): these are read-only references. Read the upstream design,
reimplement from scratch in C11, never vendor, never link third-party source.**
Nothing here is a port-completion claim; no gate is marked DONE by this report.
Reported speedups are reasons to write a gate, never proof wubuwizard has a feature.

## Summary of what is genuinely new

| # | Engine / mechanism | T1 fit | One-line reason it is not in the catalog |
|---|---|---|---|
| 1 | TensorFold (Zig/CUDA/Metal) | MED-HIGH | No Zig engine, no adaptive draft caps, no disk-backed prefix *snapshots* with exactness gate |
| 2 | NVIDIA Hyperflare (formerly Flare / spark-local-inference) | MED (design), LOW (hardware) | No build-time/serve-time split with a captured-graph "core", no caller-owned memory plan, no prefix-checkpoint pool with shutdown drain |
| 3 | llm-d 1.0 | LOW (cluster) | EG-15/EG-33 are single-process PD; no event-driven fleet KV index, EPP Filter-Score-Pick, or TTFT/TPOT SLO predictor |
| 4 | HiQCache (INT8 host-tier KV) | **HIGH** | EG-18 is BF16 tiers; EG-36 is FP8 KV; no quantized *host* representation reusing an unmodified CUDA mover |
| 5 | NanoFlow (nano-batching) | MED | EG-09 defers results; EG-32 is capacity/micro-batch. Neither splits at **operation** granularity nor partitions execution units |
| 6 | DSpark | **HIGH** | EG-24 is EAGLE-1, EG-25 is token-level families; no semi-autoregressive block drafter with Markov + confidence heads and scheduled verify budget |
| 7 | P-EAGLE / DFlash parallel drafters | MED | Catalog stops at EAGLE-3/Medusa; no deep-verifier-state **block** drafting family or Speculators training stack |
| 8 | Kimi DSA sparse indexer (K3.1 Pro) | MED | EG-44 is a drop-in tile indexer; this is FP8-key/token + 32 index heads + top-2048 exact MLA, with a packed 3-state KV pool and DCP exact top-k merge |
| 9 | QSA block-pooled indexer (Qwen3.8-Flash-Next) | LOW-MED | Per-token DSA is not block-pooled; no 128-d indexer key selecting top-512 blocks-of-4 in the catalog |
| 10 | MISA / MISA† (mixture of indexer heads) | MED | New routing axis: choose a query-dependent subset of indexer *heads* instead of tokens/blocks |
| 11 | HyScale selection replay | LOW (training-side) | Catalog has no training-time replay for learned sparse attention |
| 12 | DiffusionGemma (dLLM in vLLM) | CONDITIONAL | Catalog is autoregressive-only; no bidirectional canvas, entropy-bound denoising, or 0-bonus-token reuse of the spec path |
| 13 | I-DLM / introspective strided decoding | CONDITIONAL | No stationary-batch scheduler, mask-position-gated residual LoRA, or fused introspective verify |
| 14 | DLM-as-drafter family (DiffuSpec / Spiffy / PSD) | CONDITIONAL | Diffusion block proposals verified by an AR target are absent from EG-25 |
| 15 | ReplaySSM (cache SSM inputs, not state) | **HIGH if hybrid models are added** | No recurrent-state object at all in the catalog; EG-19 is attention KV restore-vs-recompute |
| 16 | Hymba / Zebra hybrid SSM+attention | CONDITIONAL | No meta tokens, cross-layer KV sharing, or 3–8:1 constant-state:attention layout policy |
| 17 | Cross-architecture KV-cache translation | MED | EG-19 compares restore vs recompute; this maps MLA latents, recurrent states **and** indexer keys across models |
| 18 | titan-engine | MED-HIGH | No tiered experts + prefill expert streaming + hybrid prefix cache + nvcc-free Rust→PTX kernels |
| 19 | Strata (Rust + HIP) | HIGH | No zero-allocation host, prefill launch-count amortization, or 4-way split-KV decode reduction reference |
| 20 | Magnitude (Seismic per-device kernel compile) | MED | No checked DSL → structured IR → per-device compile/tune with a 60 s first-load budget |

Non-counted observations: **NVIDIA NIM** is a container wrapper around vLLM/SGLang/
TRT-LLM (no new engine mechanism — do not port); **Mooncake Store 2026 connectors**
(TRT-LLM KVCMv2 native connector, TokenSpeed L3, AMD ATOM, Dynamo event-driven
shared-cache index, `page_head` TP-LCM head sharding) refine EG-18/EG-38 rather than
adding a new family.

---

## 1. TensorFold

- **URL**: https://github.com/ashhart/TensorFold ; deployment companion
  https://github.com/ciprianveg/gb10-tensorfold
- **Date/version**: active 2026-11-22, release 1.3.4; CUDA engines added in 0.3.0
  (2026-09-26), Apache-2.0 from 0.6.0 (2026-09-30; earlier releases MIT).
- **Language/license**: Zig 67.5% / TypeScript 18.3% / Metal 3.7% / CUDA 3.2%;
  Apache-2.0 (with `LICENSES/MIT.txt` for pre-0.6.0 code).
- **Concrete mechanisms** (README flags and file names are the exact handles):
  - `--drafter`, `--mtp-drafts N`, `--mtp-confidence P`: family-specific MTP/DSpark
    draft cap; confidence stops a draft chain before a later draft below P
    (default 0.30). `gb10-tensorfold` calls this **adaptive frows**: per-round draft
    caps that follow the live stream count, with findings/patches under
    `glm-5.3/v1/findings/` and `glm-5.3/v1/patches/` (`adaptive frows`, `dspark capture`).
  - `--kv-dtype bf16|int8|int4`: Flash Next stores K **and** V with one fp16 scale
    per 32 values; the 8-bit path is reported at 53% of bf16 cache memory, ~2.4%
    decode cost. This is decode-KV, distinct from EG-43's quantized *draft* KV.
  - `--decode-share F`: while prompts prefill, running replies keep moving for F of
    each chunk's time; new prompts start at the next chunk (fewest tokens left first).
  - `--prompt-cache-gib N`, `--checkpoint-slots N`, `--spill-gib N`,
    `--snapshot-dir DIR`, `--max-snapshots N`: retained conversation prefixes,
    disk spill of evicted prefixes instead of re-prefill, and persistent
    system-block snapshots loaded at start.
  - **AOT kernel packs** ("AOT-captured kernel packs, per shape/window") and a
    3-stage RAM-guarded build (fatbins → native+lib → host tests), so no JIT at boot.
  - **Exactness discipline**: "A draft is accepted only when it equals the token
    the same engine would produce serially"; verify kernels keep each row's
    arithmetic independent; sampling keyed on prompt/seed, absolute position and
    token ID. `tools/` contains parity gates (e.g. the ASR teacher-forced tie rule
    allowing only a top-2 swap within 0.25 logit).
- **Evidence**: on 8× GB10 (SM121) GLM-5.3 EXL3 4.04bpw, 43.9 prose tok/s and
  55.9 code tok/s at c1 vs vLLM's 26/40 (1.6×), 1M-token context, needle found at
  845,962 tokens; DCP1/P4 600K total KV, DCP2 1.2M.
- **T1 fit**: MED-HIGH for the *mechanisms* (disk prefix spill, KV8, adaptive
  drafts and exactness gates all target a small single machine). The engine's
  TP8/GB10 benchmarking setup is out of scope; read-only reference, C11 rewrite.
- **Why new**: catalog has no Zig engine, no adaptive per-round draft caps, no
  disk prefix snapshots with a serial-equivalence gate, and no AOT kernel packs.

## 2. NVIDIA Hyperflare (published as Flare; repo was `spark-local-inference`)

- **URL**: https://github.com/NVIDIA/hyperflare-inference (the former
  `flare-local-inference` / `spark-local-inference` redirects here; binary still
  named `flare`). Architecture walkthrough:
  https://veduis.com/blog/nvidia-flare-architecture-local-inference/ (2026-10-21,
  describing the 2026-10-20 release).
- **Date**: repo created 2026-10-20, renamed 2026-11-10, v0.1.0 2026-11-11;
  active 2026-11-22 with 1,338 commits.
- **Language/license**: Rust 49.6% / Python 49%; Apache-2.0, DCO sign-off.
  Research prototype; README explicitly says it is not a vLLM replacement.
- **Concrete mechanisms (exact paths/symbols from commit history and README)**:
  - **Ignition / Helios split**: Python build-time loads weights, tunes kernels,
    `torch.compile`s forwards and captures CUDA graphs into a **runtime core**;
    pure-Rust `helios` serves that core with zero Python. `docs/core-design.md`,
    `docs/model-preparation.md`, `ignition/`, `helios/`, `flare/`, `flare-cli/`.
    "Make memory and state explicit: plan buffers and execution capacities ahead
    of time. Treat conversation state—including recurrent state alongside KV
    caches—as a whole."
  - **Compressed Scale Format (CSF)** for losslessly compressed FP4 scales;
    FP8 E4M3 KV cache default in every recipe; caller-owned page pool and
    bounded request scheduler at serve time.
  - **Selene prefix-checkpoint pool** (`selene/`, `selene-manager/`, commit
    `8fbd2b7`): single-process prefix checkpoint capture with a CPU-readable
    buffer, budgeted host bytes and device mappings, checksummed puts, bounded
    retirement, and a configured **shutdown DRAM drain**.
  - **Heliopause `Root::adopt_companion` / `container::link_companion`** (commit
    `5d395de`): hard-link a received cache payload into core staging when it is
    byte-for-byte the canonical wrapper, instead of copying + fsync + rehash.
  - **Checkpoint candidate pricing** in `helios/src` (commit `6b58639`): resolve
    catalog costs once per selection, share an interval between accepted cuts,
    and aggregate whole cheapest-capacity spans.
  - **`DualBf16TCGemv`** in `ignition/recipes/kimi3/decoder.py` (commit `135ab51`):
    fuses BF16 `latent_down_rows` and FP32 router into one launch;
    `compact_head` row compaction; `_draft_stage_kv_windows` with `POS_BLOCK=16`
    in `kimi3/dspark.py`, 128-thread/64-block LSE combine for DCP4.
  - **Hash-guarded insertion-only chat-template patches** (commit `28f0833`):
    upstream Jinja templates are not vendored; nine patches guarded by exact
    source/result hashes, canonical templates fetched into the hub cache at
    preparation time.
  - Windows WinUI desktop backed by the shared Rust library; execution ABI 6 /
    prepared model contract 3 gates replay against the prepared architecture.
- **Evidence**: pool-only prefix restores cut median TTFT 21.1 s → 7.6 s on
  GLM-5.3-Flash TP4; checkpoint pricing 31 ms/1.92M lookups → 0.83 ms/48 lookups
  (4×150k-token prompts, 999 boundaries, 8192 chunk, 11 capacities);
  `DualBf16TCGemv` removed 572 captured kernels and freed 29 KV pages, with
  greedy C1 +10.9%, C4 +6.4%, C8 +1.2% on Kimi K3.1 Pro TP16/DCP8.
- **T1 fit**: LOW on hardware (SM120/SM121, 24–128 GB, 96–180 GB prepared cores).
  MEDIUM as a design reference: caller-owned memory plan, prefix-checkpoint pool
  with drain, hash-guarded templates and kernel-count reduction all map onto a
  6 GB/16 GB machine. Reimplement the *ideas* in C11; do not copy the core format.
- **Why new**: EG-16 is vLLM sleep/wake tagging, EG-14 is Jinja rendering from
  GGUF; neither has build/serve separation, a prepared-graph core, CSF scales,
  a prefix-checkpoint pool, or hash-guarded template patches.

## 3. llm-d 1.0 (cluster-level, KV-aware routing)

- **URL**: https://llm-d.ai/docs/architecture ;
  https://docs.vllm.ai/en/latest/deployment/integrations/llm-d/ ;
  https://cloud.google.com/blog/topics/ai-infrastructure/llm-d-1-0-an-open-source-stack-for-distributed-inference/
- **Date**: 1.0 released 2026-10-20; docs updated 2026-10-30.
- **Language/license**: Go/Python/Kubernetes stack; Apache-2.0 (verify per repo).
- **Concrete mechanisms** (exact components): `InferencePool` custom resource,
  L7 proxy via Gateway API Inference Extension `ext-proc`, **Endpoint Picker
  (EPP)** with a Filter-Score-Pick pipeline, **Latency Predictor** (online ML
  regression for TTFT/TPOT SLOs), **KV-Cache Indexing** (event-driven tracking of
  cache state across servers from vLLM KV events), **Prefix-Cache Aware Routing**,
  **KV Offloading** to CPU/SSD via `OffloadingConnector`/Mooncake/HiCache, and
  **P2P KV-Cache Sharing** (pull blocks from a peer's CPU tier instead of
  recomputing). `LeaderWorkerSet` and `DisaggregatedSet` for synchronized
  multi-role slices; NixlConnector RDMA for PD; wide expert parallelism.
- **Evidence**: on an agentic coding workload on Ironwood TPU (v7x),
  prefix-aware routing + host-memory offload cut median TTFT by >85% and raised
  throughput 2.4× vs a plain Kubernetes Service (Oct 2026).
- **T1 fit**: LOW — this is Kubernetes fleet coordination, not a single-box
  engine. The transferable idea is *event-driven KV residency tracking feeding a
  routing decision*, which could inform a local tier-advice policy (EG-18/EG-37).
- **Why new**: EG-10 is a single shared admission validator, EG-15/EG-33 are
  in-tree single-process PD; no catalog entry has an event-driven global KV index,
  EPP Filter-Score-Pick, or an SLO latency predictor.

## 4. HiQCache — quantized host-tier KV for HiCache

- **URL**: https://github.com/awesome-pro/hiqcache ;
  https://abhinandan.one/artifacts/building-a-quantized-hicache-for-sglang ;
  upstream PR https://github.com/sgl-project/sglang/pull/41365
- **Date**: published 2026-10-13; commits through 2026-11-10.
- **Language/license**: Python 95.2% / Shell; **Apache-2.0**.
- **Concrete mechanism**: active KV stays BF16 in GPU L1; KV evicted to CPU L2 is
  stored as INT8 in a fixed **1152-byte encoded record** with padding so the
  existing HiCache CUDA mover and page layout are unchanged. On an L2 hit the row
  is restored to BF16 before attention. No change to caching policy, radix tree,
  transfer engine or scheduler. Encoded format targets Qwen3-8B, TP=1, 8 local
  KV heads, head_dim=128, page_size=1; explicitly does not target TP>1 or
  SSM/Mamba. Restored values are not bit-identical to the original BF16 KV.
- **Evidence**: 147,456 → 82,944 bytes/token (−43.75%), 54,254 → 96,451 tokens
  (1.78×) at a fixed host budget; cache-hit rate ~57% → ~73% (+16.6 pp); ~168K
  vs ~125–127K tokens restored from L2; end-to-end GPU→CPU→GPU compressed
  restores validated. Latency improved in 3 of 4 paired runs, near parity in one;
  replay measured 57.57% / 71.97% against published 56.1–57.6% / 72.7–73.4%.
- **T1 fit**: **HIGH**. The 16 GB RAM budget is exactly where a quantized L2
  KV tier buys capacity without touching attention kernels — this is the single
  most directly portable idea in this sweep (C11 codec + aligned record + async
  restore gate).
- **Why new**: EG-18/EG-38 cover HiCache tiers and logical pages; EG-36 covers
  FP8 KV; EG-43 covers quantized *draft* KV. None covers an INT8 *host* tier with
  an alignment-preserving record that reuses an unmodified transfer kernel.

## 5. NanoFlow — nano-batching / intra-device parallelism

- **URL**: https://arxiv.org/html/2408.12757 ; https://github.com/efeslab/Nanoflow
- **Date**: arXiv:2408.12757v2, 2025-05-25 (original 2024; outside the 2026
  window but absent from the catalog).
- **License**: arXiv CC BY 4.0; repo license not verified in this sweep.
- **Concrete mechanism**: argument that end-to-end serving is *compute*-bound for
  common workloads; splits inputs at **operation** granularity into nano-batches
  and duplicates operations so a compute-bound GEMM, a memory-bound attention and
  a network-bound all-reduce can overlap on one device. A device-level pipeline
  with **execution-unit scheduling** partitions functional units; a search over
  the number, size, ordering and GPU resource allocation of nano-batches
  (topological sort + greedy search) accounts for interference. Survey detail:
  operations classified by bottleneck, sizes adjusted dynamically per resource.
- **Evidence**: 1.91× throughput over state-of-the-art serving systems; 50–72%
  of optimal (68.5% in README) on LLaMA-2-70B, Mixtral 8×7B, LLaMA-3-8B.
- **T1 fit**: MEDIUM — intra-device overlap fits a single 6 GB GPU, but needs
  multi-stream/split-kernel support and a cost model; not a first gate.
- **Why new**: EG-09 defers result drains by one step; EG-32 is a capacity and
  micro-batch scheduler. Neither splits a single operation into nano-batches nor
  partitions execution units.

## 6. DSpark — confidence-scheduled semi-autoregressive drafting

- **URL**: https://arxiv.org/abs/2607.05147 ;
  https://docs.nvidia.com/nemo/megatron-bridge/nightly/training/dspark-speculative-decoding.html ;
  https://vllm.ai/blog/2026-07-28-speculators-parallel-drafting
- **Date**: paper 2026; Megatron-Bridge RFC is `nightly` (2026-11).
- **Language/license**: paper/training reference in Python (NeMo AutoModel
  `nemo_automodel/components/speculative/dspark/`); Apache-2.0 for NeMo.
- **Concrete mechanism (paper §3 / RFC "Background: the DSpark draft")**:
  - Draft factorizes a block as `P(x|x0) = ∏_k p_k(x_k | x0, x_1..x_{k-1})`.
    `block_size` (gamma) default 7 offline / 5 in production.
  - **Parallel backbone** predicts every block position in one pass, attending
    to frozen target hidden states as context K/V with block "noise" slots as
    queries; target embedding and LM head are shared and frozen (no compressed
    draft vocab, no d2t/t2d remap). Selected decoder layers fused by learned `fc`.
  - **Sequential head** restores intra-block dependency: a **Markov head** (low
    rank, token-conditioned additive logit bias, **rank 256**) or an RNN/GRU-like
    head. Teacher-forced in training, serial at inference — fixes "suffix decay".
  - **Confidence head** `c_k = sigmoid(wᵀ[h_k ; embed(x_{k-1})])`, trained with
    BCE against `c*_k = 1 − 0.5·||p_draft − p_target||_1`, lets the serving engine
    schedule its verify budget per request and per position, with a GPU-measured
    cost table.
  - Loss: `L_ce` (0.1) + **raw probability L1** (0.9, Eq. 10) + `L_conf` (1.0),
    position weights `exp(-k/gamma)`.
- **Evidence**: Qwen3-8B at T=1 lifts acceptance length 14–23% (avg +18%) over
  DFlash across nine benchmarks (paper Table 1); DeepSeek-V4 reports +60–85%
  generation speed over its MTP-1 production baseline; 95.0 → 124.9 tok/s
  (1.31×) on Qwen3-8B in a third-party reproduction; energy paper reports 54–60%
  J/token savings at B=1.
- **T1 fit**: **HIGH** — a confidence head that trims weak drafts saves verifier
  FLOPs on a 6 GB card; the rank-256 Markov bias is cheap in C11. Requires a
  trained drafter and a calibration gate; no accuracy claim without one.
- **Why new**: EG-24 is EAGLE-1 direct-prediction, EG-25 is token-level spec
  families; EG-42 (TVP) trains with a TV objective, whereas DSpark uses a raw-L1
  acceptance proxy plus an analytical acceptance label and per-position *scheduling*.
  No block-parallel + Markov-correction drafter is catalogued.

## 7. P-EAGLE and DFlash — parallel drafting family

- **URL**: https://vllm.ai/blog/2026-07-28-speculators-parallel-drafting ;
  P-EAGLE https://arxiv.org/abs/2602.01469 ; DFlash https://arxiv.org/abs/2602.06036 ;
  https://github.com/vllm-project/speculators
- **Date**: blog 2026-07-28; papers 2026-02.
- **Language/license**: Python speculators + vLLM (Apache-2.0).
- **Concrete mechanism**: all three of P-EAGLE/DFlash/DSpark condition on the
  verifier's hidden states but predict a whole candidate block in **one** forward
  pass, decoupling drafting latency from speculation length. This permits larger,
  more expressive drafters and removes per-load tuning of K. Earlier examples are
  Medusa and PARD (`arXiv:2504.18583`); vLLM config uses
  `--speculative-config '{"method":"dflash","num_speculative_tokens":7}'`.
- **Evidence**: blog figures show consistent gains over EAGLE-3 on Qwen3-8B,
  Qwen3-30B and Gemma-4-31B (errata: original plots had a bad environment setup,
  relative conclusions unchanged). Rejection sampling preserves the verifier
  distribution exactly.
- **T1 fit**: MEDIUM — same benefits and constraints as DSpark; P-EAGLE/DFlash
  are separate drafter architectures, so one mechanism/gate each if pursued.
- **Why new**: catalog stops at EAGLE-3/Medusa and the survey-derived families;
  no deep-verifier-state *block* drafting family or training stack.

## 8. Kimi DSA sparse indexer (Kimi K3.1 Pro)

- **URL**: https://vllm.ai/blog/2026-11-04-kimi-k31-pro
- **Date**: 2026-11-04 (day-0 vLLM support).
- **Model/license**: Moonshot AI open-weight model; vLLM Apache-2.0 support.
- **Concrete mechanism**: K3.1 Pro keeps K3's 69 KDA + 24 full-attention layers;
  each MLA layer gains a DSA indexer. Config `dsa_config` fields are the exact
  handles: `index_num_heads = 32`, `index_head_dim = 128`, `index_topk = 2048`,
  `kv_lora_rank = 512`. One **FP8 index key per token, 132 bytes with scale**;
  32 heads score with `ReLU(q_h · k_t)` and weighted sum, keep top-2048, and run
  exact MLA only over those. vLLM **packs MLA latent + indexer keys + KDA state
  into one block-indexed KV pool**, so prefix caching survives the hybrid cache
  and P/D transfers it unchanged. DCP decode merges sparse candidate sets by
  exchanging *scores* (not token IDs), with a shared threshold and consistent
  tie handling: 16 → 8 KiB per query per GPU at 2048 candidates. TPU path splits
  work between TensorCore (FP8 scoring) and SparseCore (top-k + gather).
- **Evidence**: long-context decode reads ~23% of dense MLA bytes per layer;
  exact top-k DCP merge cuts traffic ~10× and speeds the attention module
  1.3–1.5× at DCP8; token-sharded prefill up to 1.69× TTFT at 340K on 8× GB300;
  fused CuTe decode 1.2–1.6× (1.9× under DCP8). FP8 latent+index 16,992 B/token
  per GPU vs 13,824 for K3; NVFP4+MXFP4 10,080 B, 1.7× more tokens on 8× GB300.
- **T1 fit**: MEDIUM — sparse indexing helps long context, but this is a
  model-specific 2.8T MoE. The transferable parts are the *packed heterogeneous
  state pool* and the score-only DCP merge (score-only merge is irrelevant on
  one GPU).
- **Why new**: EG-44 is a post-training NVIDIA tile indexer (G=QH/KH, ~200-row
  tile); this is a trained per-token FP8 key with 32 heads and exact MLA over
  top-2048, plus a 3-state packed pool. Different architecture and system design.

## 9. QSA block-pooled sparse indexer (Qwen3.8-Flash-Next)

- **URL**: https://arxiv.org/html/2611.12062v1 (§1 Table 1, §10.1, Fig. 25/27) ;
  https://arxiv.org/html/2610.32057 (CSA/QSA implementations)
- **Date**: 2026-11-09 / 2026-11-04.
- **Concrete mechanism**: Qwen3.8-Flash-Next is 36 **GDN** + 12 **QSA** layers.
  QSA is gated attention plus a **block-pooled indexer**: a 128-d indexer key per
  token, keys pooled over blocks of 4 tokens, selecting the **top-512 blocks of 4
  (≤2051 tokens)**. This differs from DSA's per-token top-2048 and from
  DeepSeek-V4 CSA's compressed KV stream. HyScale implements CSA/QSA in
  Megatron-LM and adds *selection replay* (§ on learned sparse attention).
- **Evidence**: HyScale runs CSA 2.01× and QSA 1.57× faster than Megatron-LM at
  1M tokens on 32 H800; the KV-translation paper rehearses a QSA indexer transfer
  and reports target-like needle recall from *predicted* indexer keys (one seed).
- **T1 fit**: LOW-MED — only useful if a QSA model is in scope; the pooled-block
  indexer idea could inform a cheaper local indexer (accuracy gate mandatory).
- **Why new**: no per-token/block indexer family, no GDN+QSA hybrid, in catalog.

## 10. MISA / MISA† — mixture of indexer heads

- **URL**: https://arxiv.org/html/2605.07363v1
- **Date**: 2026-05-08; CC BY 4.0; code https://github.com/MuLabPKU/TransArch
- **Concrete mechanism (§3)**: treat the DSA indexer's `H^I` heads as a
  mixture-of-experts pool. A lightweight router uses cheap **block-pooled
  indexing keys** `k̃_b^I` (Eq. 4) to choose a query-dependent subset of
  `h ≪ H^I` active heads; only those run heavy token-level scoring. Cost drops
  from `O(H^I L)` to `O(hL + H^I M)` with `M = ⌈L/B⌉ ≪ L`. MISA† keeps an
  enlarged candidate set from the routed pass and re-ranks with the original
  indexer. Drop-in: same top-k set size fed to the same sparse MLA operator.
- **Evidence**: with h=8 and **no additional training**, matches dense DSA on
  LongBench across DeepSeek-V3.2 and GLM-5 with 8× and 4× fewer indexer heads;
  >92% of DSA-selected tokens recovered per layer; green Needle-in-a-Haystack to
  128K; TileLang kernel ~3.82× faster than the original DSA indexer on H200.
- **T1 fit**: MED — post-training, head-axis routing is cheap to emulate and
  composes with EG-44; needs an accuracy gate before any speed claim.
- **Why new**: EG-44 prunes attention blocks; MISA routes *indexer heads*, a
  distinct axis from token/block selection.

## 11. HyScale selection replay (training-side)

- **URL**: https://arxiv.org/html/2610.32057
- **Date**: 2026-11-04 (v3); cs.AI.
- **Concrete mechanism**: "selection replay" applies to any learned sparse
  attention that selects by top-k, whether entries are tokens (DSA) or blocks
  (CSA/QSA). Implements DeepSeek-V4-Flash CSA and Qwen3.8-Flash-Next QSA layers
  in Megatron-LM from their papers.
- **Evidence**: 2.01× CSA and 1.57× QSA at 1M tokens on 32 H800 vs Megatron-LM.
- **T1 fit**: LOW — training infrastructure, not an inference engine; recorded
  so a later sparse-indexer gate does not conflate training replay with decode.
- **Why new**: no training-side mechanism in the catalog.

## 12. DiffusionGemma — dLLM natively in vLLM

- **URL**: https://vllm.ai/blog/2026-06-10-diffusion-gemma
- **Date**: 2026-06-10.
- **Implementation/license**: vLLM (Apache-2.0); Google DeepMind 26B dLLM on a
  Gemma4 backbone; quantized checkpoints on RedHatAI hub.
- **Concrete mechanism**: vLLM **model runner v2 `ModelState`** abstraction owns
  custom input preparation and per-request state hooks. The same Gemma4 weights
  run in two modes: **encoder** (causal attention, writes KV; runs twice per
  block — prompt prefill and block commit) and **decoder** (bidirectional
  attention, KV-read only, denoises a 256-token canvas). The spec-decoding data
  path is reused: the canvas is a set of draft tokens accepted or rejected as a
  whole; support for **sampling 0 tokens** was added and is `ModelState`-controlled.
  Scheduler, model runner and backbone are unchanged. **Entropy-bound denoising**
  accepts positions from most to least confident until accumulated entropy
  exceeds a budget; convergence requires stable argmax plus low mean entropy or
  a step cap. **Self-conditioning** feeds the previous step's full softmax as a
  probability-weighted embedding average through a gated MLP (zeroed on encoder
  passes), so renoised positions keep information.
- **Evidence**: matches Hugging Face reference accuracy; on H200 FP8 reaches
  1,288 gen tok/s at batch 1 (~6× AR baseline, ~3× MTP); H100 1,008 tok/s
  (~5× / ~2.6×). Evaluated on AIME 2025, GPQA Diamond, GSM8k.
- **T1 fit**: CONDITIONAL — attractive for low-batch latency on limited VRAM,
  but requires a dLLM checkpoint and bidirectional attention support; a separate
  decision from the AR engine.
- **Why new**: catalog has only autoregressive decode paths; no canvas,
  bidirectional decode, entropy-bound acceptance or 0-token sampling.

## 13. I-DLM / introspective strided decoding (ISD)

- **URL**: https://arxiv.org/html/2604.11035 ;
  https://github.com/Introspective-Diffusion/I-DLM
- **Date**: 2026-04-13; cs.AI; arXiv perpetual license.
- **Concrete mechanism (§ systems design, Appendix H Table 12)**: ISD alternates
  an **NP step** (append N masks, produce 1 free token via logit shift + N−1
  speculative proposals, finalize nothing) with **P steps** (introspect previous
  proposals and propose new ones) in the same forward pass. Strict causal
  attention lets it drop into SGLang as an extension and inherit paged KV,
  continuous batching and TP unchanged (`disable_radix_cache=true` required for
  DLLM KV trim; extend forward of all 2N−1 tokens captured as one CUDA graph per
  batch size with in-place attention-metadata writes). Inference customizations:
  **stationary-batch scheduler** (`NO_DECODE_LOOP`), **argmax proposals**
  (`NO_ARGMAX_PROPOSALS`), **fused Triton verify kernel** (`NO_FUSED_VERIFY`,
  replaces 7+ launches), **deferred `stream_output`** (`NO_DEFERRED_STREAM`),
  paged-only attention vs cascade (`DLLM_USE_RAGGED`), and **gated residual LoRA
  applied only at mask positions** with base weights doing verification
  (near-lossless ↔ strictly lossless continuum).
- **Evidence**: first DLM to match same-scale AR quality across 15 benchmarks;
  69.6 AIME-24, 45.7 LiveCodeBench-v6; ~3× throughput over prior DLMs, 3.1×
  over LLaDA-2.1-mini at matched MATH-500 quality; peak 925 tok/s on B200 N=8,
  685 on H100 N=8, 341 on H100 N=4.
- **T1 fit**: CONDITIONAL — the fused verify and deferred stream ideas port
  independently; the model/stride training does not.
- **Why new**: no DLM engine, stationary-batch scheduler or mask-gated LoRA in
  the catalog.

## 14. DLM-as-drafter family (DiffuSpec / Spiffy / PSD)

- **URL**: https://aclanthology.org/2026.findings-acl.1048.pdf (DiffuSpec) ;
  https://nips.cc/virtual/2026/loc/sydney/poster/150134 (PSD) ;
  https://en.papernotes.org/ICML2026/llm_efficiency/structuring_the_future_diffusion_llm_speculative_decoding_via_calibrated_draft_g/ (Spiffy)
- **Date**: 2026 (Findings ACL 2026, NeurIPS 2026, ICML 2026).
- **Concrete mechanisms**:
  - **DiffuSpec**: a diffusion LM is a *parallel* drafter for an ordinary AR
    verifier. Four-stage serving-compatible pipeline: draft a length-`k_t` block
    with per-position candidates → **Causal-consistency Path Search (CPS)** on a
    pruned lattice to extract a left-to-right, verifier-aligned path → parallel
    verification using the causal path distribution in the acceptance ratio →
    **Adaptive Draft-Length (ADL)** controller from online acceptance feedback.
    Only the drafter side changes; no target architecture change.
  - **Spiffy**: auto-speculation from the target's own distribution; a **Directed
    Draft Graph** of multi-step denoising states, offline-calibrated structure,
    dynamic pruning to a budget, parallel verification to skip steps.
  - **PSD**: training-free; one forward pass supplies confidence scores for both
    adaptive position unmasking and multi-depth drafts, then batched verification
    with hierarchical acceptance keeps the deepest consistent draft.
- **Evidence**: DiffuSpec is ACL 2026 Findings; PSD reports up to **5.5× tokens
  per forward pass** at accuracy comparable to greedy on three dLLMs.
- **T1 fit**: CONDITIONAL — matters only if dLLM drafting is in scope; ADL (acceptance
  feedback controlling draft length) is a cheap general idea worth a gate.
- **Why new**: EG-25 covers AR spec families; none uses a diffusion proposer,
  causal path search over a candidate lattice, or denoising-step speculation.

## 15. ReplaySSM — cache SSM inputs, not state

- **URL**: https://developer.nvidia.com/blog/replayssm-rethinking-state-storage-for-faster-hybrid-model-decoding/ ;
  paper https://arxiv.org/abs/2610.13778 ; https://github.com/Johnny-Liou/ReplaySSM
- **Date**: 2026-10-12.
- **Language/license**: research code + integration into TensorRT-LLM (PR #14203),
  vLLM and FlashInfer; production in GLM-5.3-Flash and Kimi-K3.
- **Concrete mechanism**: represent SSM state as a **checkpoint plus a buffer of
  recent inputs**. Outputs are computed every step from checkpoint+buffer; the
  checkpoint is written only when the buffer fills. This separates output
  computation from state updates. It also makes speculative rollback a matter of
  appending draft inputs to the buffer (no state snapshot), and limits quantized
  state error by keeping a **high-precision checkpoint for recurrence and a
  low-precision copy for reads**. Mamba-2 recurrence given in the post:
  `S_t = α_t S_{t-1} + v_t k_tᵀ`, `o_t = S_t q_t`.
- **Evidence**: ~half the state memory traffic; 1.46–1.80× faster SSM decode
  kernels at batch 256 on B300; 1.85–2.10× higher speculative throughput on
  120B-class models; three of four models within ~2 pp of FP32 on GPQA-Diamond.
- **T1 fit**: **HIGH if hybrid models enter scope** — constant-size state is
  exactly what a 6 GB/16 GB box wants, and the rollback story composes with
  EG-24/EG-25. It needs a new recurrent-state object in C11 (no catalog object
  for it today).
- **Why new**: EG-19 is attention-KV restore-vs-recompute; no catalog entry has
  an SSM state object, checkpoint+buffer replay, or state rollback for drafting.

## 16. Hymba / Zebra — hybrid SSM+attention layout policy

- **URL**: https://gunjandhanuka.github.io/research-hub/concepts/hybrid-attention-ssm ;
  https://rocm.docs.amd.com/projects/primus/en/v26.9/04-technical-guides/hybrid-models/README.html ;
  Hymba arXiv:2411.13676.
- **Date**: Hymba 2024; AMD Primus Zebra v26.9 (2026); concept page 2026.
- **Concrete mechanism**: interleave a few full-attention layers with many
  constant-state layers (Mamba/SSM, KDA, GDN). `hybrid_attention_ratio` controls
  attention-layer fraction (0.25 default = 1 attention per 3 recurrent; 0.0 pure
  recurrent, 1.0 pure MLA). Two structural styles: sequential (whole layers) and
  parallel (both inside one layer) — **Hymba** uses attention heads and SSM heads
  in the same layer with **meta tokens**, **cross-layer KV sharing** and a
  sliding window. Modern examples: Kimi-K3 3 KDA:1 MLA, Qwen3.8-Next 3 GDN:1
  attention; Zebra pairs recurrent layers with MLA + YaRN + LoRA-compressed KV.
- **Evidence**: Hymba reports **11.67× cache reduction and 3.49× throughput**
  at 1.5B; Zebra provides reproducible single-node configs from 300M to 8B.
- **T1 fit**: CONDITIONAL — only helps if the engine loads such checkpoints;
  the 3–8:1 ratio and meta-token sharing are useful design constraints.
- **Why new**: catalog assumes softmax attention with a KV cache; no constant-state
  layer type, ratio policy, meta tokens or cross-layer KV sharing.

## 17. Cross-architecture KV-cache translation (hybrid / MLA / indexers)

- **URL**: https://arxiv.org/html/2611.12062v1
- **Date**: 2026-11-09; cs.LG; CC BY 4.0; Hostinger Research.
- **Concrete mechanism (§3–§10.1, Appendices D–F)**: extend KV-Lingo beyond GQA
  to three new cache objects. **MLA latents** translate by token-wise linear
  maps (closed-form fit + self-distillation). **Recurrent states** are copied
  with a closed-form correction or synthesised by replaying the target's own
  recurrence; GDN corrections are bilinear (`A−I` median effective rank 90/128).
  **Sparse indexer keys** are predicted; Appendix F lists four supply options,
  with "predict" and "dense" rehearsed on the QSA pair. Exact long-range recall
  needs token identity: **decode token ids from the source's layer-0 MLA latent**,
  raising needle recall 0.09 → 0.99 at 2–32K and 0.83–1.00 at 64K.
  Segment-wise per-turn re-translation avoids the naive re-translation collapse.
- **Evidence**: pairs Kimi-VL-A3B ↔ Kimi-Linear-48B and Qwen3.5-27B ↔
  Qwen3.8-27B at 16–48B; hybrid→full-MLA within the paired confidence interval
  of native quality on every metric; GDN same-shape hybrids reach native level
  both ways; TTFT 18.7× and 46.8× shorter than re-reading at 64K. Maps are
  high-rank (median 395/576 for MLA), i.e. not low-rank adapters.
- **T1 fit**: MED — a 16 GB RAM box benefits from switching models without
  re-prefilling; but trained maps are per model pair, so this is a research
  direction, not a near-term gate.
- **Why new**: EG-19 compares restore vs recompute *within* one model; this
  translates cache objects *across* architectures, including recurrent states
  and sparse indexer keys.

## 18. titan-engine — MoE on one too-small consumer GPU

- **URL**: https://github.com/optim-enterprises-bv/titan-engine
- **Date**: initial public release 2026-09-29; active 2026-11-22 (155 commits).
- **Language/license**: Rust 81% / CUDA 7.5% / Metal; fork of **mistral.rs v0.9.4**;
  `NOTICE.md` carries upstream licenses — read-only reference only.
- **Concrete mechanisms**: **tiered experts** (`TITAN_TIERED_CPUS` pinning: doorbell
  thread + miss pool to P-cores), **prefill expert streaming**, MTP heads,
  **hybrid prefix cache**, and `oxide-kernels/` Rust `#[kernel]` functions compiled
  to PTX by cuda-oxide's rustc backend — GGUF dequantisation, Q8_0/K-quant
  matvecs, expert GEMVs, MMQ prefill tiles, flash-decode and a **doorbell**; PTX
  is committed so the build needs no `nvcc` and the binary links only
  `libcuda.so.1`. Indexed MoE keeps expert ids on device; PFS `O_DIRECT` staging
  of mmap'd experts and partial-K grouped MMQ are recent additions.
- **Evidence**: measured 2026-09-29 on one RTX 5080 Laptop (16 GB, 150 W), Core
  Ultra 9 285HX, 30 GB RAM, single user; Qwen3.6-35B-A3B UD-Q4_K_XL and
  Qwen3-Next-80B-A3B Q4_K_M with host experts through mmap. CPU pinning:
  0.150 → 0.110 ms per decode pass in isolation, but 0.071 vs 0.068 ms in-server
  (~40 GB/s expert streaming, ~37 GB/s DRAM ceiling), so pinning defaults off.
  Indexed MoE: 12.5 → 16.9 tok/s on one model; 5324/5350 launches bit-identical.
- **T1 fit**: MED-HIGH for mechanisms (host experts via mmap, O_DIRECT staging,
  doorbell, indexed MoE, no-nvcc build) even though the measured GPU is 16 GB.
  It is a fork — study the design, write C11 from scratch, take no code.
- **Why new**: EG-27 is mistral.rs ISQ; EG-20 is MoE expert traces. Neither has
  tiered expert streaming, a hybrid prefix cache, or Rust→PTX kernels.

## 19. Strata — from-scratch Rust + HIP local engine

- **URL**: https://mihai.ltd/blog/beating-llamacpp-from-scratch-consumer-amd ;
  https://github.com/mihailtd/strata
- **Date**: 2026-10-07/08.
- **Language/license**: safe Rust host + custom AMD HIP compute kernels; license
  not verified in this sweep.
- **Concrete mechanisms**: (1) **prefill launch-count amortization** — 280 vs
  ~3,300 kernel launches removes a 50 ms CPU driver queue stall; (2) **4-way
  parallel split-KV reduction** decode-attention kernel, inspired by llama.cpp's
  split-KV but reimplemented; (3) **zero-allocation** steady-state Rust host and
  explicit exactness gate; (4) an un-flushed 8 KB HTTP socket buffer was a
  600 ms phantom stall, and per-token SSE overhead measured <0.3%.
- **Evidence**: RX 7900 XTX (gfx1100, 24 GB, 960 GB/s) ROCm 7.2, byte-identical
  BF16 weights, 100% GPU offload, independent HTTP daemons: 83.0 vs 71.3
  (llama.cpp) vs 72.0 (Ollama) tok/s; TTFT 48.5 vs 121.0 vs 173.2 ms; gains hold
  0.8B (285.9 vs 210.6) through 9B (49.1 vs 44.6, bandwidth ceiling).
- **T1 fit**: HIGH — kernel-launch count and split-KV reduction are backend-agnostic
  and directly relevant to a 6 GB card; the socket finding is a free correctness
  win. Not a vendoring target.
- **Why new**: catalog has no from-scratch AMD/HIP engine, no zero-allocation
  host design and no launch-count amortization evidence.

## 20. Magnitude — per-device kernel compilation/tuning (Seismic)

- **URL**: https://github.com/magnitudedev/magnitude ;
  https://platform-monkey.com/reviews/magnitude-inference-engine-review ;
  https://aniketkarneai.com/blog/2026-10-11-magnitude-self-optimizing-inference-engine-seismic-compiler/
- **Date**: public alpha from 2026-06, v0.1.0 2026-09-17, launch 2026-09-30;
  repo active 2026-11-22.
- **Language/license**: Rust (8.2k stars), Apache-2.0; YC S25.
- **Concrete mechanism**: **Seismic** — numerical programs authored once in a
  checked DSL, lowered to a structured IR, then jointly selected, instantiated
  and compiled to the backend (Metal reference, CUDA via on-device NVRTC 12.9
  into CUBIN for the exact architecture, Vulkan 1.3 compute, Cranelift CPU).
  Structure is authored, parameters are fitted per device at first load with a
  **60-second budget**, then cached; the "native" path skips candidate families
  and IR and only tunes. KV cache quantized **K8V4**. Five engine generations in
  the repo (`old-inference/` frozen llama.cpp bindings, `inference-v2/` MLX,
  `inference-v3/` TileLang, `inference/` V3+Seismic shipping, `inference-v4/`
  under implementation).
- **Evidence**: project claim up to 2× Ollama / llama.cpp in agent scenarios;
  ~1 minute tuning per model; independent review calls the compiler pipeline
  real engineering but notes a 2× claim that inverted on week-old silicon and
  a root-owned package-manager admission gate — treat performance as unverified.
- **T1 fit**: MED — per-device tuning is attractive for fragmented hardware, but
  a DSL+IR compiler is far larger than the current C11 scope; the K8V4 KV idea
  and tuning-cache layout are the portable parts.
- **Why new**: no checked DSL, structured IR, first-load tuning budget or
  on-device NVRTC compilation in the catalog.

---

## Recommended framing for the gap register (not done here)

If the parent agent promotes any of these, keep one technique per gate and put the
accuracy gate before the speed claim. Highest T1 candidates in this sweep:
HiQCache (quantized host KV tier), Strata (split-KV + launch amortization),
DSpark (confidence-scheduled block drafting), ReplaySSM (if hybrid models are in
scope), and TensorFold's disk prefix snapshots. Hyperflare's caller-owned memory
plan and hash-guarded template patches are strong *design* references but its
hardware/model core format is out of scope. Do not vendor, link or copy any of
the upstream code; reimplement in C11 with gates in both Makefiles.

## Sources

- TensorFold — https://github.com/ashhart/TensorFold (accessed 2026-11-22); license
  and release history read from the file listing and `LICENSE` commit.
- gb10-tensorfold — https://github.com/ciprianveg/gb10-tensorfold (2026-11-22).
- NVIDIA Hyperflare — https://github.com/NVIDIA/hyperflare-inference (2026-11-22);
  README and commit messages for `selene`, `heliopause`, `helios`, `ignition`.
- Flare architecture article — https://veduis.com/blog/nvidia-flare-architecture-local-inference/ (2026-11-22).
- Hyperflare rename/ABI — https://huggingface.co/rickybrent/GLM-5.3-Flash-HYPERFLARE-TP2
  and https://github.com/sxuff/qwen38-flash-next-flare (2026-11-22).
- NVIDIA NIM — https://docs.nvidia.com/nim/large-language-models/latest/about-nim-llm/overview.html (2026-11-22).
- llm-d — https://llm-d.ai/docs/architecture ;
  https://docs.vllm.ai/en/latest/deployment/integrations/llm-d/ ;
  https://cloud.google.com/blog/topics/ai-infrastructure/llm-d-1-0-an-open-source-stack-for-distributed-inference/ (2026-11-22).
- Mooncake / HiCache — https://github.com/kvcache-ai/Mooncake/blob/main/README.md ;
  https://kvcache-ai.github.io/Mooncake/design/hicache-design.html ;
  https://docs.nvidia.com/dynamo/v1.4.1/knowledge-base/modular-components/backends/sg-lang/hi-cache.md (2026-11-22).
- HiQCache — https://github.com/awesome-pro/hiqcache ;
  https://abhinandan.one/artifacts/building-a-quantized-hicache-for-sglang (2026-11-22).
- NanoFlow — https://arxiv.org/html/2408.12757 ; https://github.com/efeslab/Nanoflow (2026-11-22).
- DSpark — https://arxiv.org/abs/2607.05147 ;
  https://docs.nvidia.com/nemo/megatron-bridge/nightly/training/dspark-speculative-decoding.html (2026-11-22).
- Parallel drafting — https://vllm.ai/blog/2026-07-28-speculators-parallel-drafting (2026-11-22).
- Kimi K3.1 Pro DSA — https://vllm.ai/blog/2026-11-04-kimi-k31-pro (2026-11-22).
- MISA — https://arxiv.org/html/2605.07363v1 (2026-11-22).
- HyScale / QSA — https://arxiv.org/html/2610.32057 (2026-11-22).
- DiffusionGemma — https://vllm.ai/blog/2026-06-10-diffusion-gemma (2026-11-22).
- I-DLM — https://arxiv.org/html/2604.11035 (2026-11-22).
- DiffuSpec — https://aclanthology.org/2026.findings-acl.1048.pdf ;
  PSD — https://nips.cc/virtual/2026/loc/sydney/poster/150134 ;
  Spiffy — https://en.papernotes.org/ICML2026/llm_efficiency/structuring_the_future_diffusion_llm_speculative_decoding_via_calibrated_draft_g/ (2026-11-22).
- ReplaySSM — https://developer.nvidia.com/blog/replayssm-rethinking-state-storage-for-faster-hybrid-model-decoding/ ;
  https://arxiv.org/abs/2610.13778 (2026-11-22).
- Hybrid SSM/attention — https://gunjandhanuka.github.io/research-hub/concepts/hybrid-attention-ssm ;
  https://rocm.docs.amd.com/projects/primus/en/v26.9/04-technical-guides/hybrid-models/README.html (2026-11-22).
- KV-cache translation — https://arxiv.org/html/2611.12062v1 (2026-11-22).
- titan-engine — https://github.com/optim-enterprises-bv/titan-engine (2026-11-22).
- Strata — https://mihai.ltd/blog/beating-llamacpp-from-scratch-consumer-amd (2026-11-22).
- Magnitude — https://github.com/magnitudedev/magnitude ;
  https://platform-monkey.com/reviews/magnitude-inference-engine-review (2026-11-22).
- Also observed, thin primary evidence (not counted): shimmy (pure-Rust WebGPU,
  https://everylocalai.com/tool/shimmy), Paddock (Rust/C++ with own CUDA kernels,
  https://ainews.leverageai.com.au/case/paddock-native-llm-runtime),
  local engine release tracker
  (https://localmodelwatch.tsuchitsuchi.com/en/engines-en/).
