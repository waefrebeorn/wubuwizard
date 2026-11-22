# Front G — 2026 Q3 (Aug–Oct) inference techniques missed by the November sweep

**Scope.** Single-box target: 6 GB VRAM / 16 GB RAM / SSD, reimplemented in wubuwizard's own C11. Window mined: 2026-08-01 → 2026-10-31, plus resolution of the five supplied starting leads (FlashMoE, SPICE, CoX-MoE, QuantSpec, KIVI/StreamingLLM/YOCO lineages) even where a lead predates the window.

**Method.** `web_search` for Q3 2026 arXiv IDs and primary engineering sources, then `web_extract` of arXiv HTML (11 searches, 11 primary extractions). Search snippets were used only to locate papers; every number below comes from an extracted paper. Each mechanism was diffed against `docs/ENGINE-GAP-REGISTER.md` (EG-01…EG-46) and `docs/research/engine-sweeps/front-d-recent-2026-11.md`. **No entry recommends vendoring.** Each is written as a reimplement-in-C11 spec: read the upstream *description*, write an independent C11 module and a `test_egNN_*` gate, never link or copy third-party source.

**Diff convention.** Candidate labels G1…Gn are *provisional* — no EG id is assigned here; that is the register owner's call. "Not a known gap" means: the closest EG row is named and the mechanism is shown to differ.

---

## Summary table

| # | Paper (arXiv) | Date | One-line mechanism | T1 (6 GB) fit |
|---|---|---|---|---|
| G1 | BitNest 2610.02800 | 06 Oct 2026 | W4 base nested *inside* W8 target; single physical weight+KV representation | **Highest** — 7.7 GiB peak on 16 GB part, 8 GB device tested |
| G2 | Edge0 2609.18063 | 16 Sep 2026 | Prerouter prediction *is* the routing; SSD int4 experts, unmerged recovery LoRA | **Highest** — 35B in 3 GiB, 8B in 1.5 GiB |
| G3 | SSD-LLaMA 2609.18110 | 16 Sep 2026 | Per-expert concurrent SSD reads + async H2D per completion + in-VRAM rANS; 3-tier retention decoupled from execution | High — consumer PC, ≤32 GB RAM |
| G4 | LeanStream 2609.03079 | 02 Sep 2026 | Speculate-*and-refine*: re-prioritise compute/load/retention as partial GPU results arrive | High — mobile + embedded tested |
| G5 | ReadKV 2610.11245 | 08 Oct 2026 | Progressive KV code: store W bits, fetch R<W query-adaptive prefixes | High — 8K batch-1, 39% lower latency |
| G6 | SPICE 2608.21240 | 07 Sep 2026 (v2) | Confidence-adaptive expert lookahead + LoRE surrogate substitution + weight-vs-compute miss choice | High — 2.6–3.12× TPOT on offload |
| G7 | SeqMoE 2609.12978 | 11 Sep 2026 | Expert activation as seq2seq; prefetch as job sequencing with deadlines; probabilistic Belady; sync-free graph capture | High — 96.97% hit @45% residency |
| G8 | FreeToken 2608.16157 | 17 Aug 2026 | Continuous bandwidth-adaptive mapping across layout/residency/CPU-GPU/agent state/memory | High but broad — 35B on 8 GB laptop |
| G9 | Memory-Sovereign 2608.23805 | 24 Aug 2026 | Falsifiable output-exactness certificate; split ownership/residency/addresses/scheduling | Methodology — directly strengthens "no gate, no ship" |
| G10 | SpecQuant 2609.21704 | 18 Sep 2026 | Multi-parent INT4/FP8/FP16 from shared base + complexity routing; training-free | Medium-high — on-device, no retraining |
| G11 | MemSpec 2608.10362 | 11 Aug 2026 | Memory-aware runtime draft scheduling under a budget | Medium-high — Jetson Orin Nano |
| G12 | PELM 2609.09662 | 09 Sep 2026 | Variable verification depth as a power/thermal knob with DVFS | Medium — split: depth policy in scope, DVFS out |
| G13 | ThreeH 2609.07282 | 07 Sep 2026 | Stability/access/utility horizon separation; budget-matched eval contract | Gate-design correction for EG-04 |
| G14 | SPADE 2608.13076 | 13 Aug 2026 | Edge draft / cloud verifier; only rejections trigger correction | Tier-gated T2/T3 (multi-node) |
| G15 | HySparse2 2609.26368 | 22 Sep 2026 | Two-level KV sharing (KV Bridging + Reuse); prefill exits after self-decoder | T2 — needs a trained architecture, not drop-in |
| G16 | LLM Inference in a Flash 2609.16161 | 14 Sep 2026 | Integer-only compute-in-flash + sparse dictionary-coded KV | T3 hardware-gated; KV codec extractable |

---

## G1 — BitNest: bit-nested speculative decoding

- **URL / date:** https://arxiv.org/html/2610.02800 — arXiv:2610.02800v2 [cs.AI], **06 Oct 2026** (v1 02 Oct).
- **Mechanism (§1 contributions; §2.1; App. A.6):** Build a strong W4 base *first*, then quantise only the residual error into W8 refinement bits, so the higher-precision target **contains** the draft as a fixed subset. Dual-plane storage: drafting reads only the base plane, verification reads both; the nesting principle is extended to a bit-nested KV cache. This is base-first, not independent quantisation of the target (which would break nesting).
- **Numbers:** 95.2% mean acceptance; 1.48–1.61× end-to-end vs FP16 AR. On Jetson Orin NX16: LLaMA-2-7B-32K peak PyTorch allocation **13.9 → 7.7 GiB**, max context 2K (FP16) / 6K (W8A8) → **10K**; 9.0–9.3 vs 7.8 tok/s. On NX8: Llama-3.2-3B **15.5–16.6 tok/s** vs 14.0 W8A8. Optimised GEMV lifts effective W8 bandwidth 46 → 63 GB/s.
- **T1 fit:** Best in this sweep — already measured on 8 GB and 16 GB edge devices; the memory and context numbers map directly onto 6 GB + SSD.
- **Not a known gap:** EG-25 lists token-level spec families; EG-43 quantises a *separate* draft KV while keeping target BF16; EG-27/EG-28 are quantise-on-load and lazy KV quant. BitNest has **one physical representation** serving both roles, a base-first residual construction, and dual-plane reads — none of the 46 covers nested bit planes or a single-representation self-draft.

## G2 — Edge0: prediction *as* routing

- **URL / date:** https://arxiv.org/html/2609.18063v1 — arXiv:2609.18063v1 [cs.AI], **16 Sep 2026**.
- **Mechanism (§3.1 SSD Streaming Expert Layers; §3.2 The Prerouter; §3.3 Unmerged Recovery LoRA; App. B):** A per-layer prerouter predicts layer N+1's routing from the previous token's layer-N state, giving a full-token lead so SSD reads overlap the forward pass. Crucially the prediction **is** the routing at decode — staged set == routed set, nothing dropped; quality is paid once in training instead of via runtime fallback loads. A recovery LoRA is trained on the student path and served **unmerged** as a parallel delta (merging + re-quantising to int4 erases its effect). Routing math is pinned bit-identical for softmax-top-k and sigmoid-group families. Sticky-slot incremental stacks remove the per-step tensor-assembly tax (App. B: +34% decode).
- **Numbers:** 35B MoE on one 24 GB machine at ≈**20 tok/s inside 3 GiB peak active memory**; 8B hybrid inside **1.5 GiB**. Prerouter worth +80→+84% decode on a machine where the checkpoint does not fit; incremental stacks +34%; native→full pipeline 6.8 → 12.5 tok/s in K=8 config.
- **T1 fit:** Strong — active-memory bound (3 GiB) is the property a 6 GB box needs; caveat: consumes *trained* prerouter + adapter artifacts, so the engine must load them, not train them.
- **Not a known gap:** EG-07 is SSD row reuse/coalesced reads (weights), EG-20 is expert trace caching/replacement, EG-45 is low-rank/amortised KV and MoE-CORE layer staging, EG-17 is multi-adapter slots. None covers **replacing routing with a prediction as a training contract**, an unmerged repair adapter, or sticky-slot incremental stack assembly.

## G3 — SSD-LLaMA: SSD-native three-tier expert delivery

- **URL / date:** https://arxiv.org/html/2609.18110 — arXiv:2609.18110v1 [cs.DC], **16 Sep 2026**.
- **Mechanism (§1 contributions; SSD I/O pipeline; Native SSD–RAM–VRAM Hierarchy; balanced CPU–GPU execution):** Each expert's tensors are reorganised into one aligned directly-accessible block; missing experts use concurrent direct reads, and **as each read completes** its block goes through pinned RAM to a VRAM slot without waiting for the other reads. A CUDA rANS decoder reconstructs original weights directly in VRAM, cutting both SSD and H2D traffic. SSD/RAM/VRAM retention uses observed frequency and recency, and residency is **decoupled from execution placement** (a RAM-resident expert may run on GPU). Every selected expert is executed exactly — no pruning or substitution.
- **Numbers:** Prefill **1.52–4.19×**, decode **2.10–15.58×** over baselines across three MoE families; >1 tok/s for a trillion-parameter model on one RTX 5090 with ≤32 GB RAM. Only ~5% of GLM5.2 transfer latency is hideable — the paper says so explicitly, which is a useful negative for our planner.
- **T1 fit:** High — consumer-PC target with ≤16 GB host memory; tiering is the path down to 6 GB.
- **Not a known gap:** EG-06 offload placement solver, EG-07 coalesced SSD row reuse, EG-22 pipelined offload, EG-20 expert traces. The new pieces are **per-expletion async H2D (no read-barrier)**, **in-VRAM rANS decompression to shrink both SSD and H2D bytes**, and **retention policy independent of execution device**. Row reuse is weight-level; this is expert-block delivery plus codec.

## G4 — LeanStream: speculate-and-refine

- **URL / date:** https://arxiv.org/html/2609.03079v1 — arXiv:2609.03079v1 [cs.LG], **02 Sep 2026** (MobiCom '26, DOI 10.1145/3795866.3844470).
- **Mechanism (§1 Fig.1(d); §3.3 stacked learnable hashing; fine-grained streaming control):** Instead of choosing between full-context prediction (accurate but serialises compute/I/O) and one-shot layer-wise speculation (overlaps but wrong), start with the best available prediction and **continuously refine** loading, execution priority and cache retention from partial GPU results while the preceding layer is still running. I/O streams weights in fine blocks so the GPU starts the highest-priority available sub-model as soon as compute frees, rather than waiting for the whole layer.
- **Numbers:** **4.8–7.5× less memory** at the best throughput achieved by prior work, and **1.6–2.1× higher** token throughput; implemented on both mobile and embedded platforms.
- **T1 fit:** High — exact activation-sparsity/SSD streaming regime the 6 GB box uses; needs a refinement hook in the GEMV/offload loop.
- **Not a known gap:** EG-05 is a hot/cold neuron-cluster activation predictor; EG-07 is SSD row reuse; EG-22 is a per-iteration offload policy. None covers **mid-layer refinement of three decisions (execute / load / retain) from partial results**, or the lightweight stacked-hash controller.

## G5 — ReadKV: query-adaptive progressive KV codes

- **URL / date:** https://arxiv.org/html/2610.11245v1 — arXiv:2610.11245v1 [cs.LG], **08 Oct 2026**.
- **Mechanism (§1 separate storage/read budgets; §3 Algorithm 1; §4 Theorems 1–3; App. A):** KV entries are written once as a progressive code. Store W bits/scalar but let each decode query read R<W bits: allocate **key-channel prefixes using the query**, compute attention from reconstructed keys, then allocate **value-token prefixes using that attention**. Stored entries never change. Allocation repeatedly spends the next bit where predicted error reduction is largest; Theorems 1–2 show query-dependent access strictly beats any query-independent reader at equal read budget, Theorem 3 bounds attention-output error.
- **Numbers:** Reading 4 bits on average from an 8-bit cache raises C4 perplexity ≤**0.66%**, using ~¼ of logical reads and half the retained capacity of a 16-bit cache. On 8K batch-1 A10G, ReadKV(8,2) cuts DRAM reads **62–65%** (2.80 vs 7.28 MiB) and has **39% lower latency** than TurboQuant K4/V4; (8,2) write traffic is only 0.79% of cumulative reads over 8192+128 tokens.
- **T1 fit:** High — drop-in KV read path, and the 8K/batch-1 fixture is exactly the single-user edge case.
- **Not a known gap:** EG-28 is lazy per-layer KV quantisation, EG-30 FP8 attention, EG-43 quantised *draft* KV, EG-44 a trained sparse indexer, EG-23 post-training block pruning. No 46-row covers **decoupling retained bits from per-query fetched bits**, progressive codes, or query-chosen key/value prefix allocation. KIVI lineage is a static per-channel/per-token quantiser; here the read depth is a query-time decision.

## G6 — SPICE: speculative expert prefetch + LoRE surrogates

- **URL / date:** https://arxiv.org/html/2608.21240 — arXiv:2608.21240v2 [cs.AR], **07 Sep 2026** (accepted ASP-DAC 2027).
- **Mechanism (§4.1 Algorithm 1; §4.2 miss recovery; §4.3 CPU–GPU orchestration):** A draft model aligned to the target recursively forecasts future-layer Top-K expert sets; prefetch horizon adapts between H_min (Eq. 4, transfer time ÷ per-layer compute window) and H_max, halting when confidence C=Σπ_i over selected experts falls below τ (Eq. 5). On a miss, choose among: resident shared-expert + **Low-Rank Expert (LoRE) surrogate** approximation for low-confidence misses, weight migration to GPU, or **compute-near-weights** on the CPU using host-resident weights, overlapped with GPU work. Expert loading is 73–88% of per-layer latency in their Fig. 1.
- **Numbers:** up to **3.12× TPOT** on DeepSeek-V2-Lite and Qwen2-57B-A14B; 2.04–2.70× over AdapMoE; 2.44–3.12× as prompt grows 64→1024; full system 172.5 ms (23.9% below best ablation).
- **T1 fit:** High for the MoE-offload path, with the caveat that LoRE surrogates are lossy — treat as opt-in quality/throughput trade, measured by a gate.
- **Not a known gap:** EG-20 = expert activation traces + layer-decay eviction; EG-25 = *token*-level speculative families; EG-06/EG-22 = placement/pipelining. Nothing in 46 covers **expert-set prediction with a confidence-adaptive horizon**, **surrogate substitution on a miss**, or the **migrate-weights-vs-move-computation** decision.

## G7 — SeqMoE: sequence-modelled activation + sync-free graph capture

- **URL / date:** https://arxiv.org/html/2609.12978 — arXiv:2609.12978v1 [cs.OS], **11 Sep 2026**.
- **Mechanism (§3.1 Expert Activation Sequence Modeling; §3.2 Joint Structure in Prefetch Scheduling; forecast-driven caching; graph-compatible runtime):** Encode the full layer stack's activation state per generation step and train a Transformer probe for **next-step** prediction — extending the overlap window beyond the ≤3-layer horizon of same-step methods. Formulate prefetch as **Job Sequencing with Deadlines** (confidence = objective, deadline = constraint); eviction uses a **probabilistic Belady** policy. Runtime principles: compute-transparent expert placement and **synchronization-free orchestration** for end-to-end graph capture.
- **Numbers:** at **45% expert residency**, 96.97% hit rate and 80.22% of full-load performance; eager offloading ≈2% of full-load for comparison.
- **T1 fit:** High — the graph-capture discipline is directly implementable in a single-process C11 loop; the trained probe is an artifact to load.
- **Not a known gap:** EG-20 is a per-batch trace cache with layer-decay eviction; EG-09 is an overlapping scheduler; EG-06 a placement solver. Multi-step *sequence modelling* of expert activations, deadline job sequencing, probabilistic Belady, and sync-free graph capture are all absent.

## G8 — FreeToken: bandwidth-adaptive edge full stack

- **URL / date:** https://arxiv.org/html/2608.16157v1 — arXiv:2608.16157v1 [cs.DC], **17 Aug 2026**.
- **Mechanism (§1 three failure modes; co-design of model layout/loading, expert residency, CPU–GPU execution, agentic state reuse, runtime memory):** Treats a personal machine as a unified elastic platform rather than a small GPU; continuously maps computation and model state onto whichever GPU/CPU/memory/interconnect balance is actually available, including fluctuating budgets from other apps. Explicitly handles **prefill working-set densification** (union of routes over a long prompt covers most experts) and agentic state reuse across turns.
- **Numbers:** >20 MoE models; 35B on an **8 GB laptop GPU**, 284B on a gaming desktop, 753B GLM-5.2 on one workstation.
- **T1 fit:** High at the concept level, but overlapping — isolate the bandwidth-adaptive mapping and prefill-densification handling, not the whole stack.
- **Not a known gap (partial):** Pieces touch EG-03/EG-22/EG-37/EG-39. What is new is the **continuous cross-stack remapping driven by measured bandwidth and a changing budget**; no single 46-row covers that policy. Be honest in the gate: do not claim the agent-state or arena parts as new here.

## G9 — Memory-Sovereign Inference: output-exactness certificate

- **URL / date:** https://arxiv.org/html/2608.23805 — arXiv:2608.23805v1 [cs.DC], **24 Aug 2026**.
- **Mechanism (§2 certificate tuple C=(I,B,A,E,L); authority grades; §1 four-way split):** Separates canonical storage ownership, physical residency, stable execution addresses and scheduling into a shared Plan; maps logical objects to source extents, fixed host windows, reusable GPU slots and **generations**. Defines a falsifiable evidence record: semantic-demand lower bound (distinct from scheduler requests and cumulative traffic), per-tier authority vector (hard / physical / audited / sampled), exactness equivalence + horizon, and tested reuse invariants. Explicitly reports **stopped lines** (GPT-OSS packing, Llama-3.3 admission) rather than promoting local wins.
- **Numbers:** Qwen3-Next 45.08 GiB representation, 43.59 GiB semantic-demand lower bound vs 34 GiB declared / 11 GiB host-hard + 24 GiB device; LRU64 exact against a zero-cache oracle on all 64 token IDs, all 64 complete 151,936-float logit rows, 3,408 route events. Eight-window async path = **32.3% of blocking one-window wall time (3.10×)** at identical source bytes; 33,481 `memory.max` events on buffered path vs zero on direct.
- **T1 fit:** Methodology, high leverage: turns our "no gate means not absorbed" rule into a concrete assertion set (generation counters for async buffer reuse, whole-process memory accounting, logit-row equality).
- **Not a known gap:** EG-03/EG-35 are VRAM planners; EG-29 a blob store; none defines an **execution-evidence certificate** or a semantic-demand lower bound, and none tests reuse invariants for asynchronously recycled buffers.

## G10–G16 — shorter entries

**G10 SpecQuant** — https://arxiv.org/html/2609.21704v1, arXiv:2609.21704v1 [cs.LG], **18 Sep 2026**. Training-free: derive INT4/FP8/FP16 variants from a shared base, route queries by predicted complexity, use lightweight variants for simple/factual tasks and full precision for reasoning/long context; shared weights keep speculative acceptance compatible. 35–43% speedups, <2% accuracy loss on Qwen2.5/MMLU/AlpacaEval/GSM8K. T1 high, no retraining. **Diff:** EG-27 is quantise-on-load to one precision; multi-*parent* simultaneous variants + complexity routing is new.

**G11 MemSpec** — https://arxiv.org/html/2608.10362, arXiv:2608.10362v1 [cs.OS], **11 Aug 2026**. Prediction-guided, memory-aware runtime for adaptive draft scheduling on edge; Jetson Orin Nano steady-state throughput **+40.7%** over bandit-based adaptive methods, near oracle. **Diff:** EG-41 is offline hardware/workload config search; this is per-step runtime draft scheduling under a memory budget.

**G12 PELM** — https://arxiv.org/html/2609.09662v1, arXiv:2609.09662v1 [cs.LG], **09 Sep 2026**. Adds speculative decoding and **variable verification depth** as two knobs beside DVFS; pilot: 32 layers @1.3 GHz ≈ 24 layers @0.7 GHz in speed, latter **41.7% less energy**. **Diff:** no gap treats verification depth as a power/thermal control variable. Split the work: depth policy is in-scope C11; DVFS is OS/hardware, tier it out.

**G13 ThreeH** — https://arxiv.org/html/2609.07282v1, arXiv:2609.07282v1 [cs.CL], **07 Sep 2026**. Formal Proposition 1: a bounded-state model can have infinite stability horizon while access/utility horizons stay ≤W; ThreeH is a budget-matched protocol measuring all three. 128K-token streams and delayed-binding recall. **Diff:** refines EG-04 as a *gate-design* rule — pinning sink KV proves stable streaming, not semantic memory; do not write an EG-04 gate that tests recall.

**G14 SPADE** — https://arxiv.org/html/2608.13076v1, arXiv:2608.13076v1 [cs.AI], **13 Aug 2026**. Edge draft + cloud verifier, only rejections trigger verifier correction; **76% fewer cloud calls**, zero accuracy loss, no retraining; edge is RTX 3080 12 GB. **Diff:** EG-15/EG-33 are in-tree single-node PD handoff and a connector protocol; this is a multi-node edge-cloud split → tier-gated T2/T3, no T1 work.

**G15 HySparse2** — https://arxiv.org/html/2609.26368, arXiv:2609.26368v1 [cs.CL], **22 Sep 2026**. Two-level KV sharing: YOCO-style KV Bridging across self/cross decoder plus intra-block KV Reuse; prefill can exit after the self-decoder. **Diff:** no gap covers two-level KV sharing, but it requires a trained architecture — not a drop-in for stock GGUFs; T2.

**G16 LLM Inference in a Flash!** — https://arxiv.org/html/2609.16161, arXiv:2609.16161v1 [cs.LG], **14 Sep 2026**. Integer-only execution for compute-in-flash + dictionary-coded KV (each KV vector a linear combination of static dictionary vectors); **15× lower dynamic KV traffic**, 3.1/2.7× latency/energy savings short context, 4.4/6.8× long. **Diff:** EG-07 is SSD row reuse on conventional hardware; the sparse-dictionary KV codec is extractable in software, CIM execution is hardware-gated T3.

---

## Starting leads resolved

- **FlashMoE** — https://ar5iv.labs.arxiv.org/html/2601.17063, arXiv:2601.17063, **22 Jan 2026 (outside window)**. Per-layer ~113 KB FFN cache policy combining recency+frequency to approximate Belady; +51% hit rate over LFU, +28% over ARC, +21% over LRU; up to 2.6×, 4× faster model load. Closest to EG-20 (cache replacement) and EG-07 (SSD). The *ML policy* itself is new vs the 46 but predates Q3; list as lineage, not a Q3 find.
- **CoX-MoE** — https://arxiv.org/html/2605.17889, arXiv:2605.17889v2 [cs.LG], **19 May 2026 (outside window)**. Coalesced expert execution with ordinary (not micro-) batch, selective attention offloading, static expert-aware stratification, AMX CPU–GPU; 1.7–2.4× over MoE-Lightning, 3.4–7.1× over FlexGen. Touches EG-06/EG-20; note for the offload-placement solver but not a Q3 addition.
- **QuantSpec** — arXiv:2502.10424, ICML 2025 (outside window). Hierarchical INT4 KV (upper 4-bit for draft, upper+lower → INT8 for verify), >90% acceptance, ~2.5×, ~1.3× less memory. Its Q3 descendant is **BitNest (G1)**; the November sweep's EG-43 cites QuantSpec only as related work. Do not re-report QuantSpec itself.
- **KIVI lineage (Q3):** ReadKV (G5), interface-aware NVM KV quant (arXiv:2609.05764, **04 Sep 2026** — 3.1–3.6× lower KV read energy, 8× less metadata than KIVI, hardware-gated), diffusion-LLM KVQ (arXiv:2610.39853, **29 Oct 2026** — KIVI/KVQuant adapted to reusable dLLM caches, marginal to this engine). Only ReadKV has direct single-user T1 fit.
- **StreamingLLM/attention-sink follow-up (Q3):** ThreeH (G13) and "The KV Cache Is the New Memory Wall" (arXiv:2609.30854, **25 Sep 2026**, five-domain compression taxonomy, context only). No new sink *mechanism* in-window that beats EG-04.
- **YOCO / linear-attention hybrids (Q3):** HySparse2 (G15) and "How Linear Attention Remembers" (arXiv:2609.33093, **27 Sep 2026** — mechanistic finding that in hybrids, recall shifts to the full-attention KV state; useful capacity-planning evidence, not a mechanism). YOCO-U (2604.01220) and YOCO++ (2604.13556) are April 2026 — outside window. Caveat for any hybrid work: these need a trained architecture; the C11 engine can only schedule them, not create them from stock GGUFs.

**Additional Q3 leads worth a later pass, not detailed here:** "KV Cache Is the New Memory Wall" 2609.30854 (25 Sep, taxonomy); PrefixPlace 2608.01655 (distant-node prefix placement, T2); KVBoost 2608.21362 (chunk-level reuse + SelectiveRecompute/CacheBlendRecompute, ~15% recompute; overlaps EG-08/EG-19); SpecEdge/EG-Spec (entropy-gated adaptive speculation, claimed 2.38× and 1420 MB "within 4 GB and 6 GB edge accelerators") — **manuscript on GitHub, no arXiv ID and no verifiable submission date, so not counted as a dated Q3 paper**.

---

## Rejected / caveats

- All speedups and memory figures are the **authors' reported numbers**, not measurements on wubuwizard's 6 GB box. A reported speedup is a reason to write a gate, not evidence the feature exists locally.
- SPICE LoRE, Edge0 routing replacement, and SpecQuant's low-precision parents are lossy or require model artifacts; each must be opt-in behind an accuracy gate.
- SPADE, HySparse2, compute-in-flash and SiFAR-class work are multi-node, trained-architecture or hardware-specific — record, do not start at T1.
- Nothing here authorises vendoring. Reimplement from the papers' descriptions in C11, keep the mmap zero-copy weight law (no `MADV_SEQUENTIAL`), keep `wubu_tier_advice()`/`wubu_offload_decision()` as per-step policy functions, and mark no row DONE without a gate in both Makefiles.

## Sources

[G1] https://arxiv.org/html/2610.02800
[G2] https://arxiv.org/html/2609.18063v1
[G3] https://arxiv.org/html/2609.18110
[G4] https://arxiv.org/html/2609.03079v1
[G5] https://arxiv.org/html/2610.11245v1
[G6] https://arxiv.org/html/2608.21240
[G7] https://arxiv.org/html/2609.12978
[G8] https://arxiv.org/html/2608.16157
[G9] https://arxiv.org/html/2608.23805
[G10] https://arxiv.org/html/2609.21704v1
[G11] https://arxiv.org/html/2608.10362
[G12] https://arxiv.org/html/2609.09662v1
[G13] https://arxiv.org/html/2609.07282v1
[G14] https://arxiv.org/html/2608.13076v1
[G15] https://arxiv.org/html/2609.26368
[G16] https://arxiv.org/html/2609.16161
[L1] https://ar5iv.labs.arxiv.org/html/2601.17063
[L2] https://arxiv.org/html/2605.17889
[L3] https://ar5iv.labs.arxiv.org/html/2502.10424
[L4] https://arxiv.org/pdf/2609.05764
[L5] https://arxiv.org/html/2610.39853v1
[L6] https://arxiv.org/html/2609.30854v1
[L7] https://arxiv.org/pdf/2609.33093v1
[L8] https://arxiv.org/html/2608.04074
