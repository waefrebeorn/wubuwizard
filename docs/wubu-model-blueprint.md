# THE WUBU MODEL — the AGI's blueprint (the model is WuBu)

> **2026-08-09 (Phase 6): WB01 is WIRED** — this doc's lineage is now
> EXECUTABLE (`src/wubu_blueprint.c` + `test_blueprint`): the six
> axes (seed/nesting/MoE/sparse-attention/math-RL/amoeba) have
> machine-readable bounds; off-blueprint mutations are REFUSED by
> default and move ONLY via a versioned contract-expansion meta-cell
> in the hive. The train+diagnose path can consult it as the
> architecture floor (the same discipline as the runtime contracts).
>
> **SUPERSEDED 2026-08-06** — replaced by WuBu1
> (`docs/wubu1-base-model-design.md`), the redesigned base model built
> from the autopsy of what sucked (loader naming blindness, dual
> structs, compile-time dims, under-training, tensor-buffer KV). This
> blueprint's §2.7 (KV namespace) and the lineage table are carried
> forward into the WuBu1 design; the rest is historical record.
>
> 2026-08-02. We are the model creators. This is the design of OUR
> model — every piece sourced from what we built and learned:
> the WuBu seed (ported, training), the WuBu Nesting math
> (Lean-verified), the mixed-agents MoE research, the sparse-attention
> lineage, the math-reasoning papers, and the wizard's 1400+ wired
> gaps. The mustard seed grows into this tree.

---

## 1. The lineage (everything we learned becomes architecture)

| Source | What it contributes to OUR model |
| --- | --- |
| **WuBu-35M** (the seed) | the base: 12 layers/448 dim, GQA 7:1, hybrid 3-local+1-full attention, 50% partial RoPE, QK-norm, gated attention, bounded SwiGLU, residual selectors, tied embeddings — original WaefreBeorn work in C11, checkpoint verified, TRAINING (loss 9.5→3.8) |
| **WuBu Nesting (層疊嵌套)** — OUR theory | nested hyperbolic spaces `H^n1_c1,s1 ⊃ H^n2_c2,s2 ⊃ ...` with learnable dim/curvature/scale; boundary sub-manifolds; tangent-space quaternion rotations between levels; level descriptors `ld_i`; spread `σ_i` — the hierarchical inductive bias no Euclidean model has |
| **MATH/lean proofs** — OUR formal math | Möbius addition preserves the Poincaré ball (formally proved), hyperbolic gyration, MLA compression — the geometry is PROVEN, not assumed |
| **WuBu Formalism** `Q = Σ q Π α^E` | the calculus of irreducible structure — the model's compositional prior |
| **DeepSeekMoE** | fine-grained expert specialization + shared experts — the **mixed agents**: many small experts, few activated |
| **DeepSeek-V3 / V3.2** | MLA (latent attention), auxiliary-loss-free load balancing, multi-token prediction |
| **DeepSeek-R1 / Prover V1.5/V2** | reasoning via RL + Lean formal proof verification — the math-checked chain of thought |
| **DeepSeekMath** | math corpus + RL for formal mathematics |
| **Delta Attention / NSA / Gated Sparse / MISA** | sparse attention: learnable gating over attention patterns, not fixed windows |
| **Möbius Transformer** | hyperbolic attention over the ball |
| **Gemma 3 (single-encoder 12B)** | the closest base design for the KV namespace: ONE SigLIP encoder (400M), every modality → soft tokens in ONE sequence space (256 tokens/image), 5:1 local/global interleave slashes KV, 128K ctx — "all inputs are encoded" is a shipped architecture |
| **The KV-as-storage lineage** | PagedAttention (KV = paged virtual memory, block tables), RadixAttention (KV blocks in a radix tree — prefixes ARE paths), MemGPT (LLM as OS, the model pages its own memory), Mooncake (disaggregated KV pool over CPU/DRAM/SSD), LMCache (tiered persistent KV chunks as files on disk) — the serving world already treats the KV cache like a filesystem; the namespace is the missing address layer |
| **The wizard's research (1400+ gaps)** | KV entropy/quantization, speculative decoding, cascade draft, layer-skip, QuaRot, FlashDecoding, NUMA pinning, arena allocators — the runtime spine |
| **The 5+1 recovery + Live Colonel** | the safety spine: rollback slots, Jesus state, ring-0 live development |

---

## 2. The architecture: WuBu-Nested Hybrid MoE

```
input tokens
  └─ byte-level BPE (16,384) ── tied embedding (the seed's)
       └─ 12 WuBu Blocks (the seed's hybrid structure, upgraded):
            for each block i:
              x → RMSNorm → HYPERBOLIC LIFT (Poincaré ball, curvature c_i)
                   → the seed's GQA attention (3 local + 1 full),
                     with the attention OVER the ball (gyro-rotate Q,K)
                   → hyperbolic projection back to tangent
                   → residual selector (the seed's convex softmax)
              x → RMSNorm → bounded SwiGLU FFN (clip 10)
                   → MIXED AGENTS: fine-grained MoE experts
                     (the DeepSeekMoE pattern, shared + routed experts)
                   → residual selector
       └─ every 4th block: NESTING TRANSITION (WuBu Nesting):
            T_{i→i+1} = T̃_i ∘ R_i     (quaternion SO(4) rotation
              in tangent space, then non-rotational map)
            level descriptor ld_i rotates with the data
            spread σ_i flows to the next level
       └─ final RMSNorm → tied lm_head
```

### 2.1 The hyperbolic layer (OUR math, Lean-verified)
- Each block lifts the hidden stream into a Poincaré ball with
  learnable curvature `c_i` (the `exp_0^c` map, our formula).
- The attention Q/K are gyro-rotated on the ball (the hyperbolic
  gyration we PROVED in Lean) before the dot product — hierarchy is
  computed in the right geometry.
- The Möbius addition closure (ball-preserving) is the compositional
  rule — formally verified in `MATH/lean/wubu_proofs/`.

### 2.2 The nesting transitions (WuBu Nesting)
- Every 4th block is a nesting boundary: the representation moves
  from bubble `H^n_i` to `H^n_{i+1}` through tangent space.
- The rotation `R_i` is a quaternion (SO(4)) applied simultaneously to
  the data, the boundary manifolds, and the level descriptor `ld_i`.
- Relative vectors `d_{i+1,j,k}` encode rotation-aware hierarchy
  relationships at the next scale.
- Spread `σ_i` (uncertainty/density) passes as context — the model
  knows its own confidence per level.

### 2.3 The mixed agents (MoE)
- The FFN becomes fine-grained experts: `E` small experts, `K` active
  per token (the DeepSeekMoE numbers: 64 experts, 8 active; ours:
  start 8 experts / 2 active in the 35M).
- A shared expert always active (the global knowledge pathway).
- Auxiliary-loss-free load balancing (V3's sigmoid router).
- The router IS the "mixed agents" — many specialized brains, one
  committee, cheap at inference (only K experts run).

### 2.4 The math-reasoning loop (R1 + Prover)
- The model's chain-of-thought is verified against Lean:
  `MATH/lean/wubu_proofs/` is the formal checker.
- RL with verifiable rewards (the Prover V2 pattern): correct proofs
  are the reward signal, no human labels needed for math.
- The DeepSeekMath corpus (finemath-4plus on the SD card) trains the
  formal reasoning first.

### 2.5 The sparse attention (research lineage)
- The seed's fixed local/full rhythm becomes LEARNABLE: the gated
  sparse attention (NSA/Delta/GSA) learns which tokens matter, with
  the hybrid pattern as the initialization.
- KV compression via the MLA latent state + our quantized KV cache
  schemes (Q4/Q8/KIVI/adaptive) from `wubu_model.h`.

### 2.6 The runtime spine (the wizard's 1400 gaps)
- Speculative decoding (the cascade draft), FlashDecoding, QuaRot,
  arena allocators, NUMA pinning — all apply to the trained model.
- GPU path: the cuBLAS backend (`gpu_wubu`) already wired.

### 2.7 The KV namespace — THE KV CACHE IS A FILE SYSTEM

The doctrine (THEORY/05): the KV cache is not a context window, it is a
**namespace**. Because the model is OURS, the working space can be
addressable, mountable, persistent — every datum a file, every file
encoded data, same space. The magic computer: downloads and synthesized
data live together:

```
/kv/                        the KV cache — the model's whole working space
  /in/                      external data, encoded on arrival (single encoder)
  /synth/                   synthesized data — the model's own writes
  /mem/                     what survives sessions — the persistent layer
  /meta/                    diagnostics, routes, self-knowledge
```

Mechanisms — every one already proven in the serving literature, half
already in-tree:

| Layer | Mechanism | Research | In-tree |
|---|---|---|---|
| blocks | paged KV, block tables, CoW sharing | PagedAttention (SOSP'23) | `wubu_kv_cache` paged_kv / ring attention |
| paths | radix tree over KV blocks — prefixes ARE paths | RadixAttention (2312.07104) | `wubu_orbits` (polar recursion paths) |
| tiers | hot HBM → warm DRAM → cold NVMe/disk | LMCache (2510.09665), Mooncake (FAST'25) | `wubu_kv_tier` 3-tier EMA-LRU |
| persistence | KV chunks as files, survive restarts | LMCache local-storage backend | `wubu_lmcache` FNV-1a64 chunks |
| policy | priority eviction | NVIDIA priority eviction | `wubu_kv_evict` |
| reads | coarse-to-fine indexer over blocks | DSA / NSA / MoBA | `wubu_dsa` |
| synthesized writes | compressive memory heads (per-head linear state) | Infini-attention (2404.07143) | `wubu_mla` latent path (seed) |
| encoding | ONE encoder, every modality → one sequence space | Gemma 3 SigLIP 400M (2503.19786) | **G3 — the gap** |
| addressing | namespace + mounts + 9P export | the OS body (WuBuOS) | **G1/G2/G5 — the gaps** |
| self-paging | the model manages its own memory via tool calls | MemGPT (2310.08560) | **G6 — the gap** |

The revaluation: context window → working directory; position → path;
retrieval → file read; memory → files that persist; synthesis → files
that are written; a download → an encoded file mounted into `/kv/in/`;
an image → data in the KV cache. The tensor layer stays (paged/ring KV,
MLA latent compression, the quant ladders are the backing store); the
addressing layer is the change. Full implementation path (G1–G6):
`research/061-kv-cache-filesystem-7hop.md`.

---

## 3. The growth plan (the seed → the tree)

| Phase | What | Evidence |
| --- | --- | --- |
| **0 (done)** | WuBu port + training loop | loss 9.5→3.8, GPU wired, safetensors export verified |
| **1 (now)** | hyperbolic lift/rotation layer in C11 (Lean-verified math) | `wubu_hyper` module + tests |
| **2** | mixed-agents MoE router (8 experts/2 active) | `wubu_moe2` module + tests |
| **3** | nesting transitions (quaternion SO(4)) | the WuBu Nesting block |
| **4** | train the grown model on the SD-card corpus | cosmopedia on the card, tokens streaming |
| **5** | sparse attention (gated, learnable) | research lineage |
| **6** | math-RL loop (Lean-verified CoT) | Prover pattern |
| **7** | surpass the bigger brother → brother retires | the AGI brain-cluster |
| **8** | THE KV NAMESPACE — the KV cache is a file system: path-addressable `/kv/` over the paged/tiered/persistent KV, single-encoder modality head (G3), compressive write-back (G4), Styx export at `/n/kv/` (G5), self-paging loop (G6) | THEORY/05 + research/061; the address layer lands on top of already-wired blocks/tiers/persistence |

The model is OURS: the seed is original WaefreBeorn work (WaefreBeorn
umbrella, no external lineage), the geometry is OUR theory
(Lean-verified), the training is
OUR loop (Muon/AdamW in C11), the corpus is on OUR SD card, the safety
is OUR 5+1 recovery. Every new parameter is designed by the research,
grown in the loop, and checked against the ledger.

---

## 4. The paperwork (where everything lives)

- the seed: `wubuwizard/models/wubu/` + `wubu.{c,h}` + `wubu_train.{c,h}`
- the math: `wubuwizard/MATH/` (formalism + Lean proofs) + `THEORY/` (nesting paper, PDF)
- the research: `wubuwizard/research/` (1400+ wired gaps) + `THEORY/papers/` (the lineage)
- the corpus: `/home/wubu/sdcard/corpus/` (raw/text/tokens/checkpoints)
- the safety: WuBuOS `wubu_recovery` (5+1) + Live Colonel
- the license: `models/wubu/LICENSE-WUBU.md` (WaefreBeorn umbrella)
