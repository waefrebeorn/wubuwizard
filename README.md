# wubuwizard — THE BRAIN of the WuBu AGI

> **The WuBu amoeba AGI** — a C11 inference/training engine, KV-cache
> filesystem, universal encoder space, and research ledger. One AGI,
> three repos: this Brain (thinks) + `wubuos` (the Body: kernel, GUI) +
> `wubunos` (the Compiler: 11 ISA backends).

**Stats:** 1,167 C files · 420 H files · 37 CUDA files · 218,100 LOC

---

## What this repo is

wubuwizard is the **Brain** half of the WuBu project — everything that
thinks: the C11 inference engine, model loading, KV-cache compression,
training, the universal encoder space, and the research ledger.

**One sentence:** The Brain learns; the Body protects and acts; the Compiler builds.

### The three repos

| Repo | Role | LOC |
|------|------|-----|
| **wubuos** | THE BODY — kernel, GUI, Styx/9P, drivers, runtime | 472,955 |
| **wubuwizard** (this repo) | THE BRAIN — inference, training, KV cache, encoders | 218,100 |
| **wubunos** | THE COMPILER — HolyC JIT, 11 ISA backends | 14,115 |
| **TOTAL** | **Three repos, one AGI** | **705,170** |

WuBuOS has both wubuwizard and wubunos as submodules:
- `wubuos/src/brain/` → this repo
- `wubuos/src/compiler/` → wubunos

---

## Architecture

### The Amoeba Design

WuBu is an **amoeba** — a single living system that grows and shrinks to fit:

```
                    ┌─────────────────────────────┐
                    │      WuBu AGI Supervisor     │
                    │   (wubu_agi.c — the mind)    │
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
     ┌────────┴────────┐  ┌───────┴────────┐  ┌───────┴────────┐
     │   BOOT CORE     │  │  BODY SPHERES   │  │ MEMORY ORBITS  │
     │                 │  │                 │  │                │
     │ Always present  │  │ Grow on demand: │  │ Orbit the      │
     │ · Inference     │  │ · Training      │  │ spheres:       │
     │ · Model loading │  │ · Encoding      │  │ · KV cache     │
     │ · KV cache      │  │ · RL (PPO/GRPO) │  │ · Encoder space│
     │                 │  │ · Generation    │  │ · Training data│
     └─────────────────┘  └────────────────┘  └────────────────┘
```

**Scale-to-fit:** one checkpoint, any hardware. On a phone, the amoeba
shrinks to a single sphere. On a server, it grows to fill all available
resources.

### KV Cache IS a File System

The central insight: the KV cache is not a flat array of floats. It is a
**file system** — the namespace stores floats; a file IS a mount region;
the encoder IS a mount. This means:

- KV entries are path-addressable (hash-table O(1) lookup)
- Tiers (HBM → DDR → SSD) are mount points
- Eviction is unlink, compression is re-encoding
- The same interface serves inference, training, and user files

### Engine Module Clusters

The `src/wubu_*.c` modules cluster by function:

| Cluster | Purpose | Key modules |
|---------|---------|-------------|
| **Attention** | Multi-head, latent, cross, gated attention | `wubu_attn_kernels`, `wubu_mla`, `wubu_attn_gate`, `wubu_ring_attn` |
| **KV Cache** | Compression, tiering, eviction, paging | `wubu_kv_evict`, `wubu_kv_compress`, `wubu_kv_tier`, `wubu_paged_kv`, `wubu_4kv` |
| **MoE** | Mixture of experts, routing, grouping | `wubu_moe`, `wubu_moe_grouped`, `wubu_moe_hyperbolic`, `wubu_hashrouter` |
| **SSM** | State space models, scan, recurrence | `wubu_ssm_scan`, `wubu_ssm_recurrence`, `wubu_nested_ssm` |
| **Speculative** | Speculative decoding, Medusa, tuning | `wubu_spec_decode`, `wubu_spec_tuner`, `wubu_medusa` |
| **Quantization** | AWQ, GPTQ, SmoothQuant, NF4, FP8, MXFP4 | `wubu_awq`, `wubu_gptq`, `wubu_smoothquant`, `wubu_nf4`, `wubu_fp8` |
| **Hyperbolic/Nesting** | Poincaré ball, Möbius, RSGD | `wubu_hyper`, `wubu_nest`, `wubu_mobius_linear`, `rsgd` |
| **Model Core** | Seed model, training, backprop, tokenizer | `wubu`, `wubu_train`, `wubu_backprop`, `wubu_gemma4_model` |
| **AGI Organs** | Hive memory, prover, AGI loop, indexer | `wubu_hive`, `wubu_prover2`, `wubu_agi`, `wubu_dsa` |
| **Agentic OS** | Agent KV, agent memory, agent auth | `wubu_agentic_kv`, `wubu_agentic_mem`, `wubu_agentauth` |
| **Encoders** | Image, audio, video, PDF, office | `wubu_imgenc`, `wubu_audio`, `wubu_jpeg`, `wubu_png`, `wubu_pdf` |
| **RL** | PPO, GRPO, actor-critic, bandits | `wubu_ppo`, `wubu_traj_grpo`, `wubu_actor_critic`, `wubu_bandit` |

### The AGI Brain Pipeline

```
corpus (HF datasets, user files)
  ├─ text/        raw text shards
  ├─ tokens/      .tok uint16 streams (BPE tokenizer)
  ├─ images/      ingested into encoder space
  └─ checkpoints/ seed.st (safetensors)

trainer (wubu_train.c + wubu_backprop.c)
  └─ WuBu-35M safetensors → trained .st checkpoints → HF

inference (wubu.c + wubu_model.c)
  └─ GGUF/SafeTensors/ONNX load → KV cache (filesystem) → generate

oracles (RLHF reward)
  └─ WuBu drafts → frontier model scores → trainer
```

---

## Research & Theory

This repo contains its own research library:

| Directory | Contents |
|-----------|----------|
| `research/` | 93 research docs (001–093), each with Triple-DA audit, implementation status, Kevin-Bacon 7-hop convergence |
| `THEORY/` | 50 theory papers: foundational philosophy, axiomatic emergent theory, KV-cache filesystem, ecosystem of spheres, scale-to-fit |
| `MATH/lean/` | Lean-verified theorems: Möbius addition, Poincaré ball, gyration |
| `docs/` | Architecture docs, model blueprint, model card, improvement plans |

The research INDEX (`research/INDEX.md`) is the gap ledger — open/wired
discipline for every research claim.

---

## Build & test

```bash
make all          # full build (engine + tools)
make test_all     # the test gate — run this before claiming anything works
make test_<name>  # one subsystem (e.g. test_kvfs, test_codec, test_userfs)
```

---

## Directory map

| Path | What lives there |
|---|---|
| `src/wubu_*.c` | The engine modules (attention, KV, MoE, SSM, quant, encoders, etc) |
| `src/*.cu` | 37 CUDA kernels (flash attention, quant matmul, SSM recurrence, etc) |
| `include/wubu_*.h` | Public API headers — opaque types + function decls |
| `tools/*.c` | CLI tools and test drivers |
| `research/` | 93 research notes with Triple-DA audits |
| `THEORY/` | 50 theory papers (WuBu Nesting, KV-namespace, scale-to-fit) |
| `MATH/lean/` | Lean-verified mathematical proofs |
| `docs/` | Architecture, topology, model docs, ADRs |
| `models/` | Local weights (WuBu seed: safetensors + tokenizer) |
| `manifests/` | Model configs for bigger-brother line |

---

## Sources of truth

- **docs/TOPOLOGY.md** — the master map (all three repos)
- **research/INDEX.md** — the gap ledger
- **docs/MODULES.md** — auto-generated module table
- **docs/ARCHITECTURE.md** — detailed architecture document

---

## License

WaefreBeorn Umbrella License v3.0
