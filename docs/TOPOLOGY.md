# WuBu Project Topology — the master map

> 2026-08-15. The authoritative map of the WuBu project across all
> three repositories. It supersedes scattered READMEs when they conflict.

```
╔══════════════════════════════════════════════════════════════════╗
║                    THE WUBU UNIVERSE                              ║
║                                                                  ║
║   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  ║
║   │  wubuwizard      │  │    wubuos        │  │   wubunos    │  ║
║   │  THE BRAIN       │  │    THE BODY      │  │   COMPILER   │  ║
║   │  218,100 LOC     │  │    472,955 LOC   │  │   14,115 LOC │  ║
║   │  1167 C · 37 CUDA│  │    2463 C        │  │   33 C       │  ║
║   │  93 research     │  │    414 tests     │  │   11 ISA     │  ║
║   └────────┬─────────┘  └────────┬─────────┘  └──────┬───────┘  ║
║            │                     │                    │          ║
│            └─────────────────────┼────────────────────┘          ║
║                                  │                                ║
║                   WuBuOS links both as submodules:                ║
║                   src/brain/ → wubuwizard                        ║
║                   src/compiler/ → wubunos                         ║
║                                                                  ║
║                   TOTAL: 705,170 LOC · 3 repos · 1 AGI           ║
╚══════════════════════════════════════════════════════════════════╝
```

## THE ONE-SENTENCE SPLIT

**wubuwizard = the BRAIN** (everything that thinks: inference, training,
research, math, encoders, the amoeba model).

**wubuos = the BODY** (everything that acts: the kernel, the shell,
the GUI, the firmware, the drivers, the measured-boot chain).

**wubunos = the COMPILER** (the toolchain that builds: HolyC frontend,
MIR optimizer, 11 ISA backends, self-hosting battery).

The Brain trains; the Body runs; the Compiler builds. The Live Colonel
(ring-0 REPL in wubuos) is where the Body hosts the Brain.

---

## 1. wubuwizard (THE BRAIN) — /home/wubu/wubuwizard

### 1.1 The layers

| Path | Role | Contents |
|---|---|---|
| `src/` | the ENGINE | **1,167 C modules / 37 CUDA / 420 headers** — every algorithm, every kernel, every data structure. Opaque structs, minimal includes, pure C11. |
| `include/` | the API | Public headers — opaque types + function decls |
| `tools/` | CLI + tests | CLI tools (`gen_text`, `wubu_cli`, `bench_*`) and test drivers |
| `research/` | the paper library | **93 research notes** (`001-…` to `093-…`) — each with Triple-DA, implementation status, Kevin-Bacon 7-hop convergence. `INDEX.md` = the ledger. |
| `THEORY/` | our own papers | **50 theory docs** — foundational philosophy, axiomatic emergent theory, KV-cache filesystem, ecosystem of spheres, scale-to-fit, revolver doctrine |
| `MATH/` | the proof vault | `lean/wubu_proofs/` — Lean-verified theorems (Poincaré ball, Möbius, gyration) |
| `docs/` | the brain's docs | Architecture, topology, model blueprint, model card, improvement plans, ADRs |
| `models/` | local weights | WuBu seed (safetensors + tokenizer), reference checkpoints |
| `manifests/` | model configs | Bigger-brother line configs |

### 1.2 The engine module clusters

The `src/wubu_*.c` modules cluster by theme:

| Cluster | Purpose | Key modules |
|---------|---------|-------------|
| **Attention** | Multi-head, latent, cross, gated, ring | `wubu_attn_kernels`, `wubu_mla`, `wubu_attn_gate`, `wubu_attn_tune`, `wubu_ring_attn`, `wubu_cross_attn` |
| **KV Cache** | Compression, tiering, eviction, paging, coherence | `wubu_kv_evict`, `wubu_kv_compress`, `wubu_kv_tier`, `wubu_kv_quant`, `wubu_paged_kv`, `wubu_4kv`, `wubu_kv_coherence_diag` |
| **MoE** | Mixture of experts, routing, grouping, hyperbolic | `wubu_moe`, `wubu_moe_grouped`, `wubu_moe_hyperbolic`, `wubu_latentmoe`, `wubu_ssd_moe`, `wubu_hashrouter`, `wubu_expert_choice` |
| **SSM** | State space models, scan, recurrence, chunked | `wubu_ssm_scan`, `wubu_ssm_recurrence`, `wubu_nested_ssm`, `wubu_ssm_chunked`, `wubu_ssm_delta` |
| **Speculative** | Speculative decoding, Medusa, tuning | `wubu_spec_decode`, `wubu_spec_tuner`, `wubu_spec_variants`, `wubu_medusa`, `wubu_mtp` |
| **Quantization** | AWQ, GPTQ, SmoothQuant, NF4, FP8, MXFP4, AWQ | `wubu_awq`, `wubu_gptq`, `wubu_smoothquant`, `wubu_nf4`, `wubu_fp8`, `wubu_mxfp4`, `dequant_iq2_xxs`, `quantized_matmul` |
| **Hyperbolic/Nesting** | Poincaré ball, Möbius, RSGD, ecosystem | `wubu_hyper`, `wubu_nest`, `wubu_mobius_linear`, `wubu_poincare_gqa`, `rsgd`, `wubu_ecosystem` |
| **Model Core** | Seed model, training, backprop, tokenizer | `wubu`, `wubu_train`, `wubu_backprop`, `wubu_model`, `wubu_gemma4_model`, `wubu_tokenizer_hf` |
| **AGI Organs** | Hive memory, prover, AGI loop, indexer | `wubu_hive`, `wubu_prover2`, `wubu_agi`, `wubu_dsa`, `wubu_deltanet`, `wubu_mhc` |
| **Agentic OS** | Agent KV, agent memory, agent auth | `wubu_agentic_kv`, `wubu_agentic_mem`, `wubu_agentic_os`, `wubu_agentauth`, `wubu_agentid` |
| **Encoders** | Image, audio, video, PDF, office | `wubu_imgenc`, `wubu_audio`, `wubu_jpeg`, `wubu_png`, `wubu_pdf`, `wubu_video` |
| **RL** | PPO, GRPO, actor-critic, bandits | `wubu_ppo`, `wubu_traj_grpo`, `wubu_actor_critic`, `wubu_bandit`, `wubu_reinforce` |

### 1.3 The AGI brain pipeline

```
corpus (HF datasets, user files, SD card)
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

## 2. wubuos (THE BODY) — /home/wubu/wubuos

### 2.1 The layers

| Path | Role | Contents |
|---|---|---|
| `src/kernel/` | the KERNEL | Memory, tasking, interrupts, AHCI, FAT32, TXFS, VMM, klog, libc, 20+ hardware drivers |
| `src/firmware/` | WuBuFW | UEFI from scratch: PCI, NVMe, AHCI, GOP, TPM, secureboot, chainloader |
| `src/compiler/` | HolyC compiler | WuBuNOS submodule (lexer, parser, codegen, 11 ISA backends) |
| `src/brain/` | wubuwizard | The Brain submodule (inference, training, encoders) |
| `src/jit/` | JIT engine | x86-64 encoder, regalloc, minic expression compiler |
| `src/runtime/` | Runtime | Styx/9P, VSL, containers, Arch, network, DOS emulator, archd, holyd |
| `src/gui/` | Windowing | Win98/XP chrome, theme engine, rendering |
| `src/apps/` | GUI apps | Editor, canvas, calc, notepad, cmd, music, todo, notes, explorer |
| `src/audio/` | Audio | DAW, Furnace tracker, TinySoundFont, AI plugins |
| `src/bear/` | RL world | Cartpole physics, GAAD training |
| `src/hosted/` | Hosted leg | DRM/KMS, Vulkan, Metal, main entry |
| `src/bridge/` | VSL bridge | Syscall bridge, DOS flip |
| `src/worldsim/` | World sim | GAAD world state, physics, terrain |
| `tools/` | Tools | Benchmarks, dev utilities, ISA tests, research |
| `docs/` | Docs | ADRs, compendium, research, reference |

### 2.2 The boot chain (the verified spine)

```
WuBuFW (src/firmware) measures the kernel
  -> PCR4 + AuthentiCode (TPM)
  -> chainloader reads KERNEL.ELF off the ESP
  -> SHA-256 -> attestation handoff in low RAM
  -> ExitBootServices -> crt0 -> kernel_main
  -> AGI supervisor with the root-of-trust gate LIVE
     (verified: make test_agi_metal = PASS)
```

### 2.3 The kernel's AGI organs

| Module | Role |
|---|---|
| `wubu_recovery` | The 5+1 rollback (five slots + the Jesus state) — mistakes are safe |
| `wubu_psych` | The HX-A user model + HX-B adaptive timing |
| `wubu_tutor` | HX-C learning/education |
| `wubu_hive` | The AGI's memory, kernel-side |
| `wubu_bonzi_study` | The study daemon (bonzi assistant) |

### 2.4 Hardware driver registry

| Driver class | Modules |
|--------------|---------|
| GPU | `wubu_drv_gpu`, `wubu_navi10`, `wubu_nvidia_*`, `wubu_radeon_*`, `wubu_ampere` |
| Storage | `wubu_nvme_gen4`, `wubu_nvme_gen5`, `wubu_ahci`, `fat32`, `txfs` |
| Network | `wubu_drv_net`, `wubu_wifi7`, `wubu_nicoffload` |
| Audio | `wubu_drv_hda`, `wubu_intel_*` |
| Display | `wubu_drm`, `wubu_drmx`, `wubu_fbcon`, `wubu_backlight` |
| Power | `wubu_drv_battery`, `wubu_power`, `wubu_pm` |
| USB | `wubu_xhci`, `wubu_usb4`, `wubu_uas` |
| Thermal | `wubu_thermal`, `wubu_thermalthrottle` |

---

## 3. wubunos (THE COMPILER) — /home/wubu/wubunos

### 3.1 The compilation pipeline

```
HolyC Source
     │
     ▼
┌─────────────────────────────────────────┐
│  FRONTEND  (holyc_*.c/h)                │
│  Lexer → Parser → AST → MIR Emitter     │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  MID-LEVEL IR  (wubu_mir*.c/h)          │
│  3-address code, virtual registers      │
│  Optimizer: fold, strength, DCE, CSE,   │
│    LICM, unroll, combine                │
│  Register allocation: linear-scan SSA   │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  BACKENDS  (wubu_isa_*.c)               │
│  11 ISA drivers consuming the same MIR  │
│  Native JITs: x86-64, ARM64, PTX        │
│  Interpreters: RISC-V, MIPS, 68k,       │
│    8086, 6502, Z80, 8051, AVR          │
└─────────────────────────────────────────┘
```

### 3.2 File map

| Path | Purpose |
|------|---------|
| `holyc_*.c/h` | HolyC frontend (lexer, parser, codegen) |
| `wubu_mir*.c/h` | Mid-level IR + optimizer |
| `wubu_isa_*.c` | 11 ISA backends |
| `wubu_preproc.c/h` | C preprocessor |
| `x86_peephole.c` | x86-64 peephole optimizer |
| `brainfuck.c` | Brainfuck → x86-64 JIT (proof) |
| `test_isa_driver.c` | Differential ISA tests |

---

## 4. Integration points

### How wubuos integrates both repos

```
wubuos/
  src/brain/     → git submodule → wubuwizard (THE BRAIN)
  src/compiler/  → git submodule → wubunos (THE COMPILER)
  src/jit/       → native JIT (shared with wubunos encoders)
  src/bear/      → RL training (shared with wubuwizard algorithms)
```

### Data flow

```
wubuwizard (Brain)                    wubuos (Body)
  ├─ trained weights ──────────────►  src/brain/ loads for inference
  ├─ encoder space ───────────────►  Styx namespace (/wubu/enc)
  ├─ KV cache ────────────────────►  KV filesystem (src/kvfs)
  └─ research ────────────────────►  docs/research/

wubunos (Compiler)                    wubuos (Body)
  ├─ compiled HolyC ──────────────►  ring-0 execution on kernel
  ├─ ISA backends ────────────────►  JIT engine (src/jit)
  └─ self-hosting proof ──────────►  test gate (make test_holyc)
```

---

## 5. Project statistics (verified 2026-08-15)

| Metric | wubuos | wubuwizard | wubunos | TOTAL |
|--------|--------|------------|---------|-------|
| C files | 2,463 | 1,167 | 33 | 3,663 |
| H files | 1,006 | 420 | 16 | 1,442 |
| CUDA kernels | 0 | 37 | 0 | 37 |
| Total LOC | 472,955 | 218,100 | 14,115 | 705,170 |
| Test targets | 414 | ~150 | 1 | 565+ |
| Research docs | 60 | 93 | 0 | 153 |
| Theory papers | 0 | 50 | 0 | 50 |
| ISA backends | — | — | 11 | 11 |
