# WuBu Project Topology — the master map of BOTH repositories

> 2026-08-03. The user's directive: "start cohesively organizing our
> project between our two repositories." This is the authoritative
> map: what lives where, the layer boundaries, the data flow, and the
> canonical placement rules. It supersedes scattered READMEs when they
> conflict.

```
╔══════════════════════════════════════════════════════════════════╗
║                    THE WUBU UNIVERSE                              ║
║                                                                  ║
║   ┌──────────────────────────┐   ┌──────────────────────────┐    ║
║   │   wubuwizard (THE BRAIN) │   │    wubuos (THE BODY)     │    ║
║   │   inference + training   │   │   kernel + shell + GUI   │    ║
║   │   research + math vault  │   │   firmware + attestation │    ║
║   └──────────┬───────────────┘   └──────────┬───────────────┘    ║
║              │  trained weights,            │                    ║
║              │  model cards, evals          │  Live Colonel      ║
║              ▼                              ▼  (ring-0 REPL)     ║
║   WuBu-35M (HF) ◄──────────────►  WuBuOS metal (measured boot)   ║
║                                                                  ║
║   satellites: BearRL, WuBuContainer, multi-device-os,            ║
║   mythos-fable, reactos-study, gnome-study, mujoco_local,        ║
║   physics, bytropix-*, VulkanShaderCUDA                          ║
╚══════════════════════════════════════════════════════════════════╝
```

## THE ONE-SENTENCE SPLIT

wubuwizard = the BRAIN (everything that thinks: model code,
training, research, math, inference engines, the corpus pipeline).
wubuos = the BODY (everything that acts: the kernel, the shell,
the GUI, the firmware that boots it, the measured-boot chain, the
recovery substrate, the container isolation).
wubunos = the COMPILER (the from-scratch C11 toolchain: HolyC frontend,
MIR optimizer, 14 ISA backends). Built from the BODY via the symlink
`wubuos/src/compiler -> wubunos`.

The Brain trains; the Body acts; the Compiler compiles. The Live Colonel
(ring-0 REPL in wubuos) is where the Body hosts the Brain.

---

## 1. wubuwizard (THE BRAIN) — /home/wubu/wubuwizard

### 1.1 The layers

| Path | Role | Contents |
|---|---|---|
| `src/` + `include/` | the ENGINE | **305 C modules / 21 CUDA / 302 headers** — every algorithm, every kernel, every data structure. Opaque structs, minimal includes, pure C11. |
| `tools/*.c` | the CLI + tests | **553 C tools** (348 `test_<module>.c`) — one test per module, operational CLIs (`wubu_train_cli`, `gen_text*`, `infer_*`), `repodoc/` (doc generator). |
| `tools/*.py` | the harnesses | **87 Python tools** — corpus fetch/extract (`wubu_*`), API clients (`nvidia_nim`, `openrouter_rlhf`), viz. |
| `research/` | the paper library | **45 research notes** (`001-…` to `059-…`) — each with Triple-DA, implementation status (`wired`/gap), ties. `INDEX.md` = the ledger (AN01-AN11). |
| `THEORY/` | our own papers | the WuBu Nesting (層疊嵌套) papers, foundational philosophy, axiomatic emergent theory, `papers/` (DeepSeek lineage, Möbius transformers, …). |
| `MATH/` | the proof vault | `lean/wubu_proofs/` — the Lean-verified theorems (Poincaré ball, Möbius, gyration, MLA compression). |
| `WUBUNEST_V2/` | python training prototypes | the numpy/torch nesting experiments that became the C11 `wubu_nest`. |
| `docs/` | the brain's docs | model blueprint, model card, live-stream/free-API ledger, improvement plans. |
| `vault/` | collected references | api-server notes, quantization formats, bins/tools. |
| `manifests/` | model configs | `Qwen_Qwen3.6-27B`, `Kwaipilot_KAT*`, `InternScience*` — the bigger-brother line. |
| `models/` | local weights | `wubu/` (the WuBu seed: safetensors + tokenizer), the reference checkpoints. |
| `python/` | small helpers | tokenizer extraction etc. |
| `DEMOS/ DRAFT/ DIAGRAMS/` | sketches | the early prototyping (kept for lineage; most logic now lives in `src/`). |

### 1.2 The engine modules (the 305)

The `src/wubu_*.c` modules cluster by theme (the naming convention:
`wubu_<theme>_<thing>.c`):

| Cluster | Modules (representative) |
|---|---|
| **attention** | `wubu_attn_kernels`, `wubu_attn_gate`, `wubu_attn_tune`, `wubu_attnres`, `wubu_cross_attn`, `wubu_mla` (latent KV) |
| **KV cache** (11+) | `wubu_kv_cache`, `wubu_kv_evict`, `wubu_kv_compress`, `wubu_kv_tier`, `wubu_kv_quant`, `wubu_paged_kv`, `wubu_4kv`, `wubu_ring_attn` |
| **MoE** (8+) | `wubu_moe`, `wubu_moe2`, `wubu_moe_grouped`, `wubu_moe_hyperbolic`, `wubu_latentmoe`, `wubu_ssd_moe`, `wubu_hashrouter`, `wubu_expert_choice` |
| **SSM** (4+) | `wubu_ssm_scan`, `wubu_ssm_recurrence`, `wubu_nested_ssm`, `wubu_chunked_ssm` |
| **speculative** (4+) | `wubu_spec_decode`, `wubu_spec_tuner`, `wubu_spec_variants`, `wubu_medusa` |
| **quantization** | `quantized_matmul`, `quantized_dot_generic`, `wubu_awq`, `wubu_gptq`, `wubu_smoothquant`, `wubu_nf4`, `wubu_mxfp4`, `dequant_iq2_xxs`, `wubu_tensor_store` (mixed per-role export), `gguf_reader` (TurboQuant Q2_0/TQ3_1S/TQ4_1S) |
| **hyperbolic/nesting** | `wubu_hyper`, `wubu_nest`, `wubu_mobius_linear`, `wubu_poincare_gqa`, `wubu_hyperbolic_output_proj`, `rsgd` |
| **model core** | `wubu` (the seed), `wubu_train`, `wubu_backprop`, `wubu_model`, `wubu_gemma4`, `wubu_tokenizer_hf` |
| **the AGI organs** | `wubu_hive` (memory), `wubu_moe2` (agents), `wubu_prover2` (verifier), `wubu_agi` (the loop), `wubu_deltanet` (linear mixer), `wubu_dsa` (indexer), `wubu_mhc`/`wubu_mhc_mh` (hyper-connections) |
| **agentic OS** | `wubu_agentic_kv`, `wubu_agentic_mem`, `wubu_agentic_os`, `wubu_agentauth`, `wubu_agentid` |
| **misc** | `wubu_arena`, `wubu_audio`, `wubu_bandit`, `wubu_actor_critic`, `wubu_ecs`, `wubu_hopfield`, `wubu_energy`, `wubu_freeenergy`, `thread_pool`, `tile_manager` |

The machine-generated full table (every module + purpose) is
[docs/MODULES.md](MODULES.md).

### 1.3 The AGI brain pipeline (the flow)

```
corpus (SD card: /home/wubu/sdcard/corpus/)
  ├─ text/        raw Cosmopedia shards (wubu_extract.py)
  ├─ tokens/      .tok uint16 streams (wubu_tokenc C11 BPE)
  ├─ finemath-live.tok / openmath-live.tok   (wubu_stream.py live)
  └─ checkpoints/ seed.st-NNN.st (every 10 steps, the 5+1 slots)

trainer (tools/wubu_train_cli.c + src/wubu_train.c
         + src/wubu_backprop.c)
  └─ WuBu-35M safetensors -> trained .st checkpoints -> HF
       (WaefreBeorn/WuBu-35M, weights + tokenizer + LICENSE + card)

oracles (tools/nvidia_nim.py, tools/openrouter_rlhf.py)
  └─ the RLHF reward: WuBu drafts -> frontier scores -> trainer
```

---

## 2. wubuos (THE BODY) — /home/wubu/wubuos

### 2.1 The layers

| Path | Role | Contents |
|---|---|---|
| `src/kernel/` | the KERNEL | **~90 modules**: boot/crt0, memory, tasking, interrupts (APIC/PIC/PIT), AHCI, FAT32 family (10 modules), TXFS, VMM, SMP, klog, libc, serial, swap, sync, WDT, TSS, vdso, the human HX family (`wubu_psych`, `wubu_tutor`, `wubu_bonzi_study`), the recovery (`wubu_recovery`), the AGI kernel (`wubu_agi_kernel`), the hive port (`wubu_hive`), the math (`wubu_math`). |
| `src/firmware/` | WuBuFW | the UEFI firmware from scratch (no EDK2): fw_* modules (PCI, NVMe, AHCI, XHCI, GOP, TPM, secureboot, sha256, acpi), `fw_agi` + attestation, the chainloader, `wubufw.fd` — **the measured boot chain (28/28 conformance, real kernel boots)**. |
| `src/apps/` | the GUI apps | canvas (full editor), explorer, notepad, calc, regedit, taskmgr, repl, the bonzi/comfy/cmd/control suites, the Tandem shared-desktop window. |
| `src/gui/` | the windowing | Win98/XP chrome, theme engine (`wubu_theme`), rendering. |
| `src/bridge/` | the VSL bridge | the ReactOS NT syscall -> VSL transliteration, the syscall handlers. |
| `src/compiler/` | the HolyC compiler | lexer, parser, codegen, PTX — "My Seed" (the compiler that compiles). |

> **`src/compiler` is a SYMLINK → `/home/wubu/wubunos`**, not a submodule.
> See §3. Edit compiler sources in wubunos; the OS `Makefile` reaches them
> via `$(COMP)=src/compiler`. The `.gitmodules` entry for it was stale and
> is removed.
| `src/runtime/ src/hosted/ src/shell/` | the hosted layer | the scaffold for Linux/Windows/macOS parity, the 9P namespace. |
| `src/worldsim/ src/bear/` | the RL world | cartpole physics, GAAD training, curriculum. |
| `docs/compendium/` | the institutional memory | 00-philosophy, 01-reference (GENERATED by make docs), 02-architecture, 03-learned (the prestige ledger: worked/didn't-work), 04-roadmap, 05-sources. |
| `holyc-include/` `vendor/` `reference/` | the reference | ZealOS headers, upstream comparisons. |

### 2.2 The boot chain (the verified spine)

```
WuBuFW (src/firmware) measures the kernel
  -> PCR4 + AuthentiCode (TPM)
  -> chainloader reads KERNEL.ELF off the ESP
  -> SHA-256 -> attestation handoff in low RAM
  -> ExitBootServices -> crt0 -> kernel_main
  -> AGI supervisor with the root-of-trust gate LIVE
     (verified: make test_agi_metal = PASS, measured boot green)
```

### 2.3 The kernel's AGI organs

| Module | Role |
|---|---|
| `wubu_recovery` | the 5+1 rollback (five slots + the Jesus state) — mistakes are safe |
| `wubu_psych` | the HX-A user model + HX-B adaptive timing |
| `wubu_tutor` | HX-C learning/education |
| `wubu_bonzi_study` | HX-D companion |
| `wubu_agi_kernel` | the AGI supervisor (ring-0, attestation-gated) |
| `wubu_hive` | the hive port (the AGI's memory, kernel-side) |
| `wubu_verifier` | the DA-2 fail-closed verification |
| `wubu_attest` | the root-of-trust attestation |
| `wubu_hid/input` | the human's mouse + keyboard (the human keeps control) |

---

## 3. wubunos (THE COMPILER) — /home/wubu/wubunos

The from-scratch C11 compiler that the Body runs ON the kernel. Branch
`main` on `waefrebeorn/WuBuNOS`. Single source of truth — edit compiler
sources here, never in `wubuos/src/compiler/` (which is a symlink mirror).

| Path | Role | Contents |
|---|---|---|
| `holyc_*.c/h` | FRONTEND | HolyC lexer → parser → AST → MIR emitter |
| `wubu_mir*.c/h` | MID-LEVEL IR | 3-address code, SSA, virtual regs, optimizer passes |
| `wubu_isa_*.c` + `wubu_*_interp.c` | ISA backends | 14 targets: x86-64, ARM64, MIPS, 8051, AVR, PIC, 8086, M68K, 6502, RISC-V, Z80, PTX, AMDGPU, WASM |
| `holyc_elf.c / holyc_pe.c / holyc_bin.c` | binary emitters | ELF64, PE32+, raw (portable byte-level serialization, no `__attribute__`) |
| `wubu_preproc.c` | preprocessor | `#include` / `#define` / `#if` / `##` |
| `wubu_lang_router.c` | dispatch | routes .hc/.c/.bf/.rs/.py/.js/... by policy |
| `test_gauntlet_runner.c`, `isa-test/*`, `peephole_superopt/`, `dev/` | test harness | universal gauntlet (4,450 tests × 14 targets), differential ISA tests, peephole battery |

### 3.1 Linkage into the BODY
```
wubuos/src/compiler  ──symlink──▶  wubunos   (NOT a submodule)
                                       │
                                       ▼  Makefile: $(COMP) = src/compiler
                                make holyc          # builds the compiler driver
                                make gauntlet_runner  # builds the test runner
                                make test_gauntlet  # runs 4,450 × 14 tests
```
The compiler has no standalone Makefile — it compiles from the BODY because
it needs the BODY's runtime/interpreters/9P layer. There is **no `.gitmodules`
entry for `src/compiler`** (it's a symlink; the old entry was stale and removed).

### 3.2 Placement rule for the compiler
A new ISA backend goes in `wubunos/wubu_isa_<name>.c` (+ header, + interp in
`wubunos/` or `wubuos/src/runtime/`). A new emitter goes in `wubunos/holyc_<fmt>.c`.
The Body's `mk/tests.mk` already references all of them via `$(COMP)/…`; do NOT
re-add old `tools/…` paths (those were stranded here during the 2026-08-16
re-org and rehomed).

---

## 4. THE BOUNDARIES (what goes where)

**The Brain owns (wubuwizard only):**
- ALL model code (forward/backward/training), ALL quantization, ALL
  inference engines, the tokenizer, the oracles (NVIDIA/OpenRouter),
  the corpus pipeline, the research, the math proofs.
- The hive lives HERE as the reference implementation (`src/wubu_hive.c`).

**The Compiler owns (wubunos only):**
- The HolyC frontend, MIR IR, 14 ISA backends, ELF/PE/bin emitters, the
  preprocessor, the language router, and the gauntlet harness.
- Edit compiler sources in `/home/wubu/wubunos/` only. The BODY imports
  them via the `src/compiler` symlink (§3.1).

**The Body owns (wubuos only):**
- The kernel, the firmware, the boot chain, the GUI, the shell, the
  container isolation, the recovery substrate.
- The hive lives HERE as the metal port (`src/kernel/wubu_hive.c`) —
  same API, no-heap (the kernel allocator), for the ring-0 brain.

**The bridge (both):**
- `WuBu-35M` weights flow Brain -> HF -> (Body hosts them on metal).
- The Live Colonel (Body, ring-0) loads the Brain's model file.
- The 9P namespace (`/n/kv/`, `/n/models/`) exposes the Brain's state
  to the Body's tools (per WUBUOS_INTEGRATION.md).
- `wubu_agi` (Brain: the learning loop) and `wubu_agi_kernel` (Body:
  the supervisor) are the two halves of the same AGI: the Brain
  learns, the Body protects and acts.

**Satellite repos (context, not core):**
- `BearRL` — RL training experiments (the cartpole GAAD work).
- `WuBuContainer` — container isolation prototypes (now in kernel).
- `multi-device-os`, `mythos-fable` — kernel lineage studies.
- `reactos-study`, `gnome-study` — upstream gap analyses.
- `physics`, `mujoco_local` — physics/RL grounding.
- `bytropix-*` — the bytropix integration work.
- `VulkanShaderCUDA` — the Vulkan compute path.

---

## 4. THE PLACEMENT RULES (canonical)

1. **A new algorithm goes in wubuwizard** `src/wubu_<theme>.c` +
   `include/wubu_<theme>.h` + `tools/test_<theme>.c`. No exceptions.
2. **A new kernel primitive goes in wubuos** `src/kernel/wubu_*.c`.
   If it must also run in the Brain, port it (same API, metal impl).
3. **A new compiler piece goes in wubunos** `wubu_isa_<name>.c`,
   `wubu_mir*.c`, `holyc_<fmt>.c`, etc. Edit in wubunos only; the BODY
   reaches it via `src/compiler` symlink. Never duplicate compiler files
   into `wubuos/tools/`.
4. **Research notes** go in `wubuwizard/research/NNN-name.md` with the
   Triple-DA + `wired`/gap status. Papers go in `THEORY/papers/`.
   Proofs go in `MATH/lean/wubu_proofs/`.
5. **The prestige ledger** (worked/didn't-work) goes in
   `wubuos/docs/compendium/03-learned/`.
6. **Model artifacts** (weights, cards) go on HuggingFace under
   `WaefreBeorn/`; the local copies live in `wubuwizard/models/`.
7. **Corpus data**: ACTIVE working copies live on the SSD at
   `/home/wubu/models/corpus/` (master manifest `CORPUS.md` there:
   Tier 0 pretrain tokens, Tier 1 SFT pack, Tier 2 agentic pack).
   The SD card (`/home/wubu/sdcard/corpus/`) is the COLD raw archive;
   `/home/wubu/sdcard/archive/` holds finalized cold tarballs
   (research ponds, qwen36 embeddings). Never clone git or write
   active work on the SD card (drvfs has no chmod; 256KB clusters).
   Nothing corpus goes in a repo.
8. **Secrets** live in `~/.hermes/profiles/mind-palace/secrets/`
   (0600), NEVER in any repo.
9. **Test binaries** are never committed (gitignore covers `/test_*`).
10. **The research ponds** (701 MB pure text, 7 ponds × 100 MB) are the
    READING substrate — `/home/wubu/research-ponds-work/` (SSD active,
    SD `archive/` cold). PONDS.md is the catalog; grep the ponds for
    the failing subject, sources.json maps file → paper/repo.

## 5. THE AUDIT FINDINGS (2026-08-03, from the full-repo survey)

1. **No topology doc existed** — this file fixes that. The repo roots
   had grown organically; the boundaries were implicit.
2. **The Brain's training core had 3 real gaps** (found by reading
   `src/wubu_train.c` on 2026-08-03) — since CLOSED: `wubu_backprop` (real
   backward) landed, the SFT run completed (loss 8.04 → 7.32 @ step 2000),
   and the Muon path is wired into `wubu_train`/`wubu_train_gpu`.
3. **The Body is healthy**: 468+ C files / 91 test targets / measured
   boot verified / monoliths dissolved. The Brain's `test_*` binaries
   are gitignored correctly.
4. **The hive exists in BOTH repos** — intentional (reference vs metal
   port), now documented as the boundary contract.
5. **Repo topology corrected (2026-08-16):** `wubunos` is a third,
   independent repo (the compiler), NOT a symlink to wubuos.
   `wubuos/src/compiler` is a symlink → wubunos; `.gitmodules` entry for
   it is removed. `wubuos/src/brain` is the real submodule → wubuwizard.
   All skills/docs now reference `wubuwizard/docs/TOPOLOGY.md` as the
   single source of truth instead of hardcoding `/home/wubu/...` paths.

## 6. NEXT ACTIONS (the cohesive path)

1. Wire the DeepSeek-V4 Config-I forward (`wubu_deepseek4.c`: MLA + 256-expert
   MoE + hash router + mHC + DSA from the mapped 1328 tensors — load gate
   already PASSED) and the multi-split data reader.
2. Wire the RLHF oracle rewards (NVIDIA/OpenRouter) into the trainer — the
   Brain's RLHF loop.
3. Port the trained WuBu checkpoints to the Body (Live Colonel loads the
   weights via the 9P namespace) — the Brain→Body bridge.
4. Run `make docs` so `wubuos/docs/compendium/01-reference` regenerates with
   the new modules; re-run `tools/repodoc/repodoc.py` in both repos after
   every code wave.

