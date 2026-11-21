# wubuwizard Goal — Everyman Inference (2026-11-21)

## THE GOAL

Build the best local inference engine for hardware people actually own:

- **6 GB VRAM**
- **16 GB system RAM**
- **SSD** for weight streaming

Purpose: **the everyman can AI.** Capable local inference on an ordinary machine,
not a datacenter. Every design decision is subordinate to that budget.

## THIS IS NOT A llama.cpp CLONE

GGUF is a file format we read. It is not an architecture to imitate and llama.cpp is not
a template. No vendored ggml, no copied kernel bodies, no parity-as-goal. We reimplement
from primary research in C11 and own every line. Reference outputs are for verification
only — never for design.

## ADAPTABILITY IS THE POINT

- **Model agnostic.** No model is hard-coded into the core. Architecture probes read GGUF
  metadata; family differences are compiled rows with byte-identical kernel bodies (LP-10).
  Gemma, Qwen, Inkling and whatever comes next are configurations, not forks.
- **CUDA/AMD agnostic, eventually.** CUDA is one backend, not the engine. Keep a backend
  abstraction with CPU always working; add HIP/ROCm behind the same dispatch table. Never
  let a CUDA-only intrinsic leak into a model kernel.
- **Different tiers.** The engine runs across hardware tiers, and each tier has a model
  quant and a gate. See `docs/TIERS.md`. Tier 1 is 6 GB VRAM / 16 GB RAM / SSD; higher
  tiers add VRAM and RAM, never a different codebase.
- **Acquire and learn all inference engines.** Study upstream implementations and primary
  papers, then reimplement the useful parts in our own C11. Acquisition includes weights
  and knowledge; the goal is understanding, not copying.
- **Organize all code.** Every module has one owner, one header seam, one gate. See
  `docs/CODE-ORGANIZATION.md`. AGI adapts because the structure adapts.

## WHAT WE ARE

wubuwizard is the **brain and math engine**. It aggregates the best inference techniques
from the whole WuBu family into one C11 core:

| Source | What it contributes | Status |
|--------|--------------------|--------|
| wubuwizard | Canonical C11 engine: GGUF reader, Gemma4/MoE/SSM kernels, KV spine, quant dequant, spec decode | Active |
| bytropix | Retired. Its palace artifacts are absorbed here; its C style is a reference, not a branch | Merged |
| slermes | Persistent `/goal` loop, evidence-first judge, host-side control plane | Host layer (LP-01..04) |
| big-mac | DSP/SIMD audio kernel discipline, W4A16 sampler ideas | Reference |
| Wua (Lua) | Scripting/embedding layer for host orchestration | Planned interface |
| WuBuOS | Host OS: memory hive + wubukernel; the machine wubuwizard runs inside | Platform |
| WuBuHIVE | Memory hive — tag heap, budget, frame, TLS, atomic, pool | Platform memory layer |
| WuBuKernel | Kernel primitives the brain calls | Platform compute layer |

**Eventual shape:** wubuwizard becomes "all OS" — the brain running the math for training
inside WuBuOS, with the memory hive and wubukernel beneath it. Training math and inference
math share one engine; there is no separate training stack.

## HARDWARE CONTRACT

| Resource | Budget | Consequence |
|----------|--------|-------------|
| VRAM | 6 GB | Weights hot in VRAM only up to ~4.5 GB; rest streams from RAM/SSD |
| RAM | 16 GB | Paged KV, quantized KV, no full-model host copy unless streaming requires it |
| SSD | present | Sequential weight streaming with prefetch; random I/O is the enemy |
| Precision | NVFP4 / MXFP8 / IQ2 / Q4 | Loop-aware activation counts; tail layers keep higher precision |

Model targets (in priority order, all gated by the hardware contract):

1. **Gemma 4 12B IQ2_M** — fits the 6 GB class; primary correctness target.
2. **Qwen 3.6 35B A3B IQ2_M** — MoE, low active params; KV spine target.
3. **Inkling 2 Mini** — 26.4B total / 3.4B active, 3× loop, NVFP4 + MXFP8 tail, ~20 GB
   footprint. Does **not** fit 6 GB VRAM; it is the streaming/looping stretch target and
   the reason SSD weight streaming and loop-aware activation accounting exist.
   Vendor-reported details preserved in `vault/inkling-2-mini-target.md`.

## C CODE DISCIPLINE

- C11, no stubs, no placeholder kernels, no vendored third-party libraries.
- Opaque structs at module seams; minimal includes; runtime kernel dispatch stays runtime.
- Compile-time family specialization where it keeps kernel bodies byte-identical.
- Every phase ships a real named `make test_*` gate; a doc edit is not SHIPPED.
- Windows build path stays green (MSVC + WinSDK); static linkage where needed.
- Master the C, do not route around it with wrappers.

## ENGINE AGGREGATION PLAN

See `docs/ENGINE-AGGREGATION.md` for the concrete module map, ownership, and gate list.
Rule: existing 16 LP pads (LP-01..LP-16, all gated) are the base layer. New work extends
them; it does not fork them.

## PHASE PLANE

| Phase | Deliverable | Gate |
|-------|-------------|------|
| P0 — Reorient | This goal, engine map, archived old goals | Files present + links resolve |
| P1 — Budget | 6 GB VRAM / 16 GB RAM / SSD budget enforced in allocator | `make test_budget_6gb` |
| P2 — Stream | Sequential SSD weight streaming with prefetch, no random I/O | `make test_ssd_stream` |
| P3 — Loop | Runtime middle-block loop executor (1..10 passes) | `make test_loop_depth` |
| P4 — Format | NVFP4 + MXFP8 tail decode | `make test_nvfp4_weights` |
| P5 — Hive | wubuwizard ↔ WuBuHIVE / wubukernel interface | `make test_hive_interface` |
| P6 — Train | Shared math path for training steps | `make test_training_math` |
| P7 — Everyman | End-to-end on 6 GB / 16 GB / SSD: load, decode, agentic step | `make test_everyman` |

Dependency: P1 → P2 → {P3, P4} → P5 → P6 → P7. No later phase before its gate passes.

## THE LOOP

```
read goal → pick lowest undone phase → implement one gate → run gate →
update state + WORKLIST → update palace → push → loop
```

Exit gate on every decision: Claim → Verify → Risk → Mitigate.
No ceremony in responses. A handoff is a claim, not evidence.

## FILES

- `.hermes/GOAL.md` — this file, authoritative
- `.hermes/goal-mantra.md`, `.hermes/mind-palace/goal-mantra.md` — boot pointers
- `.hermes/vault/bytropix-goal-archive.md` — old CPU-parity goal
- `.hermes/vault/inkling-2-mini-target.md` — Inkling model research note
- `docs/ENGINE-AGGREGATION.md` — aggregation map
- `docs/WORKLIST.md` — shipped pad status
