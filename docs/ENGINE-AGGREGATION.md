# Engine Aggregation — wubuwizard as the single C11 inference core

## Rule

One engine. wubuwizard owns inference and training math. Other repos are either host layers
(WuBuOS, slermes), reference material (big-mac, bytropix history), or interfaces to define
(Wua). Nothing is vendored; techniques are reimplemented in C11 and gated here.

## Ownership map

| Component | Lives in | Role | Interface to wubuwizard |
|-----------|----------|------|-------------------------|
| Inference core | wubuwizard `src/`, `include/` | Model load, quant decode, KV, sampling, spec decode | Itself |
| Goal / judge | slermes host layer, LP-01..04 | Persistent goal, evidence-first judge, drift, resample | `wubu_goal_t` host module (engine stays ignorant of the goal) |
| Memory hive | WuBuHIVE | Tag heap, budget, frame, TLS, atomic, pool | `wubu_hive_*` allocator callbacks |
| Kernel primitives | WuBuKernel | Low-level scheduler/dispatch | `wubu_kernel_*` dispatch table |
| Host OS | WuBuOS | Process/thread/IO, SSD streaming syscalls | OS abstraction layer |
| Scripting | Wua (Lua 5.4) | Orchestration, agent loops | Embed API, host-owned |
| DSP ideas | big-mac | SIMD audio kernels, sampler discipline | Read-only reference |
| Retired C style | bytropix (deleted) | Past proof runs, IQ2_XS dequant lineage | Palace artifacts absorbed, no branch |

## Layering

```
   agent / goal (slermes-style host, LP-01..04)
                    |
   Wua scripting  ---+-->  wubuwizard C11 engine  <-->  WuBuOS host
                    |          |      |      |
                    |     quant    KV    loop   |
                    |          |      |      |
                    +--- WuBuHIVE (memory) + WuBuKernel (dispatch)
```

The engine does not know what goal it serves. Goal, state and slerm logic stay in the host
module. This is why LP-01 (persistent goal + evidence-first judge) was the first pad.

## What is already aggregated (16 LP pads)

- Wave 1, host: LP-01 goal + judge, LP-02 drift steering, LP-03 verifier resampling,
  LP-04 failure-pattern bandit.
- Wave 2, KV spine: LP-05 all-or-nothing paged claim, LP-06 SWA reclaim invariant,
  LP-07 confidence-budgeted speculation, LP-08 seeded determinism, LP-09 constrained decode.
- Wave 3, kernels/formats: LP-10 family specialization, LP-11 MX/NVFP4 KV reconstruction,
  LP-12 tolerance KV fetch, LP-13 diffusion sampler, LP-14 measured-cost TTC,
  LP-15 per-shape SIMD quant decode, LP-16 state/ABI semver.

All 16 have a named `make test_*` gate. See `docs/WORKLIST.md`.

## What must be added for the everyman target

| Need | Why | Where |
|------|-----|-------|
| 6 GB VRAM budget enforcement | defines the whole design | allocator + P1 gate |
| SSD sequential weight streaming | Inkling 2 Mini is ~20 GB, cannot fit VRAM | P2 gate |
| Loop executor (1..10) | Inkling loops middle layers 3×; research variant to 10 | P3 gate |
| NVFP4 + MXFP8 tail decode | native Inkling precision | P4 gate |
| WuBuHIVE / WuBuKernel interface | brain runs inside the OS | P5 gate |
| Shared training math | wubuwizard becomes the training brain | P6 gate |
| End-to-end 6/16/SSD proof | everyman claim must be gated | P7 gate |

## Non-goals

- Not a llama.cpp fork, clone, or parity project.
- No vendored third-party libraries or copied kernel bodies.
- No separate training stack.
- No new global config file replacing per-project markdown palaces.
- No un-gated phase.

## Source of truth

`.hermes/GOAL.md` for direction. `docs/WORKLIST.md` for shipped status.
Vendor model claims are preserved in `.hermes/vault/inkling-2-mini-target.md` and are
reported, not independently benchmarked.

## Navigation for the next session

- `docs/ENGINE-GAP-REGISTER.md` — ordered TODO: 100 gaps; Q3/Q4 2026 code-mined reports `front-h/i/j`; gate names, dependencies, next action
- `docs/ENGINE-SOURCE-INDEX.md` — engine catalog with exact upstream file/symbol pointers
- `docs/research/engine-sweeps/` — seven cited reports: serving, edge, offload, time-bounded
  recent gains, second-degree traversal, new engines, Q3 papers, plus local audits

Rule: a technique is absorbed only when it has a named gate and a measured run. Research
notes and "port complete" claims do not count.
