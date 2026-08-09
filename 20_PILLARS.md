# 20-Pillar Architecture → wubuwizard / WuBuOS / slermes Mapping

> 2026-08-09 refresh (Phase 0 truth sync). The INDEX is the source of
> truth; AN27–AN39 (the AGI colony organs) changed Pillars 8/10/11/13/
> 17/20. Pillar 8 is no longer Missing — the orchestrator + trajectory
> cells + capability gate are the planner path.

## Completed / Strong (✅)
| # | Pillar | Status | Code |
|---|--------|--------|------|
| 1 | Pure-C inference engine | ✅ Complete | wubuwizard: `src/wubu_model.c`, `src/wubu_ssm.c`, `src/wubu_moe.c`, quant matmul, GGUF+safetensors readers |
| 3 | Native agent runtime (no Python) | ✅ Complete | slermes: pure C11 agent binary, `src/agent/`, Telegram/Discord/Slack/Signal/WhatsApp/Matrix gateways |
| 7 | Persistent structured memory | ✅ Complete | wubuwizard: the HIVE is the sole long-term memory substrate (`src/wubu_hive.c`) + slermes `src/tools/memory.c` |
| 8 | Hierarchical planning / multi-agent orchestration | ✅ Implemented | wubuwizard: `wubu_orch_*` in `src/wubu_agi.c` (goal decomposer -> 4 specialist lens cells code/math/tool/critique -> judge merge with critique VETO), `src/wubu_trajcell.c` (tool trajectories), `src/wubu_capgate.c` (capability-gated spawning). All hive-native. `make test_orch test_trajcell test_capgate` |
| 10 | Error recovery & self-critique | ✅ Complete | wubuwizard: `wubu_diag_recover` (failed generations -> graveyard + shrink pressure + immediate mutation cycle), `wubu_repetition.c` (repeat_penalty + DRY), `wubu_contracts.c` (runtime floor) |
| 19 | Pure-C integration bus | ✅ Complete | wubuwizard: `include/wubu_*.h` opaque-struct API, `src/wubu_*.c` modules |

## In Progress (🔄)
| # | Pillar | Status | Code |
|---|--------|--------|------|
| 2 | Custom OS kernel/RT for long-running intelligence | 🔄 Partial | WuBuOS runs as hosted binary on Linux; ZealOS kernel exists but not booting metal on this box |
| 4 | Document/productivity engine | 🔄 Stub | WuBuOffice not started in this checkout |
| 5 | Code editor / dev environment | 🔄 Stub | WuBuPad not started in this checkout |
| 6 | Unified tool surface | 🔄 Partial | wubuwizard: the Colonel channel (`src/wubu_colonel.c`) + the 9P capability caps (`wubu_agentic_os.c`); slermes: 40+ tools |
| 11 | Continuous evaluation / benchmarking | 🔄 Partial→Progressing | wubuwizard: the dual-timescale diagnose (`src/wubu_metadiag.c`) treats the loss trend + task-harness signal as fitness; the multi-hour task suite is Phase 3 of the current wave |
| 12 | Resource & stability management | 🔄 Partial | wubuwizard: `src/wubu_affinity.c`, `src/wubu_arena.c`, `src/wubu_capgate.c` (headroom gating), ds4-ssd slot-bank |
| 13 | Sandboxing / permission model | 🔄 Partial→Progressing | wubuwizard: the Colonel's deny-by-default 9P capability boundary (a request outside the agent subtree is DENIED at enqueue); full container isolation stays in WuBuOS (cgroups + seccomp-bpf) |
| 14 | Multi-platform human interface | 🔄 Partial | slermes: CLI + TUI + Wayland/X11/Win32/macOS GUI; WuBuOS: Win98/XP shell |
| 15 | Web access / external world | 🔄 Partial→Progressing | wubuwizard: the Colonel requests carry the capability path; minimal external-world tools behind the cap surface are Phase 5 of the current wave (no full-host shell) |
| 16 | Continuous autonomy (scheduling) | ✅ Implemented | slermes: `src/cron/scheduler.c`, `cronjob` tool in Hermes |
| 17 | Skill creation / self-improvement curriculum | 🔄 Partial→Progressing | wubuwizard: tool-trajectory cells (`src/wubu_trajcell.c`) + capability gaps (`src/wubu_capgate.c`) are the skill-curriculum input; versioned skill cells in the hive are Phase 4 of the current wave |
| 18 | Geometric/math research foundations | 🔄 Research | `ENCODERS/`, `THEORY/` (Poincaré, GAAD, DFT/DCT) + the Lean-backed hyperbolic floor (`src/wubu_contracts.c` runtime form) |
| 20 | Sustained autonomous productivity demo | 🔄 Progressing | the closed loop (AN27–AN39) IS the running colony: diagnose -> mutate -> validate -> archive/graveyard -> delegate -> oracle -> Body action -> recover. The multi-hour unattended harness is Phase 3 |

## Key Gaps to Close (in priority order — the current wave)
1. **The multi-hour unattended harness RUN** (Phase 3's done-definition): a real multi-hour run that improves the suite score via accepted mutations, with lineage + graveyard explaining why.
2. **The Body side of the Colonel**: WuBuOS consumes the requests (the Styx/9P namespace executor for wubu_colonel); the tool registry's registered tools get real implementations behind the cap surface.
3. **Metal**: the colony runs in ring-0 (the Live Colonel hosts it; WuBuOS boots on metal, not just WSL-hosted).
4. **The live hive walk**: walk the LIVE hive (not just the saved archive).
5. **Pillar 2 — OS kernel**: WuBuOS hosted binary works; needs metal boot on WSL for full control.
6. **Phases 1-6 are WIRED** (2026-08-09): the closed loop is the default training path (AN40), the priority store gates mutations on Fisher evidence (AN46), the autonomy harness's suite score is first-class fitness (AN41), the skill curriculum learns from experience (AN42), the deny-by-default tool registry bounds agency (AN43), and the executable blueprint refuses off-lineage mutations (AN44).

## WSL as Agnostic Accelerator Design
WSL2 on this machine is the current compute substrate:
- 6 P-cores pinned (core0=0), F16 KV-cache, llvmpipe Vulkan
- ~13 GB RAM available to the process
- RTX 4050 via `/dev/dxg` (CUDA works: 241 matmuls/step on cuBLAS, 19× vs CPU)
- The `wubu_affinity.c` pins the engine to P-cores; `gpu_wubu.cu` is the CUDA surface; `wubu_foldmath.h` has the AVX2 kernels (sincos8 10.7×, exp8 14.6×)

Next step: `wubu_accel.c` / `wubu_accel.h` — abstract the compute backend (CPU AVX2, CUDA, Vulkan) behind a uniform interface so every route through the same accelerator surface.

## WuBuOS AGI Design Orientation
WuBuOS = ZealOS kernel + Win98 shell + Styx/9P namespace + Arch containers.
The 20 pillars map directly to WuBuOS subsystems:
- Pillars 1-3 → `src/bear/` (RL training) + inference engine (wubuwizard)
- Pillar 2 → ZealOS kernel in WuBuOS (boot on metal, not yet WSL-hosted)
- Pillar 4 → WuBuOffice (C11 OOXML/ODF/PDF)
- Pillar 5 → WuBuPad (piece-table editor)
- Pillar 6 → Styx/9P tool surface
- Pillars 7-10 → the hive memory + the colony organs (diagnosis/recovery/contracts) + session recovery
- Pillars 11-13 → the Colonel capability boundary + WuBuOS container isolation (cgroups + seccomp-bpf)
- Pillars 14-16 → Slermes TUI/CLI + cron scheduler
- Pillars 17-20 → the skill curriculum + the integration goal across all repos
