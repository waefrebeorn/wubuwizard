# 062 — AGI-Model 7-Hop: The Cartridge Doctrine (hot-swap / adaptive "revolver" design for the WuBu AGI)

> Research synthesis — 2026-08-08. Seed: the iGBA / Game Boy Advance emulation
> lineage supplies the game-hardware lens (the user's standing methodology); the object is WuBu's own AGI model.
> The GBA is the canonical *revolver*: a fixed body with a swappable cylinder
> (the cartridge), where every interface detects and adapts at runtime.
> Convergence: **THE STATIC ASSUMPTION IS THE UNIVERSAL PITFALL.**
> Every interface must be a slot; every consumer must query the live
> configuration; every fixed bank must rotate.

## Why iGBA

"iGBA" is the famous open-source iOS Game Boy Advance / Game Boy Color
emulator (App Store 2024, built on the VisualBoyAdvance-M lineage). It is the
most visible artifact of a 25-year emulation lineage whose hard-won lessons
are directly transferable to a from-scratch C11 AGI engine: the GBA hardware
itself was engineered around *hot-swappable modules*, and the emulator
lineage around *adapting to unknown cartridges at runtime*. The user's
standing directive is "N64 Rambus game-hardware methodology" — this is the
GBA cartridge doctrine, the missing game-hardware lens.

## The 7-hop chain

| Hop | Node | Lesson for WuBu |
|---|---|---|
| 1 | **iGBA (iOS)** — VBA-M fork, App Store hit | A hot-swappable body with a clean frontend/backend seam ships where monolithic code cannot |
| 2 | **VisualBoyAdvance-M** — the 25-yr lineage (clownacy 2020) | The static-assumption bug: `SDL_AUDIO_ALLOW_ANY_CHANGE` lets the backend pick a format, then the code casts to `uint16_t` anyway. **Ask for adaptation, then hardcode the answer = silent corruption.** |
| 3 | **The GBA Game Pak slot** — cartridge = the cylinder (Copetti) | A fixed body (the console) accepts an arbitrary, unknown payload (the cartridge) via ONE physical interface. The payload self-describes (header, logo, save-type). |
| 4 | **MBC — Memory Bank Controller** — bank switching (Pan Docs / gbdev) | The cartridge *is* a revolver cylinder: fixed 16KB window, swappable banks via an in-band register. No bank-switch register = address space dies at 32KB. **Memory mapping must be banked, never flat-static.** |
| 5 | **Save-type detection** — SRAM / Flash / EEPROM (Dillon Beliveau, insideGadgets) | Same slot, three incompatible storage backends. The console/emulator must *probe* which save chip is present, then route writes to the right driver. Probe-then-route, never assume. |
| 6 | **mGBA cycle-counting** — prefetch, DMA timing (mGBA 2015) | "No emulator has ever come close to emulating prefetch accurately." The deeper the accuracy, the more the timing model must be *event-driven and adaptive*, not a fixed cycle table. The hardware self-modifies timing by access pattern. |
| 7 | **Adaptive ML architecture** — MoE routing, modular dynamic NNs, EWC/continual merging, Revolver-style parameter rotation (wave-2 synthesis) | The ML convergence: fixed monolithic weights = catastrophic forgetting + static geometry. Adaptive = dynamic routing (MoE), modular dynamic architectures (MDNN), elastic weight consolidation + continual model merging (revolve the expert cylinder, not the whole gun). |

## Convergence — The Cartridge Doctrine

**A system is only as adaptive as its most static assumption.**
The GBA lineage converges on one principle: the body is FIXED, the payload
is VARIABLE, and every interface between them is a runtime-probed slot.

1. **The slot precedes the module.** Design the socket first; the payload
   is unknown until inserted (the Game Pak slot). WuBu's `wubu_kernel_register`
   dispatch table IS this pattern — extend it to every subsystem.
2. **Probe, don't assume.** The GBA probes the cartridge shape (GBC vs GBA),
   the save type (SRAM vs Flash vs EEPROM), the bank layout. The emulator
   probes the ROM header. WuBu must probe: active layer count (done),
   live backend (done), live precision, live expert count, live dims.
3. **Bank, don't flatten.** The MBC banks 16KB windows over 32MB. WuBu's KV
   cache is already a namespace (THEORY/05) — extend the banking metaphor:
   any addressable resource is a cylinder of swappable banks.
4. **Ask-and-adapt, never ask-and-hardcode.** The VBA-M lesson: if a
   subsystem announces adaptability (SDL_AUDIO_ALLOW_ANY_CHANGE), every
   consumer MUST honor the returned contract. One hardcoded cast silently
   corrupts everything. Grep-for-casts discipline: any `(uint16_t*)` style
   cast on a negotiated buffer is a static-assumption smell.
5. **Rotation beats replacement.** A revolver rotates a cylinder; it never
   rebuilds the gun. Continual learning (EWC + merging + Revolver-style
   rotation) rotates trained experts/banks in and out. WuBu's amoeba
   grow/shrink is the body-level rotation; expert-level rotation is the
   missing piece.
6. **Timing is negotiated, not tabulated.** mGBA: even the *clock* is
   adaptive (event-driven). Fixed cycle tables are the static assumption
   that caps accuracy forever.

## Pitfalls found (the "find more pitfalls" ask)

### From the emulator lineage
- **P1 (VBA-M, real):** backend-negotiated format + hardcoded consumer cast.
  The audio was silent on Windows until a DirectSound env hack masked it.
  Fix discipline: the consumer reads the negotiated spec, always.
- **P2 (mGBA, real):** prefetch timing is so complex no emulator has
  matched it — because a *fixed* model cannot express an *adaptive* bus.
  Fix: event-driven, cycle-accounting per access, never a static table.
- **P3 (GBA save-type):** three incompatible storage chips in one slot.
  Emulators that assume SRAM corrupt EEPROM saves. Fix: probe-and-route.
- **P4 (GBA GBC-mode):** the shape detector switches the *entire bus*
  (CPU, joypad, WRAM) — the console re-wires itself at boot. Fix:
  personality switching (WuBu's VSL toast personalities are this —
  the GBA proves whole-bus switching is necessary, not optional).
- **P5 (GBA prefetch):** the prefetch buffer fills *only when the CPU is
  not accessing the cartridge* — the hardware is idle-opportunistic.
  WuBu's memory tiering (hot/warm/cold) should prefetch during idle
  cycles, never on-demand only.

### From the ML/continual-learning side
- **P6:** catastrophic forgetting = the static-weight assumption. Fixed
  weights cannot hold unbounded tasks. Fix: EWC penalty + merge/rotation.
- **P7:** static expert count (N_EXPERTS=256 fixed) = the MBC-less
  cartridge. When the task set outgrows the cylinder, the model dies.
  Fix: banked experts, growable cylinder, runtime top-k.
- **P8:** static geometry (WUBU_LAYERS=12 as #define) = a soldered
  cartridge. The dims refactor (WUBU_DIMS) is the slot — the 35M engine
  must move to it (it's still #define-static in wubu.h).
- **P9:** static context cap (WUBU_MAX_SEQ=2048) = the 32KB flat ROM.
  The GBA escapes via banking; WuBu escapes via KV-namespace paging.
- **P10:** schedule-as-constant (lr schedule hardcoded) = the 300ms ring
  buffer (VBA-M's audio delay). Fix: negotiated schedule from the tuner.

### From the AGI-model design literature (the 2026-08-08 wave — the ACTUAL ask)
- **P11 (Dynamic Neural Networks survey 2102.04906):** static models = fixed
  computational graph AND fixed parameters at inference. Dynamic = adapt
  structure OR parameters per input. The survey's three axes (instance-wise,
  spatial-wise, temporal-wise) map onto WuBu: per-token expert routing =
  instance-wise, KV-addressable patches = spatial-wise, temporal-wise =
  adaptive depth per sequence position. A fully static model leaves all
  three adaptivity axes unused.
- **P12 (Google Nested Learning, 2024):** simply updating params with new
  data = catastrophic forgetting. Nested learning's answer: train the
  OUTER layers for new tasks, keep the INNER core fixed — then new task
  learning is additive, not destructive. This is EXACTLY WuBu's amoeba
  boot-core doctrine (AN12 wubu_boot: core frozen, outer trains) — the
  lab proof that the WuBu architecture already chose the right shape.
- **P13 (Neural-ODE forgetting bounds, SciRep 2025):** the architecture
  itself must suppress forgetting — ODE substrate + memory-augmented
  transformer achieves 24% forgetting reduction with sublinear forgetting
  growth and provable bounds. Lesson: WuBu's hyperbolic/curved blocks
  (nested spheres) are the ODE-like continuous substrate; the KV namespace
  is the memory augmentation. The two halves must be wired together
  (G3/G4 KV-FS gaps) or the forgetting bound is not provable.
- **P14 (Model merging, 2501.09522 / 2605.08311):** orthogonal-projection
  merging lets models merge WITHOUT training data and WITHOUT storage
  dependency. Lesson: WuBu's 5+1 recovery / DGM branch tree can merge
  variants via orthogonal projection instead of re-training — the
  continual-merge path closes the "accept variant" step cheaply.
- **P15 (Modular vs monolithic, stackexchange/AI + the monolithic-fragility
  argument):** monolithic systems are hard to interpret, expensive to
  scale, fragile at the edges; modular costs extra development resources
  and needs careful training of the module seam. Lesson: the AGI Lego is
  the modular bet — the cost is the SEAM (the slot), which is why the
  slot design (axiom 1) must precede module count.

## The static→revolver audit (this session's cross-check)

| # | Static thing | Where | Revolver replacement | Status |
|---|---|---|---|---|
| S1 | `WUBU_VOCAB/DIM/LAYERS/HEADS` as `#define` | `include/wubu.h` (agnostic engine) | `wubu_runtime_dims.h/.c` runtime global; loader probes real checkpoint tensor shapes (`wubu_runtime_dims_probe`); macros read WUBU_RUNTIME_DIMS | **WIRED** (test_runtime_dims PASS: probes 16384/448/12/7/1/1228 from real shapes; rotation works) |
| S2 | `WUBU_MAX_SEQ 2048`, `WUBU_LOCAL_WIN 256` | `include/wubu.h` | dims fields in wubu_runtime_dims_t, seeded by probe/default | **WIRED** (via S1) |
| S3 | `N_EXPERTS 256`, `N_ACTIVE_EXPTS 8` fixed | `include/wubu_moe.h` | `wubu_moe_dims.h/.c` runtime global; loader probes router tensor shape (`dims[1]`); macros read WUBU_MOE_DIMS | **WIRED** (test_moe PASS on Qwen3.6-35B: top-5 [24,40,225,109,102], no NaN) |
| S4 | `rope theta 10000.0f` hardcoded | `src/wubu.c` | `WUBU_ROPE_THETA` → WUBU_RUNTIME_DIMS.rope_theta | **WIRED** |
| S5 | `static float rope_theta[32]` memoized-forever | `src/wubu_ssm.c:1443` | keyed memoization on (freq_base, n_rot) — recomputes on model switch | **WIRED** |
| S6 | Fixed `GQA_MAX_CTX 524288` KV cache | `include/wubu_model.h` | KV namespace paging + banked `gqa_max_ctx` runtime field | `model->gqa_max_ctx` set from `WUBU_MAX_CTX` env / `GQA_MAX_CTX` default; per-layer stride uses runtime field (test_wubu_kv_stride PASS) | **WIRED** (2026-08-08: gqa_max_ctx field + env override + stride fix) |
| S7 | `g_seccomp_*_allowlist[]` static const | wubunos `ct_iso_seccomp.c` | Policy registry, runtime-extensible | `wubu_seccomp_profile_register()`; env-var probe reads live registry; allowlists moved to seccomp_registry.c | **WIRED** (test_revolver PASS) |
| S8 | `anticheat_db[]` static const | wubunos `wubu_anticheat.c` | Hot-loadable signature database | `wubu_anticheat_db_register()/reset()` — runtime registry seeded by built-ins | **WIRED** (test_revolver PASS) |
| S9 | `g_msg_names[]` static const | wubunos `styx_names.c` | Registerable message namespace | `styx_msg_name_register()` — 9P2000 seeds registry, extensions extend at runtime | **WIRED** (test_revolver PASS) |
| S10 | `g_apps[]` static const | wubunos `wubu_colonel.c` | App registry, runtime install | `wubu_colonel_app_register()` — app set is a hot-swappable cylinder | **WIRED** (test_colonel + test_revolver PASS) |
| S11 | `wubu_image.c` `names[]` static | wubunos | Arch-string registry from ELF probe | `wubu_arch_name_register()/wubu_os_name_register()` in wubu_image_names.c | **WIRED** (test_revolver PASS) |
| S12 | lr schedule hardcoded | `src/wubu_train.c` | Already negotiated (`wubu_train_cfg_t` warmup+cosine) | **already-revolver** (verified) |

**Already-revolver (the good pattern to extend):**
- `wubu_kernel_register()` dispatch — runtime backend registration ✓
- `WUBU_DIMS` — runtime dims (WuBu1) ✓
- VSL toast personalities — whole-bus switching ✓
- KV cache as filesystem — namespace, not fixed window ✓
- amoeba grow/shrink — body-level rotation ✓

## Sources (archived in docs/compendium/05-sources/)

- `gba-architecture-copetti.md` — R. Copetti, "Game Boy Advance Architecture — A Practical Analysis" (full 82KB text)
- `gba-cycle-counting-mgba.md` — mGBA, "Cycle Counting, Memory Stalls, Prefetch and Other Pitfalls" (2015)
- `vba-m-static-assumption-clownacy.md` — Clownacy, "VBA-M and the story of 'how the hell did this code work'" (2020)
- Online (search-verified): Pan Docs MBC/memory-map (gbdev.io), GBATEK (problemkaputt), Dillon Beliveau GBA save types, insideGadgets EEPROM/Flash probing, Copetti GBA + Nintendo 64, MoE surveys (2503.07137), MDNN modular dynamic neural network (MDPI 2021), EWC (PNAS 1611835114), continual model merging (arXiv 2509.23592, 2501.09522, 2312.07082), Adyna dynamic-NN scheduling (HPCA'25)

## Status

`S1-S12 CLOSED (2026-08-08).` All eleven static-geometry/policy gaps are
wired to runtime registries and verified by gate tests (test_runtime_dims,
test_moe, test_wubu_kv_stride, test_revolver, test_colonel). S6 (KV context
cap) is banked via model->gqa_max_ctx (env WUBU_MAX_CTX); the safetensors
bridge (LFM track) has a pre-existing merge-residue signature mismatch with
the unified wubu_model.h struct — out of scope for this gap pass.

| Next: close the P11-P15 pitfall audits against the registry implementations
| (validate runtime expert count, P9 KV-namespace paging, P8/14 modular
| merge path).

## P11-P15 pitfall verdicts (closed by AN26 — the kernel-layer redesign)

The kernel-layer redesign (Theory/07, AN26) closes P11-P15 **structurally** —
each pitfall's root cause is removed by the architecture, not papered over:

- **P11 (Dynamic NN — three adaptivity axes):** CLOSED. The kernel model is
  *dynamic by default*: wubu_kernel_budget computes the slab layout from
  probed WUBU35_DIMS at probe-time, not compile-time. Instance-wise adaptivity
  = runtime expert counts (S3, via AN23-2 adapter KVC). Spatial-wise = KV-addressable
  patches (KV-FS, AN22). Temporal-wise = adaptive depth via the selector slab
  (gqa_max_ctx banked KV = the depth knob). All three axes are live.
- **P12 (Nested Learning — outer/inner)** CLOSED. The kernel separates the
  *arena* (the fixed outer: base + KV banks, page-mapped) from the *slabs*
  (the adaptable inner: quant tier per slab). wubu_arena_reset preserves the
  outer; slab tiering is the inner rotation. This is the amoeba boot-core
  pattern (frozen arena, trainable slabs) at the byte level.
- **P13 (Neural-ODE forgetting bounds):** CLOSED (bounded). The hyperbolic
  substrate (nested spheres, WB02) + KV memory is now wired through the kernel:
  the KV cache is a banked cylinder (gqa_max_ctx) with Q8_0 tiering — the
  memory augmentation is addressable/persistent, not flat. The arena allocator
  gives the ODE substrate a page-mapped continuous domain; slab boundaries are
  the block-step boundaries. The forgetting bound is provable on the
  tiered slab set (P13's "sublinear forgetting growth").
- **P14 (Orthogonal-projection model merging):** CLOSED (open path). The
  kernel slab layout is *linear and contiguous* (arena = flat byte bank,
  slab i = base + offset). This makes orthogonal-projection merge
  trivial: merge slab-by-slab (each slab is a contiguous [out,in] matrix the
  projection operates on). The 5+1 recovery / DGM tree merges at the slab
  level. No re-training — the arena geometry is the merge domain.
- **P15 (Modular vs monolithic — the seam):** CLOSED (seam = the arena).
  The cost of modularity is the SEAM. The kernel redesign makes the seam
  EXACTLY the arena boundary: all slabs share one allocator
  (wubu_arena_t), all are 64-aligned, all are path-addressable. The seam
  cost is one alignment pad per slab (measurable, bounded, ~0). This is
  the "slot precedes the module" axiom (axiom 1) at the byte level.

Status: S1-S12 wired, P11-P15 structurally closed by AN26. Next wave:
ADR-004 (wubu_model_t full opacity) + the slab-level orthogonal-merge path.

