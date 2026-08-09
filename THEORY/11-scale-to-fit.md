# THEORY/11 — SCALE-TO-FIT: one checkpoint, any hardware

> **The user's directive (2026-08-08):** "we're gonna need to make a scale
> on a ton of different hardware all the way down to the CM4 raspberry pi
> and all the way up to supercomputers."
>
> WuBu is the alive homogeneous amoeba: the design has NO fixed number.
> The number is the incarnation (today: the 56.4M WuBu1 seed), not the
> design. The design's lever is a RATIO (active/total params), and the
> absolute size is decided at LOAD TIME by probing the machine.

## 1. The one-sentence doctrine

**A single checkpoint, a single code path, an infinite hardware range:
the loader probes the machine (RAM, SIMD, cores, accelerator), classifies
it into a tier, and computes a load plan — how many core layers, how many
ecosystem balls, what precision cascade, what active ratio — such that
the plan fits the budget and the model boots on ANY of them.**

The model never asks "what size am I?" — it asks "what are you?" and
becomes the biggest thing that fits.

## 2. The hardware tiers (the probe, not a table)

Tiers are computed from the probe, never hardcoded as a static policy
table (Revolver discipline 3). The classification:

| Tier | RAM budget | Typical machine | What loads |
|---|---|---|---|
| `TINY`  | ≤ 512 MB | CM4 / RPi zero-class | boot core only (wubu_boot: ~6 layers, Q8/F32), ecosystem N=16, K=1–2 |
| `SMALL` | ≤ 4 GB  | RPi 5 / small SBC / old laptop | boot core + trunk of the fractal tree, N=64, K=4 |
| `MID`   | ≤ 32 GB | laptop / workstation | full WuBu1 seed (56.4M) + N=256 ecosystem, F16/Q8 cascade |
| `BIG`   | ≤ 1 TB  | server / node | full seed + all fractal layers + N=huge ecosystem, precision cascade to Q2_K leaves |
| `HUGE`  | > 1 TB  | supercomputer | everything + training-grow enabled (the body trains outward) |

The probe reads: available RAM (`wubu_mem_budget`), SIMD ladder
(`wubu_hwcaps`), CPU core count, accelerator presence (CUDA/Vulkan —
`/dev/dxg`, `nvcc`, driver check). Each feeds the plan.

## 3. The load plan (what "scale-to-fit" computes)

```
wubu_scale_plan_t {
    tier               (computed, not assumed)
    ram_budget         (bytes, probed or injected)
    core_layers        (boot core depth — wubu_boot's top-k)
    ecosystem_n        (colony size — how many balls fit)
    k_active           (balls fired per token — the ratio lever)
    precision_cascade  (F32 trunk → F16 → Q8 → Q4 → Q2 as we go out)
    weight_bytes       (bytes the plan needs for weights)
    kv_bytes           (bytes reserved for the KV namespace)
    headroom_bytes     (budget - weight - kv; must be > 0)
    ratio_active       (k_active / ecosystem_n — must be ≤ target)

    -- RESEARCH/063 axes (the 7-hop mandate) --
    bytes_per_token    (bandwidth cost: core + k_active ball params)
    watts_estimate     (avg draw under load — EnerInfer energy axis)
    energy_class       (0 low ... 3 high — the thermal budget)
    n_devices_used     (CPU-only, or CPU+NPU/GPU split — QEIL routing)
    adaptive_depth     (1 = fractal_depth is a ceiling; the runtime
                        early-exits easy tokens — PALBERT)
}
```

The planner is O(1) closed-form: given the checkpoint's geometry (probed
from tensor shapes — Revolver), it solves for the largest ecosystem N and
the deepest core that fit `ram_budget - kv_reserve`, then derives the
bandwidth, energy, and device-routing axes from the same geometry.
Research basis: research/063-scale-to-fit-7hop.md (QEIL inference-time
scaling laws, Prima.cpp heterogeneous clusters, OHQ on-chip quant,
PALBERT early exit, PROBE expert prefetch, EnerInfer energy slack).

## 4. The invariants (the gate, `test_scale_to_fit`)

The SAME checkpoint runs through simulated budgets for each tier and must
satisfy, for ALL tiers:

1. **BOOTS**: the plan loads ≥ 1 core layer + embeddings + final norm —
   the ring-0 brain exists on every tier. (CM4 is not a toy; it boots.)
2. **FITS**: `weight_bytes + kv_bytes ≤ ram_budget` (never OOM — the
   mem-budget lesson).
3. **SMALL-ACTIVE**: `k_active / ecosystem_n ≤ target_ratio` (0.125) on
   every tier — the huge-total stays cheap on the Pi too.
4. **GROWS**: `ecosystem_n(TIER+1) > ecosystem_n(TIER)` — bigger machines
   get bigger bodies, monotonically. No tier is a cliff.
5. **SAME CODE**: one `wubu_scale_plan()` call produces all five plans —
   the planner is the same function on every machine.

## 5. The load path (how the pieces compose)

```
wubu_scale_plan(hw_probe, checkpoint_meta)
        │
        ├─ boot core depth  → wubu_boot (top-k layers by wubu_bi)
        ├─ fractal depth    → wubu_model_scalable (ram_budget → depth)
        ├─ ecosystem N      → wubu_ecosystem (N, K from remaining budget)
        └─ precision        → the cascade (F32 core → Q2 leaves)

load:
  wubu_model_load (existing loader, probed dims)
    + wubu_ecosystem_init(N, K, dim)          (the body)
    + wubu_boot_image (ring-0 core)           (the spine)
    + wubu_router (slot: which physics routes) (the trigger)
```

Every piece exists TODAY: `wubu_boot`, `wubu_model_scalable`,
`wubu_ecosystem`, `wubu_router`, `wubu_mem_budget`, `wubu_hwcaps`, the
probed-dims loader. What's missing is the PLANNER that composes them —
`wubu_scale`. This document IS the spec for that module.

## 6. What we need to make everything (the build list)

| Piece | Status | Owner |
|---|---|---|
| `wubu_scale` planner (probe → plan) | ✅ shipped | THEORY/11 |
| `test_scale_to_fit` gate (5 invariants × 5 tiers) | ✅ shipped | THEORY/11 |
| bandwidth/energy/devices/adaptive axes | ✅ shipped | research/063 |
| `wubu_scale_measure` (on-chip cascade calibration) | ✅ shipped | 063-F |
| router prefetch (next-likely balls) | ✅ shipped | 063-E |
| amoeba shrink operators (depth + width + train-state) | ✅ shipped | AM01 |
| `wubu_encoder` slot (modality-agnostic base seam) | ✅ shipped | WB06/AN06 |
| `wubu_encoder_impl` (OUR OWN encoders: imgenc CC01 + audio CC02 into the shared space — made, not imported) | ✅ shipped | WB06 |
| `wubu_boot` (boot-core extractor) | ✅ shipped | AN12 |
| `wubu_model_scalable` (fractal tree + budget depth) | ✅ shipped | AN23 |
| `wubu_ecosystem` (colony, opaque) | ✅ shipped | THEORY/10 |
| `wubu_router` (the slot) | ✅ shipped | Revolver |
| `wubu_mem_budget` (RAM probe) | ✅ shipped | engine |
| `wubu_hwcaps` (SIMD ladder) | ✅ shipped | engine |
| probed-dims loader (`WUBU_DIMS`) | ✅ shipped | WuBu1 |
| CM4-leg: armv7/arm64 cross-build of the engine | ❌ later | follow-up |
| accelerator probe (CUDA/Vulkan presence) | ❌ later | follow-up |
| real embedding table for the text path (into the encoder slot) | ❌ later | follow-up |

## 7. Triple-DA (the honest audit)

**What the labs prove:** memory-bounded inference (llama.cpp tiering,
MoE active-parameter economics, paged KV) is the standard playbook —
scaling by budget is not exotic.

**What is OUR claim (carries own risk):** the SAME checkpoint, without
re-quantizing per machine, can serve a 256MB Pi AND a 1TB node from one
file. The dequant-on-demand path (wubu_poincare_gqa's `attn_*_weight_q`
blobs) is what makes this real — weights stay quantized in the file and
materialize on demand. The risk: cold-start latency on the Pi (dequant
on first touch). Mitigation: boot core is Q8/F32 (no dequant), the
cascade only affects outer layers.

**The single biggest risk: under-training** (the standing doctrine) —
a Pi-booted WuBu is a tiny model; it must not be judged as a full AGI.
The scale story is about the SAME BODY at different sizes, and the body
grows with data (the ponds).

## 8. The gate line

```
make test_scale_to_fit   # one checkpoint → 5 tiers → 5×5 invariants
```
