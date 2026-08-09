# THEORY/06 — The Revolver Doctrine: Adaptive Hot-Swap Design for the AGI Lego

> The design philosophy for WuBu's AGI Lego: a fixed body, a swappable
> cylinder, runtime-probed slots everywhere. Derived from the iGBA/GBA
> lineage (research/062). Supersedes nothing — it COMPLETES the amoeba
> philosophy with the interface doctrine the amoeba was missing.

## The one-sentence doctrine

**WuBu is a Lego body on a revolver chassis: every subsystem is a slot,
every payload is a swappable cartridge, and every consumer probes the live
contract at runtime — because THE STATIC ASSUMPTION IS THE UNIVERSAL PITFALL.**

## The three axioms (from the GBA)

1. **The slot precedes the module.** You cannot swap what has no socket.
   Design the interface FIRST (the Game Pak slot), the payload second.
   A subsystem with no registration point cannot grow, shrink, or rotate.
   → Every WuBu module gets a `*_register()` / dispatch-slot seam.
   The kernel dispatch table (wubu_kernel_register) is the reference.

2. **Probe, don't assume.** The GBA probes the cartridge (shape, save
   chip, bank layout); the emulator probes the ROM header. Nothing about
   the payload is assumed. WuBu probes: active layers (done), backend
   (done), dims (done on WuBu1), precision, expert count, save format.
   → Any hardcoded constant that describes the WORLD (not the code) is a
   static assumption to be probed.

3. **Ask-and-adapt, never ask-and-hardcode.** The VBA-M lesson: negotiate
   a contract, then honor the RETURNED contract everywhere. One hardcoded
   cast silently corrupts the negotiated format.
   → A consumer that casts a negotiated buffer to a fixed type is a bug.
   Grep for casts on `*_spec`, `*_cfg`, `*_negotiated` buffers.

## The revolver vocabulary (the AGI Lego pieces)

| Piece | What it is | Revolver form |
|---|---|---|
| **The body (frame)** | the fixed system — kernel, dispatch, the amoeba loop | never rebuilt; the cylinder rotates inside it |
| **The cylinder (revolver)** | the swappable set — experts, layers, personalities, backends | rotates a fixed NUMBER of chambers; each chamber = one loaded module |
| **The cartridge (payload)** | one module — an expert, a personality, a policy DB | self-describes on insertion (header, probe, contract) |
| **The slot (socket)** | the registration seam — `*_register()`, dispatch table, mount point | exists before any payload; unpopulated slots are legal |
| **The trigger (router)** | the selection mechanism — top-k, softmax, priority, hash | probes the live cylinder state each pull |
| **The hammer (gate)** | the validation — prover, loss gate, safety kernel | fails closed on unknown cartridge |

## The static→revolver conversion checklist (every subsystem)

Run this over ANY module before calling it done:

- [ ] **Geometry**: are dims/layers/heads `#define` or runtime (`WUBU_DIMS`)? → CLOSED S1-S5 (wubu_runtime_dims.h/.c, wubu_moe_dims.h/.c)
- [ ] **Capacity**: are counts fixed (N_EXPERTS, WUBU_MAX_SEQ) or banked? → CLOSED S2-S3, S6 (gqa_max_ctx banked)
- [ ] **Config**: is the schedule/tuning a constant or a negotiated object? → CLOSED S12 (negotiated wubu_train_cfg_t)
- [ ] **Backends**: is there a `*_register()` slot, or a hardcoded if-chain? → CLOSED S7-S11 (seccomp, anticheat, styx, colonel, image registries)
- [ ] **Data formats**: are save/format/type constants probed or assumed? → S5 (rope keyed memoization)
- [ ] **Tables**: are policy/allowlist/name tables static or registry? → CLOSED S7-S11
- [ ] **Timing**: is the clock a fixed table or event-driven adaptive? → OPEN (mGBA prefetch lesson: next wave)
- [ ] **The cast test**: does any consumer cast a negotiated buffer? → CLOSED S4 (rope theta via macro)
- [ ] **The empty-slot test**: does the module work with a slot unfilled? → registry seed lazy-init + overwrite-by-name (all S7-S11)

## How this fits the amoeba (the integration)

- The amoeba MUTATE step (grow/shrink) IS body-level rotation: the body
  swaps layers/experts in and out of the cylinder.
- The routing doctrine (everything is a routing problem) IS the trigger:
  every route probes the live cylinder.
- The KV namespace (THEORY/05) IS the banking: addressable space is a
  cylinder of swappable banks, not a flat fixed window.
- Continual learning (EWC + merge + Revolver-style rotation) IS the
  expert-level rotation: trained chambers rotate, the gun never rebuilds.
- The VSL toast personalities ARE the GBA GBC-mode bus switching: whole
  buses re-wire at personality switch, exactly like the shape detector.

## The five disciplines (the standing rules)

1. **No fixed geometry.** dims are runtime (WUBU_DIMS); a `#define` for
   model shape is a violation.
2. **No fixed capacity.** counts are banked (MBC-style); a flat
   N_EXPERTS=256 with no bank register dies when the task set grows.
3. **No static policy tables.** allowlists/anticheat/names are registries;
   a `static const` policy array is a soldered cartridge.
4. **No hardcoded schedule.** the lr/anneal schedule is a negotiated
   object from the tuner; a constant schedule is a 300ms ring buffer.
5. **No cast on negotiated buffers.** the consumer reads the returned
   contract; a `(uint16_t*)`-style cast on a negotiated spec is a bug.

## References

- research/062-igba-hotswap-revolver-7hop.md (the 7-hop + the audit)
- docs/compendium/05-sources/gba-architecture-copetti.md (the GBA full text)
- docs/compendium/05-sources/gba-cycle-counting-mgba.md (adaptive timing)
- docs/compendium/05-sources/vba-m-static-assumption-clownacy.md (the bug)
- THEORY/05-kv-cache-filesystem.md (banking = namespace)
- THEORY/03-wubu-nesting-paper.md (the spheres the cylinder lives in)
- docs/wubu-amoeba-design.md (the body the cylinder rotates in)
