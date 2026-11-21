# wubuwizard Goal Mantra — Inkling 2 Mini (Nov 21, 2026)

## THE GOAL

Make wubuwizard a local, open-weights inference engine for looped MoE transformers.
First target: **Inkling 2 Mini** — 26.4B total / 3.4B active, 3× middle-layer loop,
native NVFP4 with MXFP8 tail layers, ~20 GB footprint. Verify looped decode against a
reference implementation on local hardware.

**Full goal, source, gap map and phase gates:** `.hermes/inkling-2-mini-goal.md`

## STATE

| Metric | Value | Status |
|--------|-------|--------|
| SOTA LP-01..LP-16 pads | 16/16 gated (see `docs/WORKLIST.md`) | Shipped |
| Inkling phase P0 (reorient) | This file + full goal doc + archived old goal | Done |
| Inkling phases P1..P7 | Architecture probe → loop → NVFP4 → MoE → context → monitor → agents | Open |

## THE LOOP

read goal → pick lowest undone phase → implement one gate → run gate →
update `state.md` + `WORKLIST.md` → update palace → push → loop

Exit gate on every decision: Claim → Verify → Risk → Mitigate.
A handoff is a claim, not evidence.

## FILES

- `.hermes/inkling-2-mini-goal.md` — authoritative goal
- `.hermes/vault/bytropix-goal-archive.md` — previous goal, preserved
- `docs/WORKLIST.md` — shipped pad status
- `docs/SOTA-2026-11-ROADMAP.md` — prior lilypad set
