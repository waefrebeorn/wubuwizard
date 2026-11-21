# wubuwizard Mind Palace Goal Mantra — Inkling 2 Mini (Nov 21, 2026)

## THE GOAL

Local, open-weights inference for looped MoE transformers. Target: **Inkling 2 Mini**
(26.4B / 3.4B active, 3× middle-layer loop, NVFP4 + MXFP8 tail, ~20 GB, agentic coding).
Verify looped decode against a reference on local hardware.

**Authoritative goal doc:** `../inkling-2-mini-goal.md`

## STATE

| Metric | Value | Status |
|--------|-------|--------|
| SOTA LP-01..LP-16 | 16/16 gated (`docs/WORKLIST.md`) | Shipped |
| Inkling P0 reorient | Goal doc + both boot mantras + archive | Done |
| Inkling P1..P7 | Probe / loop / format / MoE / context / monitor / agents | Open |

## THE LOOP

read goal → pick lowest undone phase → implement one gate → run gate →
update `state.md` + `WORKLIST.md` → update palace → push → loop

Exit gate: Claim → Verify → Risk → Mitigate. No ceremony in responses.
`bytropix` is a coding reference and past proof run, not the target.

## FILES

- `../inkling-2-mini-goal.md` — authoritative goal
- `../vault/bytropix-goal-archive.md` — previous goal, preserved
- `state.md`, `plan.md`, `testing.md`, `index.md` — walkway
- `docs/WORKLIST.md`, `docs/SOTA-2026-11-ROADMAP.md` — pad status and prior roadmap
