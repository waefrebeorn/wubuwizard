# G39 — Contract Expansion Governance (post-AN54 design)

> 2026-08-09. How the contract set EVOLVES without weakening the
> floor. The rule: an expansion requires DUAL EVIDENCE — a suite
> gain + no regression on the golden-suite hash.

## The problem

The runtime contracts (AN39 + the G38 operational pack) are the
colony's floor. But the set is not static: the metadiag may want to
tighten EXP when the math drifts, or raise the TOOLFAILS ceiling when
the harness demands more tool use. If ANY code can bump a bound
freely, the floor decays silently — a mutation that "wins loss" by
suspending the ball contract would pass, and the colony would be
worse with a green gate.

## The governance rule (the dual evidence)

A contract expansion (any `wubu_contracts_add` that CHANGES a bound
or installs a new kind) must carry:

1. **A suite gain**: the harness suite score AFTER the expansion's
   window is >= the score BEFORE (the expansion helped or was
   neutral — never a regression).
2. **No golden-suite regression**: the FROZEN golden suite (a fixed
   task set with a versioned hash, B10) still passes at >= its
   recorded score. The golden suite is the fixed point the colony
   must never lose; the live harness is the moving frontier.

Both are checked over the same window, and the expansion meta-cell
(the hive record) carries the evidence:

```c
/* the expansion governance record (a hive meta-cell) */
typedef struct {
    uint32_t contract_version;   /* the version bump */
    uint32_t kind;
    float    old_bound, new_bound;
    float    suite_before, suite_after;    /* the suite gain evidence */
    float    golden_before, golden_after;  /* the golden regression evidence */
    uint8_t  accepted;                     /* 1 = both checks passed */
    uint64_t batch;
} wubu_governance_cell_t;
```

## The enforcement points

- **The diagnose gate**: when the metadiag proposes a contract
  change, the change is staged (NOT applied) until the window's dual
  evidence lands. The gate consults the governance record: accepted
  changes apply; rejected changes go to the POLICY GRAVEYARD (G44 —
  queryable "almost raised the rate" records).
- **The release gate** (`test_colony`): the golden-suite hash is part
  of the gate. A run whose golden score regressed cannot ship, even
  if the live suite improved (the live suite can game the current
  tasks; the golden suite cannot).
- **The merge**: a contract expansion does not cross the hive merge
  unless its governance record is present (the merge refuses
  ungoverned bounds from the other checkpoint — E44's signed bundle
  carries the governance cells).

## The acceptance matrix

| suite gain | golden regression | verdict |
|---|---|---|
| + | none | ACCEPT (the expansion is a real improvement) |
| + | regression | REJECT (the frontier moved but the floor broke — the expansion is suspect) |
| 0 | none | ACCEPT with caution (neutral — keep the version history) |
| - | none | REJECT (the expansion hurt — roll back) |
| any | regression | ALWAYS REJECT (the floor is the floor) |

## The implementation order

1. `wubu_golden.c`: the frozen golden suite (fixed tasks + a version
   hash — B10's cross-run baseline, checked into the blueprint).
2. `wubu_governance.c`: the staged expansion + the dual-evidence
   check + the governance meta-cell (the struct above).
3. Wire the diagnose gate: the metadiag's contract proposals go
   through the governance stage.
4. Wire `test_colony` + the merge to consult the governance records.

## The non-negotiable

The contract set NEVER changes silently. Every version bump carries
its evidence, and the floor survives the frontier. A green gate on a
weakened floor is the failure mode this design exists to prevent.
