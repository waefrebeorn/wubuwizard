# AGENT PROMPT LOCK (post-AN54 J53)

> 2026-08-09. The standing instruction for any agent entering this
> repo. Read this BEFORE touching code. It prevents re-implementing
> wired work.

## The default agent instruction

**INDEX AN27-AN54 are done. The closed-loop colony is shipped and
tested (15+ gates green, `make all` = 0 errors). The multi-hour run
owns the machine when active. Work ONLY items from the post-AN54
backlog (INDEX AN55+ / this file), and close them in the backlog
order — do not re-open AN27-AN54 as greenfield work.**

## What is DONE (do not rebuild — it is tested code)

- AN27-AN39: the colony organs (diagnosis closed loop, hive walk,
  orchestrator, RLHF oracle, Colonel, recovery, RSI engine, lineage,
  traj cells, dual-timescale diagnose, capgate, contracts)
- AN40-AN46: the phases (closed loop default, harness, skills, tool
  registry, blueprint, priority store)
- AN47-AN54: endurance + glue (checkpoint-restart, release gate,
  GEMV dispatch, live-file harness, skill→train, colonel effect,
  policy recorder, hive merge, Body handoff)
- A1/A2/A3/A7: the run recorder, report generator, time-series,
  replay verifier (the post-mortem stack)

## The current machine state

- The multi-hour endurance run may be LIVE (wubu_endurance). Do not
  kill it casually; it owns the CPU. Work on analysis tools + docs
  while it runs.
- The run writes three artifacts: `.hive`, `.prio`, `.events.jsonl`.
  Any telemetry work reads those — never invent a parallel log.

## What is OPEN (the post-AN54 backlog — work these)

- A5 kill-resume chaos script · A6 resource ledger
- B harness depth (multi-file userfs, codec stress, adversarial pack,
  long-horizon goal, curriculum schedule, golden baseline)
- C skills & training (quality decay, negative skills, composition,
  drain metrics, dedup, on/off-policy tags, MoE bias)
- D precision/bandwidth/mobile (ladder families, bandwidth oracle
  cell, thermal fitness, KV-tier policy, scale-to-fit audit, A/B)
- E federation (signed bundles, partial merge, conflict report,
  colony identity, async merge queue)
- F Body boundary (fake-Body test, cap attenuation, durable effect
  journal, sandbox policy matrix, colonel-under-harness)
- G safety/release (contracts pack, expansion governance, nightly
  gate, ship-bit policy, policy graveyard, NaN tripwire)
- H orchestration (lens priors, critique veto audit, plan DAGs,
  budgeted search, self-play)
- J docs discipline (this lock + the manual + pillars refresh)

## The non-negotiables (still in force)

1. No stubs. 2. Opaque structs at module seams. 3. Minimal includes.
4. No third party. 5. C11 strictness. 6. Verify before claiming.
7. Research discipline: a gap is only `wired` when it ships tested
   code. 8. ONE branch (`wubu-integration`); fetch+rebase before push.

## The attack order (when starting fresh)

1. `make test_colony` (the release gate) must be GREEN before and
   after your work.
2. Do the J items (docs) first — they are cheap and prevent the next
   agent from re-discovering what exists.
3. Then the A items (telemetry — read the artifacts, don't invent).
4. Heavy integration (B/C/D) only after the multi-hour run returns
   data.
