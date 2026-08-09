# The Colony Operator Manual (post-AN54 J55)

> 2026-08-09. How to start, kill, resume, hive-walk, and read policy
> reasons for a colony run. The tools: `wubu_endurance` (the run),
> `wubu_report` (the post-mortem), `wubu_timeseries` (the trend),
> `wubu_replay` (the verifier), `wubu_hive_walk` (the raw walk).

## 1. Start a run

```bash
# the multi-hour endurance run (harness + closed loop + events)
./wubu_endurance --rounds 3000 --ckpt 250 --out /home/wubu/colony-runs/run1.st

# a training run with the closed loop as the default (every batch
# diagnoses; the mutation cycle is gated by fitness + contracts +
# lineage + the priority store)
./wubu_train --init-random --tok <corpus.tok> --steps 1000 \
    --seq 128 --diag-every 1 --out /tmp/train.st
```

The run writes THREE artifacts next to `--out`:
- `<out>.hive` — the fitness archive (ledger + graveyard)
- `<out>.prio` — the priority evidence (BI + Fisher + deltas)
- `<out>.events.jsonl` — the event stream (A1: round, loss, suite,
  verdict, policy reason, contract counters, attribution)

## 2. Kill a run

```bash
kill <pid>            # SIGTERM (graceful-ish: the sidecars are saved
                      # at the last checkpoint)
kill -9 <pid>         # SIGKILL — safe: the events are fsynced per
                      # event, the sidecars at the last checkpoint
```

A SIGKILL loses at most one checkpoint interval (the `--ckpt N`
rounds between saves). Nothing else — the event stream is fsynced per
event.

## 3. Resume a run

```bash
./wubu_endurance --rounds 3000 --ckpt 250 \
    --resume /home/wubu/colony-runs/run1.st \
    --out /home/wubu/colony-runs/run1.st
```

The resume loads `.hive` (fitness cells + counters) + `.prio` (the
Fisher evidence) + the events APPEND (the recorder opens in append
mode). The run continues from the restored batch counter — it does
NOT reset. The chaos test (A5) hammers this.

## 4. Walk the hive (raw)

```bash
./wubu_hive_walk <out>.hive --accepted     # the accepted lineage
./wubu_hive_walk <out>.hive --rejected     # the graveyard (negatives)
./wubu_hive_walk <out>.hive --last 32      # the recent window
```

## 5. Read the report (the human summary)

```bash
./wubu_report <out> --out report.md        # headline + lineage top-K
                                           # + graveyard + policy + attr
./wubu_timeseries <out> --plot --csv       # did tasks improve when
                                           # loss did? (the correlation)
./wubu_replay <out>                        # did the offline decisions
                                           # match the live run?
```

## 6. Read the policy reasons

The metadiag's slow path writes a policy meta-cell every time it
moves the mutation rate / floor, with the reason code:

| code | meaning |
|---|---|
| 1 | loss rising → the colony gets MORE aggressive |
| 2 | suite failing → the colony gets MORE aggressive (task-first) |
| 3 | improving → the colony RELAXES (rate down, floor up) |

The events carry the reason per round; the report shows the policy
change table. The hive itself holds the policy meta-cells (queryable
via the walk when the live hive is exposed).

## 7. The artifacts are the whole history

There are NO ad-hoc logs. A run is fully reproducible from the three
artifacts:
- replay the events (`wubu_replay`) → the decision history
- walk the hive (`wubu_hive_walk`) → the fitness + the negatives
- report + timeseries → the human-facing summary

## 8. The release gate

```bash
make test_colony    # the "no ship without it" gate
```

harness floor + zero contract violations + the priority sidecar
present + the blueprint bounds intact. Run it before shipping
anything.
