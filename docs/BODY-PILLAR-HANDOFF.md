# AN47 #9 — The Body Pillar Handoff Pack (Brain → Body contract)

> 2026-08-09. The explicit contract between the Brain (the wubuwizard
> colony) and the Body (WuBuOS: the kernel, the Win98 shell, the
> Styx/9P namespace, Arch containers). The goal: WuBuOS consumes the
> Colonel's requests WITHOUT Brain rework — the message set + the cap
> semantics are frozen here.

## 1. What the Colonel sends (the message set)

The Brain NEVER touches the host. It sends ONE message type through
`wubu_colonel`: the **request** (the `wubu_request_t` struct in
`include/wubu_colonel.h`). The Body-side translation is the only place
real effects happen.

| Field | Meaning | The Body must |
|---|---|---|
| `req_id` | the durable request id | ack with the SAME id (idempotent retry) |
| `kind` | READ / WRITE / RUN / SPAWN / KILL | map to the capability-scoped syscall |
| `path` | the 9P namespace path (LOGICAL, e.g. `/kv/user/...`) | translate to the real path UNDER the subtree |
| `agent_subtree` | the requester's capability boundary | enforce: deny any path outside it |
| `goal_token` | the goal that drove the request | carry it in the traj cell |
| `cell_idx` | the specialist cell that decided | carry it in the traj cell |
| `priority` | 0..3 | schedule accordingly (3 = safety) |
| `cpu_ms/ram_mb/io_kb` | the resource bound (AD04) | overrun = flag, never exceed |
| `attempt` | the retry count | backoff: wait `base * 2^attempt` before retry |

## 2. What the caps MEAN (the capability semantics)

- **Deny-by-default**: nothing is runnable until `wubu_toolreg`
  REGISTERS it. An unregistered tool is denied at the gate (the
  ambient host is never open).
- **The 9P subtree**: a request's `path` must start with the
  requester's `agent_subtree` (`wubu_9p_cap_allowed`). The check runs
  on the LOGICAL namespace path; the Body maps it to the real path.
- **The throttle**: a tool that fails more than `max_fail_rate` of its
  runs is AUTO-BARRED (the metadiag raises the bar on external
  failures). The Body's failures feed `wubu_toolreg_report`.
- **The resource bound**: `wubu_resbound_check` flags overruns (the
  bitmask: 1=cpu 2=ram 4=io). Overrun = the request is killed, never
  exceeded.

## 3. What the Body must ACK (the effect contract)

Every executed request produces:
1. **The traj cell** (via `wubu_traj_record`): goal, steps, outcome,
   cost — the hive records what the colony ACTUALLY did.
2. **The outcome report** (`wubu_colonel_report(idx, ok)`): ok=1 →
   `n_done`, ok=0 → `n_failed` + the retry arms (backoff).
3. **The effect itself** (e.g. the userfs write landing in the
   namespace) — the ONLY ambient-visible change, and it is scoped to
   the subtree.

The test `test_colonel_effect` proves the whole chain: unregistered
denied → registered runs → the file lands → traj cells + cap
accounting. WuBuOS's job: implement the Body side of that chain in the
Styx/9P namespace executor (the `wubu_colonel` consumer), keeping the
message set + the cap semantics EXACTLY as above.

## 4. The handoff boundary (what unblocks the Body work)

- **Sandbox/seccomp**: the Brain only speaks `colonel + caps` — the
  sandbox wraps the Body-side executor, NOT the Brain. No Brain
  rework.
- **Metal boot**: the colony's loop (diagnosis → mutate → validate →
  archive) runs in ring-0 hosted by the Live Colonel; the Body's
  namespace executor is the only syscall surface.
- **The priority sidecar** (`.prio`) rides with the checkpoint; the
  Body's crash-recovery reads it to restore the Fisher evidence
  (which lineages the loss cares about).

## 5. The frozen message contract (do not change without a version bump)

```
wubu_colonel_request(kind, path, agent_subtree, goal, cell, prio, cpu, ram, io)
  -> req_id (>0) or -1 (denied at the 9P cap / full)
wubu_colonel_pull()      -> idx or -1 (nothing ready, backoff)
wubu_colonel_report(idx, ok) -> the outcome (traj + ledger)
wubu_colonel_save/load() -> the durable queue (AD03)
wubu_toolreg_register(name, kind, batch)  -> the ONLY enable path
wubu_toolreg_run(name, goal, cell, cost)  -> 1 = allowed (deny-default)
wubu_toolreg_report(name, ok)             -> feeds the throttle
```

The Body implements these against the real namespace; the Brain
already speaks them.
