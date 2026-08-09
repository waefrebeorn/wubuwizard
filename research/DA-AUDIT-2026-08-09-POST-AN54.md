# Triple-DA Audit — the post-AN54 wave + the run-data fixes (2026-08-09)

> The 3-pass adversarial audit (per research-driven-improvement's
> protocol) over the telemetry stack, the endurance runner, the
> colony-core fixes, and the Body/contracts work. Every finding is
> either FIXED (real code) or VERIFIED-not-a-bug. Research grounding
> from parallel waves: early-stopping patience (ICLR 2025 +
> practice guides), DB WAL group-commit (SQLite/Postgres), crash-
> consistency fsync+rename (0xkiire, arXiv 2511.18323),
> spurious-regression differencing (Afyouni 2019, Granger/Newbold),
> MoE dead-expert pruning.

## DA PASS 1 — Correctness / Logic (does the code do what it claims?)

### 1.1 [FIXED — the KILL] The endurance runner's "learning" was CIRCULAR
The runner computed BOTH the task correctness
(`cap = 0.62 + 0.006 * n_accepted`) AND the loss
(`10 - 0.35*log1p(n_accepted+1)`) as DIRECT functions of the accept
counter. The suite "climbed to 0.918" because the runner scripted it.
The A3 `corr(suite, loss) = -0.937` measured two series derived from
the same counter — a phony learning claim dressed as real.
FIX: correctness now comes from the SKILL STORE (a real colony
artifact: accepted mutations create skills, tasks match them); the
loss follows the real suite. The honest run: suite 0.511 -> 0.836
through REAL skill accumulation, replay-verified with 0 divergences.

### 1.2 [FIXED — the hidden wiring bug the honesty fix exposed]
The runner created skills for goal tokens 100..106 but the harness's
real tasks use {11,12,13,21,22,31,32} — the skills NEVER matched, the
suite was flat 0.561 forever. The DA honesty rewrite (1.1) forced the
real tokens to surface. FIX: use `harness.suite[t].goal_token`.

### 1.3 [FIXED — type-confusion, same bug class as the max_cells fix]
The amoeba's diagnose/mutate walked the WHOLE shared tissue, casting
every cell (skills, traj, fitness, meta) to `wubu_amoeba_cell_t*` and
reading `grad_norm` at garbage offsets. Cells that LOOKED dead were
ERASED from the hive — the run's skills vanished. FIX: classify only
the amoeba's OWN registry cells (the `&am->cells[i]` membership check
in diagnose pass 1+2 and mutate's mean-grad + grow/die passes).

### 1.4 [VERIFIED — not a bug] The amoeba validate's held-out rule
The live gate accepts a FLAT loss (only rejects on worsening > 0.05);
the replay verifier's rule now matches exactly (3000/3000 + the new
run 500/500, 0 divergences). The earlier "1 divergence" was the seed
round (no previous loss) — expected, now documented.

## DA PASS 2 — Integration / Soundness (is it wired in, honestly?)

### 2.1 [FIXED] The "colony learns" claim's integrity
The old run's headline numbers (accepted 2999, suite 0.918) were real
but MEANINGLESS — they proved the organs execute under a scripted
learning curve, not that the colony learns. The honest runner now
proves a REAL causal chain end-to-end: task outcomes -> per-cell
grads -> amoeba mutate -> validate -> accept -> skill -> task
correctness -> suite -> loss. Suite 0.836 with 2978/3500 real passes.

### 2.2 [FIXED] The deadlock the honest wiring exposed
With max_cells == the seed count (8), the colony is at capacity from
round 1 -> mutate can never grow -> permanent stasis -> 0 accepts.
The honest run hit this (0 accepted for 500 rounds). FIX: the runner
uses max_cells=16 (room to grow). The LIBRARY is correct; the runner's
configuration was self-defeating.

### 2.3 [VERIFIED] The release gate + merge still pass with the
type-confined amoeba (25/25 gates green after all fixes).

## DA PASS 3 — Robustness / Production (what input breaks it?)

### 3.1 [FIXED] Checkpoint saves were NOT crash-consistent
`wubu_diag_save` wrote directly to the archive path: a SIGKILL
mid-save left a half-written .hive that the resume loaded as garbage
(wrong magic -> corrupted cells). The chaos test passed only because
kills landed between saves. FIX: the universal atomic pattern —
write .tmp + fsync + rename + fsync the directory (0xkiire
crash-consistency, arXiv 2511.18323). The load now VERIFIES the
full reads (short read = refused, not garbage).

### 3.2 [FIXED] Events recorder fsynced PER EVENT (the naive pattern)
SQLite/Postgres group-commit research: batching + one fsync per batch
is the standard; per-event fsync is ~5ms each (thousands of events =
seconds of pure fsync). FIX: `wubu_events_open_batch(ev, path, 64)`
— one fsync per 64 events; a kill loses at most 64 trailing TELEMETRY
events (the .hive/.prio sidecars are the authoritative state).

### 3.3 [FIXED] Metadiag stasis was a SINGLE-SNAPSHOT test
Early-stopping research (ICLR 2025 + practice): a single flat
observation is noise; the standard is a PATIENCE WINDOW. The old band
held the policy after ONE flat trend. FIX: `stasis_window = 3`
consecutive flat slow passes before the policy holds; any real trend
resets the counter.

### 3.4 [FIXED] The timeseries correlation was the spurious-regression
class. Correlating autocorrelated LEVELS inflates significance
(Afyouni 2019 effective-dof; Granger/Newbold). FIX: report BOTH the
levels correlation AND the FIRST-DIFFERENCES correlation. Verified:
the honest run survives differencing (-0.949); the old circular run
collapses (-0.236) — the differenced method catches the artifact.

### 3.5 [VERIFIED] The skill_match pointer-as-int64 index
`wubu_skill_match` returns the cell pointer cast to int64. Verified
stable (the hive keeps slot pointers stable; the find_cell lookup by
pointer works; standalone probe + the full runner agree). Not a bug.

## The corrected claim (what the endurance runner now proves)

The multi-hour run proves: (1) the closed loop executes for thousands
of rounds without a reset; (2) the colony learns REAL tasks through
the skill store (suite 0.511 -> 0.836, 85% pass rate, replay-verified);
(3) a SIGKILL mid-run loses at most one checkpoint interval and the
archive is never corrupted; (4) the metadiag's policy changes are
patience-gated + explainable. The run NO LONGER claims a scripted
learning curve — the numbers come from real colony state.

## Sources
- Early stopping patience: ICLR 2025 instance-dependent early
  stopping; MCP Analytics early-stopping guide (relative thresholds +
  patience).
- Group commit: SQLite WAL docs, eatonphil "a write-ahead log is not
  a universal part of durability" (fsync + group commit), Medium
  WAL-engineering (5ms-per-fsync vs 5ms-per-100).
- Crash consistency: 0xkiire "Crash Consistency: fsync(), rename()",
  arXiv 2511.18323 (macOS/APFS checkpoint study — the same pattern).
- Spurious regression: Afyouni et al. 2019 (effective degrees of
  freedom of Pearson's r), Granger/Newbold (levels correlation),
  stats.stackexchange "How to use Pearson correlation correctly with
  time series".
- MoE dead experts: arXiv 2604.23036 (expert importance), aman.ai MoE
  primer (dead-expert prevention).
- Skill decay: PMC6825451 (skills decay without maintenance);
  arXiv 2502.03752 (SISL self-improving skills).
