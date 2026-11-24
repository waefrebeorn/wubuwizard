# Diagnostic coding principles

1. **One mechanism, one gate.** A named `test_eg*` target must fail when the mechanism is absent.
2. **Evidence before claim.** Exact command, printed measurements and exit code go in the register.
3. **Model-agnostic core.** Read metadata; do not hard-code a model family in a math kernel.
4. **Proxy labels.** Synthetic byte counts, hashes and distributions are named as proxies. A C11 policy gate never claims wall-clock speedup.
5. **Real benchmark logs.** Device, build, model, quant, flags, repetitions, raw stdout and mean/std/CV are required for tok/s claims.
6. **Fail closed.** Invalid shapes, NULL buffers, impossible budgets and unsupported formats return -1/0, never undefined state.
7. **No vendoring.** Clones under `~/WuBu/research/engine-sources/` are study-only. Read, reimplement in C11, gate.
8. **Deterministic fixtures.** Use hashed PRNGs and recompute hand-derived constants before blaming the implementation.
9. **Stress new gates 30 times.** One lucky run hides alignment/state faults.
10. **Full-suite tally per wave.** Rebuild and run every `test_eg*` target in one shell loop and record PASS/FAIL.

See `docs/BENCHMARK-WORKFLOW.md` for the measurement protocol and `docs/DEVILS-ADVOCATE-LEGO-LOOP.md` for the current evidence review.
