# WORKLIST — SOTA 2026-10/11 pads

> Execution view of `docs/SOTA-2026-11-ROADMAP.md`. Pads are dependency-ordered.
> Status reflects a **real gated test**, not intent: SHIPPED / PARTIAL / TODO.

| Pad | Size | Depends on | Best source | Gate | Status |
|---|---|---|---|---|---|
| LP-01 Persistent goal + evidence-first judge | M | — | slermes `/goal` docs + Sauron + self-report leakage | `make test_goal` | SHIPPED |
| LP-02 Fast/slow goal-drift steering | S | LP-01 | goal-drift steering paper | `make test_goal_steer` | SHIPPED |
| LP-03 Bounded verifier-guided resampling | M | LP-01/02 | VGSC + SafeInferCom | `make test_goal_resample` | SHIPPED |
| LP-04 Failure-pattern harness bandit | L | LP-03 | ActiveSaddler / MESH-Harness | `make test_harness_bandit` | SHIPPED |
| LP-05 All-or-nothing paged KV claim | M | — | kvcached v0.1.6 | `make test_kv_alloc_exact` | SHIPPED |
| LP-06 Shared-cell SWA reclamation | M | LP-05 | llama.cpp PR #32309 | `make test_swa_reclaim` | SHIPPED |
| LP-07 Confidence-budgeted speculation | M | LP-05 | DeepInfer DSpark + SGLang v0.5.23 | `make test_spec_budget` | SHIPPED |
| LP-08 Deterministic seeded speculation | S | LP-07 | SGLang v0.5.23 | `make test_spec_determinism` | SHIPPED |
| LP-09 Constrained decode + speculation | L | LP-07/08 | Strata PR #4222 | `make test_spec_constrained` | SHIPPED |
| LP-10 Compile-time family specialization | L | — | Splash PR #573 | `make test_kernel_identity` | SHIPPED |
| LP-11 KV reconstruction in compute format | L | LP-05 | MX/NVFP4 KV + CALFA-KV | `make test_kv_qfmt` | SHIPPED |
| LP-12 Tolerance-parameterised 9P KV fetch | M | LP-05/11 | VerKV | `make test_kv_fetch_eps` | SHIPPED |
| LP-13 Fused diffusion logits/sampler | M | — | W4A16 diffusion sampler paper | `make test_diffusion_sampler` | SHIPPED |
| LP-14 Measured-cost test-time compute | M | LP-01/03 | TTC cost audit papers | `make test_ttc_budget` | SHIPPED |
| LP-15 Per-shape SIMD quant decode + W4A | L | LP-11 | TileKernels v2.1.0 | `make test_simd_quant` | SHIPPED |
| LP-16 State/ABI semver boundary | S | — | llama.cpp v1.0 / ggml v1.0 / slermes goals | `make test_state_version` | SHIPPED |

## Immediate frontier

LP-01 is the first crossing: nothing in the goal/slerming wave can be gated until a
persistent goal object and a report-isolated judge exist. LP-05 is the independent
spine fork and can be worked in parallel; LP-06..LP-09 stand on it.

## Dependency forks

- Goal fork: LP-01 → LP-02 → LP-03 → LP-04, with LP-14 joining after LP-03.
- KV/spec fork: LP-05 → {LP-06, LP-07 → LP-08 → LP-09}, and LP-05 → LP-11 → LP-12 → LP-15.
- Independent: LP-10, LP-13, LP-16.

## Definition of SHIPPED

The named gate exists in the Makefile, compiles with the project's C11 flags, runs
without stubs, and its assertions cover the mechanism named in the pad. A documentation
edit or a partial struct is PARTIAL at best.
