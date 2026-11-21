# SOTA 2026-10/11 → WuBu lilypads

> A **lilypad** is one crossable deliverable with a gate on the far side. Order is
> **dependency**, not importance: a pad that another pad stands on goes first.
> Every pad ships a real gate: no gate, no pad.
> Source window: 2026-10-01 → 2026-11-20, from eight cited sweep reports.
> Status 2026-11-20: all 16 pads LP-01..LP-16 have a gated C11 module and a
> passing `make test_<gate>` target; gates are force-added past the `tools/test_*` ignore rule.

## Standing rules

- Read upstream, reimplement from scratch in C11. Never vendor a third-party library.[179][180]
- Opaque structs at module seams, minimal includes, no stubs, runtime kernel dispatch stays runtime.
- The engine stays ignorant of the goal it serves. Goal/state/slerm logic lives in a host module.
- "Slerming" in this roadmap means the slermes-style persistent-goal/continuation loop, not a new inference kernel.[33][133]
- No primary source was found for a distinct 2026-10/11 "slerming paradigm"; do not invent one.

## Wave 1 — goals and slerming (host layer)

### LP-01 — Persistent goal state + evidence-first judge
Opaque `wubu_goal_t` with text, status, turn budget, and a `.goal` sidecar that survives restart. The judge is deterministic: it reads a **separate evidence string** (never the agent's own completion claim), requires a verifier callback to pass, and treats an explicit blocked reason as done-with-reason.[147][148] Real user messages preempt; budget exhaustion pauses. Gate: `make test_goal` (continue / report-only rejected / pass / blocked / preempt / budget / sidecar round-trip). **Size M**. Unlocks LP-02..LP-04 and LP-14.
Shipped as `src/wubu_goal.c` + `include/wubu_goal.h` + `tools/test_goal.c`; wired into both Makefiles.

### LP-02 — Fast/slow goal-drift steering
One scalar drift score plus a tunable threshold τ: the fast path monitors every turn, the slow path rewrites the subgoal only when drift exceeds τ, and the existing judge veto stays authoritative.[91][99] Gate: `make test_goal_steer` (below τ no rewrite, above τ rewrite, veto preserved). **Size S**. Depends on LP-01.

### LP-03 — Bounded verifier-guided resampling with rollback
Cap candidate trajectories, use the LP-01 verifier to reject early, stop as soon as one valid plan exists, and restore a snapshot on rejection.[38][39] Gate: `make test_goal_resample` (bounded count, early stop, snapshot restored). **Size M**. Depends on LP-01/02.

### LP-04 — Failure-pattern harness bandit
Turn hive failure cells into arms scored by severity, fixability, breadth and side-effect risk; allocate revision effort with a bandit and keep a fixed-capacity role pool.[189][193][199] Gate: `make test_harness_bandit` (arm scoring, exploration cap, rollback on unsafe revision). **Size L**. Depends on LP-03.

## Wave 2 — KV spine (engine layer)

### LP-05 — All-or-nothing paged KV claim
Add `alloc_exact`/`can_alloc_exact`: claim all requested block IDs in caller order or leave state untouched, checking logical quota and physical page capacity together.[108][176] Gate: `make test_kv_alloc_exact` (all-or-nothing, quota, capacity, no partial mutation). **Size M**. Independent spine pad.

### LP-06 — Shared-cell SWA reclamation invariant
A cell may be reused only when it is masked for every owner under each owner's position bound; record overwritten positions per owner.[200][215] Gate: `make test_swa_reclaim` (two owners, divergent bounds, unsafe reuse refused). **Size M**. Depends on LP-05.

### LP-07 — Confidence-budgeted speculation policy
Choose draft/verification length from cumulative draft confidence and global rank; keep the target verifier unchanged.[103][104][207] Gate: `make test_spec_budget` (budget allocation, length selection, acceptance accounting). **Size M**. Depends on LP-05.

### LP-08 — Deterministic seeded speculation
Seed rejection-sampling proposals and make acceptance reproducible across runs; expose the seed in the spec context.[207][180] Gate: `make test_spec_determinism` (same seed → identical accepted prefix, different seed → different). **Size S**. Depends on LP-07.

### LP-09 — Constrained decoding survives speculation
Build masked multi-row windows so draft tokens cannot break grammar/JSON/tool constraints; advance along the draft prefix, invalidate at the first illegal draft, commit at divergence.[206] Gate: `make test_spec_constrained` (illegal draft rejected, mask intact across rows). **Size L**. Depends on LP-07/08.

## Wave 3 — kernels and formats

### LP-10 — Compile-time family specialization
Express model-family differences as compiled rows (attention variant, routing, stop tokens) instead of runtime family branches; keep the kernel bodies byte-identical.[179] Gate: `make test_kernel_identity` (1837/1843-style identity check on the local kernel set, serving output unchanged). **Size L**. Independent.

### LP-11 — KV reconstruction in compute format
Reconstruct KV directly in MX/NVFP4 with a small deterministic anchor set so attention never dequantizes to FP16 just to read the cache.[10][16] Gate: `make test_kv_qfmt` (finite dot products, max error bound, no FP16 intermediate). **Size L**. Depends on LP-05.

### LP-12 — Tolerance-parameterised KV fetch for the 9P layer
Add an `(eps, delta)` fetch gate: sketch + pivotal tokens on the fast side, error estimate decides whether to host-fetch the full page.[12] Gate: `make test_kv_fetch_eps` (error under eps avoids fetch, over delta forces it). **Size M**. Depends on LP-05/11.

### LP-13 — Fused diffusion logits/sampler
Fuse entropy, stopping and Gumbel/bit-exact sampling in one canvas×vocab pass; after W4A16 experts this is the decode bottleneck on bandwidth-bound devices.[23] Gate: `make test_diffusion_sampler` (bit-exact vs unfused reference, one pass). **Size M**. Independent.

### LP-14 — Measured-cost test-time compute budget
Charge wall-clock and KV cost to candidate counts, cap when ranking bias plateaus, and audit end-to-end speedup instead of FLOP savings.[27][30] Gate: `make test_ttc_budget` (budget respects measured cost, plateau stops early). **Size M**. Depends on LP-01/03.

### LP-15 — Per-shape SIMD quant decode + W4A streams
Autotune the native SIMD dequant path per tensor shape and stream E4M3/NVFP4 W4 activations into the existing matmul spine.[201] Gate: `make test_simd_quant` (per-shape dispatch, bit-identity on NaN/Inf, no regression on IQ2_S). **Size L**. Depends on LP-11.

### LP-16 — State/ABI semver boundary
Version the opaque state files and session serialisation the way llama.cpp/ggml v1.0 and slermes' goal state do; refuse stale versions with a clear reason.[76][110][133] Gate: `make test_state_version` (current loads, stale rejected, migration named). **Size S**. Independent.

## Accepted risk

Paged-KV fault attacks surfaced in this sweep only as an untitled stub with no extractable primary source. WuBuOS is an agentic OS insecure by design; users are told not to bank on it. Recorded once, not turned into a pad.

## What not to break

- Opaque structs and status-as-value at every module seam.
- No third-party linking — slerm means read and reimplement, never vendor.[179][180]
- Runtime kernel dispatch via `wubu_kernel_register()`; LP-10 adds compiled rows, it does not hardcode a backend.
- KV namespace remains a 9P-exportable filesystem; LP-05/12 add gates around it, not a replacement.
- The engine core stays ignorant of goals, hive cells and orchestration; LP-01..04 live in the host layer.
- `make test_all` and the colony gate remain the release truth. No pad ships without its own gate.

## Sources

[10] https://arxiv.org/abs/2611.21216
[12] https://arxiv.org/abs/2611.11319
[16] https://arxiv.org/abs/2611.10511
[23] https://arxiv.org/abs/2611.21852
[27] https://arxiv.org/abs/2610.25519
[30] https://arxiv.org/abs/2610.41970
[33] https://github.com/waefrebeorn/slermes — slermes - C11 Hermes Agent (waefrebeorn)
[38] https://arxiv.org/pdf/2611.06606 — Verifier-Guided Self-Consistency (VGSC)
[39] https://arxiv.org/pdf/2610.11223v1 — SafeInferCom: Safe Inference-Time Compute via Verifier-Guided Mid-Generation Intervention
[76] https://github.com/ggml-org/llama.cpp/releases/tag/v1.0.0 — llama.cpp v1.0.0 GitHub release
[91] https://arxiv.org/html/2610.22316 — Learning to Steer LLM Agents and Multi-Agent Systems (steering agent, goal drift)
[99] https://learnaivisually.com/ai-explained/learned-steering-fast-path-monitor-slow-path-intervention — Learned steering supervisor: fast-path monitoring vs slow-path intervention (explainer)
[103] https://github.com/deepseek-ai/DeepInfer
[104] https://github.com/RueClaw/public-data/blob/main/reviews/deepinfer.md
[108] https://github.com/ovg-project/kvcached
[110] https://github.com/ggml-org/ggml/releases
[133] https://github.com/waefrebeorn/slermes/commit/f95fb59d456894691c153665d5712bc19605c028 — slermes docs: Persistent Goals feature page
[147] https://arxiv.org/html/2610.17724v1 — Evidence-First Agentic Verification (Sauron verifier)
[148] https://arxiv.org/html/2610.29609 — Who Verifies the Agent's Success Claim? Self-Report Leakage in GUI agents
[176] https://github.com/ovg-project/kvcached/releases/tag/v0.1.6
[179] https://github.com/incoai/splash/pull/573
[180] https://github.com/synchopate/llama-modchip
[189] https://arxiv.org/abs/2610.00906
[193] https://arxiv.org/abs/2610.05300
[199] https://autosaddler-projectpage.github.io/activesaddler
[200] https://github.com/ggml-org/llama.cpp/pull/32309 — llama.cpp PR 32309: reuse SWA cells shared by several sequences
[201] https://github.com/deepseek-ai/TileKernels — deepseek-ai/TileKernels repo README
[206] https://github.com/Niko1221/Strata/pull/4222 — Strata PR 4222 tools+JSON token masks
[207] https://github.com/sgl-project/sglang/releases/tag/v0.5.23 — SGLang v0.5.23 primary GitHub release notes
[215] https://github.com/ggml-org/llama.cpp/issues/32258 — llama.cpp issue 32258 unified SWA shared-cell slot bug
