# WuBuMath Vault — status

**As of 2026-10-11.**

## Built

| File | Content | State |
|---|---|---|
| `VAULT.md` | index + reading order + human's guiding principles | complete |
| `VAULT/orientation.md` | full disk map: paper → Python → C → Lean → math_viz → cross-repo | complete |
| `VAULT/core-math.md` | every formula with code-matched conventions, stability rules, the nesting operator | complete |
| `VAULT/overlap-ledger.md` | 20+ claims with provenance, verdict (ORIGINAL/STANDARD/NOVEL COMBINATION/UNVERIFIED/DISCREPANCY), and Lean status table | complete |
| `VAULT/openai-math-index.md` | 722-manuscript corpus cloned (1.9G, depth-1) + searched: **zero content overlap**; toolchain leverage documented; false-positive traps listed | complete |
| `VAULT/klein-movable-manifolds.md` | arXiv:2410.16813 Klein HNN analysis + the movable-manifold / non-orientable / storage-adaptive proposal | complete |

## Key findings

1. **nD Möbius ball preservation — PROVEN (2026-10-11).** The critical gap
   is closed. `WubuProofs/MobiusAddND.lean` proves, for all n: if
   ‖x‖², ‖y‖² < 1/c then ‖x ⊕_c y‖² < 1/c. Load-bearing identity:
   `D² − c‖N‖² = (1−ca)(1−cb)·[(1+cp)² + c²(ab−p²)]` — the naive
   `(1−ca)(1−cb)(1+cp)²` is FALSE in nD (silently assumes p² = ab,
   Cauchy–Schwarz equality, true only in 1D). The `c²(ab−p²)` term is the
   Cauchy–Schwarz slack.
2. **MLA Frobenius error bound — PROVEN (2026-10-11).**
   `mla_compression_error_bound`: `‖H·W_K − H·W_DKV·U_K‖_F ≤
   ‖H‖_F·‖W_K − W_DKV·U_K‖_F` via mathlib's `frobenius_norm_mul` (works
   for rectangular matrices — the old "mathlib restricts to square" note
   was wrong). Eckart–Young optimality still unproven (needs SVD library).
3. Remaining unproven: MLA Eckart–Young optimality, nD
   gyration-as-rotation, fiber-bundle claim.

2. **A real internal discrepancy exists** (logged as D1 in the ledger): the
   exponential map has a `/2` in MATH/README.md and the Python docstring but
   NOT in the Python code, the C port, or the Lean proof. Code + Lean are
   consistent with each other; README is stale. Recommendation recorded:
   keep the no-/2 form, fix the docs, and state explicitly in the paper that
   WuBu's radial map is a deliberate reparameterization, not the Riemannian
   exponential map of the standard Poincaré metric.

3. **openai/math has zero overlap with WuBu's math.** Searched all 9142
   lines of CONTENTS.md + 242 Lean scope docs for hyperbolic-geometry,
   Poincaré, Möbius-addition, gyrovector, tanh, quaternion-DL terms. The
   corpus is AG/number theory/CFT frontier results. It poses no prior-art
   risk. Its value to us: Lean toolchain format (Comparator pattern) and a
   citation source for adjacent classical lemmas. "Möbius" hits are the
   Möbius *group* of the Riemann sphere (Koebe rigidity), NOT Möbius
   *addition* — false-positive trap documented.

4. **The Klein paper (2410.16813) is directly relevant** and handily
   supports the human's extension direction: straight-line geodesics make
   Einstein addition the Klein analogue of Möbius ⊕, the paper proves the
   tangent-space constructions of scalar mult and parallel transport are
   exactly Einstein ops, and the model isometry Poincaré↔Klein
   (k = 2x/(1+‖x‖²)) gives WuBu a clean per-level "model switching layer"
   beside the existing learnable curvature/scale.

5. **The movable-manifold idea decomposes into three tractable pieces**,
   all documented in klein-movable-manifolds.md:
   a. Klein levels as affine-deformable containers (cheap because geodesics
      are straight lines — deformation bends ambient space, not geometry),
   b. per-level orientation flag with O(n) transitions and reflection bits
      for Klein-like levels (quaternion SO(n) machinery no longer covers),
   c. storage-adaptive projection: model + container + parameters learned
      against a per-level bit budget, turning nesting into rate-distortion
      over geometry.

## In progress / next

- mathlib4 cache is fully built (one-time cost paid; `fastcheck.sh <file>`
  now gives ~4-5 min single-file checks).
- Next Lean targets in priority order:
  1. **nD gyration is a rotation** (HyperbolicGyration.lean — currently a
     sketch; needs the gyrator matrix shown orthogonal),
  2. **MLA Eckart–Young optimality** (blocked: FLYWHEEL has full SVD +
     Eckart–Young in Lean 4 but pins toolchain v4.26.0-rc2 while we're on
     v4.29.1 — either wait for FLYWHEEL to bump, or port the Frobenius
     tail-sum argument ourselves),
  3. **Fiber-bundle/connection claim** (math_viz/05 — highest-value
     formalization target, currently numerical only),
  4. Klein paper Lean targets (deck transformation is an involution,
     T² orientation-preserving, patch metric well-definedness).

## Not done (roadmap)

- nD Möbius addition preserves the ball — Lean proof (the critical gap)
- nD gyration is a rotation — Lean proof
- MLA compression error bound — Lean proof (needs Eckart–Young)
- Fiber-bundle/connection claim — Lean proof or honest downgrade of
  math_viz/05
- Fix the /2 discrepancy in docs (or the code)
- Literature pass: movable/morphing containers in DL; non-orientable
  hyperbolic DL; rate-distortion for hyperbolic embeddings
