# WuBuMath Orientation — map of the territory

Everything WuBuMath lives in `/home/wubu/wubuwizard`. On the
`wubu-integration` branch (research ledger).

## 1. The paper (canonical statement)

`ENCODERS/hash-mind/WuBuNestingv0.1.md` — 13-page manuscript by WaefreBeorn:
"WuBu Nesting: A Comprehensive Geometric Framework for Adaptive Multi-Scale
Hierarchical Representation with Integrated Rotational Dynamics."

Claims in one sentence: a Russian-doll stack of hyperbolic Poincaré balls
H^{n1}_{c1,s1} ⊃ H^{n2}_{c2,s2} ⊃ …, where dimension n_i, curvature c_i > 0,
and scale s_i > 0 are learnable; inter-level transitions happen in Euclidean
tangent spaces T_p(H^n_i) ≅ R^{n_i} via a learned rotation R_i ∈ SO(n_i)
(quaternions available), followed by a non-rotational dimension-adjusting map
T̃_i; each level carries learnable Boundary Sub-Manifolds B_{i,j}, a Level
Descriptor l_{d_i}, a Spread parameter σ_i for uncertainty, and intra-level
tangent flow F_i.

## 2. Python reference implementation

`ENCODERS/hash-mind/wubu_nesting_impl.py` (~1030 lines) — the operational spec:

- `HyperbolicUtils.poincare_clip` — clip to radius (1/√c)(1-ε)
- `HyperbolicUtils.scale_aware_exponential_map`
  - exp_0^{c,s}(v) = tanh(s·√c·‖v‖/2) · v/(√c·‖v‖)
  - NOTE: the paper/MATH/README.md write this WITHOUT the /2. The Python code
    and the C port agree with each other but differ from the README. This is a
    documentation inconsistency, not a math bug — the /2 form matches the
    standard Poincaré-disk metric ds² = 4 dx²/(1-|x|²)². Logged in
    overlap-ledger.md.
- `HyperbolicUtils.scale_aware_logarithmic_map`
  - log_0^{c,s}(y) = (1/s)·atanh(√c·‖y‖)·y/(√c·‖y‖)
- `hamilton_product(q1, q2)` — quaternion multiply (batched, broadcasting)
- `quat_conjugate`, `quat_rotate_via_pvq` (p·v·q double-quaternion rotation)

Numerical-stability convention: all norms computed in float32, atanh argument
clamped to (-1+ε, 1-ε), NaN→0 guards after each map.

## 3. C port (GPU/embedded)

`ENCODERS/hash-mind/c/wubu_math.h` — dependency-free C mirror of the above:
`wubu_norm`, `wubu_poincare_project`, `wubu_exp_map`, `wubu_log_map`,
`wubu_mobius_add`, `wubu_gyration`, `wubu_hamilton_product`,
`wubu_quaternion_rotate`. Internal double accumulation, float interface.

Same exp formula as Python (no /2). Same EPS = 1e-7.

## 4. Lean 4 formalization (MATH/lean/)

Two projects:

- `MATH/lean/` (package `lean`, lib `LeanCopies`) — scratch/copies.
- `MATH/lean/wubu_proofs/` (lib `WubuProofs`, exe `Main`) — the real one.
  Lean 4.29.1 + mathlib4.

  | File | Theorem | Status |
  |---|---|---|
  | `WubuProofs/PoincareBall.lean` | exp_0^c(log_0^c(y)) = y (1D, y≠0) | proven (Real.tanh_artanh) |
  | `WubuProofs/MobiusAdd.lean` | Möbius addition preserves the ball (1D) | proven via D²-cN²=(cx²-1)(cy²-1)(cxy+1)² |
  | `WubuProofs/HyperbolicGyration.lean` | gyration preserves ball + 1D gyroassociativity + gyr = identity in 1D | proven |
  | `WubuProofs/MLACompression.lean` | H·W_K − H·W_DKV·U_K = H·(W_K − W_DKV·U_K) | proven (pure linear algebra) |

  Build: `cd MATH/lean/wubu_proofs && lake build`
  Status printer: `lake exe wubu_proofs` prints the table above.

## 5. Theory / math_viz

`THEORY/math_viz/` — seven exploratory scripts each claiming to "prove" or
visualize a piece:

| Script | Claim |
|---|---|
| 01_nested_hyperbolic_spaces.py | nested Poincaré disks render correctly; φ-inspired curvature progression |
| 02_golden_ratio_decomposition.py | golden-ratio structure in level parameters |
| 03_poincare_clock.py | Poincaré disk as a clock / cyclic structure |
| 04_lie_group_nesting.py | nesting as Lie-group chain |
| 05_fiber_bundle_proof.py | WuBu nesting = principal G-bundle with Maurer–Cartan connection A = R⁻¹dR, curvature F = dA + A∧A; numerically checks parallel transport path-dependence |
| 06_symplectic_optimizer.py | symplectic optimization on the nested space |
| 07_lean_certificate.py | emit a Lean certificate from the numerics |

These are research explorations, not settled results. 05 is the most ambitious
claim and the least verified.

## 6. Cross-repo presence

- `wubuwizard/ENCODERS/hamilton-encoder-cpu/` — CPU Hamilton (quaternion)
  encoder, the rotational engine referenced by MATH/README.md.
- `wubuwizard/LLAMA-CPP-INTEGRATION/bsp-tree-cuda/` — BSP trees with
  quaternion splits (SO(4) node rotations).
- `WuBuOS` repo — runtime/BEAR lineage that consumes encoders.
- Source of truth for band crossovers lives in `wubumind_codec.py` →
  `wubu_bands.c` ({300, 4000, 10000, 16000} Hz) — that's the audio codec side,
  separate from the nesting geometry, same brand.

## 7. How to enter

1. Read `WuBuNestingv0.1.md` abstract + section 1 (the paper's own framing).
2. Read `wubu_nesting_impl.py` HyperbolicUtils (the executable truth).
3. Cross-check `wubu_math.h` (must match Python — it does).
4. Check `MATH/lean/wubu_proofs` for what is actually machine-verified.
5. Only then believe `THEORY/math_viz/*`.
