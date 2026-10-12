# WuBuMath Overlap Ledger

The comparison document. For every WuBu mathematical claim: what it is, where
it comes from in the literature, what WuBu adds, and what our machine-verified
state is. Verdicts:

- **ORIGINAL** — no known prior equivalent; WuBu's contribution.
- **STANDARD** — textbook result; WuBu implements it correctly (or not).
- **NOVEL COMBINATION** — ingredients standard, the specific combination is new.
- **UNVERIFIED** — claim asserted without proof or literature check.
- **DISCREPANCY** — WuBu's internal documents/code disagree; needs a ruling.

The purpose of this ledger is honest attribution. WuBu's value is in the
compositions and the engineering; pretending the components are new destroys
credibility, and crediting WuBu only for components throws away the real
contribution.

---

## A. Hyperbolic geometry core

### A1. Poincaré ball model of hyperbolic n-space
- WuBu: primary embedding space for every level.
- Provenance: Poincaré (1882); standard in GDL (Nickel & Kiela 2017,
  Chami et al. 2019, "Poincaré Embeddings for Learning Hierarchical
  Representations").
- WuBu's addition: nothing on the model itself.
- **Verdict: STANDARD.**

### A2. Metric ds² = 4 dx² / (c(1−|x|²)², distance 2 atanh(√c|x|)/√c
- WuBu: stated in math_viz/01; c scaling of the conformal factor.
- Provenance: standard conformal model of hyperbolic space with curvature
  −1/c. The c-scaling is itself a standard change of metric
  g → g/c (Ungar's gyrovector spaces use curvature −1 convention).
- **Verdict: STANDARD.**

### A3. Scale-aware exp/log maps
- WuBu: exp_0^{c,s}(v) = tanh(s·√c·‖v‖)·v/(√c·‖v‖), log inverse with 1/s.
- Provenance: parameterized family of radial maps. The s parameter is a
  learned "zoom". Closest literature: learnable curvature in Poincaré
  embeddings (Nickel & Kiela's curvature is global; several works learn
  per-space curvature, e.g. "Fully Hyperbolic Neural Networks" RPL,
  Skopek et al. 2020, and mixing curvatures). A per-level learnable scale
  multiplying the hyperbolic radius is a small, natural extension; not a
  named result.
- **Verdict: NOVEL COMBINATION (small).** Correct as a parameterization; NOT
  the Riemannian exp map of the standard Poincaré metric (see D1).

### A4. Möbius (gyrovector) addition ⊕_c
- WuBu: standard formula; Lean-proven to preserve the ball in 1D.
- Provenance: Ungar, "Analytic Hyperbolic Geometry" (2005); the operation is
  Möbius addition in the Poincaré ball. 1D special case is classical.
- WuBu's addition: Lean formalization of the 1D preservation theorem with an
  explicit factorization
  `D² − cN² = (cx²−1)(cy²−1)(cxy+1)²`.
- **Verdict: STANDARD result, WuBu's Lean proof is a clean contribution to
  our own trust chain.**

### A5. Gyration and gyroassociativity
- WuBu: gyr[u,v]w = (−(u⊕v))⊕(u⊕(v⊕w)); 1D gyration = identity;
  1D gyroassociativity proven in Lean.
- Provenance: Ungar's gyrogroups. The 1D degeneration to identity is
  elementary (1D Möbius addition is associative). nD gyration is a rotation
  of the ball.
- **Verdict: STANDARD; the 1D proofs are honest and complete, and correctly
  labeled as 1D-only in the code comments.** The nD "gyration is a rotation"
  claim in HyperbolicGyration.lean is a sketch, not a proof.

### A6. exp/log round trip = identity
- WuBu: PoincareBall.lean, 1D, y≠0, proven via Real.tanh_artanh.
- Provenance: elementary for the standard map; for the no-/2 convention it is
  still an exact algebraic identity (tanh(atanh z)=z).
- **Verdict: STANDARD, correctly proven in our version.**

---

## B. Quaternions and rotation

### B1. Hamilton product
- WuBu: batched implementation, Python + C.
- Provenance: Hamilton (1843). Textbook.
- **Verdict: STANDARD.**

### B2. Rotation via p·v·q double-quaternion conjugation
- WuBu: `quat_rotate_via_pvq`.
- Provenance: standard quaternion rotation; the p,q form (not just q,v,q*) is
  the "sandwich with two independent unit quaternions" used in graphics.
- **Verdict: STANDARD.**

### B3. Quaternions as inter-level rotations in tangent spaces
- WuBu: R_i ∈ SO(n_i) implemented via quaternions, applied to primary data +
  boundary representations + level descriptor simultaneously.
- Provenance: Quaternion neural networks (Parcollet et al.), and
  rotation-based tangent-space transitions appear in several hyperbolic
  works. The simultaneous application to three heterogeneous payloads is an
  engineering design, not a math result.
- **Verdict: NOVEL COMBINATION (architectural), STANDARD components.**

---

## C. The nesting architecture

### C1. Recursive nesting H^{n1}_{c1,s1} ⊃ H^{n2}_{c2,s2} ⊃ …
- WuBu: learnable dimension, curvature, and scale per level; data at level i+1
  lives in a bounded region near the origin of level i.
- Provenance: hyperbolic hierarchy embedding is standard; multi-curvature
  product spaces are standard (Skopek et al. 2020 mixes curvatures;
  Gu et al. 2019 "Learning Mixed-Curvature Representations in Product
  Spaces"). Product manifolds put geometries in PARALLEL; WuBu NESTS them.
  Nested hyperbolic spaces with per-level learnable geometry: closest
  relatives are hierarchical/recursive hyperbolic embeddings and
  "Hyperbolic Neural Network++" style work, but the explicit Russian-doll
  nesting with inter-level rotation + tangent-space transitions is not a
  named published framework as far as the current check shows.
- **Verdict: NOVEL COMBINATION — this is the core claim and the honest
  contribution. Components standard, composition new.**

### C2. Boundary Sub-Manifolds B_{i,j}
- WuBu: learnable manifolds inside each level representing scale-specific
  structures.
- Provenance: related to hyperbolic capsule/prototype ideas, and to
  boundary-aware embeddings, but no direct match found.
- **Verdict: UNVERIFIED — needs a literature pass before claiming anything.**

### C3. Relative Vectors d_{i+1,j,k}
- WuBu: rotation-aware structure codes computed in the target tangent space.
- Provenance: closest to relational/relative-position encodings; the
  "computed in the target tangent space after rotation" part is specific.
- **Verdict: UNVERIFIED.**

### C4. Intra-level tangent flow F_i and Spread σ_i
- WuBu: local dynamics + uncertainty per level.
- Provenance: hyperbolic flow/ODE models exist (e.g. hyperbolic diffusion);
  uncertainty in hyperbolic embeddings exists (probabilistic hyperbolic
  embeddings, Mathieu et al. 2019). Combination is design.
- **Verdict: NOVEL COMBINATION (design-level), UNVERIFIED as math.**

---

## D. Discrepancies inside WuBu (must be ruled on)

### D1. The /2 in the exponential map
- MATH/README.md: exp_0^c(v) = tanh(√c·‖v‖ **/ 2**)·v/(√c·‖v‖)
- wubu_nesting_impl.py docstring: same /2
- wubu_nesting_impl.py CODE: tanh(scale·√c·norm) — **no /2**
- c/wubu_math.h: no /2 (agrees with code)

Consequences:
- The /2 form is the true Riemannian exponential map for the standard
  Poincaré metric (conformal factor 2/(1−|x|²) at c=1).
- The no-/2 form is a valid radial parameterization T_0 → ball, bijective,
  and pairs exactly with the log map as written. It is a compressed-radius
  map: hyperbolic distance is effectively halved near the origin.
- Python, C, and the 1D Lean proof all use the no-/2 form consistently.
- Ruling: **code is the source of truth; README and docstring are stale.**
  Action: fix the docs, or add the /2 and re-verify. Either is defensible;
  leaving three documents disagreeing is not. Recommendation: keep the
  no-/2 form (it's what's verified and implemented), correct README +
  docstring, and note explicitly in the paper that WuBu's radial map is a
  deliberate reparameterization, not the Riemannian exp map.

### D2. "Proofs" in THEORY/math_viz
- `05_fiber_bundle_proof.py` etc. are numerical simulations. Naming them
  "*_proof.py" overstates. They verify consistency numerically (parallel
  transport path-dependence matches curvature), which is real evidence but
  not a proof.
- Action: rename or re-document as "numerical verification".

---

## E. Machine-verification status (Lean)

| Claim | Lean file | Status |
|---|---|---|
| exp∘log = id (1D) | PoincareBall.lean | ✅ proven |
| Möbius add preserves ball (1D) | MobiusAdd.lean | ✅ proven |
| **Möbius add preserves ball (nD)** | **MobiusAddND.lean** | ✅ **proven (2026-10-11)** |
| Denominator ≠ 0 in ball | MobiusAdd.lean | ✅ proven |
| Gyration preserves ball (1D) | HyperbolicGyration.lean | ✅ proven |
| Gyroassociativity (1D) | HyperbolicGyration.lean | ✅ proven |
| 1D gyration = identity | HyperbolicGyration.lean | ✅ proven |
| MLA factor identity | MLACompression.lean | ✅ proven (algebra only) |
| **MLA Frobenius error bound** | **MLACompression.lean** | ✅ **proven (2026-10-11, via `frobenius_norm_mul`)** |
| MLA error bound optimality (Eckart–Young) | — | ❌ needs SVD library (FLYWHEEL pins Lean 4.26; we're on 4.29) |
| nD gyration is rotation | GyrationND.lean | ✅ **ball preservation proven (2026-10-11); inner-product preservation future work** |
| Fiber-bundle/connection claim | — | ❌ no Lean file |

**Third result 2026-10-11: GyrationND.lean.** Two theorems:

1. **Möbius norm formula (nD)**: `‖a ⊕_c b‖² = (‖a‖² + ‖b‖² + 2⟨a,b⟩) /
   (1 + 2c⟨a,b⟩ + c²‖a‖²‖b‖²)` — proven via the same inner-product
   expansion as MobiusAddND plus the algebraic identity
   `‖N‖² = (A+B+2p)·D` (verified by `ring`).
2. **Gyration preserves the ball (nD)**: `gyr_c[u,v]w` maps the ball to
   itself — every gyration argument is a Möbius sum of in-ball points and
   Möbius addition preserves the ball, so the whole loop stays in-ball.

Ungar's full claim `⟨gyr[u,v]a, gyr[u,v]b⟩ = ⟨a,b⟩` (which implies
"gyration is a rotation") needs the explicit gyrator matrix
(`gyr[u,v]w = w + Au + Bv` with Ungar's A, B, D coefficients) — future
work, documented in the file header.

**Second critical gap closed 2026-10-11: the MLA Frobenius error bound.**
`mla_compression_error_bound`: for any hidden-state matrix H,
```
‖H·W_K − H·W_DKV·U_K‖_F ≤ ‖H‖_F · ‖W_K − W_DKV·U_K‖_F
```
via the factor identity + mathlib's `frobenius_norm_mul` — which handles
RECTANGULAR matrices (the old sketch note claiming mathlib restricted to
square matrices was wrong). Eckart–Young *optimality* (that truncated SVD
minimizes the error among all rank-d_c factorizations) is still unproven —
it needs an SVD library; FLYWHEEL has it but pins Lean 4.26 while we're on
4.29. The one-sided bound is what WuBu's argument actually needs.

**The nD Möbius preservation theorem — the critical gap the entire
WuBuNesting paper rests on — is now closed.** Proof structure (all
machine-checked, Lean 4.29.1 + mathlib):

1. `mobius_add_nd` defined as `(1/D) • (α • x + β • y)` on
   `EuclideanSpace ℝ (Fin n)`.
2. D > 0 via `D = (1+cp)² + c²(ab−p²)`, contradiction argument on cp = −1.
3. `‖(1/D) • N‖² = ‖N‖²/D²` via `norm_smul`/`norm_div`/`norm_one`.
4. `‖N‖² = α²a + 2αβp + β²b` via inner-product linearity
   (`norm_add_sq_real`, `inner_smul_left`, `inner_smul_right`).
5. **The factorization** (the hard-won piece — three failed candidates
   before the right one; the naive `(1−ca)(1−cb)(1+cp)²` is FALSE in nD
   because it silently assumes p² = ab, which only holds in 1D):
   ```
   D² − c‖N‖² = (1−ca)(1−cb)·[(1+cp)² + c²(ab−p²)]
   ```
   verified by `ring` after unfolding α, β, D and rewriting p, a, b.
6. Strict positivity: `(1−ca) > 0`, `(1−cb) > 0` from the ball hypotheses;
   bracket > 0 by contradiction — both terms zero would need cp = −1 AND
   p² = ab, but cp = −1 gives p² = 1/c² > ab (contradicting
   Cauchy–Schwarz + `ab < 1/c²`).
7. Finish via `div_lt_div_iff₀` + `linarith`.

Remaining gaps: MLA Eckart–Young optimality (needs SVD library), nD
gyration-is-rotation, the fiber-bundle claim.

Everything machine-checked is now either algebra or the full nD ball
invariant (MobiusAddND.lean, proven 2026-10-11). The remaining unproven
claims (the bundle structure, the MLA error bound, nD gyration-as-rotation)
are the roadmap.

---

## F. openai/math overlap pass

openai/math (722 manuscripts, 372 families, released Oct 6 2026, incl. Lean
formalizations) is our "has this been proven already" lookup for geometry and
hyperbolic-adjacent results. Index in `openai-math-index.md`. Method: search
the catalogue for our theorem statements (Möbius preservation, gyration,
Poincaré maps, bundle/connection results) and record family IDs that match,
so we never claim something already published there.
