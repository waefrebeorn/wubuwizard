import Mathlib
open Real

set_option maxHeartbeats 0

/-
QuaternionRotation.lean — quaternion conjugation preserves norm (the
foundation of "p·v·q is a rotation").

A unit quaternion q = w + xi + yj + zk (‖q‖ = 1) acts on ℝ³ by conjugation
v ↦ q·v·q* (or the double form p·v·q). The reason this is a rotation —
hence the WuBuNesting BSP claim that each node split is an SO(4) rotation
via quaternions (see draftPY/oldwubunest.py, the p·v·q formulation) — is
that conjugation by a unit quaternion preserves the Euclidean norm:

    ‖q·v·q*‖ = ‖v‖   for unit q

This file proves the algebraic core: for quaternions represented as
ℝ⁴ (w,x,y,z), the Hamilton product, and the key scalar identity that
makes conjugation norm-preserving. The full ‖q·v·q*‖ = ‖v‖ statement needs
the norm of a Hamilton product, which reduces (via the same
inner-product-linearity pattern as the Poincaré proofs) to:

    ‖p·q‖² = ‖p‖²·‖q‖²   (the Euclidean norm is multiplicative under
                          the Hamilton product)

This single identity — `hamilton_norm_mul` — is the load-bearing lemma:
it implies ‖q·v·q*‖ = ‖q‖·‖v‖·‖q*‖ = ‖v‖ when ‖q‖ = 1. We prove it here
by expanding the Hamilton product and verifying the resulting polynomial
identity with `ring`.
-/

/-- Quaternion as ℝ⁴: (w, x, y, z) = w + xi + yj + zk. -/
abbrev Quat := ℝ × ℝ × ℝ × ℝ

/-- Euclidean norm² of a quaternion. -/
def quatNormSq (q : Quat) : ℝ := q.1 ^ 2 + q.2.1 ^ 2 + q.2.2.1 ^ 2 + q.2.2.2 ^ 2

/-- The Hamilton product of two quaternions.
    (w1+x1i+y1j+z1k)·(w2+x2i+y2j+z2k), component by component. -/
def hamilton (q p : Quat) : Quat :=
  ( q.1 * p.1 - q.2.1 * p.2.1 - q.2.2.1 * p.2.2.1 - q.2.2.2 * p.2.2.2,
    q.1 * p.2.1 + q.2.1 * p.1 + q.2.2.1 * p.2.2.2 - q.2.2.2 * p.2.2.1,
    q.1 * p.2.2.1 - q.2.1 * p.2.2.2 + q.2.2.1 * p.1 + q.2.2.2 * p.2.1,
    q.1 * p.2.2.2 + q.2.1 * p.2.2.1 - q.2.2.1 * p.2.1 + q.2.2.2 * p.1 )

/-- The load-bearing identity: the Euclidean norm² is multiplicative under
    the Hamilton product.

        ‖p·q‖² = ‖p‖² · ‖q‖²

    This is what makes conjugation by a unit quaternion norm-preserving,
    hence a rotation. The proof expands `hamilton` and closes the
    resulting degree-6 polynomial identity with `ring`.
-/
theorem hamilton_norm_mul (q p : Quat) :
    quatNormSq (hamilton q p) = quatNormSq q * quatNormSq p := by
  unfold quatNormSq hamilton
  ring

/-- Quaternion conjugate: q* = w − xi − yj − zk. -/
def quatConj (q : Quat) : Quat := (q.1, -q.2.1, -q.2.2.1, -q.2.2.2)

/-- Conjugation preserves norm²: ‖q*‖² = ‖q‖². -/
theorem quatConj_norm (q : Quat) : quatNormSq (quatConj q) = quatNormSq q := by
  unfold quatNormSq quatConj
  ring

/-- Corollary: for a unit quaternion (‖q‖² = 1), the double conjugation
    p ↦ q·p·q* preserves norm²:

        ‖q·p·q*‖² = ‖p‖²

    This is exactly the property that makes the WuBuNesting BSP node
    split (an SO(4) rotation via p·v·q) a genuine isometry of the
    spherical cap, not an arbitrary linear map.
-/
theorem unit_quat_conj_preserves_norm (q p : Quat) (hunit : quatNormSq q = 1) :
    quatNormSq (hamilton (hamilton q p) (quatConj q)) = quatNormSq p := by
  rw [hamilton_norm_mul, hamilton_norm_mul, quatConj_norm, hunit]
  simp