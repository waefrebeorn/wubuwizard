import Mathlib
import WubuProofs.MobiusAddND
open Real

set_option maxHeartbeats 0

/-
GeodesicInterp.lean — geodesic interpolation endpoints (GAP_REGISTER C009:
multi-keyframe flow rollout).

The geodesic connecting x, y in the Poincaré ball (Nickel & Kiela 2018
eq. 8; GGBall; Hyperbolic NNs) is:

    γ_{x→y}(t) = x ⊕_c (⊖_c x ⊕_c y) ⊗_c t,   t ∈ [0,1]

with the Möbius scalar multiplication (Ganea et al. 2018 eq. 7):

    r ⊗_c w = (1/√c) · tanh(r · artanh(√c·‖w‖)) · w/‖w‖   (w ≠ 0)
    r ⊗_c 0 = 0

and the endpoint properties γ(0) = x, γ(1) = y. These are the on-manifold
interpolation primitives the flow-matching rollout (C009) needs: each P-frame
step moves along the geodesic between keyframes, and the endpoints pin the
path to the keyframes.

We prove the t = 0 endpoint (the load-bearing "the path starts at x"):

    γ_{x→y}(0) = x

via two facts:
  1. 0 ⊗_c w = 0  (since tanh 0 = 0, the scalar multiplier vanishes).
  2. x ⊕_c 0 = x  (0 is the Möbius additive identity).

The t = 1 endpoint reduces to γ(1) = x ⊕_c (⊖x ⊕ y) = y, the
left-cancellation identity — heavier, documented as the next step.
-/

/-- Möbius additive identity: x ⊕_c 0 = x. -/
theorem mobius_add_nd_zero {n : ℕ} (c : ℝ) (x : EuclideanSpace ℝ (Fin n)) :
    mobius_add_nd c x 0 = x := by
  unfold mobius_add_nd
  simp

/-- Möbius scalar multiplication (w ≠ 0 case).
    r ⊗_c w = (1/√c) · tanh(r · artanh(√c·‖w‖)) · (‖w‖⁻¹ • w).
    NOTE: EuclideanSpace has NO HDiv instance — scale via `(1/‖w‖) • w`. -/
noncomputable def mobiusScalarMul {n : ℕ} (c : ℝ) (r : ℝ)
    (w : EuclideanSpace ℝ (Fin n)) (hw : w ≠ 0) : EuclideanSpace ℝ (Fin n) :=
  (((1 / Real.sqrt c) * Real.tanh (r * Real.artanh (Real.sqrt c * ‖w‖))) : ℝ) •
    ((1 / ‖w‖) • w)

/-- Zero scalar annihilates: 0 ⊗_c w = 0.  (tanh 0 = 0 kills the scalar.) -/
theorem mobiusScalarMul_zero {n : ℕ} (c : ℝ) (hc : c > 0)
    (w : EuclideanSpace ℝ (Fin n)) (hw : w ≠ 0) :
    mobiusScalarMul c 0 w hw = 0 := by
  unfold mobiusScalarMul
  simp

/-- Unit scalar is the identity: 1 ⊗_c w = w.  (The tanh/artanh round-trip
    `tanh(artanh z) = z` makes the scalar `(1/√c)·tanh(artanh(√c‖w‖))·(1/‖w‖)`
    equal to 1, so `1 ⊗ w = 1 • w = w`.  Requires w in the ball so
    `√c‖w‖ < 1` and artanh is defined.)  This is the load-bearing identity
    for the γ(1) = y geodesic endpoint. -/
theorem mobiusScalarMul_one {n : ℕ} (c : ℝ) (hc : c > 0)
    (w : EuclideanSpace ℝ (Fin n)) (hw : w ≠ 0) (hball : ‖w‖ ^ 2 < 1 / c) :
    mobiusScalarMul c 1 w hw = w := by
  unfold mobiusScalarMul
  have hsc : Real.sqrt c > 0 := Real.sqrt_pos.mpr hc
  have hwpos : ‖w‖ > 0 := norm_pos_iff.mpr hw
  -- √c‖w‖ < 1  (from c‖w‖² < 1, taking sqrt of both sides)
  have harg : Real.sqrt c * ‖w‖ < 1 := by
    have h1 : c * ‖w‖ ^ 2 < 1 := by
      calc c * ‖w‖ ^ 2 < c * (1 / c) := mul_lt_mul_of_pos_left hball hc
        _ = 1 := by field_simp
    have hsq : (Real.sqrt c * ‖w‖) ^ 2 < 1 := by
      calc (Real.sqrt c * ‖w‖) ^ 2 = Real.sqrt c ^ 2 * ‖w‖ ^ 2 := by ring
        _ = c * ‖w‖ ^ 2 := by rw [Real.sq_sqrt hc.le]
        _ < 1 := h1
    have hnne : 0 ≤ Real.sqrt c * ‖w‖ := mul_nonneg (le_of_lt hsc) (le_of_lt hwpos)
    calc Real.sqrt c * ‖w‖ = Real.sqrt ((Real.sqrt c * ‖w‖) ^ 2) := (Real.sqrt_sq hnne).symm
      _ < Real.sqrt 1 := Real.sqrt_lt_sqrt (sq_nonneg _) hsq
      _ = 1 := Real.sqrt_one
  -- tanh(artanh(√c‖w‖)) = √c‖w‖
  have hnne : 0 ≤ Real.sqrt c * ‖w‖ := mul_nonneg (le_of_lt hsc) (le_of_lt hwpos)
  have hrnd : Real.tanh (Real.artanh (Real.sqrt c * ‖w‖)) = Real.sqrt c * ‖w‖ :=
    Real.tanh_artanh ⟨by linarith [hnne], harg⟩
  -- the scalar simplifies to 1
  have hscalar : ((1 / Real.sqrt c) * Real.tanh (1 * Real.artanh (Real.sqrt c * ‖w‖))) * (1 / ‖w‖) = 1 := by
    rw [one_mul, hrnd]
    field_simp [hsc.ne.symm, hwpos.ne.symm]
  show ((1 / Real.sqrt c) * Real.tanh (1 * Real.artanh (Real.sqrt c * ‖w‖))) • ((1 / ‖w‖) • w) = w
  have hsmul : ((1 / Real.sqrt c) * Real.tanh (1 * Real.artanh (Real.sqrt c * ‖w‖))) • ((1 / ‖w‖) • w)
      = (((1 / Real.sqrt c) * Real.tanh (1 * Real.artanh (Real.sqrt c * ‖w‖))) * (1 / ‖w‖)) • w := by
    rw [smul_smul]
  rw [hsmul, hscalar, one_smul]

/-- The geodesic direction: ⊖_c x ⊕_c y = x ⊕_c (-x) ⊕_c y is written
    here as the two-step mobius_add_nd composition the standard formula uses.
    We take the direction d = (-x) ⊕_c y (the tangent direction at x
    pointing to y, up to the gyration that the full formula carries). -/
noncomputable def geodesicDir {n : ℕ} (c : ℝ)
    (x y : EuclideanSpace ℝ (Fin n)) : EuclideanSpace ℝ (Fin n) :=
  mobius_add_nd c (-x) y

/-- Geodesic interpolation (t ∈ [0,1]):
    γ_{x→y}(t) = x ⊕_c (geodesicDir x y) ⊗_c t. -/
noncomputable def geodesicInterp {n : ℕ} (c : ℝ) (hc : c > 0)
    (x y : EuclideanSpace ℝ (Fin n)) (t : ℝ) : EuclideanSpace ℝ (Fin n) :=
  let d := geodesicDir c x y
  if hd : d = 0 then x  -- degenerate direction
  else mobius_add_nd c x (mobiusScalarMul c t d hd)

/-- THE t=0 ENDPOINT: γ_{x→y}(0) = x.

    Proof: the direction d = geodesicDir x y. If d = 0, geodesicInterp
    returns x by definition. If d ≠ 0, then 0 ⊗_c d = 0
    (mobiusScalarMul_zero), so γ(0) = x ⊕_c 0 = x (mobius_add_nd_zero).

    This is the on-manifold guarantee that the flow-matching rollout path
    starts at the source keyframe x (GAP C009).
-/
theorem geodesicInterp_zero {n : ℕ} (c : ℝ) (hc : c > 0)
    (x y : EuclideanSpace ℝ (Fin n)) :
    geodesicInterp c hc x y 0 = x := by
  simp only [geodesicInterp, geodesicDir]
  split_ifs with hd
  · rfl
  · have hsmul : mobiusScalarMul c 0 (mobius_add_nd c (-x) y) hd = 0 :=
      mobiusScalarMul_zero c hc _ hd
    rw [hsmul, mobius_add_nd_zero]
