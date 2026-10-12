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
