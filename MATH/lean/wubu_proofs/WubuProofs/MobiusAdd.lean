import Mathlib
open Real

set_option maxHeartbeats 0

/-
MobiusAdd.lean — Möbius Addition Preserves the Poincaré Ball (1D)
Ball condition: x^2 < 1/c

Uses factorization: D² - c·N² = (c·x²-1)(c·y²-1)(c·x·y+1)²
-/

noncomputable def mobius_add_1d (c x y : ℝ) : ℝ :=
  ((1 + 2 * c * x * y + c * y ^ 2) * x + (1 - c * x ^ 2) * y) /
  (1 + 2 * c * x * y + c ^ 2 * x ^ 2 * y ^ 2)

lemma denom_nonzero (c a b : ℝ) (hc : c > 0) (ha : a^2 < 1/c) (hb : b^2 < 1/c) :
    1 + 2*c*a*b + c^2*a^2*b^2 ≠ 0 := by
  have hsq : 1 + 2*c*a*b + c^2*a^2*b^2 = (1 + c*a*b)^2 := by ring
  rw [hsq]
  refine pow_ne_zero 2 ?_
  intro hzero
  have : c*a*b = -1 := by nlinarith
  have h_sq_bound : (c*a*b)^2 < 1 := by
    calc
      (c*a*b)^2 = c^2 * (a^2 * b^2) := by ring
      _ < c^2 * ((1/c)*(1/c)) := by
        refine mul_lt_mul_of_pos_left ?_ (by positivity : 0 < c^2)
        nlinarith
      _ = 1 := by field_simp [hc.ne.symm]
  have : (c*a*b)^2 = 1 := by simp [this]
  nlinarith

theorem mobius_add_preserves_ball (c x y : ℝ) (hc : c > 0) (hx : x ^ 2 < 1 / c) (hy : y ^ 2 < 1 / c) :
    (mobius_add_1d c x y) ^ 2 < 1 / c := by
  dsimp [mobius_add_1d]
  set N := ((1 + 2 * c * x * y + c * y ^ 2) * x + (1 - c * x ^ 2) * y) with hN
  set D := (1 + 2 * c * x * y + c ^ 2 * x ^ 2 * y ^ 2) with hD
  have hx_c : c * x ^ 2 < 1 := by
    calc
      c * x ^ 2 < c * (1 / c) := mul_lt_mul_of_pos_left hx hc
      _ = 1 := by field_simp [hc.ne.symm]
  have hy_c : c * y ^ 2 < 1 := by
    calc
      c * y ^ 2 < c * (1 / c) := mul_lt_mul_of_pos_left hy hc
      _ = 1 := by field_simp [hc.ne.symm]
  have hDpos : D > 0 := by
    have hsq : D = (1 + c*x*y)^2 := by ring
    rw [hsq]
    have h_nonzero : 1 + c*x*y ≠ 0 := by
      intro hzero
      have h_cxy_eq_neg1 : c*x*y = -1 := by nlinarith
      have h_sq_bound : (c*x*y)^2 < 1 := by
        calc
          (c*x*y)^2 = c^2 * (x^2 * y^2) := by ring
          _ < c^2 * ((1/c)*(1/c)) := by
            refine mul_lt_mul_of_pos_left ?_ (by positivity : 0 < c^2)
            nlinarith
          _ = 1 := by field_simp [hc.ne.symm]
      have h_sq_eq_1 : (c*x*y)^2 = 1 := by
        calc
          (c*x*y)^2 = (-1)^2 := by rw [h_cxy_eq_neg1]
          _ = 1 := by norm_num
      nlinarith
    exact sq_pos_iff.mpr h_nonzero
  have h_factor : D ^ 2 - c * N ^ 2 = (c * x ^ 2 - 1) * (c * y ^ 2 - 1) * (c * x * y + 1) ^ 2 := by
    dsimp [N, D]; ring
  have h_pos : D ^ 2 - c * N ^ 2 > 0 := by
    rw [h_factor]
    have h_factor1 : c * x ^ 2 - 1 < 0 := by nlinarith
    have h_factor2 : c * y ^ 2 - 1 < 0 := by nlinarith
    have h_sq_nonneg : (c * x * y + 1) ^ 2 > 0 := by
      refine sq_pos_iff.mpr ?_
      intro hzero
      have h_cxy_eq_neg1 : c * x * y = -1 := by nlinarith
      have h_sq_lt_1 : (c*x*y)^2 < 1 := by
        calc
          (c*x*y)^2 = c^2 * (x^2 * y^2) := by ring
          _ < c^2 * ((1/c)*(1/c)) := by
            refine mul_lt_mul_of_pos_left ?_ (by positivity : 0 < c^2)
            nlinarith
          _ = 1 := by field_simp [hc.ne.symm]
      have h_sq_eq_1 : (c*x*y)^2 = 1 := by
        calc
          (c*x*y)^2 = (-1)^2 := by rw [h_cxy_eq_neg1]
          _ = 1 := by norm_num
      nlinarith
    have : (c * x ^ 2 - 1) * (c * y ^ 2 - 1) > 0 := by nlinarith
    nlinarith
  have h : (N / D) ^ 2 < 1 / c := by
    field_simp [hDpos.ne.symm, hc.ne.symm]
    nlinarith [h_pos]
  exact h

/-- 1D Möbius left-cancellation: (-x) ⊕_c (x ⊕_c y) = y.

    This is the scalar reduction of the gyro-left-cancellation law (Ungar
    2005; Ganea et al. 2018) — the property that makes Möbius subtraction
    well-defined from the left. It is load-bearing for the geodesic γ(1)=y
    endpoint (GeodesicInterp) and for the C bridge [T10].

    Proof: the outer Möbius addition is a rational function; cross-multiply
    (`div_eq_iff`, using the outer denominator ≠ 0 from ball membership of
    -x and x⊕y), substitute the inner Möbius addition, clear the inner
    denominator with `field_simp`, and close the resulting polynomial
    identity with `ring`. (Plain `field_simp; ring` leaves a residual
    denominator-inverse that doesn't match — the cross-multiply route
    avoids that friction.) -/
theorem mobius_add_1d_left_cancel (c x y : ℝ) (hc : c > 0)
    (hx : x ^ 2 < 1 / c) (hy : y ^ 2 < 1 / c) :
    mobius_add_1d c (-x) (mobius_add_1d c x y) = y := by
  have hden : 1 + 2 * c * x * y + c ^ 2 * x ^ 2 * y ^ 2 ≠ 0 := denom_nonzero c x y hc hx hy
  have hz_ball : mobius_add_1d c x y ^ 2 < 1 / c := mobius_add_preserves_ball c x y hc hx hy
  have hden2 : 1 + 2 * c * (-x) * (mobius_add_1d c x y)
      + c ^ 2 * (-x) ^ 2 * (mobius_add_1d c x y) ^ 2 ≠ 0 :=
    denom_nonzero c (-x) (mobius_add_1d c x y) hc (by rw [neg_sq]; exact hx) hz_ball
  show ((1 + 2 * c * (-x) * (mobius_add_1d c x y) + c * (mobius_add_1d c x y) ^ 2) * (-x)
        + (1 - c * (-x) ^ 2) * (mobius_add_1d c x y))
       / (1 + 2 * c * (-x) * (mobius_add_1d c x y) + c ^ 2 * (-x) ^ 2 * (mobius_add_1d c x y) ^ 2) = y
  rw [div_eq_iff hden2]
  have hzdef : mobius_add_1d c x y
      = ((1 + 2 * c * x * y + c * y ^ 2) * x + (1 - c * x ^ 2) * y)
        / (1 + 2 * c * x * y + c ^ 2 * x ^ 2 * y ^ 2) := rfl
  rw [hzdef]
  field_simp [hden]
  ring
