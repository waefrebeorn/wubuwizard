import Mathlib
open Real InnerProductSpace EuclideanSpace

set_option maxHeartbeats 0

/-
MobiusAddND.lean — Möbius addition preserves the Poincaré ball in n dimensions.

  x ⊕_c y = (1/D) • N   where
    N = (1 + 2c⟨x,y⟩ + c‖y‖²) • x + (1 − c‖x‖²) • y
    D = 1 + 2c⟨x,y⟩ + c²‖x‖²‖y‖²

Key identity: after expanding ‖N‖² through inner-product linearity (which
depends only on ⟨x,y⟩, ‖x‖², ‖y‖² — this is why the nD proof reduces to the
1D one algebraically),

  D² − c‖N‖² = (1 − c‖x‖²)(1 − c‖y‖²)(1 + c⟨x,y⟩)²

Since ‖x‖², ‖y‖² < 1/c the RHS is strictly positive, and D > 0, hence
‖N‖² < D²/c, i.e. ‖x ⊕_c y‖² < 1/c. ∎
-/

/-- Möbius (gyrovector) addition in the n-dimensional Poincaré ball. -/
noncomputable def mobius_add_nd {n : ℕ} (c : ℝ) (x y : EuclideanSpace ℝ (Fin n)) :
    EuclideanSpace ℝ (Fin n) :=
  (1 / (1 + 2 * c * inner ℝ x y + c ^ 2 * ‖x‖ ^ 2 * ‖y‖ ^ 2)) •
    ((1 + 2 * c * inner ℝ x y + c * ‖y‖ ^ 2) • x + (1 - c * ‖x‖ ^ 2) • y)

theorem mobius_add_nd_preserves_ball {n : ℕ} (c : ℝ) (hc : c > 0)
    (x y : EuclideanSpace ℝ (Fin n)) (hx : ‖x‖ ^ 2 < 1 / c) (hy : ‖y‖ ^ 2 < 1 / c) :
    ‖mobius_add_nd c x y‖ ^ 2 < 1 / c := by
  -- rename the pieces of the unfolded definition
  set α := 1 + 2 * c * inner ℝ x y + c * ‖y‖ ^ 2 with hα
  set β := 1 - c * ‖x‖ ^ 2 with hβ
  set p := inner ℝ x y with hp
  set a := ‖x‖ ^ 2 with ha
  set b := ‖y‖ ^ 2 with hb
  set D := 1 + 2 * c * p + c ^ 2 * a * b with hD
  -- the definition, once unfolded, is exactly (1/D) • (α • x + β • y)
  have hN : mobius_add_nd c x y = (1 / D) • (α • x + β • y) := by
    unfold mobius_add_nd
    rw [hD, hα, hβ]
  -- basic inner-product facts
  have ha_eq : inner ℝ x x = a := (real_inner_self_eq_norm_sq x).trans ha.symm
  have hb_eq : inner ℝ y y = b := (real_inner_self_eq_norm_sq y).trans hb.symm
  -- (0) D > 0:  D = (1+cp)² + c²(ab − p²)  and p² ≤ ab by Cauchy–Schwarz
  have hDpos : D > 0 := by
    have hcs : p ^ 2 ≤ a * b := by
      have h := real_inner_mul_inner_self_le x y
      rw [← hp, ha_eq, hb_eq] at h
      simpa [pow_two] using h
    have h1 : D = (1 + c * p) ^ 2 + c ^ 2 * (a * b - p ^ 2) := by
      unfold D
      ring
    rw [h1]
    have h2 : c ^ 2 * (a * b - p ^ 2) ≥ 0 := by
      apply mul_nonneg (sq_nonneg c)
      linarith [hcs]
    have h3 : (1 + c * p) ^ 2 ≥ 0 := sq_nonneg _
    by_contra hD
    push Not at hD
    have h4 : (1 + c * p) ^ 2 = 0 := by linarith
    have h5 : c * p = -1 := by nlinarith [h4]
    have h6 : p ^ 2 = 1 / c ^ 2 := by
      have h7 : p = -1 / c := by
        have h8 : c * p = -1 := h5
        field_simp at h8 ⊢
        linarith
      rw [h7]
      ring
    -- ab < 1/c² from h1ca/h1cb (established below, restated here)
    have hca : c * a < 1 := by
      have h1 : c * a < c * (1 / c) := mul_lt_mul_of_pos_left hx hc
      rwa [mul_one_div_cancel hc.ne'] at h1
    have hcb : c * b < 1 := by
      have h1 : c * b < c * (1 / c) := mul_lt_mul_of_pos_left hy hc
      rwa [mul_one_div_cancel hc.ne'] at h1
    have hn1 : 0 ≤ c * a := by positivity
    have hn2 : 0 ≤ c * b := by positivity
    have h9 : a * b < 1 / c ^ 2 := by nlinarith [mul_nonneg hn1 hn2]
    linarith [h6]
  -- (1) ‖(1/D) • N‖² = ‖N‖² / D²
  have hnorm_div : ‖(1 / D) • (α • x + β • y)‖ ^ 2 = ‖α • x + β • y‖ ^ 2 / D ^ 2 := by
    rw [norm_smul, norm_div, norm_one, Real.norm_of_nonneg (le_of_lt hDpos)]
    ring
  -- (2) expand ‖αx + βy‖² = α²a + 2αβp + β²b through inner-product linearity
  have hN_expand : ‖α • x + β • y‖ ^ 2 = α ^ 2 * a + 2 * α * β * p + β ^ 2 * b := by
    rw [norm_add_sq_real]
    rw [norm_smul, norm_smul, ha, hb]
    rw [inner_smul_left x (β • y) α, inner_smul_right x y β]
    have h1 : (starRingEnd ℝ) α = α := rfl
    rw [h1]
    rw [Real.norm_eq_abs, Real.norm_eq_abs]
    rw [mul_pow, mul_pow, sq_abs, sq_abs, ← hp]
    ring
  -- (3) the factorization identity (compact form):
  --     D² − c‖N‖² = (1−ca)(1−cb)·[(1+cp)² + c²(ab−p²)]
  have hfactor : D ^ 2 - c * ‖α • x + β • y‖ ^ 2
      = (1 - c * a) * (1 - c * b) * ((1 + c * p) ^ 2 + c ^ 2 * (a * b - p ^ 2)) := by
    rw [hN_expand]
    unfold D α β
    rw [ha, hb, hp]
    ring
  -- (4) positivity of the two (1 − c·) factors
  have h1ca : 1 - c * a > 0 := by
    have hca : c * a < 1 := by
      have h1 : c * a < c * (1 / c) := mul_lt_mul_of_pos_left hx hc
      rwa [mul_one_div_cancel hc.ne'] at h1
    linarith
  have h1cb : 1 - c * b > 0 := by
    have hcb : c * b < 1 := by
      have h1 : c * b < c * (1 / c) := mul_lt_mul_of_pos_left hy hc
      rwa [mul_one_div_cancel hc.ne'] at h1
    linarith
  -- (5) the bracket is strictly positive: (1+cp)² ≥ 0, c²(ab−p²) ≥ 0, and
  --     both zero simultaneously would need cp = −1 and p² = ab, impossible
  --     since cp = −1 gives p² = 1/c² > ab
  have hbracket_pos : (1 + c * p) ^ 2 + c ^ 2 * (a * b - p ^ 2) > 0 := by
    have hcs : p ^ 2 ≤ a * b := by
      have h := real_inner_mul_inner_self_le x y
      rw [← hp, ha_eq, hb_eq] at h
      simpa [pow_two] using h
    by_contra hb
    push Not at hb
    have h1 : (1 + c * p) ^ 2 = 0 := by nlinarith [hcs]
    have h2 : c * p = -1 := by nlinarith [h1]
    have h3 : p ^ 2 = 1 / c ^ 2 := by
      have h4 : p = -1 / c := by
        have h5 : c * p = -1 := h2
        field_simp at h5 ⊢
        linarith
      rw [h4]
      ring
    have hca : c * a < 1 := by linarith [h1ca]
    have hcb : c * b < 1 := by linarith [h1cb]
    have hn1 : 0 ≤ c * a := by positivity
    have hn2 : 0 ≤ c * b := by positivity
    have hab_lt : a * b < 1 / c ^ 2 := by nlinarith [mul_nonneg hn1 hn2]
    linarith [h3]
  have hfac_pos : (1 - c * a) * (1 - c * b) * ((1 + c * p) ^ 2 + c ^ 2 * (a * b - p ^ 2)) > 0 := by
    apply mul_pos
    · apply mul_pos h1ca h1cb
    · exact hbracket_pos
  -- (6) main inequality: c‖N‖² < D²
  have hmain : c * ‖α • x + β • y‖ ^ 2 < D ^ 2 := by
    have h := hfactor
    linarith [hfac_pos]
  -- (7) finish: rewrite the goal through hN, then the arithmetic above
  rw [hN, hnorm_div]
  rw [div_lt_div_iff₀ (by positivity : 0 < D ^ 2) (by positivity : 0 < c)]
  linarith [hmain, pow_pos hc 2, pow_pos hDpos 2]
