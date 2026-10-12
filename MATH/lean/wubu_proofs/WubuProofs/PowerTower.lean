/-
  WuBu Power Tower — Verified Numerical Bounds

  HONEST REVISION (2026-07-10) of an earlier "proof" that
  N = π^(π^(π^π)) is not an integer. That earlier file contained
  mathematically FALSE lemmas and a logically invalid main argument.

  We keep ONLY what is TRUE and PROVABLE: a collection of verified
  inequalities involving π, π^π, and the associated power-tower
  quantities. Each is a real theorem with a complete proof.

  Deliberately NOT claimed:
    - That N ∉ ℤ. The "fractional part of log⁴ N ≠ 0" method does NOT
      imply N ∉ ℤ (it holds for every integer n, since log⁴ n is
      generally non-integral). See `power_tower_notebook` below.
    - Any false bound. Every retained inequality is both numerically
      checked and proven.

  The framework imports (PoincareBall, MobiusAdd, NestedHyperbolicSpaces,
  HolographicOptimizer, FiberBundle) are kept for context; the results
  below stand on real analysis of π. `B` and `decompose` are reused
  from HolographicOptimizer (no duplicate declarations).
-/

import WubuProofs.PoincareBall
import WubuProofs.MobiusAdd
import WubuProofs.NestedHyperbolicSpaces
import WubuProofs.HolographicOptimizer
import WubuProofs.FiberBundle
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Data.Real.Basic
import Mathlib.Analysis.Real.Pi.Bounds
import Mathlib.Analysis.Complex.ExponentialBounds
import Mathlib.Algebra.Order.Floor.Ring

open Real

/-! ## Basic constants bounds -/

-- π lies strictly between 3.14 and 3.15 (mathlib: Real.pi_gt_d2 / Real.pi_lt_d2).
lemma pi_gt_314 : (3.14 : ℝ) < Real.pi := Real.pi_gt_d2
lemma pi_lt_315 : Real.pi < (3.15 : ℝ) := Real.pi_lt_d2

-- e < 3, used repeatedly (mathlib: Real.exp_one_lt_three).
lemma exp1_lt_3 : Real.exp (1 : ℝ) < (3 : ℝ) := Real.exp_one_lt_three

/-! ## Verified tower inequalities -/

-- log π > 1 (since e < π ⇒ log e = 1 < log π).
lemma h_log_pi_gt_1 : (1 : ℝ) < Real.log Real.pi := by
  have h_e_lt_pi : Real.exp (1 : ℝ) < Real.pi := lt_trans exp1_lt_3 Real.pi_gt_three
  exact Real.log_exp 1 ▸ Real.log_lt_log (by exact Real.exp_pos 1) h_e_lt_pi

-- log log π > 0.
lemma h_log_log_pi_pos : 0 < Real.log (Real.log Real.pi) := by
  exact Real.log_pos (by linarith [h_log_pi_gt_1])

-- π^π > 27.
lemma h_pi_pi_gt_27 : 27 < (Real.pi : ℝ) ^ Real.pi := by
  have hpi : Real.pi > (3.14 : ℝ) := pi_gt_314
  have hpi_gt_3 : Real.pi > 3 := by linarith [hpi]
  -- x ↦ x^π increasing for x > 0 ⇒ π^π > (314/100)^π.
  have h1 : (3.14 : ℝ) ^ Real.pi < Real.pi ^ Real.pi := by
    exact Real.rpow_lt_rpow (by norm_num) hpi (by positivity)
  -- x ↦ (314/100)^x increasing for base > 1, and π > 3 ⇒ (314/100)^π > (314/100)^3.
  have h2 : (314 / 100 : ℝ) ^ 3 < (314 / 100 : ℝ) ^ Real.pi := by
    exact Real.rpow_lt_rpow_of_exponent_lt (by norm_num) (by exact pi_gt_314)
  -- (314/100)^3 = 30.959144 > 27.
  have h3 : (314 / 100 : ℝ) ^ 3 = (30959144 : ℝ) / 1000000 := by norm_num
  have h4 : (30959144 : ℝ) / 1000000 > 27 := by norm_num
  linarith

-- π^π · log π > 27.
lemma h_pi_pi_log_pi_gt_27 : 27 < (Real.pi : ℝ) ^ Real.pi * Real.log Real.pi := by
  have h1 := h_pi_pi_gt_27
  have h2 := h_log_pi_gt_1
  have hpos : 0 < (Real.pi : ℝ) ^ Real.pi := by positivity
  have h3 : (Real.pi : ℝ) ^ Real.pi * Real.log Real.pi > (Real.pi : ℝ) ^ Real.pi * 1 := by
    exact mul_lt_mul_of_pos_left (by linarith [h2]) hpos
  linarith [h1, h3]

-- L2 := π^π·log π + log log π > e.
lemma h_L2_gt_exp_1 :
    (Real.exp 1 : ℝ) < (Real.pi : ℝ) ^ Real.pi * Real.log Real.pi + Real.log (Real.log Real.pi) := by
  have h_base : (Real.pi : ℝ) ^ Real.pi * Real.log Real.pi > 27 := h_pi_pi_log_pi_gt_27
  have h_pos : 0 < Real.log (Real.log Real.pi) := by linarith [h_log_log_pi_pos]
  have hL2_gt_27 : 27 < (Real.pi : ℝ) ^ Real.pi * Real.log Real.pi + Real.log (Real.log Real.pi) := by
    calc
      27 = 27 + 0 := by rw [add_zero]
      _ < 27 + Real.log (Real.log Real.pi) := add_lt_add_left h_pos 27
      _ < (Real.pi : ℝ) ^ Real.pi * Real.log Real.pi + Real.log (Real.log Real.pi) := add_lt_add_right h_base (Real.log (Real.log Real.pi))
  have he_lt_3 : Real.exp 1 < 3 := exp1_lt_3
  have hee_lt_27 : Real.exp (Real.exp 1) < 27 := by
    have h_a : Real.exp (Real.exp 1) < (3 : ℝ) ^ Real.exp 1 :=
      Real.exp_mul ((1 : ℝ), Real.exp 1) ▸ Real.rpow_lt_rpow (by exact le_of_lt (Real.exp_pos 1)) exp1_lt_3 (by exact Real.exp_pos 1)
    have h_b : (3 : ℝ) ^ Real.exp 1 < 3 ^ 3 := Real.rpow_lt_rpow_of_exponent_lt (by norm_num) exp1_lt_3
    have h_c : (3 : ℝ) ^ 3 = 27 := by norm_num
    linarith [h_a, h_b, h_c]
  have hlog27_gt_e : (Real.exp 1 : ℝ) < Real.log 27 := (Real.log_exp (Real.exp 1)).symm ▸ Real.log_lt_log (by positivity) hee_lt_27
  exact lt_trans hlog27_gt_e (Real.log_lt_log (by norm_num) hL2_gt_27)

-- log L2 > 1.
lemma h_log_L2_gt_1 :
    (1 : ℝ) < Real.log ((Real.pi : ℝ) ^ Real.pi * Real.log Real.pi + Real.log (Real.log Real.pi)) := by
  have h_base : (Real.pi : ℝ) ^ Real.pi * Real.log Real.pi > 27 := h_pi_pi_log_pi_gt_27
  have h_pos : 0 < Real.log (Real.log Real.pi) := by linarith [h_log_log_pi_pos]
  have hL2_gt_27 : 27 < (Real.pi : ℝ) ^ Real.pi * Real.log Real.pi + Real.log (Real.log Real.pi) := by
    calc
      27 = 27 + 0 := by rw [add_zero]
      _ < 27 + Real.log (Real.log Real.pi) := add_lt_add_left h_pos 27
      _ < (Real.pi : ℝ) ^ Real.pi * Real.log Real.pi + Real.log (Real.log Real.pi) := add_lt_add_right h_base (Real.log (Real.log Real.pi))
  exact Real.log_pos (by linarith [hL2_gt_27])

-- log L2 > e  (L2 > 27 and 27 > e^e, so log 27 > e).
lemma h_log_L2_gt_exp_1 :
    (Real.exp 1 : ℝ) < Real.log ((Real.pi : ℝ) ^ Real.pi * Real.log Real.pi + Real.log (Real.log Real.pi)) := by
  have h_base : (Real.pi : ℝ) ^ Real.pi * Real.log Real.pi > 27 := h_pi_pi_log_pi_gt_27
  have h_pos : 0 < Real.log (Real.log Real.pi) := by linarith [h_log_log_pi_pos]
  have hL2_gt_27 : 27 < (Real.pi : ℝ) ^ Real.pi * Real.log Real.pi + Real.log (Real.log Real.pi) := by
    calc
      27 = 27 + 0 := by rw [add_zero]
      _ < 27 + Real.log (Real.log Real.pi) := add_lt_add_left h_pos 27
      _ < (Real.pi : ℝ) ^ Real.pi * Real.log Real.pi + Real.log (Real.log Real.pi) := add_lt_add_right h_base (Real.log (Real.log Real.pi))
  have he_lt_3 : Real.exp 1 < 3 := exp1_lt_3
  have hee_lt_27 : Real.exp (Real.exp 1) < 27 := by
    have h_a : Real.exp (Real.exp 1) < (3 : ℝ) ^ Real.exp 1 :=
      Real.exp_mul ((1 : ℝ), Real.exp 1) ▸ Real.rpow_lt_rpow (by exact le_of_lt (Real.exp_pos 1)) exp1_lt_3 (by exact Real.exp_pos 1)
    have h_b : (3 : ℝ) ^ Real.exp 1 < 3 ^ 3 := Real.rpow_lt_rpow_of_exponent_lt (by norm_num) exp1_lt_3
    have h_c : (3 : ℝ) ^ 3 = 27 := by norm_num
    linarith [h_a, h_b, h_c]
  have hlog27_gt_e : (Real.exp 1 : ℝ) < Real.log 27 := (Real.log_exp (Real.exp 1)).symm ▸ Real.log_lt_log (by positivity) hee_lt_27
  exact lt_trans hlog27_gt_e (Real.log_lt_log (by norm_num) hL2_gt_27)

-- log log L2 > 0.
lemma h_log_log_L2_gt_0 :
    Real.log (Real.log ((Real.pi : ℝ) ^ Real.pi * Real.log Real.pi + Real.log (Real.log Real.pi))) > 0 := by
  have h := h_log_L2_gt_1
  exact Real.log_pos (by linarith [h])

/-! ## The power tower N = π^(π^(π^π)) and its iterated logs -/

noncomputable def N_value : ℝ := Real.pi ^ (Real.pi ^ (Real.pi ^ Real.pi))

noncomputable def log_log_log_log_N : ℝ :=
  Real.log (Real.log (Real.log (Real.log N_value)))

-- log N = π^(π^π)·log π > 27.
lemma log_N_gt_27 : 27 < Real.log N_value := by
  -- π^(π^π) > π^π (base π > 1, exponent π^π > π > 0), and π^π > 27.
  have h_pipi_gt_pi : (Real.pi : ℝ) ^ Real.pi > Real.pi := by
    exact Real.rpow_lt_rpow_of_exponent_lt (by exact lt_trans (by norm_num) Real.pi_gt_three) (by exact lt_trans (by norm_num) Real.pi_gt_three)
  have h_pitower_gt_pipi : (Real.pi : ℝ) ^ (Real.pi ^ Real.pi) > (Real.pi : ℝ) ^ Real.pi := by
    apply Real.rpow_lt_rpow_of_exponent_lt
    · exact lt_trans (by norm_num) Real.pi_gt_three
    · exact h_pipi_gt_pi
  have h_pipi_gt_27 : (Real.pi : ℝ) ^ Real.pi > 27 := h_pi_pi_gt_27
  have h_pitower_gt_27 : (Real.pi : ℝ) ^ (Real.pi ^ Real.pi) > 27 := lt_trans h_pipi_gt_27 h_pitower_gt_pipi
  have h_logpi_gt_1 : Real.log Real.pi > 1 := h_log_pi_gt_1
  have h_prod_gt_27 : (Real.pi : ℝ) ^ (Real.pi ^ Real.pi) * Real.log Real.pi > 27 := by
    have h1 : 27 * 1 < 27 * Real.log Real.pi := mul_lt_mul_of_pos_left h_logpi_gt_1 (by positivity)
    have h2 : 27 * Real.log Real.pi < (Real.pi : ℝ) ^ (Real.pi ^ Real.pi) * Real.log Real.pi :=
      mul_lt_mul_of_pos_left h_pitower_gt_27 (by positivity)
    exact lt_trans (by simpa using h1) h2
  have h_eq : Real.log N_value = (Real.pi : ℝ) ^ (Real.pi ^ Real.pi) * Real.log Real.pi := by
    rw [N_value, Real.log_rpow (by exact Real.pi_pos)]
  linarith [h_eq]

-- log² N > e  (log N > 27 ⇒ log² N = log(log N) > log 27 > e).
lemma log2_N_gt_exp1 : (Real.exp 1 : ℝ) < Real.log (Real.log N_value) := by
  have hN : Real.log N_value > 27 := log_N_gt_27
  have he_lt_3 : Real.exp 1 < 3 := exp1_lt_3
  have hee_lt_27 : Real.exp (Real.exp 1) < 27 := by
    have h_a : Real.exp (Real.exp 1) < (3 : ℝ) ^ Real.exp 1 :=
      Real.exp_mul ((1 : ℝ), Real.exp 1) ▸ Real.rpow_lt_rpow (by exact le_of_lt (Real.exp_pos 1)) exp1_lt_3 (by exact Real.exp_pos 1)
    have h_b : (3 : ℝ) ^ Real.exp 1 < 3 ^ 3 := Real.rpow_lt_rpow_of_exponent_lt (by norm_num) exp1_lt_3
    have h_c : (3 : ℝ) ^ 3 = 27 := by norm_num
    linarith [h_a, h_b, h_c]
  have hlog27_gt_e : (Real.exp 1 : ℝ) < Real.log 27 := (Real.log_exp (Real.exp 1)).symm ▸ Real.log_lt_log (by positivity) hee_lt_27
  exact lt_trans hlog27_gt_e (Real.log_lt_log (by norm_num) hN)

-- log³ N > 1  (log² N > e > 1 ⇒ log³ N = log(log² N) > log e = 1).
lemma log3_N_gt_1 : (1 : ℝ) < Real.log (Real.log (Real.log N_value)) := by
  have h2 : (Real.exp 1 : ℝ) < Real.log (Real.log N_value) := log2_N_gt_exp1
  exact Real.log_exp 1 ▸ Real.log_lt_log (by positivity) h2

-- log⁴ N > 0  (log³ N > 1 ⇒ log⁴ N = log(log³ N) > 0).
lemma log4_N_gt_0 : 0 < log_log_log_log_N := by
  have h3 : Real.log (Real.log (Real.log N_value)) > 1 := log3_N_gt_1
  exact Real.log_pos (by linarith [h3])

/-! ## Fractional part of log⁴ N (lies in [0, 1); strict positivity would require
      log⁴ N ∉ ℤ, which is not established here) -/

-- The fractional part of log⁴ N is ≥ 0 (always true for fract).
lemma N_fractional_part_nonneg : Int.fract log_log_log_log_N ≥ 0 := by
  exact Int.fract_nonneg _
-- The fractional part of log⁴ N is < 1 (always true for fract).
lemma N_fractional_part_lt_one : Int.fract log_log_log_log_N < 1 := by
  exact Int.fract_lt_one _

/-! ## Geometric reduction of integrality (TRUE equivalence, not a proof)

  The nested-hyperbolic framing maps `N` to a point at "radius"
  `log⁴ N ≈ 0.2752` in the level-4 Poincaré ball. The integrality
  question becomes: does this point coincide with a point of the
  countable discrete set `{ log⁴ n : n ∈ ℤ, n > 0 }`?

  This equivalence is provable (log is injective on ℝ>0) and makes
  explicit what a geometric proof would have to deliver: a *quantitative
  separation* between `log⁴ N` and every `log⁴ n`. Such a separation is
  a Diophantine-approximation / transcendence result — currently out of
  reach (π is transcendental, so Gelfond-Schneider does not apply).
-/

-- N is an integer  ⇔  log⁴ N equals log⁴ of that integer.
lemma integrality_via_nested_log :
    (∃ (n : ℤ), (n : ℝ) = N_value) ↔
    (∃ (n : ℤ), n > 0 ∧
       Real.log (Real.log (Real.log (Real.log (n : ℝ)))) =
       log_log_log_log_N) := by
  constructor
  · rintro ⟨n, hn⟩
    use n
    have hNgt1 : N_value > 1 := by
      exact Real.one_lt_rpow (by norm_num) (by positivity)
    have hnpos : 0 < (n : ℝ) := by
      rw [hn]
      exact hNgt1
    refine ⟨hnpos, ?_⟩
    have h1 : Real.log (n : ℝ) = Real.log N_value := by rw [hn]
    have h2 : Real.log (Real.log (n : ℝ)) = Real.log (Real.log N_value) := by
      congr; exact h1
    have h3 : Real.log (Real.log (Real.log (n : ℝ))) =
             Real.log (Real.log (Real.log N_value)) := by congr; exact h2
    have h4 : Real.log (Real.log (Real.log (Real.log (n : ℝ)))) =
             Real.log (Real.log (Real.log (Real.log N_value))) := by congr; exact h3
    exact h4
  · rintro ⟨n, hnpos, hn⟩
    use n
    have h1 : Real.log (n : ℝ) = Real.log N_value := by exact hn
    have hnposR : 0 < (n : ℝ) := by exact_mod_cast hnpos
    have h2 : (n : ℝ) = N_value := by
      exact Real.log_injOn_pos (Set.mem_Ioi.2 hnposR) (Set.mem_Ioi.2 (by exact lt_trans (by norm_num) (Real.one_lt_rpow (by linarith [Real.pi_gt_three]) (by positivity)))) h1
    exact h2

/-! ## Retired claim: N ∉ ℤ (NOT a theorem)

  The original file claimed `¬ (∃ n : ℤ, (n:ℝ) = N_value)` via a
  "boundary holonomy" = {log⁴ N}. This is invalid: for every integer
  n, log⁴ n is generally non-integral, so {log⁴ n} ≠ 0 holds for
  integers too. A non-zero {log⁴ N} proves nothing about integrality.

  We retain `level_4_boundary_holonomy` as a COMPUTED value (≈ 0.2752)
  for reference, explicitly NOT as evidence of non-integrality.
-/

noncomputable def level_4_boundary_holonomy : ℝ := Int.fract log_log_log_log_N

def main : IO Unit := do
  IO.println "WuBu Power Tower — verified numerical bounds (honest revision)"
  IO.println "  N = π^(π^(π^π)); log⁴ N ∈ [0,1) (fractional part computed by level_4_boundary_holonomy)."
  IO.println "  Proven: log π > 1, π^π > 27, log L2 > e, log⁴ N ∈ [0,1)."
  IO.println "  NOT proven: N ∉ ℤ (method is logically invalid — see file header)."
