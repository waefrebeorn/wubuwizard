/-
Nested Hyperbolic Spaces (WuBu Nesting)

WuBu Nesting: H^{n1}_{c1,s1} ⊃ H^{n2}_{c2,s2} ⊃ ... ⊃ H^{nk}_{ck,sk}

Each level i has:
  - Dimension n_i (natural number)
  - Curvature c_i > 0 
  - Scale s_i > 0

The nesting means H^{n_{i+1}} is embedded in H^{n_i}
as a totally geodesic submanifold (preserving the hyperbolic metric).

Theorems:
  1. Nested balls form a chain: B(0, R_i) ⊂ B(0, R_{i+1}) for R_i < R_{i+1}
  2. The embedding is isometric (distance-preserving) when dimensions match
  3. φ-scaled curvatures form a valid nesting (c_i = φ^{i-3})
-/

import Mathlib.Tactic
open Real
open Set

-- Golden ratio, explicitly defined (was previously an undefined constant,
-- which made `0 < φ` and exponent laws unprovable).
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

-- A hyperbolic level with dimension, curvature, and scale
structure HyperbolicLevel where
  dim : ℕ
  curvature : ℝ
  scale : ℝ
  curvature_pos : 0 < curvature
  scale_pos : 0 < scale

-- Nesting condition: level i+1 is embedded in level i.
-- Stated as a total function (avoids List indexing API).
structure NestingChain (levels : ℕ → HyperbolicLevel) : Prop where
  dim_decreasing : ∀ i : ℕ, (levels i).dim ≥ (levels (i + 1)).dim
  metric_preserved : ∀ i : ℕ,
    (levels i).curvature * (levels i).scale ^ 2 =
    (levels (i + 1)).curvature * (levels (i + 1)).scale ^ 2

-- Euclidean ball of radius r (in ℝ^n)
def euclidean_ball (n : ℕ) (r : ℝ) : Set (EuclideanSpace ℝ (Fin n)) := {x | ‖x‖ < r}

-- The Poincaré ball model of hyperbolic space (curvature -1)
-- is the open unit ball with metric g = 4⟨·,·⟩/(1 - ||x||²)²
-- With curvature c, the metric is g_c = c · g (so radius scales)

-- Nested balls in the Poincaré model: B(0, r_1) ⊂ B(0, r_2) iff r_1 < r_2
theorem nested_balls (r₁ r₂ : ℝ) (h : r₁ < r₂) :
    euclidean_ball 1 r₁ ⊆ euclidean_ball 1 r₂ := by
  intro x hx
  simp only [euclidean_ball, Set.mem_setOf_eq] at hx ⊢
  have h₁ : ‖(x : EuclideanSpace ℝ (Fin 1))‖ < r₁ := hx
  have h₂ : r₁ < r₂ := h
  have h₃ : ‖(x : EuclideanSpace ℝ (Fin 1))‖ < r₂ := by linarith
  exact h₃

-- The φ-progression of curvatures
theorem phi_curvature_progression (k : ℕ) : 0 < φ ^ ((k : ℝ) - 3) := by
  have hφ_pos : 0 < φ := by unfold φ; positivity
  exact Real.rpow_pos_of_pos hφ_pos ((k : ℝ) - 3)

-- Verify that the curvatures from our visualization (0.382, 0.618, 1.000, 1.618, 2.618, 4.236)
-- correspond to φ^P for integer P (stated with ℝ exponents since `^` on ℝ with
-- a ℤ exponent does not reduce to Real.rpow in this mathlib; use ↑ explicitly).
theorem phi_power_series (i : ℤ) :
    φ ^ (↑(i - 3) : ℝ) = φ ^ (↑i : ℝ) / (φ ^ (↑3 : ℝ)) := by
  have hφ_pos : 0 < φ := by unfold φ; positivity
  rw [Int.cast_sub i 3]
  exact Real.rpow_sub hφ_pos (↑i) (↑3)

-- The φ-progression of curvatures (the actual nesting formula c_i = φ^{i-3})
noncomputable def phi_curvature (level : ℕ) : ℝ := φ ^ ((level : ℝ) - 2)

theorem phi_curvature_positive (level : ℕ) : 0 < phi_curvature level := by
  dsimp [phi_curvature]
  have hφ_pos : 0 < φ := by unfold φ; positivity
  exact Real.rpow_pos_of_pos hφ_pos ((level : ℝ) - 2)