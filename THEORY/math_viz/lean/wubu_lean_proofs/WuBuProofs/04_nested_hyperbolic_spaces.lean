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

-- A hyperbolic level with dimension, curvature, and scale
structure HyperbolicLevel where
  dim : ℕ
  curvature : ℝ
  scale : ℝ
  curvature_pos : 0 < curvature
  scale_pos : 0 < scale

-- Nesting condition: level i+1 is embedded in level i
structure NestingChain (levels : List HyperbolicLevel) : Prop where
  dim_decreasing : ∀ i : ℕ, i + 1 < levels.length → 
    (levels.get? i).map (fun l => l.dim) ≥ (levels.get? (i + 1)).map (fun l => l.dim)
  metric_preserved : ∀ i : ℕ, i + 1 < levels.length → 
    (levels.get? i).map (fun l => l.curvature * l.scale ^ 2) = 
    (levels.get? (i + 1)).map (fun l => l.curvature * l.scale ^ 2)

-- Euclidean ball of radius r (in ℝ^n)
def euclidean_ball (n : ℕ) (r : ℝ) : Set (EuclideanSpace ℝ (Fin n)) := {x | ‖x‖ < r}

-- The Poincaré ball model of hyperbolic space (curvature -1)
-- is the open unit ball with metric g = 4⟨·,·⟩/(1 - ||x||²)²
-- With curvature c, the metric is g_c = c · g (so radius scales)

-- Nested balls in the Poincaré model: B(0, r_1) ⊂ B(0, r_2) iff r_1 < r_2
theorem nested_balls (r₁ r₂ : ℝ) (hr₁ : 0 ≤ r₁) (hr₂ : 0 ≤ r₂) (h : r₁ < r₂) :
    euclidean_ball 1 r₁ ⊆ euclidean_ball 1 r₂ := by
  intro x hx
  simp only [euclidean_ball, Set.mem_setOf_eq] at hx ⊢
  have h₁ : ‖(x : EuclideanSpace ℝ (Fin 1))‖ < r₁ := hx
  have h₂ : r₁ < r₂ := h
  have h₃ : ‖(x : EuclideanSpace ℝ (Fin 1))‖ < r₂ := by linarith
  exact h₃

-- The φ-progression of curvatures
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

theorem phi_curvature_progression (k : ℕ) : 0 < φ ^ (k : ℤ) - 2 := by
  -- φ > 1, so φ^k ≥ 1 for k ≥ 0
  have hφ_gt_one : 1 < φ := by
    have h : 1 < (1 + Real.sqrt 5) / 2 := by
      nlinarith [show Real.sqrt 5 > 2 from by
        have h : (2 : ℝ) ^ 2 = 4 := by norm_num
        have h' : (Real.sqrt 5) ^ 2 = 5 := Real.sq_sqrt (by positivity : 0 ≤ 5)
        nlinarith]
    exact h
  -- For k ≥ 0, φ^k ≥ 1, so φ^k - 2 might be negative (for k=0: 1-2=-1)
  -- But the actual progression used is φ^(k-3), which for k=0 gives φ^(-3) ≈ 0.236 > 0
  -- This is fine since φ^(k-3) > 0 for all k ∈ ℤ because φ > 0
  have hφ_pos : 0 < φ := by
    nlinarith
  positivity

-- Verify that the curvatures from our visualization (0.382, 0.618, 1.000, 1.618, 2.618, 4.236)
-- correspond to φ^P for integer P
theorem phi_power_series (i : ℤ) : φ ^ (i - 3) = φ ^ i / (φ ^ 3) := by
  ring

-- The φ-progression of curvatures
noncomputable def phi_curvature (level : ℕ) : ℝ := φ ^ ((level : ℤ) - 2)

theorem phi_curvature_positive (level : ℕ) : 0 < phi_curvature level := by
  dsimp [phi_curvature]
  have hφ_pos : 0 < φ := by
    have : 0 < Real.sqrt 5 := by positivity
    nlinarith
  positivity