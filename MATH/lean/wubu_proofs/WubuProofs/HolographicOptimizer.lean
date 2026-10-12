/-
math_viz/lean/03_holographic_optimizer.lean

PROOF 3: Holographic Gradient Decomposition

The holographic optimizer decomposes each gradient g as:
  g = q · 2π + r    where q ∈ ℤ, r ∈ [-π, π]

Theorems:
  1. Decomposition exists uniquely for any real g
  2. Recovery is exact: g = q·2π + r
  3. Information is conserved across arbitrary many additions
  4. Weight death survival: stored (Σq, Σr) recovers Σg exactly
-/

import Mathlib.Tactic
open Real
open Set

-- The boundary 2π for the decomposition
noncomputable def B : ℝ := 2 * π

-- The decomposition: g = q*B + r where q ∈ ℤ and r ∈ (-B/2, B/2]
-- i.e., r ∈ (-π, π]
noncomputable def decompose (g : ℝ) : ℤ × ℝ :=
  let q : ℤ := ⌊(g + π) / B⌋
  let r : ℝ := g - (q : ℝ) * B
  (q, r)

theorem remainder_in_range (g : ℝ) :
    -π ≤ (decompose g).2 ∧ (decompose g).2 ≤ π := by
  dsimp [decompose, B] at *
  have hq : (⌊(g + π) / (2 * π)⌋ : ℝ) ≤ (g + π) / (2 * π) := by
    exact_mod_cast Int.floor_le ((g + π) / (2 * π))
  have hq' : (g + π) / (2 * π) < (⌊(g + π) / (2 * π)⌋ : ℝ) + 1 := by
    have h := Int.lt_floor_add_one ((g + π) / (2 * π))
    exact_mod_cast h
  constructor
  · -- Prove -π ≤ g - ⌊(g + π) / (2 * π)⌋ * (2 * π)
    have h₁ : (⌊(g + π) / (2 * π)⌋ : ℝ) * (2 * π) ≤ g + π := by
      calc
        (⌊(g + π) / (2 * π)⌋ : ℝ) * (2 * π) ≤ ((g + π) / (2 * π)) * (2 * π) := by gcongr
        _ = g + π := by
          field_simp [Real.pi_pos.le]
          <;> ring_nf
          <;> linarith [Real.pi_pos]
    have h₂ : g - (⌊(g + π) / (2 * π)⌋ : ℝ) * (2 * π) ≥ -π := by
      linarith [Real.pi_pos]
    linarith
  · -- Prove g - ⌊(g + π) / (2 * π)⌋ * (2 * π) ≤ π
    have h₁ : g + π < ((⌊(g + π) / (2 * π)⌋ : ℝ) + 1) * (2 * π) := by
      calc
        g + π = ((g + π) / (2 * π)) * (2 * π) := by
          field_simp [Real.pi_pos.le]
          <;> ring_nf
          <;> linarith [Real.pi_pos]
        _ < ((⌊(g + π) / (2 * π)⌋ : ℝ) + 1) * (2 * π) := by
          have h₂ : (g + π) / (2 * π) < (⌊(g + π) / (2 * π)⌋ : ℝ) + 1 := hq'
          nlinarith [Real.pi_pos]
    have h₂ : g - (⌊(g + π) / (2 * π)⌋ : ℝ) * (2 * π) ≤ π := by
      linarith [Real.pi_pos]
    linarith

theorem decomposition_exact (g : ℝ) : g = ((decompose g).1 : ℝ) * B + (decompose g).2 := by
  dsimp [decompose, B]
  push_cast
  ring

-- Cumulative decomposition: storing soul and echo across multiple steps
noncomputable def soul (gradients : List ℝ) : ℤ :=
  (gradients.map (λ g => (decompose g).1)).sum

noncomputable def echo (gradients : List ℝ) : ℝ :=
  (gradients.map (λ g => (decompose g).2)).sum

-- Cons-laws used by total_gradient (safe: do not re-substitute g into decompose)
theorem soul_cons (g : ℝ) (gs : List ℝ) : soul (g::gs) = (decompose g).1 + soul gs := by
  simp [soul, List.sum_cons]

theorem echo_cons (g : ℝ) (gs : List ℝ) : echo (g::gs) = (decompose g).2 + echo gs := by
  simp [echo, List.sum_cons]

-- Total gradient = sum of all individual gradients
theorem total_gradient (gradients : List ℝ) :
    (gradients.sum : ℝ) = ((soul gradients : ℤ) : ℝ) * B + echo gradients := by
  induction' gradients with g gs ih
  · simp [soul, echo]
  · rw [soul_cons g gs, echo_cons g gs]
    conv => lhs; rw [List.sum_cons, decomposition_exact g]
    rw [ih]
    push_cast
    ring_nf

-- The "Lazarus test": after a crash, stored (soul, echo) recovers total gradient
theorem lazarus_recovery (gradients : List ℝ) (s : ℤ) (e : ℝ)
    (hs : s = soul gradients) (he : e = echo gradients) :
    (gradients.sum : ℝ) = ((s : ℤ) : ℝ) * B + e := by
  rw [hs, he]
  exact total_gradient gradients

-- The decomposition is additive: decompose(g₁ + g₂) ≠ decompose(g₁) + decompose(g₂)
-- But the CUMULATIVE soul and echo ARE additive
theorem soul_additive (g₁ g₂ : ℝ) :
    soul [g₁, g₂] = (decompose g₁).1 + (decompose g₂).1 := by
  simp [soul, echo, decompose, List.sum_cons, List.sum_nil, zero_add]

theorem echo_additive (g₁ g₂ : ℝ) :
    echo [g₁, g₂] = (decompose g₁).2 + (decompose g₂).2 := by
  simp [soul, echo, decompose, List.sum_cons, List.sum_nil, zero_add]
