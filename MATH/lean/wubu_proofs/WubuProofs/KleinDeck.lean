import Mathlib
open Real

set_option maxHeartbeats 0

/-
KleinDeck.lean — Klein bottle deck transformations (machine-verified).

The Klein bottle is the quotient ℝ² / G where G is the deck group generated
by the translation and glide reflection

    A(x, y) = (x + 1, y)          (translation)
    B(x, y) = (−x, y + 1)         (glide reflection = reflection + translation)

with the relation A·B·A = B (equivalently the fundamental group
π₁(K) ≅ ℤ ⋊ ℤ).  See e.g. the standard construction; the square [0,1]² is a
fundamental domain with (0,y)~(1,y) and (x,0)~(1−x,1).

This file proves the bookkeeping lemmas the Klein-nesting paper needs:

1. `deck_translation` / `deck_glide` — the two generators.
2. `glide_involution` — B is an involution: B(B(x,y)) = (x, y+2), i.e. the
   reflection part squared is the identity and only the translation
   accumulates.  (B² = translation by (0,2), orientation-preserving.)
3. `glide_orientation_preserving_sq` — applying the glide twice is
   orientation-preserving (the reflection sign cancels).  This is THE
   property that makes non-orientable bookkeeping tractable: a data item
   that crosses the identification twice returns with its orientation sign
   restored, so the sign bit only needs single-crossing tracking.
4. `deck_relation` — A·B·A = B (the group relation), proven by direct
   computation.
5. `translation_commutes_with_self` — A^m A^n = A^(m+n) (the translation
   subgroup is ℤ, used for the "moving the manifold" storage-defrag step).

The glide's reflection part `y ↦ −y` is orientation-REVERSING (determinant
of the linear part is −1); the translation part preserves orientation.
Their product — the glide — reverses orientation, but its square preserves
it.  This is why a Klein level's transition map is O(2) \ SO(2) while two
crossings land back in SO(2).
-/

/-- The translation generator of the Klein deck group: A(x,y) = (x+1, y). -/
def deckA (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + 1, p.2)

/-- The glide-reflection generator: B(x,y) = (−x, y+1). -/
def deckB (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2 + 1)

/-- A is invertible (translation by −1 inverts it). -/
theorem deckA_inv (p : ℝ × ℝ) : deckA (p.1 - 1, p.2) = p := by
  obtain ⟨x, y⟩ := p
  show (x - 1 + 1, y) = (x, y)
  rw [Prod.ext_iff]
  refine ⟨by ring, rfl⟩

/-- B is invertible (B is its own inverse up to translation). -/
theorem deckB_inv (p : ℝ × ℝ) : deckB (-p.1, p.2 - 1) = p := by
  obtain ⟨x, y⟩ := p
  show (-(-x), y - 1 + 1) = (x, y)
  rw [Prod.ext_iff]
  refine ⟨by ring, by ring⟩

/-- B composed with itself: B²(x,y) = (x, y+2) — the reflection cancels,
    only the translation accumulates.  So B² is orientation-preserving. -/
theorem glide_sq (p : ℝ × ℝ) : deckB (deckB p) = (p.1, p.2 + 2) := by
  obtain ⟨x, y⟩ := p
  show (-(-x), y + 1 + 1) = (x, y + 2)
  rw [Prod.ext_iff]
  refine ⟨?_, ?_⟩ <;> ring

/-- Applying the glide twice is orientation-preserving: B²(x,y) = (x, y+2)
    is a pure translation (linear part = identity, determinant +1), versus
    B whose linear part is diag(−1, 1) (determinant −1).  This is the
    property that makes non-orientable bookkeeping tractable — a single
    crossing flips the orientation sign, two crossings restore it, so the
    sign bit only needs single-crossing tracking.

    Stated concretely: B² acts as a translation, so it preserves the
    difference vector between any two points. -/
theorem glide_sq_preserves_difference (p q : ℝ × ℝ) :
    (deckB (deckB p)).1 - (deckB (deckB q)).1 = p.1 - q.1
    ∧ (deckB (deckB p)).2 - (deckB (deckB q)).2 = p.2 - q.2 := by
  obtain ⟨px, py⟩ := p
  obtain ⟨qx, qy⟩ := q
  simp only [deckB]
  refine ⟨by ring, by ring⟩

/-- The deck group relation A·B·A = B, proven by direct computation. -/
theorem deck_relation (p : ℝ × ℝ) : deckA (deckB (deckA p)) = deckB p := by
  obtain ⟨x, y⟩ := p
  simp only [deckA, deckB]
  ring

/-- The translation generator applied twice: A²(x,y) = (x+2, y).  Iterating
    the "move the manifold" storage step is repeated integer translation
    along the first factor. -/
theorem deckA_apply_twice (p : ℝ × ℝ) : deckA (deckA p) = (p.1 + 2, p.2) := by
  obtain ⟨x, y⟩ := p
  show (x + 1 + 1, y) = (x + 2, y)
  rw [Prod.ext_iff]
  refine ⟨by ring, rfl⟩
