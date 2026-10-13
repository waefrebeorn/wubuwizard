# Klein Non-Orientable Storage Layer — Theory Note

> How the Klein-bottle deck bookkeeping (`lean/WubuProofs/KleinDeck.lean`)
> sits inside the WuBu Nesting chain, and what storage capability each proven
> theorem enables. Companion to `docs/LINKAGE.md`.

## 1. Where the Klein level sits

The WuBu Nesting framework nests hyperbolic manifolds of increasing
curvature (the φ-progression in `NestedHyperbolicSpaces.lean`). Each level
is an orientable Poincaré ball — Möbius addition, gyration, parallel
transport all live there, and the proven invariants (ball preservation,
gyration-as-rotation, PT isometry) keep every operation on-manifold.

The **Klein level** is different in kind: it is *non-orientable*. It is not
another Poincaré ball; it is a quotient ℝ² / G where G is the deck group

    A(x, y) = (x + 1, y)          (translation)
    B(x, y) = (−x, y + 1)         (glide reflection)

with the single relation **A·B·A = B** (π₁(K) ≅ ℤ ⋊ ℤ). The Klein level
sits *beside* the hyperbolic nesting chain as a **storage substrate** —
the place where data that has crossed an odd number of identifications
lives with its orientation sign flipped.

## 2. Why non-orientable storage is useful

An orientable manifold cannot store a "handedness" bit for free: every
loop preserves orientation, so there is nowhere to put the bit. A
non-orientable manifold can: a path that crosses the identification an odd
number of times returns with its orientation sign flipped. That sign bit
is **free storage capacity** — it costs no extra dimension, only the
bookkeeping to track which side of the identification a datum is on.

This is the same reason natural image patches concentrate on a Klein
bottle (Carlsson et al.; Love et al.): the data has a genuine
non-orientable structure, and the Klein bottle is the minimal orientable-
cover-free space that holds it. For WuBu storage, the Klein level is where
"the non-graphed portions" of a manifold live — the parts that don't fit
the orientable hyperbolic chain.

## 3. Theorem → storage-capability map

Every theorem in `KleinDeck.lean` enables a concrete storage primitive:

| Theorem | Storage capability |
|---|---|
| `deckA` / `deckB` (generators) | The two fundamental moves: shift the manifold (A) and cross the identification (B) |
| `glide_sq` / `glide_sq_is_translation` (B² = translation by (0,2)) | **Orientation restoration**: two crossings return to the orientable side, so the sign bit only needs *single-crossing* tracking, not full path history |
| `deck_relation` (ABA = B) | The group is ℤ ⋊ ℤ; any word in A, B reduces, so storage addressing is canonical |
| `glide_sq_preserves_difference` | B² preserves inter-point distances → data laid out across two crossings stays metric-consistent |
| `deckA_iterate_two` / `deckA_iterate_three` / `deckA_defrag_commutes` | **Defrag by translation**: A^k = +k in x, and the moves commute (⟨A⟩ ≅ ℤ free abelian rank 1), so defrag is order-independent |
| `klein_orientation_bookkeeping` | The sign bit: B flips it, A preserves it, B² restores it, ABA = one net flip — a single bit per datum |
| `glide_sq_in_translation_subgroup` | B² lands in the translation subgroup → even-crossing data is addressable by pure translation |

## 4. The orientation-sign bookkeeping (the core primitive)

The load-bearing lemma is `klein_orientation_bookkeeping`. It states the
sign rules together:

- **B² is orientation-preserving**: `deckB (deckB p) = (p.1, p.2 + 2)` — a
  pure translation, linear part = identity, determinant +1.
- **A is orientation-preserving**: `deckA p = (p.1 + 1, p.2)` — pure
  translation, determinant +1.
- **ABA = B**: `deckA (deckB (deckA p)) = deckB p` — one full cycle is a
  single net flip, matching the group relation.

Together these say: a data item's orientation sign is **one bit**, flipped
by each B crossing and restored by pairs. You never need to store the full
path — only the parity of B-crossings. This is what makes non-orientable
bookkeeping *tractable* rather than exponential.

## 5. Defrag primitive: "moving the manifold"

The storage-defrag step is iterated A-translation. `deckA_iterate_two`
(A² = +2) and `deckA_defrag_commutes` (A³ = +3, order-independent) prove
that shifting the fundamental domain by k units moves every point's
x-coordinate by exactly k while y is untouched, and that the moves
commute (the subgroup ⟨A⟩ ≅ ℤ is free abelian of rank 1). Defragging the
Klein level is therefore a sequence of integer translations along x whose
order doesn't matter — a clean, parallelizable primitive.

## 6. Linkage to the C implementation

The C bridge (`src/tests/test_lean_bridge.c` [K]) verifies the deck
generators satisfy ABA=B, B²=translation by (0,2), and A^5=+5 in x over
1000 points — the exact bookkeeping the storage layer relies on. The
`examples/klein_deck_demo.c` primitive (Phase 2.4 target) will visualize
the orientation-sign flip/restore across a sequence of glides.

## 7. Open: capacity advantage via non-orientability

The *existence* of free sign-bit capacity is proven (the bookkeeping
theorems). The *quantitative* capacity advantage — how many bits the Klein
level stores per unit of the orientable cover — requires a rigorous
"stored bits" definition and the capacity-ratio experiment
(`experiments/klein_capacity_experiment.py`, present). This is the next
measured step, documented as future work per charter rule 3.
