# ~ WuBu ~ Math Vault

Orientation + overlap ledger for WuBuMath domain expansion, worldsim optimization,
and AGI work. This vault maps WuBu's internal math against external bodies of
mathematics so we know what is already proven, what overlaps known results, and
what is genuinely original.

## Reading order

1. [orientation.md](orientation.md) — where WuBuMath lives on disk and what each
   piece is (paper, Python impl, C port, Lean proofs, theory viz).
2. [core-math.md](core-math.md) — the actual math, with code-matched notation:
   nested Poincaré balls, exp/log maps, Möbius addition, gyration, quaternion
   rotations, MLA compression, fiber-bundle claim.
3. [overlap-ledger.md](overlap-ledger.md) — theorem-by-theorem comparison of WuBu
   results vs known literature. Each row states claim, known provenance, verdict
   (original / standard / novel combination), and where our Lean proof stands.
4. [openai-math-index.md](openai-math-index.md) — index of the 722 openai/math
   manuscripts (372 families) relevant to hyperbolic geometry, geometry, and
   formalization; used as a "has this been proven already?" lookup.
5. [status.md](status.md) — current vault state, key findings,
   what's verified vs narrative, and the roadmap.
6. [klein-nesting-paper-design.md](klein-nesting-paper-design.md) — THE NOVEL
   PAPER: nested Klein manifolds as movable storage containers, the
   Gauss–Bonnet constraint (stated up front, routed around via patches),
   storage-theft mechanism, orientation machinery, novelty table, and
   Lean targets. Read this first if you want the new work.

## Guiding principles (from the human)

- The Poincaré ball is a basic manifold network base; other manifolds fit other
  things. Nesting is a sphere thing unless non-orientable shapes (Klein bottle)
  are involved — in which case Poincaré-ball assumptions must be re-derived.
- WuBuMath is a full dynamic system, not a single trick.
- Comparisons must be concrete: theorem vs theorem, formula vs formula, with
  provenance. No vibes.
