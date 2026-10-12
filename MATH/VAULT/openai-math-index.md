# openai/math — index for WuBu overlap checks

Corpus cloned (depth-1) to `MATH/external/openai-math`. 719 manuscripts, 372
families, released 2026-10-06 by an unnamed internal OpenAI frontier model.
~42% of top-line results have Lean formalizations. Structure:

- `CONTENTS.md` — manuscript map, one line per family (`**NNN. Title.** …`),
  372 families, with per-paper links into `preprints/`.
- `lean/docs/NNN.md` — 242 formalization scopes, each naming the exact
  theorems proven and linking a Comparator file
  (`lean/ComparatorChallenges/NNN.lean`) that states them machine-checkably.
- `preprints/` — PDFs + sources.
- `lean/OAI/` — the internal Lean library the formalizations build on.

## What we actually need from this corpus

**Finding so far: zero overlap with WuBuNesting's mathematical content.**

Searched CONTENTS.md (all 9142 lines) and all 242 Lean docs for:
`hyperbolic geometry|hyperbolic space|Poincaré ball|Poincaré disk|Klein model|
Möbius addition|gyrovector|Einstein addition|Einstein midpoint|tanh|geodesic|
exponential map|isometry|conformal|quaternion (DL sense)|manifold (DL sense)|
bundle (connection sense)`.

Results:

- The corpus is dominated by algebraic geometry, number theory, conformal
  field theory, combinatorics, and analysis at the research frontier
  (Fujita freeness, Beauville splitting, Koebe/Brennan circle domains,
  Campana conjectures, Vlasov–Maxwell, spin-glass formulas).
- "Hyperbolic" hits are algebraic-geometric (hyperbolic Kähler manifolds,
  hyperbolic curves over p-adic fields, hyperbolic metric in the Penrose
  inequality) — NOT the constant-negative-curvature DL geometry WuBu uses.
- Möbius hits are conformal-map theory (Koebe rigidity: every conformal
  equivalence between circle domains is Möbius) — the Möbius *group* of the
  Riemann sphere, not Möbius *addition* in the Poincaré ball. Same surname,
  different objects. Do not confuse when citing.
- Quaternion hits are number-theoretic (quaternion division algebras,
  quaternionic division points in Zilber–Pink). No rotation/quaternion-DL
  content.
- No tanh, no gyrovector, no Einstein addition, no Poincaré-ball metric, no
  hyperbolic embedding, no learnable-curvature content at all.

**Conclusion: the openai/math corpus neither validates nor threatens any
WuBuNesting claim.** It is not a prior-art risk for us. Its use to WuBu is
narrower and still real:

1. **Lean toolchain leverage.** Their `lean/OAI` library and Comparator
   format (`lean/docs/*.md` + `ComparatorChallenges/*.lean`) is a working
   template for how to state and verify a theorem library at scale. Worth
   copying the *format* for our own `wubu_proofs` when it grows past 4 files.
2. **The Conformal/Kähler hyperbolic-adjacent families** (071 Koebe, 072
   Brennan, 051 Kobayashi canonical-ampleness on hyperbolic Kähler) are
   machine-verified results in *adjacent* geometry. If a WuBu claim ever
   needs a classical lemma that happens to be in their corpus, we can cite
   their Comparator file instead of re-proving. (None needed today.)
3. **Proof-artifact hygiene.** Their convention of pairing every claim with
   a scope document is the antidote to WuBu's current "theorem names in
   Main.lean status printer" approach. Adopt it.

## Closest-by-name families (recorded to prevent false-positive citation)

| Family | Title | Why it is NOT WuBu overlap |
|---|---|---|
| 071 | Koebe's circle-domain conjecture (Lean) | Möbius *group* of Riemann sphere, conformal rigidity of domains. Not Möbius addition ⊕_c. |
| 072 | Brennan's conjecture, integral means (Lean) | Conformal bijections to the disk, area integrability of \|φ'\|^s. Different objects. |
| 051 | Kobayashi canonical-ampleness, hyperbolic Kähler | "Hyperbolic" = Brody-hyperbolic complex manifold. Not Poincaré model. |
| 041 | Hyperkähler SYZ | Hyperkähler ≠ hyperbolic. |
| 052 | Tangent splittings, Beauville splitting | Tangent-bundle decompositions of Kähler manifolds. Adjacent in spirit to our bundle claim, different setting. |

## Procedure for future overlap checks

```bash
cd MATH/external/openai-math
grep -inE "<your theorem keywords>" CONTENTS.md lean/docs/*.md
```

Run this BEFORE claiming any WuBu theorem is novel. If a family matches,
open `lean/docs/NNN.md`, read the Scope section, and cite the Comparator
file in the overlap ledger rather than re-deriving.
