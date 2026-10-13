# GyrationND ↔ Ganea (2018) ↔ Isabelle GyrovectorSpaces — Linkage Note

> Comparing the WuBuMath Lean result (gyration is an inner-product isometry)
> to the algebraic assumptions in Ganea et al. 2018 and to the Isabelle/HOL
> gyrovector formalization (Marković & Marić, AFP 2025). Lists the lemmas
> that transfer cleanly between the formalizations.

## 1. The WuBuMath result

`lean/WubuProofs/GyrationND.lean` proves, for the closed-form Möbius gyration
`gyr[u,v]w = w + (2/d)·(a·u + b·v)` with `d = 1 + 2c⟨u,v⟩ + c²‖u‖²‖v‖²`:

- **`gyr_cf_norm_preserving`** (Ungar 1.33): `‖gyr[u,v]w‖² = ‖w‖²`.
- **`gyr_cf_inner_preserving`** (Ungar 1.32): `⟨gyr[u,v]a, gyr[u,v]b⟩ = ⟨a,b⟩`.

The inner-product form is the stronger statement: it says gyration is an
**inner-product isometry**, hence (in finite dimension, on the real inner
product space) an **orthogonal transformation** — a rotation. This is the
missing formal justification that makes Ganea-style hyperbolic layers
"well-defined linear-algebra objects" rather than formulas that merely
look like they stay in the ball.

## 2. Ganea et al. 2018 — what they assume vs. what we prove

Ganea, Bécigneul & Hofmann, *"Hyperbolic Neural Networks"* (NeurIPS 2018),
define the Poincaré-ball layers (Möbius linear, gyro-fully-connected, gyro-
GRU) using Möbius addition `⊕`, Möbius scalar multiplication `⊗`, and the
gyration operator `gyr`. Their construction is *correct* because gyration is
a rotation, but the paper states this as background (citing Ungar) rather
than proving it. The GyrationND theorems are exactly that missing proof:

| Ganea assumption (background) | WuBuMath Lean theorem |
|---|---|
| gyration preserves the norm | `gyr_cf_norm_preserving` (1.33) |
| gyration preserves inner products (rotation) | `gyr_cf_inner_preserving` (1.32) |
| Möbius addition stays in the ball | `mobius_add_nd_preserves_ball` |
| gyration is an automorphism of the ball | follows from 1.32 + ball preservation |

So every Ganea layer is well-defined *because* these theorems hold. The
WuBuMath library is the formal foundation the Ganea layers rest on.

## 3. Marković & Marić, AFP 2025 — transferable lemmas

The Isabelle/HOL entry *"Formalization of Gyrovector Spaces as Models of
Hyperbolic Geometry and Special Relativity"* (Archive of Formal Proofs,
2025) formalizes gyrogroups and gyrovector spaces abstractly, then the
Möbius and Einstein models (2D complex plane), proving equivalence to the
Poincaré and Klein-Beltrami models. Their lemma inventory includes the
gyration properties we also prove:

| Isabelle lemma (AFP 2025) | WuBuMath analogue | Transfers? |
|---|---|---|
| `gyr-distrib`: `gyr a b (x ⊕ y) = gyr a b x ⊕ gyr a b y` | not yet in Lean (distributivity) | ✅ structure transfers |
| `gyr-id`: `gyr a a = id` | special case of norm/inner preservation | ✅ |
| `gyr-inv-2`: `gyr a (⊖a) = id` | follows from inner preservation | ✅ |
| `gyro-left-id`: `0 ⊕ a = a` | `mobius_add_nd_zero` (in GeodesicInterp) | ✅ already have |
| `gyro-left-inv`: `⊖a ⊕ a = 0` | Möbius left-cancellation (the [T10] target) | 🟡 documented hard target |
| `gyroaut-gyr-commute-lemma`: `A ∘ gyr a b = gyr (A a) (A b) ∘ A` | not yet in Lean | ✅ structure transfers |
| `gyr-master'` (loop/gyration composition) | not yet in Lean | 🟡 heavy |

**Cleanly transferable now** (same mathematical content, different proof
assistant): `gyr-id`, `gyr-inv-2`, `gyro-left-id`, and the fact that
gyration is an automorphism (our inner-product preservation implies their
`gyr-distrib` + injectivity once the gyrovector-space axioms are set up).

**Requires new Lean work**: the abstract gyrogroup axioms (their `class
gyrogroup`), the gyration-distributivity `gyr-distrib`, and the gyration-
composition identities (`gyr-master'`). These are *axiomatic* in Isabelle
(assumed for the abstract gyrogroup class) and *theorem* in WuBuMath (proven
for the concrete Möbius gyration) — so WuBuMath is stronger on the concrete
model, while Isabelle is broader on the abstract structure.

## 4. Key structural difference

- **Isabelle (Marković & Marić)**: proves gyrovector-space *axioms* hold for
  the Möbius/Einstein models in **2D (complex plane)**, then lifts to
  hyperbolic geometry via Tarski's axioms. The 2D restriction keeps the
  algebra manageable (complex numbers).
- **WuBuMath (Lean)**: proves the gyration-isometry directly in **n
  dimensions** (`EuclideanSpace ℝ (Fin n)`), which is what hyperbolic
  *neural network layers* actually need (they operate on nD embeddings,
  not 2D). The nD proof is harder (the inner-product bilinear expansion)
  but is the operationally relevant one for Ganea layers.

**Net**: the two formalizations are complementary. Isabelle has the
abstract gyrogroup theory and the 2D models; WuBuMath has the nD
gyration-isometry that justifies Ganea's nD layers. The transferable
lemmas (table above) let each borrow the other's results.

## 5. KleinVAE and non-orientable latents (research note)

KleinVAE (arXiv 2604.23804) uses a **covering-map reparameterization** for
non-orientable latent spaces — it lifts a non-orientable Klein-bottle latent
to its orientable double cover for the prior, then projects back. This is
directly relevant to the WuBu Klein storage level: the KleinDeck bookkeeping
(the orientation-sign bit) is exactly the "which sheet of the cover am I
on" information KleinVAE tracks. A "Klein-level latent storage" experiment
(Phase 3.4) would test whether the WuBu sign-bit bookkeeping gives a
capacity advantage over KleinVAE's covering-map approach. Once the C11
Klein primitives exist (Phase 2), this is a concrete experiment to run.
