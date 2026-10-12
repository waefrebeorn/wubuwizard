# Klein Models & Movable Manifolds — Research Notes

Two ideas landed in this vault, both aimed at extending WuBuNesting past
Poincaré:

1. **The Klein model** (arXiv:2410.16813, Mao/Gu/Werner/Zou, NeurIPS 2024
   Symmetry & Geometry workshop) — a third hyperbolic coordinate model with
   straight-line geodesics and Einstein addition/midpoints, complementing
   Poincaré (Möbius) and hyperboloid.
2. **Movable Klein / adaptive non-orientable nesting** — WaefreBeorn's
   extension: manifold shapes that *move* and *adapt* to storage, i.e. the
   nesting container is not a fixed ball but a learned, morphing region —
   and where orientability fails (Klein-bottle-like), the Poincaré-ball
   assumptions must be re-derived rather than transplanted.

---

## 1. Why Klein matters to WuBuNesting

The human's note: "the poincare sphere is a basic manifold math network base;
other shapes fit other things; but nesting is a sphere thing unless klein
bottle nested shapes."

WuBuNesting as built is 100% Poincaré. The Klein paper is the strongest
existing evidence that adding a second coordinate model is both tractable and
useful:

- **Straight-line geodesics.** In Klein, geodesics are Euclidean chords, so
  the inter-level "Relative Vectors" d_{i+1,j,k} and boundary sub-manifolds
  B_{i,j} become affine/Einstein constructions instead of Möbius ones. Our
  rotation machinery (quaternions in tangent space) can stay.
- **Einstein addition / scalar multiplication** are the Klein analogues of
  Möbius ⊕_c and λ-c-scalar multiplication. The paper *proves* the tangent-
  space constructions of scalar multiplication and parallel transport are
  exactly Einstein ops — i.e. the same structural result WuBu relies on for
  Möbius has a clean Klein counterpart.
- **Einstein midpoint** construction is directly relevant to "levels that
  adapt" — a midpoint between two adaptive levels is a first-class object,
  not an approximation.

Practical consequence for WuBu: a **hybrid nesting** where coarse levels
(where straight-line geometry and cheap linear ops matter — storage
projection, bulk movement) use Klein, and fine levels (where conformal
volume growth and Möbius addition matter — hierarchy) use Poincaré. The
isomorphism between models is a radial map:

- Poincaré → Klein:  `k = 2x / (1 + ‖x‖²)` (for curvature −1, unit ball)
- Klein → Poincaré:  `x = k / (1 + √(1 − ‖k‖²))`

Both stay in the open unit ball. This map is the "model switching layer" —
learnable per level, exactly like curvature and scale are today.

## 2. Movable manifolds — the actual new idea

### 2.1 The problem with fixed nesting

Current WuBuNesting: levels are balls with learnable c_i, s_i, n_i. The
*shape* is fixed (ball). Data is projected INTO a fixed container. That means
the storage budget is fixed by geometry, not by content: a level with 90% of
its mass in one corner still allocates as if uniform.

The idea: **the container is a learned, morphing region**. Level i is a
subset S_i of the ambient space with a geometry (metric) that adapts to the
data distribution AND to the storage budget. Nesting becomes
S_1 ⊃ S_2 ⊃ … ⊃ S_k where each S_i is:

- a deformable region (learned radial/angular deformation of the ball),
- with a possibly different coordinate model per level (Poincaré / Klein /
  hyperboloid — whichever fits that level's job),
- with morphisms between levels that are not just rotations but genuine
  diffeomorphisms with learned parameters.

### 2.2 Why "moving Klein" is the right handle

Klein gives us the tool that makes morphing containers cheap:

- In Poincaré, deforming the metric changes geodesics nonlinearly; Möbius
  addition and gyration both warp.
- In Klein, geodesics are straight lines in the ambient Euclidean space. A
  **deformation of the container is then an affine/linear map on the ambient
  space** (plus a radial warping), and Einstein addition composes cleanly.
  The morphism S_i → S_{i+1} can be written as a matrix + radial function,
  which is exactly what GPU code wants.

So: **movable manifolds = Klein levels whose containers are learned affine
deformations of the unit ball.** The straight-line property means the
deformation doesn't break the geodesic structure — it bends the ambient
space, not the geometry.

### 2.3 Non-orientable levels

The human's "klein bottle nested shapes" is the sharp end. A Klein bottle
K² is non-orientable: there is no consistent global orientation. Consequences
if a nesting level is Klein-like:

- Quaternion rotation R_i assumes orientation-preserving maps (SO(n));
  on a non-orientable level the transition "rotation" must be allowed to
  include reflections (O(n) \ SO(n)), and the double-cover story changes
  (S³ → SO(3) no longer covers O(3)).
- Gyration, as defined via Möbius addition on an oriented ball, has no
  direct meaning; the 1D proofs trivially don't apply.
- Boundary Sub-Manifolds and Relative Vectors need a local-frame convention
  that patches with a sign flip across some loop — the defining feature of
  non-orientability.

Design proposal for WuBu: **orientation flag per level**. Each level carries
`orientable : bool` and, when false:
- transitions use O(n_i) (quaternions + a reflection bit),
- the BSP node split gains an orientation-flip child,
- exp/log maps are defined chart-locally with an explicit transition sign.

This is implementable and testable. It is also genuinely novel as far as the
literature pass shows — most hyperbolic DL work never leaves the oriented
Poincaré/hyperboloid ball, and Klein-model work (including the paper above)
is oriented.

### 2.4 Storage-adaptive projection

The "adapt to storage" half: level i gets a storage budget b_i (bits or
parameters). The container S_i and the coordinate model are then chosen to
make the required distortion achievable within b_i. Concretely:

- Projection of level i+1's points into level i must minimize distortion
  subject to the tangent-space code length ≤ b_i.
- Distortion measure: hyperbolic distance ratio
  `d(S_i-origin-metric) preserved vs original`, or better, the geodesic
  distortion under the morphism.
- Coordinate model per level is then a *storage decision*: Klein when
  straight-line + linear ops buy precision per bit; Poincaré when the
  conformal volume growth is needed for hierarchy.

This turns nesting from "fixed geometry + learned parameters" into
"geometry, model, and parameters all learned against a storage budget" —
which is what the human means by "nested storage projections."

## 3. Relation to openai/math

Use the openai/math corpus (cloned to MATH/external/openai-math) as the
prior-art check for:
- Klein/Einstein geometry results (model isometries, Einstein addition
  associativity — expect classical results there),
- manifold/morphism and non-orientable results,
- anything already machine-verified that we'd otherwise re-derive.

"Razzle dazzle" = when a WuBu claim can be lifted onto an already
machine-verified lemma family from that corpus, cite it and skip the
re-proof; when it can't, that's where our Lean work has value.

## 4. Open questions to resolve next

1. Is there published work on *learnable/morphing* Klein containers in DL?
   (lit pass needed — initial search says no, but confirm)
2. Does non-orientable hyperbolic nesting admit a consistent gyrovector
   structure, or must gyration be dropped on those levels?
3. Storage-adaptive projection: is there a known rate-distortion result for
   hyperbolic embeddings we can anchor to?
4. Does the fiber-bundle claim (math_viz/05) survive mixed Poincaré/Klein
   levels? The connection 1-form A = R⁻¹dR assumed a single model.
