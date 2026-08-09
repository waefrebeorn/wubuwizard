# THEORY/10 — The Ecosystem of Spheres: Small Active, Huge Total

> "a lot of the big models have a very big number and then they have a
> small number which is their active parameter... the secret to getting
> effective usage is the smallest active parameter but an effective large
> system." — WaefreBeorn

> "our amoeba structure needs to have ecosystem and it needs to be a full
> ecosystem AGI of our spheres, and we just need to have a bunch of spheres
> and use physics... to do it very effectively and efficiently using
> physics. That way we can have as much data as possible in a small space
> while effectively having our AGI function." — WaefreBeorn

> "there is also fractal stacking on the surface of the sphere on top of
> the hyperbolic embedding we can really cram a lot into the empty space."
> — WaefreBeorn

---

## 0. The thesis

WuBu is not one model. It is an **ecosystem of spheres** — a colony of
small hyperbolic balls, each a miniature expert with learnable curvature,
interacting through physics (Möbius dynamics, gyration, gravitational
attraction on the manifold). At inference, **physics routes** — only the
spheres whose potential wells the input falls into fire their parameters.
Total parameters can be enormous; active parameters stay tiny. Data is
crammed into the empty space three ways at once: hyperbolic interior
(exponential volume growth), sphere-surface fractal stacking (space-filling
curves on the ideal boundary), and nested ball-in-ball layers.

This is the MoE principle taken to its architectural conclusion: DeepSeek
V3 keeps 671B total with only **37B active per token** (5.5% active ratio),
and that small-active/large-total split is what made it trainable at scale.
WuBu does the same with *physics as the router* and *geometry as the
compression* — no learned routing weights, no dense megamodel.

## 1. The active-parameter doctrine (the industry proof)

| Model | Total | Active | Active % | Routing |
|-------|-------|--------|----------|---------|
| DeepSeek V3 | 671B | 37B | 5.5% | 8 of 256 experts/token |
| Mixtral 8x7B | 47B | 13B | 27% | 2 of 8 experts/token |
| WuBu-35M (old) | 35M | 35M | 100% | dense |
| WuBu1 ecosystem (target) | ~56M+ | ~7M | ~12% | physics |

The scaling insight (HuggingFace MoE blog, DeepSeek-V3 tech report): MoE
models train far cheaper than dense models of the same *total* size, and
inference compute is proportional to *active* params. The ceiling is
storage (weights on disk/RAM), not compute. So the design rule:

> **Maximize total knowledge stored; minimize parameters touched per
> forward.** The ratio active/total is the design lever, and physics
> (not a learned router) sets it.

## 2. The ecosystem: many spheres, physics routing

A colony of N balls. Each ball k has:
- radius / learnable curvature c_k (already in `wubu_nested_ssm.h`)
- a small expert core: q/k/v/o projections + gate_up/down + per-head norms
  (the `wubu1_block_t` layout)
- a hive slot (stable pointer, `wubu_hive_insert/erase`)
- a **potential function** φ_k(x) = gyro-distance² from the ball center
  to the input, weighted by curvature

### Physics as the router (no learned routing weights)

1. **Input x** (a token embedding, a query, a KV file) enters the shared
   tangent space at the colony origin.
2. **Gravitational pull**: each ball exerts a force
   `F_k = -∇ φ_k(x)` — the gyro-gradient of its potential. The input is
   "attracted" to the balls whose learned regions it resembles.
3. **Top-k by potential well depth**: only the K balls with the deepest
   wells fire their parameters (K small, e.g. 2-4 of 256). This is
   DeepSeek's 8/256, done with physics instead of a learned gate.
4. **Möbius composition**: fired balls contribute
   `y = y ⊕_c (w_k · exp_0^{c_k}(proj_k(x)))` — Möbius addition of their
   outputs on the ball, then project back to tangent for the next layer.
   (Already proven in Lean: `MobiusAdd.lean`, `HyperbolicGyration.lean`.)

The router has **zero learned parameters** — it is closed-form physics
evaluated on the existing geometry. This is the "efficiently using
physics" directive.

### The hive IS the ecosystem

- **GROW** = a new ball mitoses from a parent (curvature inherited,
  gate-zeroed daughter — the `wubu_grow_*` operator)
- **SHRINK** = a dead ball (low utilization, negative loss-delta) is
  erased; its weights recycle to the freelist
- **SPECIATE** = balls drift apart in curvature space (c_k learnable) so
  the colony develops ecological niches — math balls, code balls, agent
  balls — exactly like an ecosystem filling niches

## 3. Density: cramming data into empty space

The user's "as much data as possible in a small space" is satisfied three
layers deep:

### 3a. Hyperbolic interior — exponential volume growth
A ball of hyperbolic radius R has volume growing **exponentially** with R
(Euclidean: polynomial). Sarkar's theorem: **any tree embeds in the
Poincaré ball with arbitrarily low distortion, in just 2 dimensions** —
while Euclidean space needs unbounded dimensions for the same distortion.
Hierarchical knowledge (concept trees, topic hierarchies, code/module
trees) packs into ~2 hyperbolic dimensions what would need a huge
Euclidean dim. The nested SSM already runs K balls with K curvatures.

### 3b. Fractal stacking on the sphere surface — the infinite shelf
The sphere at infinity of hyperbolic space is the **ideal boundary** —
every geodesic ends there. On that boundary:
- Space-filling curves (Hilbert, Peano — fractal dimension 2 filling 2D
  area) stack infinitely many addresses onto the 1D boundary
- The boundary of a 3D ball is a 2-sphere; Kleinian groups act on it —
  fractal limit sets tile the sphere (Wikipedia: "the sphere at infinity
  of hyperbolic 3-space")
- **Stacking rule**: each ball's *surface* carries a fractal index —
  a space-filling curve maps the ball's boundary to a dense address space
  where KV files, learned patterns, and expert fingerprints live.
  Multiple fractals can be stacked on the same surface (different
  space-filling curves, different granularities) — the surface is
  measure-zero in volume but carries unbounded *addressable* data.

So a single ball holds: exponential interior volume + fractal-indexed
surface = two independent capacity dimensions in one geometric object.

### 3c. Nesting — ball-in-ball
WuBu1's nesting transition (every K-th block, quaternion rotation on
boundary manifolds + level descriptors) makes each sphere a *shell* in a
deeper colony: the surface fractal of sphere A is the interior of sphere
A+1. Cram, then cram into the crams.

## 4. What this means for the existing code (implementation hooks)

| Piece | Exists | Ecosystem role |
|-------|--------|----------------|
| `wubu_hyper.c` (mobius_add, exp_0, log_0, gyro) | ✅ Lean-verified | the physics engine |
| `wubu_nested_ssm.c` (K balls, K curvatures) | ✅ | the colony skeleton |
| `wubu_moe_hyperbolic.c` (hyperbolic experts) | ✅ | sphere expert cores |
| `wubu_poincare_gqa.c` (gyro-rotated attention) | ✅ | within-ball attention |
| `wubu_moe.c` (256 experts) | ✅ | candidate sphere pool |
| `wubu_hive` (grow/shrink/specialize) | ✅ | ecosystem lifecycle |
| `wubu_ecosystem.c` (physics router: potential-well top-K, zero learned params) | ✅ NEW | the user's "use physics" |
| surface fractal index (space-filling addr) | ✅ NEW | the "cram into empty space" |

**Gate for the physics router**: `test_ecosystem` — build a 64-ball colony,
feed synthetic clusters, assert that (a) inputs land in the balls whose
region they resemble (top-k purity > 90%), (b) only K balls' params are
touched (active-param counter < 12% of total), (c) shrink+grow keeps the
colony stable (no thrash). **PASS** (purity 100%, active 6.25%, no thrash,
ASan clean).

**Gate for fractal stacking**: `test_fractal_index` — a space-filling
curve on the sphere surface assigns unique addresses to ≥ 2^20 surface
sites in < 1MB of curve table; address collision rate 0; round-trip
address→point→address exact. **PASS** (Hilbert bijection 1M/1M, box-dim
2.000, stacking 94% distinct).

## 5. Triple-DA

1. **Correctness**: Möbius/gyro math is Lean-proven. The new parts are
   the potential-well router (convexity of gyro-distance — needs the
   math library proof or a numeric check) and the fractal index
   (space-filling curves are constructive, verifiable by round-trip).
2. **Efficiency**: physics router is O(N) potential evals over N balls
   (all closed-form, vectorizable) — cheaper than a learned router's
   softmax over N experts with N×dim weights. Active-param ratio is a
   hard measurable (count bytes touched per forward).
3. **Risk**: ball collapse (all curvature → 0, colony → dense model) —
   mitigate with curvature floor + diversity regularizer; input to
   wrong ball (cold-start) — mitigate with a mandatory 1-2 always-on
   "heart" balls (the colony's core trunk, WuBu1 §3.2 boot core dense).

## 6. Source grounding
- DeepSeek-V3 tech report (arXiv 2412.19437): 671B/37B, 256 experts
- HuggingFace "Mixture of Experts Explained" blog
- Sarkar (2011) tree embeddings in Poincaré ball, arbitrary low distortion
- Wikipedia "Space-filling curve" (sphere at infinity / Kleinian groups)
- Hyperbolic deep learning surveys (arXiv 2507.17787): exponential
  volume growth = compact embedding of hierarchies
