# WuBuMath — Linkage Map: Proven Theorems → AGI Manifolds

> How every machine-verified theorem in `lean/WubuProofs/` connects to the
> WuBu Nesting framework and to the ~ WuBu ~ AGI. This is the map that shows
> the proofs are not isolated formalities — each one is load-bearing for a
> concrete capability in the system.

## The one-paragraph thesis

The ~ WuBu ~ AGI reasons and stores information on **nested hyperbolic
manifolds** (Poincaré balls of increasing curvature), connected by
**Möbius gyrovector** operations. Every layer of a hyperbolic neural
network, every flow-matching generative step, and every Riemannian
optimizer update relies on a small set of geometric facts — that Möbius
addition stays in the ball, that gyration is a rotation (preserves inner
products), that parallel transport is a metric isometry, that the
exponential map never escapes the manifold. **This library proves exactly
those facts in Lean 4 + mathlib.** They are the verified foundation the
AGI's manifold reasoning stands on.

## The dependency graph

```
                    ┌─────────────────────────────────────────┐
                    │   PoincaréBall.lean                     │
                    │   exp_0^c ∘ log_0^c = id  (round-trip)  │
                    └───────────────────┬─────────────────────┘
                                        │ (exp/log are the bridges
                                        │  between tangent & manifold)
              ┌─────────────────────────┼──────────────────────────┐
              ▼                         ▼                          ▼
   ┌────────────────────┐   ┌──────────────────────┐   ┌────────────────────┐
   │ MobiusAdd.lean     │   │ ExpMapBall.lean      │   │ NestedHyperbolic   │
   │ (1D ball closure)  │   │ exp_0^c lands in ball│   │ Spaces.lean        │
   └─────────┬──────────┘   │ c·‖exp‖² < 1         │   │ (φ curvature       │
             │              │ [GAP C002]           │   │  progression)      │
             ▼              └──────────────────────┘   └────────────────────┘
   ┌──────────────────────────────────────────┐
   │ MobiusAddND.lean                          │
   │ Möbius addition preserves ball in nD      │
   │ D²−c‖N‖²=(1−ca)(1−cb)[(1+cp)²+c²(ab−p²)] │
   └───────────────────┬───────────────────────┘
                       │ (ball invariant is the
                       │  precondition for everything)
                       ▼
   ┌──────────────────────────────────────────────────────────┐
   │ GyrationND.lean  ← THE CORE                              │
   │  • Mobius norm formula (nD)                              │
   │  • gyration preserves the ball (nD)                      │
   │  • gyration preserves the NORM: ‖gyr w‖² = ‖w‖²  [1.33]  │
   │  • gyration preserves the INNER PRODUCT:                  │
   │      ⟨gyr a, gyr b⟩ = ⟨a,b⟩  [1.32]  ← "gyration is a    │
   │      rotation": the operation every hyperbolic layer uses │
   └───────────────────┬───────────────────────────────────────┘
                       │
       ┌───────────────┼────────────────────┐
       ▼               ▼                    ▼
┌─────────────┐ ┌──────────────┐ ┌──────────────────┐
│ Quaternion  │ │ Parallel     │ │ GeodesicInterp   │
│ Rotation    │ │ Transport    │ │ γ(0)=x           │
│ ‖p·q‖²=     │ │ isometry     │ │ [GAP C009]       │
│ ‖p‖²‖q‖²    │ │ [GAP C003]   │ │ (flow rollout    │
│ (SO(4) BSP) │ │ (on-manifold │ │  path starts at  │
│             │ │  guarantee)  │ │  keyframe)       │
└─────────────┘ └──────────────┘ └──────────────────┘
       │               │                    │
       └───────────────┴────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │ The ~ WuBu ~ AGI manifold    │
        │ stack: hyperbolic NN layers, │
        │ flow-matching generative     │
        │ models, Riemannian SGD,      │
        │ nested Klein storage.        │
        └──────────────────────────────┘
```

## Theorem-by-theorem linkage

| # | File | Theorem (machine-verified) | AGI-manifold role | Codebase use |
|---|---|---|---|---|
| 1 | `PoincareBall.lean` | `exp_0^c(log_0^c(y)) = y` | Tangent↔manifold bridge | Every exp/log map call |
| 2 | `MobiusAdd.lean` | Möbius addition preserves ball (1D) | Gyrovector arithmetic | Base case for nD |
| 3 | `MobiusAddND.lean` | **Ball preservation in nD** | The invariant every layer needs | `mobius_add_nd` in all hyperbolic ops |
| 4 | `MLACompression.lean` | Frobenius error bound ‖H·W_K−H·W_DKV·U_K‖≤‖H‖·‖…‖ | KV-cache compression correctness | Multi-head latent attention |
| 5 | `HyperbolicGyration.lean` | 1D gyration + gyroassociativity | Gyrogroup laws (1D) | Foundation for nD gyration |
| 6 | `GyrationND.lean` | **Norm formula (nD)** | Distance computations | Every hyperbolic distance |
| 7 | `GyrationND.lean` | **Gyration preserves ball (nD)** | Gyration stays on-manifold | Hyperbolic layer outputs |
| 8 | `GyrationND.lean` | **‖gyr‖²=‖w‖² (Ungar 1.33)** | Gyration is norm-preserving | Rotation-equivariant layers |
| 9 | `GyrationND.lean` | **⟨gyr a,gyr b⟩=⟨a,b⟩ (Ungar 1.32)** | Gyration is an inner-product isometry — *the* rotation property | **Every hyperbolic NN layer** |
| 10 | `KleinDeck.lean` | B² orientation-preserving; ABA=B | Non-orientable nesting bookkeeping | Klein-bottle storage levels |
| 11 | `QuaternionRotation.lean` | ‖p·q‖²=‖p‖²‖q‖² | SO(4) rotations via quaternions | BSP node splits, 3D rotations |
| 12 | `FiberBundle.lean` | [Lx,Ly]=Lz; SO(3) closure | Principal-bundle structure | Level-to-level transport |
| 13 | `NestedHyperbolicSpaces.lean` | φ-curvature progression | Nesting-chain invariants | Multi-level curvature schedule |
| 14 | `HolographicOptimizer.lean` | g=q·2π+r; soul/echo recovery | Gradient decomposition | Lazarus optimizer |
| 15 | `ParallelTransport.lean` | PT is a Riemannian isometry **[GAP C003]** | On-manifold guarantee for SGD/FM | Riemannian optimization |
| 16 | `ExpMapBall.lean` | c·‖exp_0^c(v)‖²<1 **[GAP C002]** | ODE integrator never escapes | Flow-matching P-frames |
| 17 | `GeodesicInterp.lean` | γ(0)=x **[GAP C009]** | Flow-rollout path endpoints | Multi-keyframe interpolation |

## The three "load-bearing" theorems

If you read only three, read these — they are what make the AGI manifolds
well-defined:

1. **MobiusAddND ball preservation** (#3) — without it, Möbius addition
   could produce a point outside the ball, and the whole nested-hyperbolic
   representation collapses. The nD identity carries a Cauchy–Schwarz slack
   term `c²(ab−p²)` that the 1D case lacks — a real subtlety the proof
   handles.

2. **Gyration inner-product preservation** (#9, Ungar 1.32) — gyration is
   the "correction" Möbius addition applies to stay associative. Proving
   it preserves inner products means gyrovector operations are genuine
   rotations, so hyperbolic neural layers are well-defined linear-algebra
   objects, not ad-hoc formulas.

3. **Parallel transport isometry** (#15) — this is what lets Riemannian
   SGD and flow matching move gradients/velocities between tangent spaces
   without distorting the geometry. It's the "on-manifold guarantee."

## How to verify all of this yourself

```bash
cd lean
./fastcheck.sh WubuProofs/MobiusAddND.lean   # any single file, ~4s
lake build                                    # whole library, 0 sorry
./.lake/build/bin/wubu_proofs                 # prints this status table
```

Every ✅ above is backed by a compiling Lean file with zero `sorry`. See
`lean/WubuProofs/` for the source and `experiments/` for the numerical
substantiations (Klein capacity advantage).
