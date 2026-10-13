# WuBuMath

**The home of all WuBu数学 (WuBu Math) — and the verified mathematical
foundation of the ~ WuBu ~ AGI manifolds.**

WuBuMath has two layers:

1. **A machine-verified formal core** (`lean/WubuProofs/`) — 17 theorems
   proven in Lean 4 + mathlib, zero `sorry`. This is the geometric
   foundation the ~ WuBu ~ AGI's nested hyperbolic manifolds stand on:
   Möbius gyrovector arithmetic, gyration-as-rotation, parallel-transport
   isometry, Klein-bottle deck bookkeeping, quaternion rotations. Every
   hyperbolic neural layer, flow-matching step, and Riemannian optimizer
   update in the WuBu stack relies on facts proven here.
   **→ See [`docs/LINKAGE.md`](docs/LINKAGE.md) for the theorem →
   capability → codebase map, and
   [`lean/README.md`](lean/README.md) for the full status table.**

2. **A pure C11 implementation** (`src/`) — the same geometry, implemented
   in dependency-free C11: Hamilton quaternion encoder, hyperbolic
   operators, nested encoders, Riemannian SGD, parallel transport, plus
   the VHF audio/media pipeline and a slermed JAX compute core. Zero
   Python, zero PyTorch, zero external deps. **The C code is correct
   *because* the Lean theorems hold** — where they overlap, the Lean side
   is the source of truth.

## Why this exists

The ~ WuBu ~ AGI reasons and stores information on **nested hyperbolic
manifolds** (Poincaré balls of increasing curvature). That representation
is only sound if a small set of geometric facts hold exactly: that Möbius
addition stays in the ball, that gyration preserves inner products (it's a
rotation), that parallel transport is a metric isometry, that the
exponential map never escapes the manifold. **WuBuMath proves exactly
those facts** — so the AGI's manifold reasoning rests on verified ground,
not on formulas that merely look right.

## The verified core (`lean/WubuProofs/` — 0 `sorry`)

| Theorem | Why it matters for the AGI |
|---|---|
| Möbius addition preserves the ball (nD) | The invariant every hyperbolic layer needs |
| **Gyration preserves the inner product** (Ungar 1.32) | Gyration is a rotation → hyperbolic NN layers are well-defined |
| Gyration preserves the norm (Ungar 1.33) | Rotation-equivariant layers |
| **Parallel transport is a Riemannian isometry** | On-manifold guarantee for Riemannian SGD / flow matching |
| Exponential map lands in the ball | Flow-matching ODE integrator never escapes the manifold |
| Geodesic interpolation endpoints | Flow-rollout path starts at the keyframe |
| Klein deck group (B² orientation-preserving, ABA=B) | Non-orientable nesting / movable-manifold storage |
| Quaternion norm multiplicativity ‖p·q‖²=‖p‖²‖q‖² | SO(4) rotations, BSP node splits |

Plus: exp/log round-trip, MLA KV-compression Frobenius bound, SO(3) fiber
bundle, φ-curvature nesting progression, holographic optimizer.

**Verify it yourself:**
```bash
cd lean
./fastcheck.sh WubuProofs/GyrationND.lean   # single file, ~4s
lake build                                   # whole library, 0 sorry
./.lake/build/bin/wubu_proofs                # prints the theorem table
/usr/bin/python3.12 ../examples/wubu_manifold_demo.py   # numerical demo
```

## The C11 implementation (`src/`)

All 64 tests pass:
```bash
make test
```
```
[Color Manifold]     6/6   [Hamilton Encoder]   1/1
[Circular L1 Loss]   3/3   [VHF Decoder]        1/1
[PRNG]               2/2   [VHF Audio Pipeline] 6/6
[Q-Controller]       3/3   [Slermed JAX core]  35/35
[Positional Encode]  1/1
[Loss Computation]    2/2   TOTAL: 64/64 PASS
```

Only needs GCC with C11. Runs on Linux/macOS/WSL.

## Project structure

```
WuBuMath/
├── lean/                         # THE VERIFIED CORE (Lean 4 + mathlib)
│   ├── WubuProofs/               #   17 proven theorems, 0 sorry
│   ├── fastcheck.sh              #   single-file check (~4s, olean-cached)
│   ├── leanfind.sh               #   mathlib lemma lookup (0 build cycles)
│   └── README.md                 #   full status table
├── docs/
│   ├── LINKAGE.md                #   theorem → AGI-manifold → codebase map
│   ├── theory/                   #   WuBu_Nesting, WuBuHypCD, math_viz, papers
│   ├── GAP_REGISTER_1000.md      #   verified gap register
│   └── SOTA_ROADMAP_50_GROUPS.md
├── examples/
│   └── wubu_manifold_demo.py     #   numerical demo of the proven theorems
├── src/                          # C11 IMPLEMENTATION (dependency-free)
│   ├── math/  model/  train/     #   color, Hamilton, VHF, Q-controller
│   ├── jax/                      #   slermed JAX compute core (35 tests)
│   └── tests/                    #   64 tests
├── include/
└── Makefile
```

## Origin

Primary source: [`vhf_audio.py`](https://github.com/waefrebeorn/bytropix/blob/master/AUDIO/wubusynth/vhf_audio.py)
from [bytropix](https://github.com/waefrebeorn/bytropix). WuBuMath is the
reconciliation — all WuBu math comes home, and the hyperbolic/nesting core
is now formally verified rather than merely implemented.

## Part of the WuBu Family

| Repo | Purpose |
|------|---------|
| [WuBuOS](https://github.com/waefrebeorn/WuBuOS) | The AGI body — OS kernel, runtime, compiler (the manifolds run here) |
| [bytropix](https://github.com/waefrebeorn/bytropix) | Inference engine (C server, GPU kernels) |
| **WuBuMath** | **All math + the verified AGI-manifold foundation** ← you are here |

## License

This project is licensed under the **Waefrebeorn Umbrella License v3.0**.
See the [LICENSE](LICENSE) file for the full license text.

The Waefrebeorn Umbrella License is a custom source-available license.
It is not OSI-approved and not FSF-approved.
