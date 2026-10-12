# WuBuMath Core — the math itself, matched to code

This file states each mathematical object once, with the exact convention used
by the Python implementation and the C port. The Lean proofs use the same
conventions unless noted.

## Nested spaces

```
H^{n1}_{c1,s1} ⊃ H^{n2}_{c2,s2} ⊃ … ⊃ H^{nk}_{ck,sk}
```

- n_i ∈ ℕ — dimension of level i (learnable)
- c_i > 0 — curvature (learnable). Larger c = steeper = faster volume growth.
- s_i > 0 — scale (learnable). Inner levels zoom in: s_i = 1/1.2^i in the viz.

Convention in math_viz/01: curvatures follow φ^{i-2} (golden progression,
φ = (1+√5)/2). This is aesthetic in the viz; in the real encoder the
parameters are learned.

## Poincaré ball model

Open ball of radius 1/√c in R^n. Metric:

```
ds² = 4·dx² / (c·(1 − |x|²)²)
```

so the conformal factor is λ(x) = 2/(√c·(1−|x|²)) and

```
d_c(0, x) = 2·atanh(√c·|x|) / √c
```

Boundary |x| = 1/√c is at infinite hyperbolic distance. Practical code always
clips to (1−ε) of that radius.

## Projection

```
poincare_clip(x) = x · min(1, (1−ε)/max(√c·‖x‖, (1−ε)))
```

i.e. if ‖x‖ > (1/√c)(1−ε), rescale radially. Linear, idempotent-ish (one pass).

## Exponential map (scale-aware)

```
exp_0^{c,s}(v) = tanh(s·√c·‖v‖) · v / (√c·‖v‖)
```

Maps tangent vector v at origin → point in ball. Standard form has tanh(√c·‖v‖);
the scale-aware form multiplies the hyperbolic radius by s. NOTES:

- Python `scale_aware_exponential_map` docstring says `tanh(s*√c*‖v‖ / 2)`
  but the code computes `tanh(scale * sqrt_c * norm)` — no /2. Code wins.
- MATH/README.md writes `tanh(√c·‖v‖ / 2)` — matches the standard curvature-c
  Poincaré disk. Code wins; README is the outlier.
- C port `wubu_exp_map` agrees with Python exactly (no /2).

Consequences: exp_0^{c,1} is the textbook map for metric with conformal
factor 2/(1−|x|²) when c=1. With the /2 in the tanh it would correspond to
metric 4/(1−|x|²)² at the origin — the actual Poincaré disk metric. The code's
no-/2 version is a "compressed" radius map that still lands inside the ball
and is a bijection T_0 → ball; it is a valid parameterization but is not the
Riemannian exponential map of the standard Poincaré metric. Treat as WuBu's
own convention, documented, not an error. (Logged in overlap-ledger.md.)

## Logarithmic map

```
log_0^{c,s}(y) = (1/s)·atanh(√c·‖y‖) · y / (√c·‖y‖)
```

Clamp √c‖y‖ to [−1+ε, 1−ε] before atanh. Exact left inverse of the no-/2 exp
above (verified numerically; the 1D Lean proof PoincareBall.lean proves the
round trip with the same no-/2 structure).

## Möbius addition (Gyrovector addition)

```
x ⊕_c y = [ (1 + 2c⟨x,y⟩ + c‖y‖²)·x + (1 − c‖x‖²)·y ] /
          (1 + 2c⟨x,y⟩ + c²‖x‖²‖y‖²)
```

Gyrovector-space addition (Ungar). In 1D this reduces to the closed form in
`WubuProofs/MobiusAdd.lean`:

```
mobius_add_1d(c,x,y) = [ (1 + 2cxy + cy²)x + (1 − cx²)y ] /
                       (1 + 2cxy + c²x²y²)
```

Denominator = (1 + cxy)² > 0 whenever x², y² < 1/c. Proven in Lean.

## Gyration

```
gyr_c[u,v] w = (−(u ⊕_c v)) ⊕_c (u ⊕_c (v ⊕_c w))
```

A rotation of the ball onto itself. In 1D it is the identity
(`gyration_1d_is_identity`, proven in Lean). In nD it is a proper SO(n) element
and is what makes Möbius addition non-associative in a controlled way:
(u⊕v)⊕w ≠ u⊕(v⊕w) in general; gyroassociativity is
u⊕(v⊕w) = gyr[u,v](u⊕(v⊕w))... i.e. the gyration measures the failure of
associativity. Lean proves gyroassociativity explicitly in 1D
(`gyroassoc_1d`).

## Quaternions / Hamilton product

```
q = w + xi + yj + zk,  p = a + bi + cj + dk

q·p = (wa − xb − yc − zd)
    + (wb + xa + yd − zc)i
    + (wy − xz + yb + wa)j      [note: code uses w1*y2 − x1*z2 + y1*w2 + z1*x2]
    + (wz + xy − xv + yw)k
```

Implementation in `hamilton_product` (Python) and `wubu_hamilton_product` (C)
are component-identical.

Unit quaternions ≅ S³ ≅ SU(2) → double cover of SO(3). Double-quaternion
conjugation p·v·q with p = q⁻¹ performs rotation by 2·angle(q).

Role in WuBu: the inter-level rotation R_i lives in SO(n_i) and is
implemented as quaternions (SO(4) for BSP node splits). Rotation happens in
the Euclidean tangent space T_p(H^n_i) ≅ R^{n_i}, not inside the ball.

## Inter-level transition (the nesting operator)

Per level i → i+1:

1. Embed both in tangent spaces at origins.
2. Apply learned rotation R_i (quaternion) to primary data, boundary
   representations, and the Level Descriptor l_{d_i}.
3. Apply non-rotational map T̃_i adjusting features and dimension n_i → n_{i+1}.
4. Compute Relative Vectors d_{i+1,j,k} in the target tangent space
   (rotation-aware structure codes).
5. Optional intra-level tangent flow F_i; level Spread σ_i for uncertainty.

## MLA compression identity

DeepSeek-style Multi-head Latent Attention KV compression:

```
H·W_K − H·W_DKV·U_K = H·(W_K − W_DKV·U_K)
```

Pure algebra (right-distributivity), proven in Lean
(`WubuProofs/MLACompression.lean`). WuBu's contribution here is pairing it
with the hyperbolic geometry, not the identity itself. The error-bound claim
(‖W_K − W_DKV·U_K‖_F controls KV compression error) is an Eckart–Young /
low-rank argument — narrative in WuBu, sketch in Lean.

## The fiber-bundle claim (math_viz/05)

WuBu nesting as a principal G-bundle:

- Flag manifold F = H^{n1} ⊃ H^{n2} ⊃ … ⊃ H^{nk}
- Inter-level rotations R_i define a connection ∇
- Connection 1-form A = Σ R_i⁻¹ dR_i (Maurer–Cartan)
- Curvature F_∇ = dA + A∧A
- Numerically: parallel transport path-independence ⟺ F = 0 (flat)

Currently: numerical simulation only, no Lean proof. This is the highest-value
target for formalization because it is the most original claim and the least
supported.

## Stability conventions (matter for reimplementation)

- EPS = 1e-7 (C) / 1e-5–1e-8 per-call (Python)
- Norms accumulated in float32 (Python) / double (C)
- atanh argument clamped strictly inside (−1, 1)
- After every map: finite check, NaN→0
- Poincaré projection applied after exp to guarantee the ball invariant
