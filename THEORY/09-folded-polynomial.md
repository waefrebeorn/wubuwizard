# THEORY/09 — The Folded Polynomial: One Poly, Multiple Outputs

> "because we designed the model, we can align everything and redesign
> everything to make every trick work so that everything is the best."
> — WaefreBeorn

The **folded polynomial** (Silas Lock, via Kaze Emanuar) is a general
principle for speeding up the AGI: compute **multiple related outputs** from
a single polynomial evaluation, using symmetry identities + a single
`sqrt`/`rsqrt` refinement step. The sqrt does double duty — it derives the
complementary function AND normalizes the output vector.

> "i meant sound in both ways like a sound concept. this can speedup the AGI"
> — WaefreBeorn clarifying that "folded polynomial" is a *sound concept*
> (both literally — DSP/audio — and figuratively as a general computational
> optimization for the AGI engine).

## The Pattern (General)
Whenever the AGI engine needs a PAIR of related math functions on hot paths,
fold them:

| Pair | Identity | Fold Technique |
|------|----------|----------------|
| sin + cos | sin²+cos²=1 | 2nd-order poly on [0,π/4] + sqrt(1-y²) |
| GELU + GELU' | derivative relation | shared poly + sqrt |
| tanh + sech | tanh²+sech²=1 | 2nd-order poly + sqrt(1-y²) |
| exp + log | exp(-x) = 1/exp(x) | poly + reciprocal (rsqrt NR) |
| x^(1/5) + x^(4/5) | x = x^(1/5) · x^(4/5) | poly + sqrt |

## CPU SIMD Application (WuBu-35M hot paths)
1. **RoPE tables** (`wubu_fold_sincos`): currently `sincosf` × 2 calls.
   Folded poly: 1 poly eval + 1 sqrt → sin AND cos pair.
2. **Activation functions** (GELU/tanh): the GQA attention uses GELU in
   some configs; tanh in the hyperbolic path. Both have a "pair" form.
3. **Poincaré embeddings** (`wubu_poincare_gqa.c`): Möbius transforms need
   sinh/cosh pairs — foldable via cosh²−sinh²=1.
4. **`wubu_foldmath.h`**: the fold abstraction layer. Add
   `wubu_fold_sin_cos_poly()` as the canonical folded implementation.

## Original Technique (N64 sin/cos)
The algorithm (Silas Lock):
1. Fold angle x → [0, π/4] via octant bit-math (fast integer shifts/masks)
2. Evaluate a SINGLE 2nd-order polynomial on the folded angle
3. Use `sqrt(1 - result²)` to derive the complementary value
4. Un-fold: swap sin/cos based on octant, flip signs as needed

**Result**: 1.5 cycles faster, 3× more accurate than 3rd+4th-order separate
polynomials (63.5 vs 65 cycles on N64, 200 calls/frame).

## Estrin vs Horner for the Folded Poly
- **Horner** (`a0 + x*(a1 + x*a2)`): serial deps, fine for degree ≤ 4
- **Estrin** (parallel tree): `D[i] = C[2i] + C[2i+1]*x`, then tree-reduce)
  — breaks serial deps, better ILP on modern CPUs

For a 2nd-order poly, Horner is fine (2 multiplies, 2 adds, minimal deps).
The real win is the **sqrt replacing a second poly** — that's the "folded" part.

## Triple-DA Verdict
1. **Correctness**: Trig identities are exact; the poly fits 1/8 the range.
   For non-trig pairs (exp/log, tanh/sech), the identity-based folding is
   mathematically sound.
2. **Applicability**: Direct target = `wubu_fold_sincos` in `src/wubu.c`
   (builds the RoPE cos/sin tables). Also: `wubu_oscillator.c`,
   `wubu_poincare_gqa.c`, `wubu_foldmath.h`.
3. **Trade-offs**: sqrt costs ~10-20 cycles on x86; but saves a full
   polynomial evaluation. Use `_mm256_rsqrt23_ps` + 1 Newton-Raphson
   refinement for speed-critical paths.

## Source
- Video: https://youtu.be/hffgNRfL1XY (Kaze Emanuar, "The Folded Polynomial")
- Technique: Silas Lock (Discord submission to Kaze's server)
- Transcript: cached at `~/.hermes/profiles/mind-palace/cache/web/www.youtube.com-da3046a6e0.md`
