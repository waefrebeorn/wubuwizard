# Poincaré + TurboQuant retrieval: cheap storage with a bound

Two pieces of mathematics that solve different halves of the same problem, and one place where they do **not** compose
the way the intuition suggests. Read the caveat before the pipeline.

## The two pieces

**Poincaré ball embeddings collapse the dimension you need.** Hyperbolic space embeds hierarchical/tree-like structure
in `O(log n)` dimensions with bounded distortion, where Euclidean space needs `O(n)` for exact embedding. For
catalogue-like data — directories, taxonomies, "is-a" and "part-of" chains — this is the difference between 32 and 768
dimensions. WuBuMath already has the operators: `wubu_poincare_dist`, `wubu_mobius_add`, `wubu_expmap`/`wubu_log_map`,
`wubu_poincare_project`, `wubu_hyper_exp/log`, `wubu_hyp_vq`, and the `wubu_gaad_encoder`; wubuwizard's encoder tree
(`ENCODERS/hash-mind/c/wubu_math.c`, `include/wubu_hyper.h`) carries the same family. So the *representation* side is
built.

**TurboQuant makes per-coordinate quantisation near-optimal without training a codebook.** (Zandieh, Daliri, Hadian,
Mirrokni — arXiv 2504.19874; ICLR 2026.) The construction is deliberately boring and that is the point:

1. apply a fixed random orthogonal rotation `R` to the vector (data-oblivious, can be seeded, never fitted);
2. after rotation the coordinates are approximately i.i.d. Beta-distributed, so each one can be scalar-quantised with a
   **Lloyd-Max codebook computed once from that distribution** — no dependence on the data at all;
3. for **inner-product** work, quantise `x` to `(b-1)` bits MSE-optimally, take the residual `r = x - x̂`, and encode only
   the **sign** of `r` projected through a second random matrix (QJL); the estimator that follows is **unbiased**.

Guarantees, quoted from the paper, for `x` on the unit sphere:

```
MSE-optimal        D_mse(Q)  <= (sqrt(3)*pi/2) * 4^-b    ~= 2.7 * 4^-b
inner-product      D_prod(Q) <= sqrt(3)*pi^2 * ||y||^2 / (d * 4^b)
information bound  D_mse(Q)  >= 4^-b        (any quantiser, any reconstruction)
```

The upper bound is within a factor **2.7** of the lower bound at every bit-width and dimension, which is what
"near-optimal" means here. Note the `4^-b` dependence: **each extra bit quarters the error**, so bit-width is the cheap
knob, not dimension.

## Storage, from the bound's own geometry

Per-item cost is `d*b/8` bytes plus a 2-byte fp16 norm (needed to undo the unit-sphere normalisation):

```
dims  bits  bytes/item   1M items   D_mse     IP err (of ||y||)
16    2     6.0          6.0 MB     0.170     0.258
16    4     10.0        10.0 MB     0.011     0.065
32    2     10.0        10.0 MB     0.170     0.183
32    3     14.0        14.0 MB     0.043     0.091
32    4     18.0        18.0 MB     0.011     0.046
64    3     26.0        26.0 MB     0.043     0.065
128   4     66.0        66.0 MB     0.011     0.023
```

A 768-dimension f32 embedding costs 3,072 B, so 1M items is 3.07 GB. **32 dimensions at 2 bits is 10 B/item = 10 MB,
307x smaller**, and 32 dimensions at 4 bits is still 170x smaller with 0.011 MSE. The dimension reduction from Poincaré
is what makes the per-item figure absurdly small; the bit-width is what makes the error tolerable. They multiply.

Recommended operating point: **d = 32, b = 3 or 4** (14-18 B/item). At b = 2 the inner-product error is 0.18-0.24 of
`||y||`, which is too coarse to rank reliably; b = 3 halves it and b = 4 halves it again for 4-8 extra bytes.

## The caveat that decides the design

**TurboQuant's guarantees are Euclidean, and Poincaré distance is not.** The bounds above are on `||x - x̂||^2` and on
`<y, x>` in the ambient coordinates. On the Poincaré ball the distance is

```
d(u,v) = arcosh(1 + 2c||u-v||^2 / ((1-c||u||^2)(1-c||v||^2)))
```

so the same coordinate error produces a *small* distance error near the origin and an **unbounded** one as `||u|| -> 1`.
Quantising ball coordinates uniformly and then ranking by hyperbolic distance therefore has no useful error bound at the
boundary, which is exactly where hierarchies put their leaves. Three honest options:

1. **Score by inner product in the tangent space at the origin** (the log map), where the metric is flat. Then
   TurboQuant's unbiased inner-product bound applies *directly and exactly* as stated, and the hyperbolic embedding is
   doing dimensionality work rather than scoring work. This is the design to build first: it is the only one where the
   guarantee survives contact with the metric.
2. **Rank by a radius-capped hyperbolic distance**, `||u|| <= 1-eps`, and bound the error with the exp-map Lipschitz
   constant on that ball. Defensible, but `eps` becomes a tuning parameter that silently trades recall for a bound.
3. **Quantise in the tangent space, then exp-map back** for a hyperbolic score. Strictly better than quantising the ball
   directly, because the error is measured where the metric is Euclidean — but the projection back through `exp`
   re-introduces the boundary blow-up, so it needs the same `eps` handling as (2).

`wubu_hyp_vq` (the existing hyperbolic VQ) sits in a different trade: it needs a **codebook**, which must be trained and
stored, and `n_codebook == n_vecs` for losslessness (per the manifold-math skill). TurboQuant needs **no codebook at
all** — only a seed. That is the practical reason to add it: for online or ever-growing catalogs, a data-oblivious
quantiser has no fitting step to invalidate when the corpus changes.

## Pipeline

```
index:  item -> encoder (Poincaré/GAAD)            -> u in B^d, d = 32
        u -> log map at origin                     -> t in R^d   (flat coordinates)
        t -> t/||t||, store ||t|| as fp16          -> unit vector + norm
        unit -> R (seeded random rotation)         -> z
        z -> Lloyd-Max codebook, b bits/coordinate -> code     (3-4 bits/dim)
        z - z_hat -> QJL sign bits (only if ranking by inner product matters)

query:  same first four steps, no quantisation of the query
        score = TurboQuant inner-product estimator:
                  <q, x> ~= ||x|| * ( <z_hat, q_z> + sqrt(pi/2)/d * ||r|| * <S q_z, sign(S r)> )
        top-k by score, then optionally re-rank the k candidates in f32 hyperbolic distance
```

The re-rank is where accuracy is recovered cheaply: quantised scoring proposes, exact hyperbolic distance disposes. With
k = 100 candidates out of 1M, the exact stage touches 100 x 128 B and the quantised stage touches 10 MB.

## What to build, in order

1. **Measure the encoder output distribution first.** The Beta assumption holds after rotation for *any* input
   distribution, but the per-vector norm distribution does not, and the fp16 norm is where the dynamic range is lost.
   Dump 10k encoder outputs, check `||t||` spread, and check whether the ball radius is concentrated near 1 (if so,
   option 2's `eps` is doing most of the work and the design needs revisiting).
2. **Orthogonal rotation in C11**: a seeded block-diagonal random rotation of 8x8 blocks is enough (it keeps the
   Beta-approximation and costs `O(d)` instead of `O(d^2)`); generate with a fixed splitmix64 seed so index and query
   agree bit-for-bit.
3. **Lloyd-Max codebook as a constant table** — the Beta distribution depends only on `d`, so a 3-bit and a 4-bit table
   can be precomputed once and pasted as static arrays, exactly as the GGUF k-quant tables are.
4. **QJL sign sketch** with the same seeded matrix.
5. **Verify against exact**: for every quantised score, compute the exact inner product and check the error distribution
   against the bound `sqrt(3)*pi^2/(d*4^b)`. If the measured error exceeds it, the rotation or the codebook is wrong —
   the bound is a testable prediction, not a hope.

## Position

Storage side: WuBuMath/WuBuOS already have Poincaré operators and a hyperbolic VQ with CABAC two-tier compression, so
the *encoding* is not the missing piece. The missing pieces are the **data-oblivious rotation + Lloyd-Max table + QJL
residual**, which is a few hundred lines of C11 with no training loop, and the **decision about which metric scores the
query** — that decision is what makes or breaks the retrieval quality, and option 1 above is the only one where
TurboQuant's bound is exactly the right bound.

Not yet verified anywhere in this repo: the encoder's output-norm distribution, and whether 32 dimensions is enough for
the target catalogue. Both are one dump of 10k vectors away, and both should be measured before any of the above is
written.

## Verification: the bound is real, measured independently

`tools/turboquant_probe.c` builds the actual pipeline — fixed random orthogonal rotation (Gram-Schmidt on a Gaussian
matrix, seeded), the coordinate density `(1-t^2)^((d-3)/2)` integrated numerically, a Lloyd-Max codebook derived from
that density by 1-D k-means, quantise/dequantise, and measures `E||x - x_hat||^2` and the inner-product error over
4,000 random unit vectors. No codebook training, no data fitting — every table is a constant of `(d, b)`.

    d=32, 4000 trials
    bits  meas D_mse   bound D_mse   verdict   meas IP err   bound IP
    1     0.3525       0.6802        PASS      0.10476       0.36545
    2     0.1111       0.1700        PASS      0.05941       0.18272
    3     0.0322       0.0425        PASS      0.03142       0.09136
    4     0.0086       0.0106        PASS      0.01644       0.04568

and the same at d = 16, 64, 128 (MSE at b=2: 0.1064 / 0.1147 / 0.1164), i.e. **dimension-free MSE**, which is the
paper's central claim and is now reproduced here.

Against the information-theoretic lower bound `4^-b`, the measured error is

    b=1  1.41x    b=2  1.78x    b=3  2.06x    b=4  2.20x

so this implementation sits **inside** the paper's guaranteed 2.7x factor, and close to the floor at low bit-widths.
Inner-product error falls as `1/sqrt(d)` as predicted: at b=2 it is 0.081 / 0.059 / 0.043 / 0.030 for d = 16 / 32 / 64
/ 128, in every case well under the bound.

What this buys, stated as numbers rather than adjectives: at **d=32, b=3** the per-item cost is 14 B with 0.032 MSE, so
1M items occupy **14 MB** against 3.07 GB for 768-dim f32 — 219x — and the error is a *proven* quantity, not an
empirical hope. At b=4 it is 18 MB and 0.009 MSE, 170x smaller.

What it does not buy: nothing here says anything about retrieval *quality* on real embeddings. The error is on random
unit vectors. Whether 3.2% squared coordinate error preserves the neighbour ordering of real encoder outputs is a
separate measurement that needs actual embeddings, and the caveat section above is still the architectural decision.
