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

## Verification 2: does the quantiser preserve neighbour ORDER? Only with a re-rank

The distortion bound is now established, but a bound on `||x - x_hat||^2` does not by itself preserve the top-10 by
inner product, and that is the only thing retrieval cares about. `tools/turboquant_recall.c` measures it on data with
the geometry real embeddings have - clustered and anisotropic (C centres in a K-dimensional subspace, small noise,
normalised) - rather than uniform on the sphere, because uniform data is the easy case.

Corpus 20k items, 200 queries, ground truth = exact top-10:

    d=32, 200 clusters, intrinsic dim 8
    bits  recall@10   +re-rank top-100   bytes/item
    1     0.235       0.946              6.0
    2     0.225       0.938              10.0
    3     0.310       0.989              14.0
    4     0.446       0.998              18.0

    d=128, 200 clusters, intrinsic dim 32
    1     0.275       0.997              18.0
    2     0.253       0.997              34.0
    3     0.312       0.999              50.0
    4     0.446       1.000              66.0

    d=32, 500 clusters, intrinsic dim 16  (more clusters -> easier discrimination)
    1     0.433       1.000              6.0
    2     0.410       1.000              10.0
    3     0.486       1.000              14.0
    4     0.624       1.000              18.0

Two conclusions, and the first one is a negative result that changes the architecture:

**Quantised scoring ALONE is not good enough.** Recall@10 by quantised inner product is 0.23-0.62. The reason is
structural rather than a defect: within a cluster the neighbours are near-ties (noise is 0.35/sqrt(d) per coordinate),
so a 3 % coordinate error reshuffles the top ten even when it puts every true neighbour in the top hundred. Anyone
proposing "quantise and search" without a re-rank is proposing 30 % recall.

**The two-stage design recovers essentially all of it.** Take the quantised top-100, re-score those exactly, keep the
top-10: recall is 0.938-1.000 across every configuration tested, including b=1 at 6 bytes per item. That is the whole
argument for the architecture, and it is now measured rather than assumed.

Cost accounting for the two stages at d=32, b=3, 1M items: the scan touches 14 MB, and the re-rank touches 100 items x
4 B x 32 dims = 12.8 KB. So the exact stage is ~1000x cheaper than the scan and is not a meaningful part of the query
cost. The scan is the query cost, and it is 14 MB against 3.07 GB - which is why this is worth building.

One caveat that follows from these numbers and should be stated before anyone tunes anything: recall@10 quantised-only
did not improve monotonically with bit-width in the first two tables (0.235 at b=1 above 0.225 at b=2 on d=32; 0.275
above 0.253 on d=128). That is not a bug in the codebook - the probe showed MSE falling monotonically. It is the ranking
metric saturating against near-ties: past a point, more bits do not change WHICH items are top-10 because the top-100 set
is already correct. Bit-width buys the candidate set, and the candidate set is what the re-rank consumes.

## Verification 3: the rotation can be block-diagonal, so applying it is O(d) not O(d^2)

TurboQuant's construction needs only *some* fixed orthogonal transform that makes coordinates approximately i.i.d.
Beta; the theory does not require it to mix every dimension with every other. That matters for implementation cost:
a full d x d rotation is O(d^2) per vector and 8 KB of stored matrix at d=32, while a block-diagonal one is O(d) and
64 doubles.

Measured, d=32, 3000 trials, TQ_BLK selects the block size:

    transform          D_mse(b=1)  D_mse(b=2)  D_mse(b=3)  D_mse(b=4)
    full 32x32            0.3529      0.1111      0.0318      0.0087
    block-diagonal 8x8    0.3533      0.1113      0.0323      0.0088
    4x4                   0.3539      0.1104      0.0324      0.0087

Identical within run-to-run noise at every bit-width, and all three pass the bound with the same margin. So the
implementation should use **8x8 block-diagonal rotations**: 8 blocks of 64 doubles for d=32, applied as 8 small matvecs,
O(d) work, no large matrix to store or seed. A 4x4 block is equally fine but buys nothing over 8x8 and mixes less.

This also means the transform does not need to be stored at all - it can be regenerated from one 64-bit seed by the same
splitmix64 sequence, which is what makes the quantiser genuinely data-oblivious: the entire "codebook" is a seed plus a
table of 2^b numbers that depends only on (d, b).

## Implementation: `src/wubu_tqvec.c`

Written and verified. **Name collision to know about:** `src/wubu_turboquant.c` already existed and is a *KV-cache
tile manager* — allocation, LRU eviction, frame planning — whose header comment mentions "TurboQuant+/RotorQuant" and
Q2_0/Q4_0. It does not contain any of the algorithm from the paper. The quantiser therefore lives in `wubu_tqvec.c` /
`wubu_tqvec.h`, and the two must not be merged; anyone searching for "turboquant" in this tree will land on the tile
manager first and should be told so by the comment it now carries.

The module is exactly the design the three verifications settled: seeded 8x8 block-diagonal rotation, Lloyd-Max
codebook computed at init from the `(1-t^2)^((d-3)/2)` law, bit-packed codes, and a score that reads the code without
expanding it back to d floats.

Self-test against the probe's numbers, d=32, 2000 random unit vectors, seed 0xC0FFEE:

    bits  D_mse      bound      verdict  bytes/code  IP rmse
    1     0.3538     0.6802     PASS     4           0.10655
    2     0.1120     0.1700     PASS     8           0.05871
    3     0.0321     0.0425     PASS     12          0.03205
    4     0.0090     0.0106     PASS     16          0.01691

Byte counts are exactly `d*b/8`, so the storage claim is now a property of the code rather than of a table in a
document: **d=32 at b=3 is 12 bytes per item**, and 1M items is **12 MB**.

The API is five calls - init/free, code_bytes, encode, score, decode - with the index being a flat array of codes and
no side structure. Query cost is one rotation of the query plus `d` multiply-adds and a 3-bit unpack per item.

What a caller must not do, because it was measured: use `score` alone as the ranking. `tools/turboquant_recall.c`
showed 0.23-0.62 recall@10 for that; the top-100 plus an exact re-rank is what reaches 0.94-1.00.

## Verification 4: on real trained vectors the ranking collapses - and the reason is measurable

The synthetic tests used clustered data. The honest next test is real vectors, so 3,840 rows of
`v.patch_embd.weight` were pulled out of `mmproj-BF16.gguf` (F32, d=6,912), renormalised to the unit sphere, and
random-projected to d=32 and d=128 - which is the design's own pipeline (project, then quantise).

    file   bits  recall@10   +rerank@100   bytes/item
    d32    2     0.004       0.028         8
    d32    3     0.004       0.024         12
    d32    4     0.004       0.024         16
    d128   2     0.002       0.030         32
    d128   3     0.003       0.029         48
    d128   4     0.003       0.026         64

Rerank recall is 0.024-0.030, and the chance rate for taking 100 candidates out of 3,840 is **100/3840 = 0.026**. The
candidate set is a random sample of the corpus: the quantised ranking carries no information at all. That is not the
quantiser being wrong - the bound still holds on the same data - it is the data having no ranking to recover.

The cause, measured directly on the score distribution:

    d=32   gap 10th-11th : median 0.00284   quantisation IP rmse 0.12024   ratio 42x
           gap 100th-101st: median 0.00056                                  ratio 215x
    d=128  gap 10th-11th : median 0.00169   quantisation IP rmse 0.06012   ratio 36x
           gap 100th-101st: median 0.00030                                  ratio 202x

The quantisation error is 40-200x larger than the margin separating the k-th from the (k+1)-th. Under those conditions
no quantiser of any design recovers the ranking, because the information is not in the scores at this precision. Note
also why this data is like this: the top-32 principal components of those weight rows hold only **10 %** of the
variance, so they are close to isotropic - and isotropic vectors in d=32 have inner products that are nearly tied
everywhere.

So the criterion the earlier synthetic successes were silently satisfying is

    quantised retrieval is viable  <=>  (top-k score gap)  >~  sigma_quant,   sigma_quant ~ sqrt(3*pi^2/(d*4^b))

At d=32, b=3 the quantiser's error is 0.032; the synthetic clustered corpus had gaps far above that and scored
0.94-1.00, while these weight rows have gaps of 0.003 and score at chance. Same code, same bound, opposite outcome,
decided entirely by the gap distribution of the corpus.

**This is a go/no-go test that must be run on the real corpus before any index is built**: compute the top-k gap
distribution and compare it to `sqrt(3*pi^2/(d*4^b))`. If the median gap is an order of magnitude below that number,
drop to a larger d or a larger b - or conclude, as here, that the ranking was never recoverable and the dimension went
one step too far. It costs one pass over the corpus and it predicts the result before anything is written.

What remains genuinely unverified: whether a WuBu/GAAD Poincare encoder produces vectors whose gap distribution passes
that test. Nothing in this document can substitute for the measurement, because the whole point of these last numbers is
that the outcome is a property of the corpus and not of the quantiser.

## RETRACTION and corrected result: real vectors PASS; the previous section's conclusion was my bug

The section above concluded that real trained vectors collapse to chance (0.024-0.030 rerank recall) and built a
"gap < quantisation error" criterion on that. **That conclusion is withdrawn. It was a defect in the harness, not a
property of the data.**

The tell was internal inconsistency: `tools/turboquant_realvec.c` measured correlation between exact and quantised
scores of **0.99994 at b=8** while simultaneously reporting 0.004 recall@10. A 0.99994-correlated scoring function
cannot miss the top-10; the two numbers cannot both be real. The bug: after computing the ground-truth top-k the code
overwrote the score array with quantised scores but never reset the index permutation, so the second selection sort
permuted an already-permuted index array. Recall was comparing the truth against a scrambled ordering.

With `idx[i] = i` restored in the quantised loop, the same data, same code, same seed:

    real trained vectors, 3840 rows of v.patch_embd.weight, projected
    d=32                                      d=128
    bits  recall@10  +rerank  bytes/item      bits  recall@10  +rerank  bytes/item
    2     0.506      0.981    8               2     0.572      0.994    32
    3     0.705      0.999    12              3     0.744      1.000    48
    4     0.839      1.000    16              4     0.861      1.000    64
    5     0.903      1.000    20              5     0.923      1.000    80
    6     0.938      1.000    24              6     0.951      1.000    96
    7     0.957      1.000    28              7     0.962      1.000    112
    8     0.973      1.000    32              8     0.978      1.000    128

So the design **passes on real trained vectors**: at d=32, b=3 the scan is 12 bytes per item, quantised-only recall@10
is 0.705, and top-100 plus an exact re-rank is **0.999**. At b=4 it is 1.000 for 16 bytes.

And the earlier structure finding survives, now with the right explanation: those vectors are near-isotropic with
top-32 energy of 10 %, their k-th gaps really are small (median 0.003 at d=32), and that is exactly why quantised-only
recall is 0.5-0.97 rather than ~1.0 - the near-ties are real. What was wrong was the inference that the candidate SET
is also lost. It is not: the re-rank reaches 0.999, which is only possible if the top-100 contains the true neighbours.
This is the same division of labour measured on synthetic data - **bit-width buys the candidate set, the re-rank
converts it into recall** - now confirmed on real weights.

Monotonicity is also restored: recall@10 rises 0.506 -> 0.973 with each bit on d=32, which is the `4^-b` error law
asserting itself. The flat-at-chance table above showed no such trend, which in hindsight was the second tell that the
measurement was broken: a quantiser whose error falls 16x cannot produce a recall curve that does not move.

The lesson, recorded because it is the kind that repeats: a result that is simultaneously surprising and internally
inconsistent is a bug until proven otherwise. The gap-vs-error ratio was computed correctly and is still a legitimate
diagnostic for predicting quantised-ONLY recall, but it does not predict the two-stage pipeline, and it must not be used
as a go/no-go gate - the go/no-go is the rerank number, which is what the harness now reports.

## Verification 5: the ball-vs-tangent question, and why scoring must be inner-product

The design argued for scoring in the tangent space at the origin because the Poincare metric diverges at the boundary.
`tools/tq_ball_vs_tangent.c` measures it. Ground truth is always the exact hyperbolic distance
`d_H(u,v) = arccosh(1 + 2||u-v||^2/((1-||u||^2)(1-||v||^2)))` on the original f32 ball vectors; two pipelines are
compared on the same real 3840x32 data, with radii spread over tangent norms 0.4-3.0 (ball radius 0.197-0.905):

    pipeline   bits  recall@10   +rerank@100   bytes/item
    ball       2     0.275       0.734         8
    tangent    2     0.392       0.764         8
    ball       3     0.392       0.740         12
    tangent    3     0.521       0.744         12
    ball       4     0.521       0.742         16
    tangent    4     0.623       0.744         16

Two findings, one expected and one not:

**Tangent-space quantisation beats ball-space quantisation at every bit-width** - 0.392 vs 0.275 at b=2, 0.521 vs
0.392 at b=3, 0.623 vs 0.521 at b=4. Quantising where the metric is Euclidean is strictly better, which is what the
design predicted, and it converges as bits rise (0.744 vs 0.742 at the rerank stage) because both approach lossless.

**The unexpected one: hyperbolic-distance scoring caps at ~0.74 rerank recall, against 0.999 for inner-product
scoring on the same data.** That gap is not quantisation error - the rerank stage here scores the exact `d_H` on the
original f32 vectors, so a 0.74 means the candidate set itself is wrong 26 % of the time. The reason is structural:
`d_H` is dominated by the radial term, so two items at different radii are far apart regardless of direction, and a
small coordinate error near the boundary (radius 0.905 here) moves the distance much more than it moves a dot product.
Ranking a quantised index by hyperbolic distance throws away the property that makes quantisation survivable.

So the architecture decision is settled by measurement rather than by argument: **encode with Poincare geometry, but
score with an inner product in the tangent space, and never rank a quantised index by hyperbolic distance.** The Poincare
ball does its work in the representation - few dimensions, hierarchy - and the inner product does its work in the
retrieval, where TurboQuant's unbiased estimator and the measured 0.999 rerank recall both live.

Caveat, stated because the radii here were assigned, not measured: the radius spread (0.197-0.905) is synthetic. A real
WuBu/GAAD encoder's radius distribution is the thing that decides how much of the boundary penalty you actually pay, and
that is still the outstanding measurement. If real encoders put many items near radius 1, the 0.74 figure gets worse and
the case for tangent-space scoring gets stronger, not weaker.

## The WuBu encoder itself: two defects, one fixed, and the outstanding question answered as "not yet"

The last open item was the real encoder's radius distribution. Running the shipped code
(`WuBuMath/src/model/wubu_gaad_encoder.c`, the GAAD WuBu nested Poincare encoder) on 400 random 64x64 images:

    ||latent|| : min 0.0022  mean 0.0022  max 0.0022      <- identical for every input everywhere
    cos(lat0,lat1) = 1.00000000    cos(lat0,lat2) = 1.00000000    cos(lat1,lat2) = 1.00000000

The encoder was returning a **constant vector**. The cause is not training and not precision, it is line 181:

    float rgb[3]={0.5f,0.3f,0.7f};     /* and `image` is never read again in this function */

`wubu_gaad_encode` ignored its own `image` argument and pushed a hardcoded colour through the network, so every input
produced a bit-identical latent. `wubu_gaad_encode_full` does sample the image correctly (it indexes
`image[(iy*IMG_DIM+ix)*3]` at each spiral point), so the single-latent path was simply missing the same step.

Fixed in WuBuMath: `wubu_gaad_encode` now averages the same spiral-sampled pixels that `encode_full` reads, falling back
to the whole image when `num_spiral == 0`. Verified after the change:

    cos(lat0,lat1) = 0.624570   cos(lat0,lat2) = 0.720172   cos(lat1,lat3) = 0.715404
    ||latent|| : 0.000334 / 0.000473 / 0.000583 / 0.000864  for inputs scaled 1x/2x/3x/4x

so the output now responds to the input and scales with it.

**But the radius question is still not answered, and now we know why it cannot be from this code as it stands.** After
the fix the untrained encoder produces `||latent|| ~ 5e-4`, i.e. a Poincare radius of ~4e-4 - everything collapses to the
origin, because `wubu_gaad_init` sets random weights and the header contains no weight-loading path at all. An untrained
encoder has no geometry to preserve: at radius 4e-4 the hyperbolic metric is Euclidean to any precision that matters, so
"does Poincare storage work" is untestable on it.

This is the honest state of the last link:

  - the quantiser is verified against its bound, on clustered synthetic data, and on real trained vectors (0.999 rerank
    at 12 B/item);
  - tangent-space scoring is verified to beat ball-space scoring, and hyperbolic-distance scoring is verified to be
    unsafe on quantised data;
  - the encoder that was supposed to supply the vectors returns a constant, which is now fixed, but it is untrained and
    collapses to the origin, so its radius distribution remains unmeasured.

Training it (`WuBuMath/examples/train_gaad.c` exists) is the prerequisite, and it is a larger job than any single step
in this document. Until then the retrieval design is complete and verified but has no encoder that produces usable
geometry - and no amount of quantiser work substitutes for that.

## Verification 6: TurboQuant does NOT beat q8_0 for K-side at equal bits - the previous recommendation was wrong

The previous section proposed putting TurboQuant on the K side because attention scores are inner products and
TurboQuant's estimator is unbiased while q8_0's row-fitted `amax/127` scale is biased. That argument is plausible and
measurably wrong. `wubu_tqvec_score` against `kvq8_quantize_row`+`dequantize` on the same 2000 key/query pairs,
d=128, keys unit-normalised but non-isotropic across dimensions (per-dim scale 0.5-2.0):

    quantiser          bits/elem  IP rmse      IP bias      bytes/row
    turboquant         8          0.001389     -0.000049    128
    turboquant         6          0.002933     0.000123     96
    turboquant         4          0.009111     0.000343     64
    q8_0 (current KV)  8.5        0.000557     -0.000005    136

**q8_0 is 2.5x more accurate than TurboQuant at 8 bits, for 6 % more space.** And its bias is not merely small but
smaller than TurboQuant's (-5e-6 against -4.9e-5). The reason is structural: q8_0 fits a per-row scale to the actual
row amplitude, while TurboQuant's codebook is a function of `(d,b)` only and cannot adapt - data-obliviousness is the
property that makes it cheap to deploy, and it is also the property that loses here. With 256 levels and a row-fitted
scale you are already close enough to lossless that adaptivity wins.

What TurboQuant does buy, visible in the same table: **a 4-bit and 6-bit option that does not exist in the tree today**,
at 64 and 96 bytes per row. Whether 0.0091 IP rmse is tolerable for attention is a separate question and would have to be
measured through the model's logits, not here - but the honest framing is "a cheap lower-bit option", not "a better
quantiser than q8_0".

The meta-lesson is the same one this document keeps producing: the theoretical argument (unbiased estimator for an inner
product) was right about the mechanism and wrong about the conclusion, because it compared bias and ignored that the
competing estimator's bias is negligible at 8 bits. Both claims were measurable; only one was measured.

## Verification 7: attention weights barely move even at 4 bits - the low-bit KV tier is worth building

The question that decides whether an engine integration is worth the build: does quantising K change the attention
distribution? Asked directly - 128 keys (8 of them high-norm "sink" keys, since attention sinks are real), 100 random
queries, `softmax(q.K^T/sqrt(d))`, TV distance and argmax stability:

    quantiser          b/elem    tv_dist     max_dp     top1_moved
    turboquant         4         0.000310    0.003970   0.130
    turboquant         6         0.000104    0.002419   0.040
    turboquant         8         0.000043    0.001960   0.020
    q8_0 (current KV)  8.5       0.000019    0.000202   0.020

At **4 bits the total-variation distance between the true and quantised attention distribution is 3.1e-4** - the
distribution moves by 0.03 %. For a 64-byte row against q8_0's 136, that is 2.1x smaller for a change that is three
orders of magnitude below anything that could flip a token on its own.

Two honest qualifications, both visible in the table:

- **`top1_moved` is 13 % at 4 bits** against 2 % for q8_0. The argmax of attention jumps a tenth of the time. Those
  jumps are between the eight near-tied sink keys, which is exactly the situation where "which sink won" does not matter
  because the sinks carry similar content - but it is a real effect and it is the mechanism by which low-bit KV could
  hurt long-context behaviour even while the distribution looks untouched.
- **TurboQuant at 8 bits is still worse than q8_0** (max_dp 0.00196 against 0.000202), consistent with the previous
  section. The low-bit tier is the reason to build this, not a replacement for the incumbent.

So the engine work is justified: a 4-bit K tier at 64 B/row with 3e-4 attention drift is a real memory win on a 6 GB
card, and the remaining question - do the 12 oracle tokens survive - is now a specific, falsifiable engine test rather
than a hope. The prediction from this table is that they should, with the sink-argmax churn as the most likely failure
mode.

## Verification 8: the 4-bit KV tier is IN THE ENGINE and it works - with two module bugs and one retraction on the way

`IN2_KV_TQ=<bits>` now selects a TurboQuant KV tier alongside the existing `IN2_KV_Q8`. Measured on the real engine,
1024-position cache, same 12-token oracle:

    tier            KV at 1024 ctx   vs f32    answer
    f32 (baseline)  576.0 MB         1.0x      the Ottoman Empire was Constantinople, which was later re   12/12 tokens
    IN2_KV_TQ=4      72.6 MB         7.9x      the Ottoman Empire was Constantinople, which was also known  10/12 tokens
    IN2_KV_TQ=6     108.6 MB         5.3x      the Ottoman Empire was Constantinople, which was also known  10/12 tokens
    IN2_KV_TQ=7     126.6 MB         4.5x      the Ottoman Empire was Constantinople, which was also the    10/12 tokens

The first ten generated tokens are identical to the f32 oracle at 4, 6 and 7 bits; the divergence is at the tail, and the
answer is correct throughout. For **7.9x less KV memory** on a 6 GB card that is the memory win the whole exercise was
for, measured in the engine rather than in a probe.

Three defects were found by doing it, two of them mine:

**1. `wubu_tqvec_decode` silently did nothing above d=256.** It held a `double z[256]` stack buffer with
`if (c->d > 256) return;`. The KV row dim is 1024, so decode returned without writing its output and the engine read
uninitialised scratch. Found because the engine segfaulted before emitting a token. Replaced with a heap buffer.

**2. The codebook grid did not follow the distribution's scale.** `tqv_build_codebook` integrated on a fixed `[-1,1]`
grid, but a unit vector's coordinates occupy `|t| <= ~1/sqrt(d)` - at d=1024 that is `|t| <= 0.1`, so 8192 grid points
put roughly 800 samples underneath 256 centroids. The grid is now built in `u = t*sqrt(d)` where the mass actually is.
Stretching it past the support initially produced `pow(negative, fractional)` = NaN, which is how the second half of
this bug was found - the d=32 self-test went to NaN while d=1024 looked perfect, the opposite of the usual direction.

**3. RETRACTION: the block-diagonal 8x8 rotation is not good enough for real data.** The earlier "Verification 3"
concluded block-diagonal was identical to full rotation within noise, and it was - **on random unit vectors**, which are
already isotropic. Real K rows are not: they carry a large shared component (norm 24.87 over 1024 dims). A block
rotation only mixes inside each 8-wide block, so such a row stays concentrated, the rotated coordinates are not i.i.d.
Beta, and the codebook is fitted to the wrong law. Measured on an actual K row:

    block-diagonal 8x8   max round-trip err 0.603  (2.4 % of the row norm)
    signs + Hadamard     max round-trip err 0.039  (0.16 %)          15x better

and with the block version the engine produced fluent nonsense - "Kazakhstan, Astana, is a city that has" for a question
about the Ottoman capital. The module now uses `R = (1/sqrt(d)) H S`: random signs then a fast Walsh-Hadamard
transform, orthogonal, `O(d log d)`, still regenerable from one seed, and it mixes every coordinate with every other.
Re-verified after the change: d=32 still passes the distortion bound at every bit-width (0.3522/0.1115/0.0324/0.0090) and
d=1024 round-trip is monotone over 300 structured rows (worst 0.0108/0.0040/0.0015 for b=4/6/8).

The lesson is the same shape as the earlier harness bug: a measurement that agrees with my expectation, on data I
generated to suit it, is not evidence about real data. Both times the real input disagreed.

**Open defect, not fixed:** `IN2_KV_TQ=5` and `=8` produce wrong output in the engine ("Kazakhstan, Astana...") while the
module round-trip at those widths is correct and monotone - so the fault is in the wiring, not the quantiser. Both are
now clamped to a verified width (5->6, 8->7) with a one-shot warning on stderr rather than being allowed to emit wrong
tokens. Refusing outright was tried first and crashed, because the KV init runs once per visit and the failure path
freed state that later visits still used.

## Verification 9: RETRACTION - there is no memory-safety defect; the tier works, and the bit-width floor is measured

The previous section claimed "an unresolved memory-safety defect" because the set of bit-widths that produced correct
output differed between builds. **That was wrong.** The working set moved because the codebook changed between those
builds - I edited `tqv_build_codebook` twice in that window - not because of layout. Attributing my own code changes to
nondeterminism was the error, and it sent me looking for corruption that was never there.

`IN2_TQ_VERIFY` settles it: every stored row is mirrored in f32 and each read is compared against that shadow. On the
rows that actually matter (norm ~24.87):

    KV_K visit=0 pos=0 el=0  max read-vs-shadow error 0.30697  (1.23 %)
    KV_V visit=0 pos=0 el=0  max read-vs-shadow error 0.27535  (1.11 %)
    KV_K visit=0 pos=1 el=1  max read-vs-shadow error 0.39192  (1.58 %)
    KV_V visit=0 pos=1 el=1  max read-vs-shadow error 0.32900  (1.32 %)

at b=4 - which is exactly what the standalone module predicts (b=4 on a unit vector: 0.0106-0.0147, times 24.87 =
0.26-0.37). Reads are correct; the error is plain quantisation error. No slot mismatch, no corruption.

What that also explains is the real constraint: **this model's greedy output changes when a stored row carries ~1.6 %
error, and does not at ~0.5 %.** Measured end to end, same prompt, same binary:

    IN2_KV_TQ=3,4   clamped to 6   "the Ottoman Empire was Constantinople, which was also kn"   correct
    IN2_KV_TQ=6     108.6 MB       "the Ottoman Empire was Constantinople, which was also kn"   correct, 10/12 tokens
    IN2_KV_TQ=7     126.6 MB       "the Ottoman Empire was Constantinople, which was also th"   correct
    IN2_KV_TQ=8     126.6 MB       "the Ottoman Empire was Constantinople, which was also kn"   correct
    f32             576.0 MB       "the Ottoman Empire was Constantinople, which was later re"  12/12 tokens

So the tier ships as **6 bits and above**: 108.6 MB against 576.0 MB is **5.3x less KV memory** with the answer correct
and ten consecutive tokens identical to the f32 oracle. Below 6 bits the row error crosses the model's sensitivity and
the answer goes wrong, so those widths clamp to 6 with a one-shot warning rather than silently producing a confident
wrong answer.

The general principle this gives, which is the transferable part: **the bit-width floor is a property of the model's
sensitivity to row error, not of the quantiser's distortion bound.** The bound says b=4 is near-optimal for its bit
budget; the model says 1.6 % is too much. Both are true, and only the end-to-end run can tell you where the line is.

## Context capacity: 256k is now real, and 1M has an arithmetic answer

With the 6-bit KV tier in place, the engine's own allocation print at two long-context settings, both running to
completion:

    ctx       f32 KV      6-bit KV     + model 7.90   result
    131072    8704.0 MB   1640.5 MB     9.54 GB       runs
    262144   16896.0 MB   3184.5 MB    11.08 GB       runs

So **262,144 tokens of context fits in a 16 GB box** where f32 needed 16.9 GB of KV alone before the model was loaded.
That is the single largest capacity gain measured in this project, and it costs token 10 of 12 versus the f32 oracle on
the short prompt (b=6 is the width where row error stays under the model's sensitivity).

### Why 1M is not a bigger version of the same thing

Full visits are 8 of the 72 and they index the *whole* context, so their KV is `8 x ctx x 1024 x 2` elements:

    ctx        f32 GB    q8 GB    q6 GB    q6 + model   q6 GB read per token
    262144      17.72      4.71     3.32     11.22        3.32
    524288      34.90      9.27     6.54     14.44        6.54
    1048576     69.26     18.40    12.99     20.89       12.99

At 1M the 6-bit tier alone is 12.99 GB, and with the 7.90 GB model that is **20.89 GB against a 16 GB machine**. Budgeting
1.5 GB of slack, the KV allowance is about 6.6 GB, which at 1M and 8 full visits permits

    6.6e9 / (8 x 1e6 x 1024 x 2) = 0.40 bytes per element  =  3.2 bits

**3.2 bits is below the 6-bit floor that this model was just measured to need**, and below it the answer goes wrong
rather than degrading gracefully. So 1M context is not reachable by tuning the quantiser: the requirement is
architecturally out of reach by a factor of two in bits, and bits are the one thing already at the limit.

### The two ways it becomes reachable, both fidelity trades

Memory scales linearly in the number of full visits, so:

  - **4 full visits instead of 8** -> 6.50 GB at 1M, fits; 2 full visits -> 3.25 GB, comfortable. The other 6 visits
    keep their 1024-token sliding window. Cost: less global mixing, unknown quality effect.
  - **pool the far history** in the full visits (merge adjacent key/value pairs as positions age), which makes their
    footprint O(window + n/group) rather than O(n): at 4:1 pooling, 1M costs the same as 250k, 3.25 GB, constant.

Neither can be chosen from arithmetic. Both need a long-context fidelity measurement on this model - a needle/recall
probe at 128k and 256k to establish that the tier works at length at all, then the same probe to price the trade. That is
the next piece of work, and it is a measurement task rather than a coding one.

### Tandem GPU, honestly priced

Per-token decode is bandwidth-bound, not FLOP-bound: at 6 bits, 1M costs 12.99 GB of KV traffic per step.

    CPU 8 threads (measured 5.65 GB/s)  -> 2.30 s/token   0.44 tok/s
    GPU resident (measured 59.05 GB/s)  -> 0.22 s/token   4.55 tok/s
    PCIe 4.0 x8 streaming (~12-13 GB/s) -> ~1.0-1.1 s    ~0.9 tok/s

The GPU figure is only reachable if the KV is resident, and 12.99 GB does not fit in 6 GB of VRAM, so the real tandem
number at 1M is the PCIe row: **about 1 s/token, an order of magnitude better than CPU-only and an order worse than
resident.** That is the case for the tiered design: keep the hot window resident on the GPU and stream only the cold
history, which is exactly the strata/tiering pattern, and it makes the PCIe row the floor rather than the ceiling.
None of the three rows above has been measured end to end at 1M - they are bandwidth arithmetic on measured throughput,
which is why the tier work has to come before any claim about 1M speed.

## Verification 10: the wall is prefill throughput, not KV memory - and the 6-bit tier currently costs 4.7x of it

`IN2_TEXT_FILE` now lets the prompt come from a file (argv cannot carry a long-context prompt on Windows), and one
token per byte caps the encode buffer. That made prefill measurable for the first time, and the numbers reframe the 1M
question completely.

Same prompt, one token generated, wall clock:

    prompt    f32 KV     6-bit KV   TQ penalty
    ~11 tok   5.16 s    14.94 s     2.9x
    ~22 tok   9.88 s    46.88 s     4.7x

Two findings, both unwelcome:

**1. Prefill costs ~0.45 s/token at f32 and ~2.1 s/token at 6 bits, with no batching.** So filling a 262,144-token
context would take roughly 262144 x 2.1 s = **6.4 days** at 6 bits before the model could answer anything. Filling a 1M
context is not a slow operation, it is an impossible one. Memory was the problem I expected; throughput is the problem
that actually blocks the goal, and it blocks 256k just as hard as 1M.

**2. TurboQuant's decode-on-read is the cost.** The 6-bit tier is 2.9x slower at 11 positions and 4.7x slower at 22, so
the penalty grows with position - exactly what per-row decode costs, since every read runs a Hadamard unrotate and a
nearest-centroid lookup before the dot product. The memory win of Verification 9 is real; the speed cost is 4.7x and
rising, which makes the tier a memory-for-speed trade that is only worth paying at contexts where the alternative is not
fitting at all.

That also invalidates the tandem-GPU arithmetic in the previous section. I priced 1M decode at 12.99 GB/step against
measured bandwidth and got ~1 s/token. But at 22 positions the engine already spends 2.1 s per token on 6-bit data
while touching ~1 MB of KV per step - three orders of magnitude under the bandwidth figure. **The engine is not
bandwidth-bound; it is per-element-bound**, because every KV element is decoded before it is used. Bandwidth arithmetic
on this engine answers the wrong question.

### Consequently, in priority order

1. **Batched prefill.** The model is currently run one token per forward pass. Chunking the prompt (matmuls over a
   token block instead of one row) is the only change that makes long context reachable at all; without it, no amount
   of memory or quantisation matters. This is now the top item and it is worth 100x+ on prefill, not percent.
2. **Score in the compressed domain.** Feed the query into the rotated basis once per visit and dot it against packed
   codes using the centroid table - never decode a row to use it. That removes the 4.7x penalty and is what the paper's
   inner-product variant is for; I implemented only the MSE variant, which forces decode.
3. Only then re-price 1M, where the remaining question is the fidelity trade (fewer full visits vs pooled history) from
   the previous section.

Measuring prefill changes the plan. The previous section's conclusion - "1M is short by a factor of two in bits" - is
still arithmetically true, but it is not the binding constraint; a 1M context that fits in RAM and takes six days to
fill is not a feature.

## Verification 11: three retractions - my prefill figures were inflated 4.8x, batching buys nothing, and AVX2 buys nothing

The previous section reported prefill at "0.45 s/token at f32 and 2.1 s/token at 6 bits" and concluded that a 262k
context would take 6.4 days to fill. **All three numbers were wrong**, for one reason: I derived tokens-per-prompt from
a character count instead of reading the value the engine prints. It prints `prefix N` in every step line, and the
prompts were 49 and 97 tokens, not the ~11 and ~22 I divided by.

The engine's own figures:

    prompt        prefix   wall      s/token   vs decode (0.17)
    s4  f32         49      4.484 s   0.0915    1.9x faster
    s4  6-bit       49     14.234 s   0.290     1.7x slower
    s8  f32         97      9.093 s   0.0937    1.8x faster
    s8  6-bit       97     46.141 s   0.4757    2.8x slower

So prefill is **0.093 s/token (10.7 tok/s) at f32**, not 0.45 - a 4.8x overstatement. And 262,144 tokens at the 6-bit
rate is **~35 hours**, not 6.4 days. The conclusion that throughput rather than memory blocks long context survives,
since 35 hours is still not a feature, but the magnitude was wrong by nearly 5x and the corrected figure is what any
planning should use.

**Retraction 2: batched prefill buys nothing on this machine.** I claimed chunking the prompt into token blocks was worth
"100x+". A probe with identical weights and identical total MACs, two loop orders, 64 tokens through 4096x4096 f32:

    (a) token-outer (what the engine does)   0.65 s   1.66 GMAC/s
    (b) row-outer   (each weight read once)  0.64 s   1.67 GMAC/s
    speedup 1.01x

Row-outer moves 64 MB of weights against token-outer's 4.3 GB and takes the same time, which proves the kernel is
**ALU-bound, not weight-bound**. Amortising weight reads cannot help when the arithmetic is the limit. Planning a
refactor on the assumption that it would was a mistake, and the probe cost two minutes against the refactor's hours.

**Retraction 3: AVX2 does not help the engine either.** The same probe with `-mavx2 -mfma -ffast-math` goes 1.66 ->
10.81 GMAC/s, a 6.5x gain, so the capability is there. But the engine built with `-mavx2 -mfma` produces **bit-identical
oracle tokens and identical prefill wall time** (10064 ms vs 9875 ms, inside noise). The engine's cost is therefore not
in vectorisable f32 arithmetic at all - it is in the IQ2 dequantisation and the expert gather, which the flag does not
touch. Adding the flag is safe (parity held) but worthless on its own; keep it for the kernels that do guard on
`__AVX2__`, and do not expect it to move the number.

What the phase dump does say, at 97 tokens in 9.09 s: MoE layers run ~0.075-0.089 s each with `[expert-major]`, and the
HEAD is 0.089 s for 201024 x 2304. Twenty-four MoE layers at ~0.08 s is ~1.9 s of the 9.09 s, so MoE is a fifth of
prefill and the dense matvecs are the rest - both dominated by dequantising IQ2 weights element by element.

The next optimisation target is therefore **dequantisation throughput**, not loop order, not SIMD flags, and not
batching: the engine converts a 7.90 GB IQ2_XXS weight set to f32 to use it, at 1.17 G w/s. Every one of the three
things I proposed last message attacks the wrong layer of the problem.

## Where the time actually is: MoE, and the IQ2 dot kernel underneath everything

Phase ledger for a 49-token prefill (IN2_TIME=1, aggregated over the step): **MoE is 1.99 s of the 4.48 s, 44 %**, at
30+ invocations averaging 40-53 ms each with `[expert-major]`. HEAD is 0.089 s for 201024 x 2304. The rest is the dense
matvecs. So prefill time is one thing seen from two angles: many small IQ2 weight sweeps.

At 8 experts of 2304x1152 twice per layer, one MoE call is ~85 MMAC in ~40 ms = **2.1 GMAC/s**, which is the same rate
the dense path reports (1.17 G w/s). Everything in the engine runs at that rate, which is why no single phase stands out
as "the slow one" - they are all the same kernel.

Four hypotheses are now eliminated by measurement, not argument:

| hypothesis | test | result |
|---|---|---|
| weight-read bound, so batching wins | 64 tokens, token-outer vs row-outer | **1.01x** - ALU-bound |
| not vectorised, so `-mavx2` wins | engine built `-mavx2 -mfma` | bit-identical tokens, **no time change** |
| kernels scalar, AVX2 unused | `grep IQ_AVX2` | `#define IQ_AVX2 __attribute__((target("avx2,fma,f16c")))` - **already vectorised** |
| MoE not threaded across experts | `IN2_THREADS` 1/4/8 | 12.26 s -> 6.10 s -> 5.14 s - **already threaded (2.4x)** |

The fourth row is worth reading carefully: 8 threads on a 6C/12T Ryzen gives 2.4x, not 8x, on work that is ALU-bound.
That sublinearity is itself evidence for contention inside the kernel rather than for spare capacity.

### The one remaining lever, and its honest size

The engine is at **~2 GMAC/s** with AVX2, threading, and a fused dequant-and-dot already in place (`DQ to_f32 calls 5492`
against `borrow hit 7128` per step means weights are being reused from a dequant cache rather than re-expanded per token,
so the dequantisation is not the repeated cost it looked like - the packed dot is). A well-tuned IQ2_XXS dot on Zen4
reaches roughly an order of magnitude more, and the reason ours does not is that IQ2 is a *lookup* format: every
8-element block indexes a 256-entry grid by 19 bits and applies per-element signs, so the inner loop is repeated gather
and sign-mask work that generic vectorisation cannot turn into a clean FMA chain. Closing that gap is a dedicated kernel
project - write the block dot so the grid lookup is hoisted and batched (maddubs against the grid table, ggml's
`iq2_xxs` approach), then verify it block-for-block against the current one before it goes anywhere near the forward.

That is the decision taken here: **the next work item is the IQ2 dot kernel, not more plumbing.** Every cheaper
hypothesis has now been measured and closed, and the four retractions in this document are all cases where I assumed
instead of measuring. The order of operations for the kernel is fixed in advance: extract one real IQ2 tensor and its
activations, time the current `wubu_iq_rows` against a standalone replacement, and only integrate if the replacement is
both faster and bit-identical over that tensor.

## The tandem GPU is currently doing zero work on this model

Measured, `IN2_GPU_DENSE=1 IN2_GPU_TRACE=1`, one 49-token step:

    NT-GPU refused ty=14 rc=-99 actb=2628 wb=5806080  (blk.0.attn_q.weight)
    NT-GPU refused ty=8  rc=-99 actb=2448 wb=2506752  (blk.0.attn_k.weight)
    NT-GPU refused ty=13 rc=-99 actb=2628 wb=4866048  (blk.3.attn_q.weight)
    NT-GPU refused ty=12 rc=-99 actb=2628 wb=3981312  (blk.18.attn_q.weight)
    NT-GPU refused ty=18 rc=-99 actb=2628 wb=2709504  (blk.20.attn_q.weight)

five refusals, **zero accepted calls**, and MoE never attempts the GPU at all. The DLL loads (`wubu_cuda.dll`, repo
root) and the path engages, but every tensor type this model uses for its dense projections - Q6_K, Q8_0, Q5_K, Q4_K,
IQ3_XXS - is outside what the kernel accepts. So on Inkling-2-Mini the GPU has been idle for the entire project, while
the CPU does 100 % of the work at ~2 GMAC/s.

That reframes "tandem GPU acceleration" from an optimisation into a **capability gap**: there is no tandem yet. The
measured split is:

    phase              share of prefill      where it runs
    MoE (24 x ~40 ms)      44 %              CPU only, never offered to the GPU
    dense matvecs          rest              CPU only, every type refused
    HEAD (0.089 s)         ~2 %              CPU

and the foundation underneath both is the IQ2 dot at ~2.1 GMAC/s.

### A wrong number, discarded

I wrote a GGUF tensor-size audit to quantify the expert weights and it reported **66.56 GB for an 8.49 GB file** -
misaligned after the KV section, so the tensor table it walked was garbage. Not reported as a finding and not used.
What the file and the engine do say:

    file                         8,486,236,352 bytes  (8.49 GB)
    engine working set           4406 MB
    d=2304  layers=48  core=[6,18)  loops=3  n_expert=128  used=8  d_ff_expert=1152

With 128 experts of 3 x (2304 x 1152) per MoE layer, the experts are by construction the overwhelming majority of the
model's bytes even at IQ2/IQ1 widths, and the engine's own 4406 MB working set against an 8.49 GB file says most of it
is not resident. That is consistent with MoE being both the largest thing in the model and the slowest thing per byte,
but the exact split needs a parser that is checked against the file size before it is trusted.

### What this changes about the plan

The IQ2 dot kernel is still the foundation, but it is not the shortest path to the goal. **Making the GPU accept the
types this model actually uses** is worth more than a 2x kernel, because it converts an idle 6 GB device into a
contributor and moves the work that is 100 % serial today onto a device that measured 59 GB/s. Concretely:

1. Enumerate the type ids the CUDA kernel accepts versus the type ids present in the model, per tensor role. Today that
   intersection is empty for dense projections and MoE is not offered at all.
2. Add MoE expert matvecs to the GPU path (`wubu_iq_rows` already takes a row range, so an expert is a sub-range - the
   wiring is smaller than it looks), with the 8 active experts per layer staged per token.
3. Only then re-measure, because every number in this document above about GPU speed was measured on kernels this model
   never reaches.

## The type-coverage enumeration: 32 % of the experts are offloadable today, and the biggest share is two simple formats

Tensor sizes derived from the GGUF tensor-table offsets rather than from a block-size table, which is self-validating:
the sizes sum to **8.474 GB against a data area of 8.474 GB** (file 8.486 GB), and the table end offset lands on
11978661, the same number the engine prints. The earlier 66.56 GB figure came from my block-size table and was wrong.

    role            tensors      GB      share
    expert               88    7.251     85.6 %
    other (dense, embd) 646    1.223     14.4 %

    expert bytes by type                       GB     share of experts   GPU kernel today
    40  NVFP4                                1.624        22.4 %        no
    42  Q2_0                                 1.338        18.5 %        no
    16  IQ2_XXS                               1.226        16.9 %        yes
    17  IQ2_XS                                1.178        16.2 %        no
    22  IQ2_S                                 1.088        15.0 %        yes
    19  IQ1_S                                 0.796        11.0 %        no

The kernel accepts `{8, 16, 22, 23}`, so **31.9 % of expert bytes are of a type it already handles** (IQ2_XXS + IQ2_S)
and 68.1 % are not. MoE is also never offered to the GPU at all, so today the working share is zero, not 32 %.

Expert layers: 24 (`blk.0..blk.23`), 302.14 MB each, 8 of 128 experts active per token, so

    18.88 MB of expert weights per layer  x  24 layers  =  453 MB touched per token

of which ~185 MB is NVFP4/Q2_0, ~145 MB is IQ2_XXS/IQ2_S, ~124 MB is IQ2_XS/IQ1_S.

### The decision this supports

The two largest expert formats are **NVFP4 and Q2_0 - 40.9 % of all expert bytes - and they are the two easiest formats
on the list**: NVFP4 is 4-bit values with one scale per 64 elements, Q2_0 is 2-bit values with an fp16 scale per 64
elements. Neither needs the 256-entry grid lookup and sign-masking that makes IQ2 slow on the CPU, so both vectorise
cleanly on the GPU. By contrast IQ2_XXS/IQ2_S are already supported and are only 32 % of bytes.

So the highest-leverage GPU work is not "port the IQ2 kernel" and not "wire up MoE for the types it already has". It is:

1. **NVFP4 and Q2_0 matvec kernels on the GPU** - 41 % of expert bytes, trivial unpack, and they are exactly the formats
   the CPU is worst at (2.1 GMAC/s scalar with per-element dequant).
2. **Offer MoE expert row ranges to the GPU** - `wubu_iq_rows` already takes `r0/r1`, so an expert is a sub-range; the
   wiring is a dispatch decision, not a new interface.
3. Keep the IQ2 path on the CPU initially; it is 45 % of expert bytes but the format that is hardest to vectorise, and
   the CPU already has a tuned integer path for it.

That ordering is driven by measured bytes and measured formats, and it puts the GPU on the 41 % that is cheapest to
implement before the 45 % that is hardest.

## Measured per-format throughput: the 8x gap is in two formats that are 35 % of the work

`IN2_TIMETY=1` on a 49-token prefill gives per-type cost accounting straight from the matvec paths:

    type                     calls     time    share     rate
    42  Q2_0                  1618   2.049 s   22.0 %   16.76 GOPS
    16  IQ2_XXS                779   1.626 s   17.5 %   21.76 GOPS
    17  IQ2_XS                 712   1.471 s   15.8 %   22.63 GOPS
    19  IQ1_S                  481   1.342 s   14.4 %   12.40 GOPS
    40  NVFP4                  677   1.230 s   13.2 %   12.69 GOPS
    22  IQ2_S                  323   0.853 s    9.2 %   17.08 GOPS
    8   Q8_0                  2849   0.448 s    4.8 %   94.54 GOPS
    14  Q6_K                   686   0.116 s    1.2 %  104.77 GOPS
    13  Q5_K                   252   0.043 s    0.5 %  100.48 GOPS
    12  Q4_K                   224   0.031 s    0.3 %  104.88 GOPS
    23  IQ4_XS                  25   0.086 s    0.9 %    5.39 GOPS
    18  IQ3_XXS                 21   0.009 s    0.1 %  115.61 GOPS

Two facts matter here and they point the same way.

**1. The K-quants run at ~105 GOPS and the low-bit formats run at 12-23 GOPS: an 8x gap.** Q4_K, Q5_K, Q6_K and Q8_0
are a small fraction of the time precisely because they are fast, not because there is little of them - Q8_0 makes 2849
calls, more than any other type, and still costs only 4.8 %.

**2. NVFP4 + Q2_0 are 35.2 % of the time at 12.69 and 16.76 GOPS**, and they are plain formats - no grid lookup, no sign
masks. That is the largest single block of time in the engine and the least defensible one.

### Why they are slow, read off the source

`dot_row_q2_0_avx2` (wubu_dequant_fp4.c:163) walks a 64-element block in **eight separate groups of 8**:

    broadcast one 16-bit word -> _mm256_srlv_epi32 by {0,2,...,14} -> and 3 -> sub 1 -> cvtepi32_ps -> fmadd_ps(x)

that is a variable shift, an add, a sub, an int-to-float convert and an FMA per 8 elements, **plus a horizontal
reduction per block** (`_mm_add_ps` + two `_mm_hadd_ps` + scalar accumulate, once per 64 elements). Compare the K-quants,
which use integer `_mm256_maddubs_epi16` / `_mm256_madd_epi16` with int32 accumulation: no per-element float conversion,
no per-block reduction, 32 elements retired per instruction.

The values themselves are exactly `{-1,0,1,2}`, so the convert-and-subtract is pure overhead - a 4-entry
`_mm256_permutevar8x32_ps` table gives the same floats bit-exactly, and the reduction can be hoisted out of the block
loop into one final horizontal sum.

So the concrete next change, in order:

1. Rewrite `dot_row_q2_0_avx2` and the NVFP4 equivalent in the K-quant shape: keep a persistent vector accumulator across
   the whole row, replace `cvtepi32_ps`+`sub` with the permutation table, and reduce once at the end.
2. A/B it behind an env switch and gate on both **rate** (`IN2_TIMETY`) and **parity** (the 12-token oracle), because
   changing accumulation order changes rounding and parity is the invariant.
3. Only if the integer-maddubs restructure is needed after that, quantise the activations once per matvec as the Q8_0
   path already does - that is the step that would close a full 6x rather than a fraction of it.

Everything above is measured on this box, in this engine, on the model's real tensor layout.

## Null result: the Q2_0 micro-optimisations buy 0.2 %, so the cost is elements-per-instruction

Two changes were implemented and A/B'd against the reference kernel behind `IN2_Q20_V2`:

  - the `cvtepi32_ps` + `sub 1` pair replaced by a 4-entry `_mm256_permutevar8x32_ps` table (bit-identical, since
    {-1,0,1,2} are all exactly representable), and
  - the per-block horizontal reduction removed, each block's sum folded into a persistent accumulator with one FMA and
    reduced once at the end of the row instead of twice per 64 elements.

Measured, `IN2_TIMETY=1`, same 49-token prefill:

    variant              Q2_0 time     Q2_0 rate      total
    reference            2.055 s       16.71 GOPS     4.5 s
    v2                   2.051 s       16.74 GOPS     4.5 s

**0.2 %, which is noise.** Parity was preserved exactly (12/12 oracle tokens), so the change was correct - it was simply
aimed at the wrong instruction. Reverted rather than left in the tree behind a flag, because dead complexity that
measures flat is worse than no change at all.

What this rules out: the per-block reduction and the int-to-float conversion are not the cost. Both are now known to be
free, which leaves the expansion itself - `srlv` producing **8 elements per five-uop sequence**, against the K-quants'
`maddubs`/`madd` retiring **32 elements per instruction**. That 4x difference in elements-per-instruction matches the
measured 6x rate gap (16.7 vs ~105 GOPS) better than anything else examined.

So the only change with a plausible 4-6x on Q2_0 and NVFP4 is structural: quantise the activations to int8 once per
matvec, exactly as the Q8_0 path already does, and run the 2-bit/4-bit fields through `_mm256_maddubs_epi16` with int32
accumulation. That is a rewrite of the dot, not a tuning of it, and it is the next piece of work - with the measured
target being Q2_0 from 16.7 to somewhere near 90 GOPS, which would remove roughly a third of prefill time.

Recording the null result matters as much as the positive ones: it is the fourth cheap hypothesis this document has
tested and closed, and each one cost minutes where the alternative was hours of a refactor built on a guess.

## The int8/maddubs probe: hoisting the activation quantisation is the whole difference

`tools/q2_dot_lab.c` runs the reference float-activation Q2_0 dot against an int8-activation + `maddubs` version on
identical synthetic blocks (18-byte layout, 64 fields per block, 512 rows of 2304 in-features):

    float-activation   12.43 GMAC/s
    int8 + maddubs      4.69 GMAC/s     0.38x

The int8 version came out **2.6x slower**, and the reason is structural rather than a dead end: that probe quantises the
activations **inside the row loop**, so the same 2304 values are re-quantised 512 times. Doing that per row costs more
than the dot it feeds. This is exactly why the engine's Q8_0 path reaches 94.54 GOPS while Q2_0 sits at 16.71 - the
q8_0 path quantises the activation vector **once per matvec** into `in2_q8_0_blk` and then sweeps many rows against
that buffer, whereas type 40/42 never reach that machinery at all.

The measured ordering is therefore:

    94.54 GOPS   type 8   q8_0 activations, quantised once per matvec, integer maddubs
    12-23 GOPS   types 16/17/19/22/40/42   float activations, per-row expansion, no shared quantisation
     4.69 GMAC/s int8 without hoisting   - worse than either

So the fix is not "write an int8 kernel". It is **make types 40 and 42 live in the same place type 8 lives**: implement
them as rows kernels in `wubu_iq_c11.c` taking the already-quantised `in2_q8_0_blk` activation buffer, so they inherit
the once-per-matvec quantisation and the maddubs sweep that Q8_0 already proves runs at ~95 GOPS. Today
`wubu_iq_supported(42)` is false, which is why the `q8_0_act` branch in `wubu_inkling2_fwd.c:386` - which already
handles `ty == 40 || ty == 42` - is never taken for them, and a second, float-activation implementation in
`wubu_dequant_fp4.c` does the work instead at a quarter of the rate.

That is a well-scoped change with a reference implementation sitting next to it in the same file, and it is the next
piece of work. The probe also stands as a warning about the measurement itself: had I only measured the int8 variant
against nothing, I would have concluded that int8 activations were slower than floats, which is the opposite of what the
Q8_0 number in the same engine shows.

## RETRACTION: the Q2_0 rewrite ceiling is 1.79x, not 4-6x - the unpack is over half the cost

The previous section promised a maddubs rewrite worth "4-6x" on Q2_0 and NVFP4 and "roughly a third of prefill time".
That was inferred from the rate gap alone (16.7 vs ~105 GOPS) and is wrong. `tools/q2_unpack_lab.c` measures the
floor directly: the same row loop with the multiply and FMA removed, keeping only the 2-bit expansion:

    full Q2_0 dot    12.75 GMAC/s
    unpack only     56 % of that time
    => best case if the multiply and FMA were completely free: 1.79x

The K-quants hit 105 GOPS because their weights are already bytes - Q4_K is 4-bit and unpacks with one shift and mask
per 32 values, Q8_0 needs no unpack at all. Q2_0 is 2-bit, so producing one lane per element costs a variable shift,
an and, a subtract and a pack before any arithmetic can happen, and that unpack is 56 % of the dot. **A perfect
multiply cannot beat 1.79x**, whatever instruction it uses.

So the realistic prize on Q2_0 is ~1.8x on 22 % of prefill - about 10 % overall, and less than that in wall time once
threading contention is included. NVFP4 (13.2 %, 12.69 GOPS) would need a separate float-LUT path for a similar
fraction. Both are multi-hour SIMD rewrites that change accumulation order, so each needs a parity gate, and together
they are worth roughly 15-20 % - not a third.

### Revised priority

Given that, the ordering from the type-coverage section stands but its justification changes: moving NVFP4 and Q2_0 to
the GPU is worth more than rewriting them on the CPU, because on the GPU the unpack problem disappears entirely (a
4-bit or 2-bit value is one index into a constant table held in shared memory, and the matrix-vector multiply can be
done at one thread per output row with no per-element expansion at all). On the CPU the unpack is an irreducible 56 %
of the work; on the GPU it is a table lookup.

So:

1. **GPU kernels for NVFP4 (22.4 % of expert bytes) and Q2_0 (18.5 %)** - the two largest formats, the two simplest,
   and the two where the CPU's structural disadvantage vanishes.
2. Route MoE expert row ranges to them (`wubu_iq_rows` already takes r0/r1).
3. Leave the CPU Q2_0 kernel alone unless a 1.8x on 22 % is the only thing left to do.

The general lesson, which has now cost me five corrections in this document: **a rate gap between two formats is not a
speedup opportunity until you measure what the slow one is actually spending its time on.** Q4_K is 6x faster than
Q2_0 not because its arithmetic is better but because its *format* stores values one per nibble instead of four per
byte. Chasing the arithmetic was chasing the wrong half of the gap.

## RETRACTION: the GPU was not idle - I was benchmarking a stale DLL

The previous section reported the GPU doing **zero work** (5 refusals, 0 accepted calls) and concluded that "there is no
tandem yet". That measurement was invalid. `wubu_cuda.dll` on disk was dated Nov 24 20:04 while
`src/wubu_cuda_quant.cu` was dated Nov 25 23:31: **the DLL was 27 hours older than its own source**, so every type
added to the quantiser in that window was absent from the library the engine loaded. The refusals I read as "the kernel
does not accept these types" were "this DLL has never heard of these types".

Rebuilding it (`mingw32-make -f Makefile.win wubu_cuda.dll`, nvcc 13.5, sm_89) produces **706,048 bytes against the
stale 657,408** - 48 KB of kernels that had never been executed. With the fresh DLL:

    NT-GPU ok ty=8 (blk.0.attn_k.weight)
    GPU tier: engaged (blk.0.attn_k.weight)

So the GPU does engage, and the parity claim survives it: 12/12 oracle tokens identical with `IN2_GPU_DENSE=1`.

### But engaging is not helping

Same prompt, 97-token prefill, wall clock:

    GPU off   9895 ms
    GPU on   10515 ms      -6.2 %

One matvec type (8, Q8_0) is accepted, on one tensor; everything else is still refused (types 12, 13, 14, 18), and the
engagement overhead costs more than the work it takes over. So the conclusion that the GPU is not contributing stands -
but the reason is coverage and overhead, not a library that refuses everything, and the specific claim "zero accepted
calls" was wrong.

### The recurring trap

This is the **second time** in this project that a stale `wubu_cuda.dll` produced a wrong conclusion about the GPU. The
DLL is not rebuilt by `tools/build_gen_fast.sh` (which builds only the C11 engine), so any engine-only rebuild leaves
the DLL behind, and the engine fails *quietly* - it does not report "DLL older than source", it reports per-type
refusals that read exactly like missing kernel support.

Rule from here on: **before any statement about GPU behaviour, compare the DLL's mtime against `src/wubu_cuda_*.cu`,
and rebuild if it is older.** The check is one `ls`, and skipping it has now cost two wrong findings.

## The tandem GPU is real, and it is worth 7-21 % on decode - not on prefill

Everything above measured prefill, where it turns out the GPU cannot engage at all. The dispatch gate in
`wubu_inkling2_fwd.c` requires **`nn == 1`** - one activation row - so a prompt batch falls through to the CPU by
construction. My "GPU on is 6.2 % slower" reading was taken on a 97-token prefill, i.e. on the one phase where the device
is guaranteed to do nothing but pay its initialisation. Measuring decode instead:

    32 tokens   GPU off  7765 ms  (4.43 tok/s)
                GPU on   6650 ms  (5.36 tok/s)     +21 %, ids identical across all 32

    24 tokens, interleaved A/B on an idle box
                off 5510 / 5676 ms      on 5140 / 5112 ms      ~7-10 %

So the device path works, preserves the token sequence exactly, and is worth roughly 7-21 % depending on length. That is
the first measured tandem win in this project, and it came from testing the right phase rather than the convenient one.

### Why it is only 5 % of the work

`IN2_TIMETY=1` on decode, per type:

    type                 share    rate
    16  IQ2_XXS          18.0 %   10.19 GOPS
    42  Q2_0             17.5 %   10.20 GOPS
    17  IQ2_XS           15.4 %   11.26 GOPS
    40  NVFP4            13.3 %    6.11 GOPS
    22  IQ2_S            11.1 %    6.81 GOPS
    19  IQ1_S            10.9 %    7.96 GOPS
    23  IQ4_XS            6.8 %    5.79 GOPS
     8  Q8_0              5.0 %   43.95 GOPS

The GPU accepts **only type 8**, and the trace confirms it (`NT-GPU ok ty=8`; types 12, 13, 14, 18 refuse with rc=-1).
The six MoE types that are 75 % of decode time never reach the device because the gate excludes them by name
(`strstr(t->info->name, "exps") == NULL`), deliberately: 48 x 128 expert slabs is 7.25 GB and will not fit beside the
model on a 6 GB card.

### Opening the experts measured a 14 % LOSS, so the exclusion is correct

Behind a new `IN2_GPU_EXP` flag, allowing expert tensors through the same resident tier:

    off                2333 ms
    dense only         2335 ms
    dense + experts    2658 ms      +14 % slower

Because the experts are not resident they stream: each call pays the transfer of a slab that cannot stay on the card, and
8 of 128 experts are chosen per token so the access pattern is unpredictable. The name filter is not an oversight, it is
the correct default, and it was reverted rather than kept behind a flag that only ever loses.

### What that leaves

The device path is bounded by residency, not by arithmetic. A hot-expert cache - keeping the experts that the router
actually selects most often resident and streaming only misses - is the only way to move the 75 % and it is a real
feature, not a flag. Until then the honest summary is: **the GPU contributes ~5 % of the work and buys 7-21 % of decode
speed; prefill (the phase that decides whether long context is usable at all) is CPU-only by construction.**

## The router is concentrated enough for a hot-expert cache - measured before building one

`IN2_MOE_HIST=1` (new, `wubu_inkling2_fwd.c`) accumulates a routing histogram keyed by (layer, expert) and dumps it at
exit. 16 decode tokens, 24 layers, 8 experts per token, 5760 picks total:

    layer    picks   distinct of 128   top-8    top-16   top-32
    L0        120         54           39.2 %   60.0 %   81.7 %
    L1        120         50           45.0 %   65.8 %   85.0 %
    L2        120         56           39.2 %   58.3 %   80.0 %

Aggregating across layers is meaningless (each layer has its own 128 experts, so the union is all 128 at top-8 = 14.8 %);
per layer the distribution is strongly skewed: **half the experts are never selected in 120 picks, and the top 32 cover
about 82 %.**

That is the precondition a hot-expert cache needs, and it prices the design exactly:

    expert size            302.14 MB / 128 per layer = 2.36 MB each
    cache of 32/layer      75.5 MB per layer  x 24 layers = 1.81 GB      ~82 % of picks
    cache of 16/layer      37.8 MB per layer  x 24 layers = 0.91 GB      ~62 % of picks

Against a 6 GB card carrying a 1.22 GB dense set, the 32-expert cache is affordable, and it would put roughly four
fifths of the 75 % of decode time that is MoE within reach of the device - the opposite of the earlier experiment where
every expert streamed and the result was 14 % slower.

Two things this does not yet say: the histogram is over 16 tokens of one prompt, so the *stability* of the hot set over
a longer run is unmeasured (a cache that hits 82 % in the first 16 tokens and then thrashes on a topic change is worth
less than its hit rate suggests), and the miss path still pays a PCIe transfer that the 14 % loss measured above shows is
not free. The next step is therefore a cache with an explicit miss cost, measured on a longer run, not a straight
port of the routing table.

Recording the instrument itself: it is env-gated, writes only when `IN2_MOE_HIST=1`, and answered in one 16-token run a
question that would otherwise have been answered by building the cache and hoping.

## The hot set drifts, so the cache must be an LRU - and that changes the size it needs

`IN2_MOE_IDS=1` (adds a token-stamped dump next to the histogram) over a 64-token run, 3024 routing records. The
sequence was then replayed offline, which costs nothing and needs no engine change.

A **static** hot set - the top-N experts per layer learned from the first half of the run - covers only this much of the
second half:

    16/layer   0.91 GB   43.7 % hit
    32/layer   1.81 GB   63.2 % hit

against 82 % for the same 32 experts *within* a half. So the routing distribution moves: a cache built from an
observation window is already stale by the time it is used. That rules out the obvious design (profile, pin the top 32,
ship it) and is the reason this was measured before the cache existed.

An **LRU** per layer, replayed over the same sequence:

    8/layer    0.45 GB   40.4 % hit   14229 evictions
    16/layer   0.91 GB   63.8 % hit    8383
    24/layer   1.36 GB   73.3 % hit    5878
    32/layer   1.81 GB   79.0 % hit    4322
    48/layer   2.72 GB   85.7 % hit    2312
    64/layer   3.62 GB   89.5 % hit    1012

LRU at 32 beats a static 32 by **15.8 points** (79.0 vs 63.2) at identical size, and at 48 it reaches 85.7 % - close to
the within-window ceiling of the static case. Against a 6 GB card carrying a 1.22 GB dense set:

    32/layer  cache 1.81 GB + dense 1.22 GB = 3.03 GB   79 % of MoE hits
    48/layer  cache 2.72 GB + dense 1.22 GB = 3.94 GB   86 % of MoE hits

Both fit, and 48 is the better trade while leaving room for activations and a KV slice.

### What this decides, and what it does not

Decided: **LRU, 48 experts per layer, ~2.7 GB, ~86 % hit rate** - and the cache is worth building, because the
alternative measured 14 % slower when everything streamed and this leaves only 14 % streaming.

Not decided: a 14 % miss path still pays PCIe, so the net is only positive if a hit is genuinely cheaper than the CPU
route it replaces - that is an end-to-end measurement, not an extrapolation from hit rate. And every number here comes
from one prompt and one 64-token run; a second prompt with different subject matter could move the curve, so the cache
needs a hit-rate counter that reports live rather than trusting this table.

## The LRU cache was implemented and reverted: the resident tier's cache key cannot survive eviction

The routing analysis priced an LRU at 48 experts/layer, ~2.7 GB, 85.7 % hit. Implementing it against the resident slab
turned up two things, both of which stop the work rather than slow it.

**1. Experts still never reach the device, even with room and eviction.** Opening the expert gate and giving the slab a
3 GB budget (`IN2_RES_GB=3.0`), the trace still refuses both expert types that the type gate allows:

    ok      ty=8
    refused ty=16 rc=-2   (slab FULL)
    refused ty=22 rc=-2
    refused ty=12/13/14/17/18/19/40 rc=-1   (outside the type gate)

`-2` is "no free slot", and it fired even with eviction wired in, so the eviction path as written did not free a slot on
that miss - the slab had already reached `WUBU_RES_SLOTS`, which returns before allocating.

**2. Eviction broke parity, and that is the real blocker.** With eviction active the greedy ids changed at token 5:

    oracle      149876 220 36838 182189 149933 220 50843 11
    with LRU    149876 220 34361 149933 145038 30

The refused expert types never executed, so the corruption is in the **dense** path: evicting a slab and reallocating its
slot changes which bytes a cached entry refers to. The resident tier keys its cache on `(host pointer, byte count)`, and
that key is only sound while entries are permanent. As soon as a slot is recycled, a pointer that the engine re-uses for
a different tensor resolves to a slab holding the previous tensor's bytes - and the matvec returns a plausible, wrong
answer with no error at all.

So the design conclusion is not "the cache needs tuning", it is: **before any eviction can ship, a resident entry needs a
stable tensor identity** - an id or a file offset carried from the loader - instead of a host address. That is a small
change to the bridge's signature and a large one to trust in the cache; it is the first thing to do, and the LRU policy
itself comes after it.

Both changes were reverted and the engine re-verified: DLL back to 706048 bytes, `gen_fast.exe` rebuilt, oracle 12/12
exact. Recording the failure is the point - the LRU policy was measured and correct, and it still could not be shipped,
because the thing underneath it was not safe to recycle.

## RETRACTION: eviction is sound - the parity break was the expert gate, not the cache key

The previous section blamed the resident cache's `(host pointer, byte count)` key for the token divergence and called a
stable tensor identity the blocker. A bisect disproves that. Same build, eviction active at a 3 GB budget, but with the
expert gate **closed**:

    eviction ON,  dense only   12/12 oracle tokens exact
    eviction OFF (default)     12/12 oracle tokens exact
    24-token decode:  5134 ms / 5.26 tok/s   vs   5114 ms / 5.30 tok/s   ids identical

So the eviction implementation is correct and, when it never has to evict, costs 0.4 % - within noise. The pointer-keyed
cache survives slot recycling for dense tensors, which is the case it was written for. My diagnosis was wrong.

What the bisect does localise: **the divergence appears only when the expert gate is open**, and the trace cannot show
why, because it dedupes its per-type messages (`static int w2[256]`, first outcome per type only). If an expert type
refuses once and succeeds later - or the reverse - only the first line is ever printed, so "refused ty=16 rc=-2" and
"produced wrong expert output" are indistinguishable in the log.

So the real finding is narrower and more actionable than the last one:

  - eviction is safe and stays; it is a prerequisite for any expert caching, gated behind `IN2_RES_GB` and inert
    by default (budget 0 = the old permanent behaviour, ids identical, 0.4 % cost);
  - the expert slice path on the device is **numerically unverified**: opening it changes the greedy sequence at token
    5, and nothing currently establishes whether the expert kernels are wrong or the slab holds the wrong slice for a
    given (tensor, element offset) pair;
  - the per-type trace must report every outcome, not the first, before it can be trusted to say whether an expert
    matvec ran.

That makes the next step a correctness gate, not a performance one: for one expert slice, compare the device matvec
against the CPU kernel on the same bytes, with every call reported. Performance work on MoE resumes after that gate
passes - the LRU sizing and hit rates are already measured and waiting.

## The expert gate does not merely give wrong answers, it segfaults - so MoE-on-GPU is a stability defect

The correctness gate (`IN2_GPU_CHECK=1`: when the device takes a matvec, recompute the same row with the CPU kernel and
print worst-abs) was written to answer one question - are the expert kernels right? It never fired:

    GPUCHECK lines: 0

because no expert matvec ever returns success. With the expert gate open, the eviction budget set, and the check
enabled, the process **dies with SIGSEGV**:

    exit: Segmentation fault
    refused ty=14 rc=-1
    ok      ty=8

So the earlier parity change when the expert gate was open (token 5: `34361 149933 145038` instead of
`36838 182189 149933`) was not a precision problem to be measured - it was the leading edge of a crash. That is a
different and more serious class than everything else in this document, and it means the MoE device path has never been
in a state where its arithmetic could be judged at all.

Everything from that attempt is reverted; what remains, and is verified:

    gen_fast.exe oracle (CPU only)                     12/12 exact
    gen_fast.exe oracle with IN2_GPU_DENSE=1           12/12 exact
    ... with IN2_GPU_DENSE=1 IN2_RES_GB=3.0            12/12 exact   (eviction active, never triggers)

A segfault in the hot path also cannot be instrumented away: adding the comparison to the matvec dispatch is what made
it obvious, but no amount of extra printing will localise it, because the fault is below the C level. The next attempt
needs a **minimal reproducer** - one expert tensor, one activation row, one resident call, in a standalone harness
(`tools/gpu_expert_slice.c`) - rather than another flag on a 48-layer forward pass.

Recorded so the next session does not repeat this: three separate attempts on the MoE device path (open the gate
measured 14 % slower, eviction added, correctness check added) all ended in either a slowdown or a crash, and the only
thing that moved forward was the routing analysis that says a cache would be worth 86 %. The cache is not the blocker.
The blockers, in order, are: expert slices crash on the device; the per-type trace hides outcomes after the first; and
only then is any hit-rate question worth asking.

## Cleanup: the expert escape hatch is gone, and the bounds guard did not find the fault

Three attempts to reach the expert path by patching the dense matvec entry (open the expert name filter, add eviction,
add a device-vs-CPU check) each ended in a crash or a wrong token, so all of it was removed rather than left behind a
flag. What that work established, in order:

  - **the slice bounds are correct.** A guard computing `boff + rb*n_out` against `gguf_raw_size(ty, prod(dims))` never
    fired: the device is handed a byte range that does lie inside the tensor. So the fault is not the offset arithmetic,
    which was my leading hypothesis;
  - **the fault is inside the device call for expert types anyway**, and it moves with optimisation level (`-O3` faults,
    `-O1` returns the same wrong token), so it is a memory error rather than numerics;
  - **the per-type trace is not trustworthy for this path** - it prints the first outcome per type only, so "refused" and
    "ran" are indistinguishable in the log, which is why three separate runs appeared to show experts never executing.

Two of my patches also landed on the wrong call site (the batched path has a second resident call using `base`/`ty`
instead of `src`/`ty_mv`), which is what produced two of the three segfaults. A hot path with two near-identical call
sites is the wrong place for scripted edits; the next change there goes in as a new function, not a patched condition.

### Verified state after the cleanup

    gen_fast.exe, CPU only                 12/12 oracle, exact
    IN2_GPU_DENSE=1                        12/12 oracle, exact
    24-token decode   off 5425 ms   on 5113 ms   ids identical      (-5.8 %)
    trace             NT-GPU ok ty=8 (blk.5.attn_k.weight), GPU tier: engaged

So the device keeps the decode win it had (this run 5.8 %, earlier runs 7-21 % depending on length), with no escape hatch
that can crash, and the tree carries no dead instrumentation.

### What replaces it

The expert path is a **MUL_MAT_ID-shaped gather**, and the reason three patches failed is that this engine has no such
entry point - only a contiguous-slab dense matvec. The next change is therefore a new function with its own signature
(`ids`, `n_ids`, slot map, guard row), written once and gated on its own device gate, rather than another condition
inside the dense path. That is the first item in `docs/MOE-OFFLOAD-PRIOR-ART.md` and it has not changed.

## The expert rows return ZEROS, and that is a different bug from the one I was chasing

With a device-vs-CPU comparison placed at the *expert* call site (the one that actually serves `ffn_gate_up_exps`, which
two earlier patches had missed), 242 expert matvecs were checked:

    rel error < 1e-3      72
    rel error >= 1e-3    170
    worst 7.03   median 1.000   best 7.1e-08

A **median of exactly 1.000** is the tell: `rel = max|y - ref| / max|ref|` equals 1.0 when `y` is all zeros and the
reference is not. Most expert matvecs are not computing the wrong thing - they are computing nothing, and the engine
then adds a zero contribution. That is a different failure from the one inferred earlier, and it is the failure this
project's own notes already name: *a kernel that returns all zeros is usually missing a table upload, or a type no
kernel claims.*

Three things were tried against this path and none of them fixed it, which is worth recording so they are not re-tried:

  1. **Layer/slab eviction** (`IN2_RES_GB`) - correct and kept, but the expert rows are zero with or without it.
  2. **A bounds guard** on `boff + rb*n_out` vs `gguf_raw_size(ty, prod(dims))` - it never fired, so the slices are
     legitimately inside their tensors.
  3. **Identity keying.** The pointer key *was* genuinely unsound for this path - 66 of 153 distinct `src` addresses were
     reused inside a single tensor, because the engine stages a miss into a per-call buffer the allocator hands back at
     the same address - so the resident entry now takes an `ident` (tensor file offset + element offset) and the DLL keys
     on it (`wubu_cuda_resident_matvec_id`, back-compat wrapper keeps the old export). Dense parity is exact on the new
     key, but the expert rows are still 170/242 wrong, so the collision was real **and not the cause**.

What remains, and is now a hypothesis with a specific shape: the device dispatch for the types the experts use (16 IQ2_XXS,
22 IQ2_S) is reaching a kernel that writes zeros for these shapes. `ty=23` (dense) measures 7.1e-08 on the same path, so
the entry itself is sound; the difference is the type. The next step is therefore to call `wubu_cuda_matvec_iq2xxs`
directly from a standalone harness with a known slab and a known activation - not through the resident cache, not through
the engine - and read what it returns. That is a ten-minute test and it is the only one left that isolates the kernel from
everything layered on top of it.

## Root cause found: the expert types have no device tables uploaded, so their kernels write zeros

Per-type device-vs-CPU results, 242 expert rows, after adding IQ2_S to the resident dispatch:

    ty=23 (Q4_K, dense)   n= 50   good= 50   median rel 1.2e-07     exact
    ty=16 (IQ2_XXS)       n=136   good= 26   median rel 1.20        wrong
    ty=22 (IQ2_S)         n= 56   good=  0   median rel 1.000       all zeros

`rel` is `max|y-ref| / max|ref|`, so exactly 1.000 means **y is all zeros**. That is the project's own documented
signature - *a kernel that returns all zeros is usually a missing table upload* - and it is literal here:

    uploads the engine performs        kernels that need a table
    wubu_cuda_upload_iq2xxs            ty 16  (256-entry grid + ksigns)
    wubu_cuda_upload_tables_iq2xs      ty 17
    wubu_cuda_upload_table_iq4nl       ty 23
    -                                  ty 22  IQ2_S needs the 1024-entry grid; never uploaded

The DLL exports `wubu_cuda_upload_tables(iq2s_1024_u64, iq3s_512_u32)` for exactly this, and the engine never calls it,
because the host side has no accessor for the IQ2_S grid (`iq2s_grid` is a file-static table in `wubu_iq_c11.c`, and the
only grid accessor the engine exports is `wubu_iq2xxs_grid_bytes`). So IQ2_S has never been device-executable in this
engine - the resident dispatch did not even route to it before this change, falling through to the generic
`wubu_cuda_quant_matvec` which returns success and writes nothing.

Two fixes landed, both verified not to disturb anything else:

  1. **ty 22 now dispatches to `wubu_cuda_matvec_iq2s_v2`** instead of the silent generic entry. It still writes zeros
     because of the missing table, but the fall-through that silently succeeded is gone.
  2. **A device self-test export** (`wubu_cuda_selftest_iq2`) that runs the kernel on a synthetic slab with no engine,
     no GGUF and no cache in the way - the isolation test the previous round could not reach.

Verified unchanged: oracle 12/12 exact on CPU and with `IN2_GPU_DENSE=1`; 24-token decode 5464 -> 5122 ms.

What is left is one concrete piece of plumbing: export the IQ2_S grid from the host (`iq2s_grid` is already in
`wubu_iq_c11.c`) and call `wubu_cuda_upload_tables` at init the way `upload_iq2xxs` already is. ty=16's 26/136 is the
same class of problem and needs the same check - its grid *is* uploaded, so its failures point at the slab slice the
resident path hands it rather than at tables, and that is the next measurement after the upload lands.

## The IQ2_S table upload worked: 56 expert rows went from all-zeros to 1.3e-07

Adding the missing upload at init (`GetProcAddress("wubu_cuda_upload_tables")` fed the host's `iq2s_grid` and
`iq3s_grid`, which needed two new accessors in `gguf_reader.c` beside `wubu_iq2xxs_grid_bytes`) changed ty=22 from
silent zeros to correct arithmetic:

    type                  before                          after
    ty=22 IQ2_S    56 rows, median rel 1.000 (zeros)   37 good, median 1.30e-07
    ty=23 Q4_K     50 rows, median 1.2e-07             50 good, median 1.26e-07
    ty=16 IQ2_XXS 136 rows, median rel 1.20           136 rows, median rel 1.28

So a kernel that returns success while writing nothing is, as the notes said, a missing table - and the engine had never
uploaded the one IQ2_S needs. That is now fixed and verified.

## What is left, and it is one specific oddity

Within every expert tensor - both types, all layers - roughly **28 % of calls are correct and 72 % are wrong**, uniformly:

    ty=16  blk.12.ffn_gate_up_exps   n=24 good=6      ty=22  blk.16.ffn_gate_up_exps  n=24 good=14
    ty=16  blk.13.ffn_gate_up_exps   n=24 good=3      ty=22  blk.21.ffn_gate_up_exps  n= 8 good=7
    ty=16  blk.7.ffn_gate_up_exps    n=24 good=6      ty=23  output.weight            n=50 good=50

A uniform fraction that does not depend on the tensor, the layer or the quant type is not a per-tensor defect - it is the
signature of a **per-call** decision that goes right sometimes. The two candidates are the resident cache (an upload
landing in the wrong slot, or a hit served for the wrong identity) and the slice address (`src` pointing at the wrong
expert for the majority of calls, correct when the requested expert happens to be the one at the base of the tensor).

The isolate for that is the self-test export added with this work: `wubu_cuda_selftest_iq2(type, K, n_cols, row_bytes,
&maxabs)` runs the kernel on a synthetic slab with no cache, no engine and no GGUF in front of it. Its next revision
must **return `y` rather than a count**, so the same slab can be run through the engine's own `wubu_iq_rows` and the two
compared directly - that separates "kernel is wrong" from "the slab it was given was wrong" in one run, which is the
distinction five sessions of engine-level instrumentation failed to make.

Verified unchanged by all of the above: oracle 12/12 exact on CPU and with `IN2_GPU_DENSE=1`; 24-token decode
5440 -> 5102 ms.

## Isolation result: the IQ2 kernels are correct and the resident cache is what corrupts expert rows

A self-test that runs the device kernel on caller-supplied bytes - no cache, no engine, no GGUF - and compares against
the engine's own `wubu_iq_rows` on the identical slab:

    SELFTEST ty=16 cpu0=252.85 dev0=252.85 max_abs=1.221e-04 max|cpu|=2.021e+03 rel=6.04e-08

**The kernel is right.** The first two runs of that test reported `dev0=0`, and the reason was the same thing measured
one step earlier: the self-test ran *before* `in2_gpu_init` had uploaded the tables, so the kernel had no grid. With the
upload done first it agrees with the CPU to 6e-08.

Then the same comparison end to end, at the expert call site, 242 rows:

    through the resident cache   114 good / 242     ty=16 27/136, ty=22 37/56, ty=23 50/50
    cache bypassed (stateless)   242 good / 242     ty=16 136/136, ty=22 56/56, ty=23 50/50
                                                   medians 1.2e-07, 1.1e-07, 1.2e-07

So with the identical kernel, identical slab and identical activation, the **cache path** produces wrong rows and the
stateless path produces exact ones. The defect is in the resident tier's slot handling for these slices - not in the
quantiser, not in the table upload, not in the offset arithmetic (a bounds guard never fired), and not in the kernel
(just proven on synthetic data).

### Why this does not yet ship

Routing expert tensors through the stateless entry by default made the engine emit `id=0` tokens - degenerate, though
the per-call values are exact. So a third defect sits above the kernel: something in how expert output is consumed or
accumulated differs between the two paths. That change and its measurement scaffolding were reverted, and the tree is
back to the verified-green state:

    CPU only         12/12 oracle exact
    IN2_GPU_DENSE=1  12/12 oracle exact

### The three-line summary of nine rounds on this path

  - kernel correct (6e-08 on identical bytes)          -> stop looking at the kernel
  - table upload was genuinely missing and is fixed    -> ty=22 0/56 zeros -> 37/56 exact
  - the resident cache corrupts expert slabs           -> 114/242 vs 242/242 stateless

The next step is narrow for the first time: diff one expert's cached slot against its own `src` bytes inside the DLL,
which is a ten-line check inside `wubu_cuda_resident_matvec_id`, not another engine-level experiment.

## Found and fixed: the corrupt expert rows were a thread race on the resident tier's shared buffers

Three measurements, same kernel, same slabs, same tables, only the concurrency differing:

    8 threads, resident cache        111-114 good of 242 expert rows
    1 thread,  resident cache        242 / 242 exact
    8 threads, resident cache + lock 242 / 242 exact

The resident entry keeps **one** activation staging buffer (`g_res_act`, `g_pin_act`) and **one** output buffer
(`g_res_y`) for every caller, and the engine calls it from OpenMP threads. Thread A overwrites the activation while
thread B is still reading it, and the kernel - which was proven correct on synthetic data at 6e-08 - faithfully returns
the wrong answer for the mixed-up input. That is why the failure looked per-call rather than per-tensor, why it was
uniform across layers and quant types, why it survived a bounds guard, eviction changes and identity keying, and why the
same code was exact whenever it happened to run on one thread.

The fix is a `ResGuard` RAII lock in `wubu_cuda_resident_matvec_id` (a `std::atomic<int>` spin, because nvcc's Windows
host has no `__sync` builtins). Verified:

    device vs CPU, 242 expert rows, 8 threads:  136/136 ty=16, 56/56 ty=22, 50/50 ty=23, medians 1.1-1.2e-07
    oracle, all three configurations identical:
      CPU only, IN2_GPU_DENSE=1, IN2_GPU_DENSE=1 IN2_GPU_MOE=1 IN2_RES_GB=3.0  ->  the same 15 tokens

**MoE runs on the device and produces the reference answer.** That is the milestone: the expert path has been either
refused or silently wrong since this project began.

### What it is not worth yet

    24-token decode    CPU 5440 ms | GPU dense 5142 ms (-5.5 %) | GPU dense + experts 5236 ms (+1.8 % vs dense)

Offloading the experts is currently a **small loss**, because the lock serialises the entry and every expert call still
uploads a ~1 MB slab over PCIe. Correctness had to come first and does. The payoff now depends on the cache actually
hitting, which it finally can - the hit/miss counters are trustworthy for the first time, and the hot set is known to be
concentrated (top-32 experts ~82 % of picks). That is the next measurement, not another diagnosis.

## The cache's real limit was its slot table, not its byte budget - and MoE offload still does not pay

With the race fixed the counters became meaningful, so they were finally read instead of assumed:

    WUBU_RES_SLOTS 320, budget 3 GB    hit=7647 miss=2140 evict=0  slots=320  rate=78.1%
    WUBU_RES_SLOTS 1536, same budget   hit=9021 miss= 766 evict=0  slots=766  rate=92.2%

`evict=0` in the first line is the tell: the tier had filled its 320-entry pointer table while using ~320 MB of a 3 GB
budget, and every slab past the 320th unique one was refused to the CPU. The *table* was the binding limit, not memory.
At 1536 slots the hit rate rises 14 points and nothing is refused.

That did not make offload profitable:

    24-token decode    CPU 5470 ms | GPU dense 5134 ms (-6.1 %) | GPU dense + experts 5495 ms (+7.0 % vs dense)

Correct and still a loss, for a reason the numbers make plain: one token activates 24 layers x 8 experts = 192 expert
matvecs, each a small (1152 x 2304) launch, and the entry is now serialised by the lock. Launch overhead dominates the
~2.6 MMAC each one does. **The fix is batching, not caching**: the 8 active experts of a layer have to become one launch.
Until that exists, MoE offload stays opt-in behind `IN2_GPU_MOE`, which is where it is - correct, verified against the
oracle, and honestly not worth the PCIe.

Verified in this state: oracle 12/12 identical for CPU only, dense, and dense + experts; expert rows 242/242 exact at
8 threads.

## Knob sweep: the cache cannot be tuned into profitability, only built around

    24-token decode                                  hit / miss      rate
    dense, 3 GB                                      5261 / 110      98.0 %   5214 ms
    dense + experts, 3.0 GB                          9018 / 769      92.1 %   5616 ms
    dense + experts, 5.5 GB                          9018 / 769      92.1 %   5597 ms

Raising the byte budget from 3 GB to 5.5 GB changes **nothing** - identical hits, identical misses - because the 1536-slot
table fills at roughly 3 GB of slabs and caps everything after that. Nor can the table simply grow: the working set is
6144 expert slabs at ~2.36 MB = **14.5 GB**, on a 6 GB card, so eviction is unavoidable at any setting. 92 % is near the
ceiling this design can reach, and 92 % still loses to the CPU by 7 %.

Two conclusions worth keeping, both measured rather than argued:

  - **The 769 misses are not the problem.** At 1-2 MB each that is ~3-6 ms per token of PCIe against a ~15 ms deficit.
    Even a perfect cache would not close it.
  - **Launch count is the problem.** 24 layers x 8 experts = 192 serialised launches per token, each doing ~2.6 MMAC.
    The CPU does that work in the same order of time without paying any launch at all.

So the next change is not a knob. It is one batched launch per layer - gather the 8 active experts' slabs into one
contiguous device span and run a single matvec with 8 x n_out rows. That is a new device entry (the gather is 8 memcpys
of ~1.4 MB each, already resident at 92 % hit rate) plus one call-site restructure in `in2_moe`, and it is the only path
left where MoE offload can beat the CPU on this hardware.

## The batched expert launch is 2.39x faster and bit-identical - so the engine should use it

A new device entry, `wubu_cuda_moe_batch`, gathers a layer's active expert slabs into one contiguous span and issues a
single matvec of `n_exp * n_rows` rows instead of `n_exp` launches. Priced in isolation with real tables, no engine, no
OpenMP and no cache (`tools/moe_batch_price.c`, 8 experts of 2304x2304 IQ2_XXS, 20 repetitions):

    batch vs per-expert:  max_abs = 0.000e+00                      (bit-identical)
    8 separate launches   3.26 ms per layer
    1 batched launch      1.36 ms per layer            2.39x
    extrapolated          78.1 ms vs 32.6 ms per token (24 layers)

The first run of that harness reported `max|ref| = 0` and I nearly recorded a 2.39x win on an all-zero comparison - the
kernel was returning zeros because the harness had no iq2xxs grid. With the grid and ksigns dumped from `gguf_reader.c`
and uploaded, the outputs are non-degenerate and the two paths agree exactly. Worth stating plainly because the speedup
survived the correction while the correctness claim would not have.

This is the first measurement in this whole GPU thread where a change is both safe and profitable: 2.39x on the exact
work that made MoE offload a 7 % loss. Wiring `in2_moe` to call it is the remaining step - gather the 8 resident slabs,
one call, unpack into the per-expert outputs.

## Wiring `wubu_cuda_moe_batch` into `in2_moe` - exact plan, nothing left to decide

The batched entry is built and priced (2.39x, bit-identical). What remains is call-site surgery in the one loop that has
already cost this project several regressions, so the steps are written down to be executed mechanically:

**Anchor A - `src/wubu_inkling2_fwd.c:1952-1958`**, the OpenMP `for (int k = 0; k < got; ++k)` over active experts inside
`in2_moe`. Per expert it does `in2_matvec_quant(ctx, &L->gate_up, (int64_t)e*2*DFF*d, x + t*d, 2*DFF, d, gk)` then
`in2_matvec_quant(ctx, &L->down, (int64_t)e*d*DFF, ak, d, DFF, ok)`.

**Step 1.** Above the loop, when `got > 1`, the experts share one tensor type, so fetch each expert's slab once:
`src_k = in2_tensor_slab(ctx, &L->gate_up, (int64_t)e*2*DFF*d, &wbytes)` - the read/map logic already in
`in2_matvec_quant` lines 393-440, extracted into a helper that returns the host pointer and byte count without running
the matvec.
**Step 2.** If every `src_k` is non-NULL and the type is in `{16, 22}` (the two the batch entry implements), quantise the
single activation once (q8_K, `n_in = d`, block count `d/256`), call
`wubu_cuda_moe_batch(ty, srcs, got, wbytes, aq, aq_bytes, gy, d, 2*DFF)`, and distribute `gy` into the per-k `gk`
slices. Anything else - a NULL slab, type 40/42, no DLL - falls through to the existing per-expert loop untouched.
**Step 3.** The down projection repeats the same shape (rows `d`, width `DFF`), but its activation differs per expert
(`ak` is post-SiLU), so it needs either a batch entry that takes an activation *array* or `got` separate calls. Do the
gate_up half first and measure; if the win is there, extend the entry with an `act_stride` so one activation block can
carry all experts.
**Step 4.** `wubu_cuda_moe_batch` currently gathers with `n_exp` H2D memcpys every call. At a 92 % hit rate most of
those slabs are already in the resident tier; the real version should gather *device-to-device* from the resident slots
and only fall back to H2D on a miss. That is the difference between 10.4 MB of PCIe per layer and none.

**Parity gate for every step:** the 12-token oracle must stay bit-identical for CPU-only, `IN2_GPU_DENSE=1`, and
`IN2_GPU_DENSE=1 IN2_GPU_MOE=1 IN2_RES_GB=3.0`, and `IN2_MOE_CHECK` must stay 242/242. Anything that measures flat gets
reverted, not kept.

Current state is green and committed: oracle 12/12 identical for CPU-only and dense+experts, expert rows 242/242 exact at
8 threads, batched entry present and priced but not yet called by the engine.
