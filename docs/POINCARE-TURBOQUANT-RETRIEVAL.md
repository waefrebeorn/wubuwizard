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
