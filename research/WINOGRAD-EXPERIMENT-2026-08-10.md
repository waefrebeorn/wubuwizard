# Winograd F(2,3) experiment — 2026-08-10

**Result: PARITY PASSED, SPEED FAILED. Reverted to direct im2col GEMM.**

## The experiment
Added `wubu_sd_conv2d_q_wino()` (env-gated `SD_WINOGRAD=1`): Lavin F(2,3)
2x2-tile 3x3 conv — 16 mults/tile/ci vs 36 direct (2.25x fewer MACs).
Transforms: U = B^T d B (input, 64 adds), V = G g G^T (filter, precomputed
per conv), Y = A^T M A (output). Serial co-chunk outer (V-chunk L2-resident),
tiles parallel inner.

## Parity — the risk we took: PASSED
- x86: Winograd vs direct: max diff=1, mean 0.018, **0 pixels >1**
- ARM: Winograd vs x86-direct: max diff=1, **0 pixels >1**
The F32 transform-space adds kept every pixel within the established
1/255 parity bar. Accumulation-order risk did NOT materialize.

## Speed — FAILED (memory-bound box, traffic went UP)
VAE decode: 913s (Winograd) vs 492s (direct). Per conv:

| conv | direct | wino | ratio |
|---|---|---|---|
| up.3.upsample 64x64 C512 | 15.4 | 37.2 | 2.4x worse |
| up.2.upsample 128x128 C512 | 80.9 | 184.5 | 2.3x worse |
| up.1.upsample 256x256 C256 | 107.9 | 130.2 | 1.2x worse |
| up.1.block.0.conv1 C512->256 | 62 | 135.9 | 2.2x worse |

### Why it lost
1. **V is 1.8x bigger than W**: C_out x C_in x 16 floats vs x9. This CM4
   (A72, 1MB shared L2) is memory-bandwidth-bound — the F16 direct GEMM
   runs at 18 GFLOP/s = the L2 wall. Fewer MACs can't win when the
   transform-space traffic is larger.
2. **U recomputed per co-chunk**: C_out/COB (up to 64x for C512) redundant
   input transforms.
3. **Gather reads per tile**: 16 scattered 1MB-strided channel reads per
   (tile,ci) — the same gather-latency problem as im2col, now 16 not 9.
4. Naive vectorization: M accumulation auto-vectorized (28 fmla) but the
   transform loops stayed scalar-ish.

## The path that COULD make it win (not taken)
Transform-domain GEMM (ncnn style): stage U per tile-strip, then run the
16 transform points as 16 real GEMMs (V_p [C_out x C_in] x U_p [C_in x
tiles]) with the existing fast kernels. Estimated: up.1.upsample ~14s wall
vs 27s direct (F32-x GEMM at 11.6 GFLOP/s). F16-U would break parity;
F32-U keeps it but halves the GEMM speed. ~2x on upsample convs, ~460s VAE
total. Real project (~150 lines + staging buffer), uncertain payoff on a
bandwidth-bound part.

## Decision
Keep direct im2col GEMM as the shipped path (492s VAE). Winograd code
remains in src/wubu_sd_ops.c gated behind SD_WINOGRAD=1 for future
experiments (parity-safe if we ever get an L2-fatter device).
