/* attn_wall.c - how fast is this engine's attention score loop, per key, and what does that cap at 1M?

   The score loop in wubu_inkling2_fullseq.c accumulates in double:
       double acc = 0.0; for (e < HD) acc += (double)q[e] * (double)k[e];
   which gcc does vectorise (AVX-512 vcvtps2pd/vmulpd) but cannot reassociate, so the reduction stays a dependent
   chain. This measures it directly instead of arguing about it.

   Build/run (standalone, not part of the engine build):
       gcc -O3 -std=c11 -march=znver4 -static -o attn_wall.exe tools/attn_wall.c -lm
       ./attn_wall.exe 3

   Measured on this box (Ryzen 5 8645HS), 1,048,576 keys of 128 dims, single thread:
       0.060 s per (head,visit) pass = 57 ns/key = 2.24 GMAC/s
       64 such passes (8 full visits x 8 kv heads) x 2 for QK and PV = 7.67 s/token => 0.13 tok/s

   That is the real 1M wall and it is arithmetic, not memory: the bandwidth estimate of ~4.3 tok/s assumes the
   attention itself is free. It says sequence-dimension compression or selection is mandatory, not optional.
*/
/* Time the real attention score pattern at 1M keys, so the long-context compute wall is measured and not argued.
 *
 * Per token at 1M context this engine does 8 full visits x 8 kv heads = 64 (query head, visit) pairs, each scoring
 * 1,048,576 keys of 128 dims, plus the same shape again for the value weighted sum. This measures one such pass over
 * an out-of-cache key array, which is the honest cost per pair.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define HD 128
#define KEYS 1048576

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double) t.tv_sec + 1e-9 * (double) t.tv_nsec;
}

__attribute__((noinline))
static double score(const float *q, const float *k)
{
    double acc = 0.0;
    for (int e = 0; e < HD; ++e) acc += (double) q[e] * (double) k[e];
    return acc;
}

int main(int argc, char **argv)
{
    const int reps = argc > 1 ? atoi(argv[1]) : 4;
    float *q = (float *) malloc(HD * sizeof(float));
    float *kv = (float *) malloc((size_t) KEYS * HD * sizeof(float));
    if (!q || !kv) { printf("alloc failed\n"); return 1; }
    for (int i = 0; i < HD; ++i) q[i] = 1.0f / (float) (i + 1);
    for (size_t i = 0; i < (size_t) KEYS * HD; ++i) kv[i] = (float) ((i % 97) - 48) * 0.01f;

    double sum = 0.0, best = 1e9;
    for (int r = 0; r < reps; ++r) {
        const double t0 = now();
        for (int j = 0; j < KEYS; ++j) sum += score(q, kv + (size_t) j * HD);
        const double dt = now() - t0;
        if (dt < best) best = dt;
        printf("rep %d: %.3f s for %d keys (%.1f ns/key, %.2f GMAC/s)\n",
               r, dt, KEYS, 1e9 * dt / KEYS, (double) KEYS * HD / dt / 1e9);
    }
    /* one full query head across all visits: 64 pairs per token at 1M, plus the value pass */
    printf("\nbest %.3f s per (head,visit) QK pass\n", best);
    printf("64 pairs (8 visits x 8 kv heads) QK+PV ~= %.2f s/token => %.2f tok/s ceiling from attention ALONE\n",
           2.0 * 64.0 * best / 1.0, 1.0 / (2.0 * 64.0 * best / 1.0));
    printf("(sink %.1f)\n", sum);
    free(q); free(kv);
    return 0;
}
