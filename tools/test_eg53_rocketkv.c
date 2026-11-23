/* test_eg53_rocketkv.c -- EG-53 gate: prompt-KV eviction vs accuracy. */
#include <stdio.h>
#include <math.h>

#include "wubu_rocketkv.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define P 16
#define PS 4
#define D 8
#define NQ 3

static double relerr(const float *a, const float *b, int n) {
    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; i++) { num += fabs(a[i] - b[i]); den += fabs(a[i]); }
    return num / den;
}

int main(void) {
    float keys[P*PS*D], q[NQ*D], score[P];
    float full[D], sel[D], naive[D];
    int all[P], keep[P], nkeep;
    const int important[4] = { 9, 11, 13, 15 };

    for (int p = 0; p < P; p++) {
        int hot = 0;
        for (int k = 0; k < 4; k++) if (p == important[k]) hot = 1;
        for (int t = 0; t < PS; t++)
            for (int d = 0; d < D; d++)
                keys[((size_t)p * PS + t) * D + d] =
                    (hot ? 1.4f : 0.2f) + 0.03f * (float)((t * 7 + d * 3) % 5);
    }
    for (int i = 0; i < NQ; i++)
        for (int d = 0; d < D; d++) q[i*D+d] = 1.0f + 0.01f * (float)(i + d);

    CHECK(wubu_rocketkv_importance(keys, P, PS, D, q, NQ, score) == 0);
    nkeep = wubu_rocketkv_select(score, P, 0.5f, keep);
    printf("EG-53 kept=%d/%d prompt_bytes=%zu full_bytes=%zu ratio=0.5\n",
           nkeep, P, (size_t)nkeep * PS * D * sizeof(float),
           (size_t)P * PS * D * sizeof(float));
    CHECK(nkeep == P / 2);
    for (int k = 0; k < 4; k++) {
        int found = 0;
        for (int i = 0; i < nkeep; i++) if (keep[i] == important[k]) found = 1;
        CHECK(found);
    }

    for (int i = 0; i < P; i++) all[i] = i;
    CHECK(wubu_rocketkv_decode(keys, all, P, PS, D, q, full) == 0);
    CHECK(wubu_rocketkv_decode(keys, keep, nkeep, PS, D, q, sel) == 0);
    CHECK(wubu_rocketkv_decode(keys, all, P / 2, PS, D, q, naive) == 0);
    printf("EG-53 relerr selected=%.5f naive_first_half=%.5f\n",
           relerr(sel, full, D), relerr(naive, full, D));
    CHECK(relerr(sel, full, D) < 0.05);
    CHECK(relerr(naive, full, D) > relerr(sel, full, D) + 0.05);

    CHECK(wubu_rocketkv_importance(NULL, P, PS, D, q, NQ, score) == -1);
    CHECK(wubu_rocketkv_select(score, P, 1.5f, keep) == -1);
    CHECK(wubu_rocketkv_decode(keys, keep, 0, PS, D, q, sel) == -1);

    if (failures) {
        fprintf(stderr, "test_eg53_rocketkv: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg53_rocketkv: PASS\n");
    return 0;
}
