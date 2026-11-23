/* test_eg58_specprefill.c -- EG-58 gate: prefill cut vs answer fidelity. */
#include <stdio.h>
#include <math.h>

#include "wubu_specprefill.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define N 64
#define D 8
#define DD 4
#define KEEP 16

static double cosine(const float *a, const float *b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < D; i++) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    return dot / (sqrt(na) * sqrt(nb) + 1e-12);
}

int main(void) {
    float emb[N*D], qd[DD], q[D], score[N];
    float full[D], comp[D], base[D];
    int all[N], keep[KEEP], naive[KEEP];

    for (int i = 0; i < N; i++) {
        int salient = (i >= N / 2 && i % 2 == 0);
        for (int d = 0; d < D; d++) {
            float v = 0.02f * (float)((i * 3 + d) % 7);
            if (d < DD) v += salient ? 1.0f : 0.05f;
            emb[(size_t)i * D + d] = v;
        }
        all[i] = i;
        if (i < KEEP) naive[i] = i;
    }
    for (int d = 0; d < DD; d++) qd[d] = 1.0f;
    for (int d = 0; d < D; d++) q[d] = 1.0f;

    CHECK(wubu_specprefill_score(emb, N, D, DD, qd, score) == 0);
    CHECK(wubu_specprefill_select(score, N, KEEP, keep) == 0);
    CHECK(wubu_specprefill_answer(emb, all, N, D, q, full) == 0);
    CHECK(wubu_specprefill_answer(emb, keep, KEEP, D, q, comp) == 0);
    CHECK(wubu_specprefill_answer(emb, naive, KEEP, D, q, base) == 0);
    printf("EG-58 prefill tokens %d->%d (%.2f), fidelity=%.5f naive=%.5f\n",
           N, KEEP, (double)KEEP / N, cosine(full, comp), cosine(full, base));
    CHECK(cosine(full, comp) > 0.95);
    CHECK(cosine(full, comp) > cosine(full, base) + 0.05);

    CHECK(wubu_specprefill_score(emb, N, D, D + 1, qd, score) == -1);
    CHECK(wubu_specprefill_select(score, N, N + 1, keep) == -1);
    CHECK(wubu_specprefill_answer(emb, keep, 0, D, q, comp) == -1);

    if (failures) {
        fprintf(stderr, "test_eg58_specprefill: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg58_specprefill: PASS\n");
    return 0;
}
