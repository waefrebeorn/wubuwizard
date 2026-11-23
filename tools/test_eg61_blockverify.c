/* test_eg61_blockverify.c -- EG-61 gate: accepted prefix + exact correction. */
#include <stdio.h>
#include <math.h>

#include "wubu_blockverify.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define V 8
#define NB 4
#define NS 200000

static unsigned long long st = 0x98765432ULL;
static double urand(void) {
    unsigned long long z = (st += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    return (double)(z >> 11) * (1.0 / 9007199254740992.0);
}

int main(void) {
    const float p[V] = { 0.25f, 0.20f, 0.15f, 0.10f, 0.10f, 0.08f, 0.07f, 0.05f };
    const float q[V] = { 0.20f, 0.18f, 0.17f, 0.12f, 0.11f, 0.09f, 0.08f, 0.05f };
    const float same[V] = { 0.25f, 0.20f, 0.15f, 0.10f, 0.10f, 0.08f, 0.07f, 0.05f };
    long long cnt[V] = { 0 };
    double maxdev = 0.0, acc_mismatch = 0.0, acc_same = 0.0;

    for (int it = 0; it < NS; it++) {
        int draft[NB], out[NB + 1], acc = -1, em;
        float u[2 * NB];
        for (int i = 0; i < NB; i++) {
            float accq = 0.0f;
            double u0 = urand();
            draft[i] = V - 1;
            for (int v = 0; v < V; v++) {
                accq += q[v];
                if ((float)u0 < accq) { draft[i] = v; break; }
            }
            u[2 * i] = (float)urand();
            u[2 * i + 1] = (float)urand();
        }
        em = wubu_block_verify(p, q, V, draft, NB, u, out, &acc);
        CHECK(em >= 1 && em <= NB + 1 && acc >= 0 && acc <= NB);
        if (em > 0 && out[0] >= 0 && out[0] < V) cnt[out[0]]++;
        acc_mismatch += acc;

        for (int i = 0; i < NB; i++) { draft[i] = 0; u[2 * i] = 0.0f; u[2 * i + 1] = 0.0f; }
        em = wubu_block_verify(p, same, V, draft, NB, u, out, &acc);
        CHECK(em == NB && acc == NB);
        acc_same += acc;
    }
    for (int i = 0; i < V; i++) {
        double dev = fabs((double)cnt[i] / NS - p[i]);
        if (dev > maxdev) maxdev = dev;
    }
    printf("EG-61 first-token max|p_hat-p|=%.5f; accepted same=%.4f mismatch=%.4f\n",
           maxdev, acc_same / NS, acc_mismatch / NS);
    CHECK(maxdev < 0.01);
    CHECK(acc_same / NS > acc_mismatch / NS + 0.5);

    {
        int draft[NB] = { 0, 1, 2, 3 }, out[NB + 1], acc;
        float u[2 * NB] = { 0 };
        CHECK(wubu_block_verify(p, q, V, draft, 0, u, out, &acc) == -1);
        draft[0] = V;
        CHECK(wubu_block_verify(p, q, V, draft, NB, u, out, &acc) == -1);
    }

    if (failures) {
        fprintf(stderr, "test_eg61_blockverify: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg61_blockverify: PASS\n");
    return 0;
}
