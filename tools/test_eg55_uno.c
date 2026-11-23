/* test_eg55_uno.c -- EG-55 gate: shared-KV gated LoRA + lossless sampling. */
#include <stdio.h>
#include <math.h>

#include "wubu_uno.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define DIM 8
#define RANK 2
#define NS 200000

static unsigned long long rng_state = 0x12345678ULL;
static double urand(void) {
    unsigned long long z = (rng_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    return (double)(z >> 11) * (1.0 / 9007199254740992.0);
}

int main(void) {
    wubu_uno_t m;
    float x[DIM], y0[DIM], y1[DIM], base[DIM];
    const float p[8] = { 0.25f, 0.20f, 0.15f, 0.10f, 0.10f, 0.08f, 0.07f, 0.05f };
    const float q[8] = { 0.20f, 0.18f, 0.17f, 0.12f, 0.11f, 0.09f, 0.08f, 0.05f };
    long long cnt[8] = { 0 };
    double maxdev = 0.0;

    CHECK(wubu_uno_init(&m, DIM, RANK, 64) == 0);
    printf("EG-55 one KV allocation: %d floats, ptr=%p\n", m.n_kv, (void *)m.kv);
    for (int i = 0; i < DIM; i++) x[i] = (float)cos(0.3 * i);
    for (int i = 0; i < DIM; i++) {
        for (int j = 0; j < DIM; j++) m.W[i*DIM+j] = 0.1f * (float)((i + j) % 5);
        base[i] = 0.0f;
        for (int j = 0; j < DIM; j++) base[i] += m.W[i*DIM+j] * x[j];
    }
    for (int i = 0; i < DIM; i++)
        for (int r = 0; r < RANK; r++) {
            m.A[i*RANK+r] = 0.2f * (float)((i + r) % 3);
            m.B[r*DIM+i] = 0.15f * (float)((i * 2 + r) % 4);
        }

    CHECK(wubu_uno_draft_step(&m, x, -100.0f, y0) == 0);
    CHECK(wubu_uno_draft_step(&m, x, 100.0f, y1) == 0);
    for (int i = 0; i < DIM; i++) {
        CHECK(fabs(y0[i] - base[i]) < 1e-5f);
        CHECK(fabs(y1[i] - base[i]) > 1e-4f);
    }
    printf("EG-55 gate closed = backbone exact; gate open adds low-rank delta\n");

    for (int it = 0; it < NS; it++) {
        double u0 = urand(), u1 = urand(), u2 = urand();
        float acc = 0.0f;
        int tok = 7, emit = -1;
        for (int i = 0; i < 8; i++) {
            acc += q[i];
            if ((float)u0 < acc) { tok = i; break; }
        }
        if (wubu_psi_sample(p, q, 8, tok, (float)u1, (float)u2, &emit) < 0 ||
            emit < 0 || emit > 7) {
            failures++;
            break;
        }
        cnt[emit]++;
    }
    for (int i = 0; i < 8; i++) {
        double dev = fabs((double)cnt[i] / NS - p[i]);
        if (dev > maxdev) maxdev = dev;
    }
    printf("EG-55 emitted max|p_hat-p|=%.5f over %d samples\n", maxdev, NS);
    CHECK(maxdev < 0.01);

    {
        int emit = -1;
        CHECK(wubu_psi_sample(p, q, 8, 8, 0.5f, 0.5f, &emit) == -1);
        CHECK(wubu_psi_sample(p, q, 8, 0, 1.0f, 0.5f, &emit) == -1);
    }
    CHECK(wubu_uno_init(&m, 0, RANK, 64) == -1);
    wubu_uno_free(&m);
    CHECK(m.kv == NULL);

    if (failures) {
        fprintf(stderr, "test_eg55_uno: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg55_uno: PASS\n");
    return 0;
}
