/* test_eg49_readkv.c -- EG-49 gate: 8-bit store / 4-bit fetch with PPL delta. */
#include <stdio.h>
#include <math.h>

#include "wubu_readkv.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define N 1024
#define V 64

static double ce_from_logits(const float *logits, const float *p_ref) {
    float mx = logits[0];
    double denom = 0.0, ce = 0.0;
    for (int i = 1; i < V; i++) if (logits[i] > mx) mx = logits[i];
    for (int i = 0; i < V; i++) denom += exp(logits[i] - mx);
    for (int i = 0; i < V; i++)
        ce -= (double)p_ref[i] * (double)(logits[i] - mx - logf((float)denom));
    return ce;
}

int main(void) {
    float kv[N], o4[N], o8[N], logits[V], l4[V], l8[V], pref[V];
    wubu_readkv_t b;
    double e4 = 0.0, e8 = 0.0, norm = 0.0;
    double ce4, ce8, delta4;
    size_t b4, b8;

    for (int i = 0; i < N; i++) {
        kv[i] = sinf(0.017f * (float)i) * (0.4f + 0.6f * fabsf(cosf(0.003f * i)));
    }
    CHECK(wubu_readkv_store(kv, N, &b) == 0);
    CHECK(wubu_readkv_fetch(&b, 4, o4) == 0);
    CHECK(wubu_readkv_fetch(&b, 8, o8) == 0);
    for (int i = 0; i < N; i++) {
        e4 += fabs(o4[i] - kv[i]);
        e8 += fabs(o8[i] - kv[i]);
        norm += fabs(kv[i]);
    }
    b4 = wubu_readkv_fetched_bytes(&b, 4);
    b8 = wubu_readkv_fetched_bytes(&b, 8);
    printf("EG-49 relerr4=%.5f relerr8=%.5f fetched4=%zu fetched8=%zu ratio=%.4f\n",
           e4 / norm, e8 / norm, b4, b8, (double)b4 / (double)b8);
    CHECK(b4 == 512 && b8 == 1536);
    CHECK(e8 / norm < 0.01);
    /* 16-level bucket midpoint error: bounded by half a bucket (~6-7% here). */
    CHECK(e4 / norm < 0.08);

    {
        float mx = 0.0f, den = 0.0f;
        for (int i = 0; i < V; i++) {
            logits[i] = cosf(0.31f * (float)i) * 2.0f;
            if (logits[i] > mx) mx = logits[i];
        }
        for (int i = 0; i < V; i++) { float e = expf(logits[i] - mx); pref[i] = e; den += e; }
        for (int i = 0; i < V; i++) pref[i] /= (float)den;

        /* Treat the reconstructed KV values as logits of a toy head. */
        for (int i = 0; i < V; i++) {
            l8[i] = o8[i] * 2.0f;
            l4[i] = o4[i] * 2.0f;
        }
        ce8 = ce_from_logits(l8, pref);
        ce4 = ce_from_logits(l4, pref);
        delta4 = ce4 - ce8;
        printf("EG-49 ppl8=%.4f ppl4=%.4f delta_nats=%.5f\n",
               exp(ce8), exp(ce4), delta4);
        CHECK(fabs(delta4) < 0.20);
    }

    CHECK(wubu_readkv_fetch(&b, 3, o4) == -1);
    CHECK(wubu_readkv_fetched_bytes(&b, 3) == 0);
    CHECK(wubu_readkv_store(NULL, N, &b) == -1);
    wubu_readkv_free(&b);
    CHECK(b.coarse == NULL && b.fine == NULL);

    if (failures) {
        fprintf(stderr, "test_eg49_readkv: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg49_readkv: PASS\n");
    return 0;
}
