#include <stdio.h>
#include <math.h>
#include "wubu_inkling2_kv.h"

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); ++fails; } } while (0)

int main(void)
{
    in2_kv_t kv;
    /* window 4 over 16 rows: slot reuse is the thing that silently corrupts a sliding cache */
    CHECK(in2_kv_init(&kv, 2, 8, 4, 16) == 0, "init");

    for (int pos = 0; pos < 10; ++pos) {
        float k[8], v[8];
        for (int i = 0; i < 8; ++i) { k[i] = (float)(pos * 100 + i); v[i] = -(float)(pos * 100 + i); }
        for (int vis = 0; vis < 2; ++vis) {
            in2_kv_k(&kv, vis, pos)[0] = k[0];
            in2_kv_v(&kv, vis, pos)[0] = v[0];
            in2_kv_store_raw(&kv, vis, k, v);
        }
        in2_kv_tick(&kv);
    }

    /* ring wrap: pos 5 lands in slot 1, and only the newest `window` positions stay resident */
    CHECK(in2_kv_k(&kv, 0, 5) == kv.k + 1 * 8, "pos 5 slot");
    CHECK(in2_kv_resident(&kv, 5) == 0, "pos 5 should have been overwritten (n_stored=10, window=4)");
    CHECK(in2_kv_resident(&kv, 6) == 1, "pos 6 resident");
    CHECK(in2_kv_resident(&kv, 9) == 1, "pos 9 resident");
    CHECK(in2_kv_resident(&kv, 10) == 0, "pos 10 never stored");

    /* visits must not alias: visit 1 has its own rows */
    CHECK(in2_kv_k(&kv, 1, 9) != in2_kv_k(&kv, 0, 9), "visit aliasing");

    /* conv history: hist(1) is the most recent raw row (pos 9), hist(3) two stores before that (pos 7) */
    CHECK(fabsf(in2_kv_hist_k(&kv, 0, 1)[0] - 900.0f) < 1e-3f, "hist back=1 = pos 9, got %g", in2_kv_hist_k(&kv, 0, 1)[0]);
    CHECK(fabsf(in2_kv_hist_k(&kv, 0, 3)[0] - 700.0f) < 1e-3f, "hist back=3 = pos 7, got %g", in2_kv_hist_k(&kv, 0, 3)[0]);
    CHECK(fabsf(in2_kv_hist_v(&kv, 0, 2)[0] + 800.0f) < 1e-3f, "hist v back=2 = -pos 8");

    /* full-attention layer: window 0 means every stored position stays valid, slot = pos */
    in2_kv_t fk;
    CHECK(in2_kv_init(&fk, 1, 8, 0, 16) == 0, "init full");
    for (int pos = 0; pos < 10; ++pos) { in2_kv_k(&fk, 0, pos)[0] = (float)pos; in2_kv_tick(&fk); }
    CHECK(in2_kv_resident(&fk, 0) == 1, "full layer keeps pos 0");
    CHECK(in2_kv_k(&fk, 0, 7) == fk.k + 7 * 8, "full slot = pos");

    /* The correctness crux: running the 4-tap conv one row at a time with cached history must equal running it
     * over the whole sequence at once. This is what breaks in a naive incremental decoder, and it breaks quietly -
     * only the first few keys of each step differ. */
    {
        enum { N = 9, CH = 5 };
        float x[N][CH], w[CH * 4], batch[N][CH], inc[N][CH];
        for (int i = 0; i < N; ++i)
            for (int c = 0; c < CH; ++c) x[i][c] = sinf((float)(i * CH + c) * 0.37f);
        for (int c = 0; c < CH; ++c)
            for (int kk = 0; kk < 4; ++kk) w[c * 4 + kk] = cosf((float)(c * 4 + kk) * 0.21f);

        /* reference: whole-sequence conv, as the engine does today */
        const float scale = 1.0f / sqrtf(4.0f + 1.0f);
        for (int t = 0; t < N; ++t)
            for (int c = 0; c < CH; ++c) {
                float acc = 0.0f;
                for (int tap = 0; tap < 4; ++tap) { const int s = t - tap; if (s < 0) continue; acc += w[c * 4 + tap] * x[s][c]; }
                batch[t][c] = (acc + x[t][c]) * scale;
            }

        /* incremental: one row at a time, history kept exactly the way the cache keeps it */
        in2_kv_t ck;
        CHECK(in2_kv_init(&ck, 1, CH, 0, 16) == 0, "init conv cache");
        for (int t = 0; t < N; ++t) {
            /* window base, oldest first; unfilled slots are zeros, which is the batch version's padding */
            const float *hist = in2_kv_hist_k(&ck, 0, 3);
            in2_kv_conv_row(hist, x[t], CH, w, 4, inc[t]);
            in2_kv_store_raw(&ck, 0, x[t], x[t]);
            in2_kv_tick(&ck);
        }
        float worst = 0.0f;
        for (int t = 0; t < N; ++t)
            for (int c = 0; c < CH; ++c) {
                const float e = fabsf(batch[t][c] - inc[t][c]);
                if (e > worst) worst = e;
            }
        CHECK(worst < 1e-6f, "incremental conv != batch conv, worst |diff| %.3e", worst);
        printf("conv row-vs-batch worst |diff| = %.3e over %d rows\n", worst, N);
        in2_kv_free(&ck);
    }

    /* full-attention layers use the dynamic conv: tap set = stat + a projection of that position's own input.
     * Only the input row is position-local, so the row-at-a-time form must match the batch form exactly. */
    {
        enum { N = 7, CH2 = 6, NH2 = 3, K2 = 4 };
        const int hd = CH2 / NH2;
        float x[N][CH2], stat[CH2 * K2], dw[N][NH2 * K2], batch[N][CH2], inc[N][CH2];
        for (int i = 0; i < N; ++i)
            for (int c = 0; c < CH2; ++c) x[i][c] = cosf((float)(i * 3 + c) * 0.29f);
        for (int c = 0; c < CH2 * K2; ++c) stat[c] = sinf((float)c * 0.11f) * 0.5f;
        for (int t = 0; t < N; ++t)
            for (int h = 0; h < NH2; ++h)
                for (int j = 0; j < K2; ++j) dw[t][h * K2 + j] = cosf((float)(t * 7 + h * 3 + j) * 0.17f);

        const float sc = 1.0f / sqrtf(2.0f * K2 + 1.0f);
        for (int t = 0; t < N; ++t)
            for (int h = 0; h < NH2; ++h)
                for (int d = 0; d < hd; ++d) {
                    const int c = h * hd + d;
                    float acc = 0.0f;
                    for (int j = 0; j < K2; ++j) {
                        const int src = t - j;
                        if (src < 0) continue;
                        acc += (stat[c * K2 + j] + dw[t][h * K2 + j]) * x[src][c];
                    }
                    batch[t][c] = (acc + x[t][c]) * sc;
                }

        in2_kv_t ck2;
        CHECK(in2_kv_init(&ck2, 1, CH2, 0, 16) == 0, "init dyn cache");
        for (int t = 0; t < N; ++t) {
            in2_kv_conv_row_dyn(in2_kv_hist_k(&ck2, 0, 3), x[t], CH2, stat, dw[t], NH2, K2, inc[t]);
            in2_kv_store_raw(&ck2, 0, x[t], x[t]);
            in2_kv_tick(&ck2);
        }
        float worst = 0.0f;
        for (int t = 0; t < N; ++t)
            for (int c = 0; c < CH2; ++c) { const float e = fabsf(batch[t][c] - inc[t][c]); if (e > worst) worst = e; }
        CHECK(worst < 1e-6f, "dynamic conv row-vs-batch worst |diff| %.3e", worst);
        printf("dynamic conv row-vs-batch worst |diff| = %.3e over %d rows\n", worst, N);
        in2_kv_free(&ck2);
    }

    /* ---- per-visit capacity: the layout that makes a long context affordable ----
     * Visit 0 is a full-attention visit sized to the 16-position context; visit 1 is a sliding visit whose ring is
     * only 4 rows. This is the shape the engine uses (8 visits at the context, 64 at the window), reduced so it can
     * be checked by eye: if either the sizing or the ring rule is wrong, one of these fails. */
    {
        in2_kv_t pv;
        int rows[2] = { 16, 4 };
        CHECK(in2_kv_init_pervisit(&pv, 2, 8, 4, rows) == 0, "init per-visit");
        CHECK(in2_kv_bytes(&pv) == (size_t)(16 + 4) * 8u * sizeof(float) * 2u,
              "per-visit reservation should be (16+4) rows, got %zu bytes", in2_kv_bytes(&pv));
        CHECK(pv.n_pos_max == 16, "position bound is the largest visit, got %d", pv.n_pos_max);

        for (int pos = 0; pos < 16; ++pos) {
            float *kp = in2_kv_k(&pv, 0, pos);   /* full visit: every position must be accepted and distinct */
            CHECK(kp != NULL, "full visit rejected pos %d", pos);
            if (kp) kp[0] = (float) pos * 7.0f;
            float *ks = in2_kv_k(&pv, 1, pos);   /* sliding visit: accepts the write, wraps inside 4 rows */
            CHECK(ks != NULL, "sliding visit rejected pos %d", pos);
            if (ks) ks[0] = -(float) pos;
        }
        /* the full visit keeps all 16 positions: no wrap, no aliasing */
        for (int pos = 0; pos < 16; ++pos) {
            const float *kp = in2_kv_k(&pv, 0, pos);
            CHECK(kp && fabsf(kp[0] - (float) pos * 7.0f) < 1e-6f,
                  "full visit lost pos %d (got %g)", pos, kp ? kp[0] : -999.0f);
        }
        /* the sliding visit kept only its last 4 writes: positions 12..15 own slots 0..3, and every older position
         * that hashed to the same slot now reads back the newest writer of that slot - which is exactly what a ring
         * cache does, and what a test has to assert rather than "pos 3 still holds 3". */
        CHECK(fabsf(in2_kv_k(&pv, 1, 15)[0] + 15.0f) < 1e-6f, "sliding pos 15");
        CHECK(fabsf(in2_kv_k(&pv, 1, 12)[0] + 12.0f) < 1e-6f, "sliding pos 12 = slot 0");
        CHECK(fabsf(in2_kv_k(&pv, 1, 3)[0] + 15.0f) < 1e-6f,
              "pos 3 shares slot 3 with pos 15 and must read the newest writer, got %g", in2_kv_k(&pv, 1, 3)[0]);
        /* and past the position bound the accessor refuses rather than wrapping silently */
        CHECK(in2_kv_k(&pv, 0, 16) == NULL, "pos 16 is past n_pos_max and must be refused");
        CHECK(in2_kv_k(&pv, 1, -1) == NULL, "negative pos must be refused");
        in2_kv_free(&pv);
    }

    in2_kv_free(&kv); in2_kv_free(&fk);
    printf(fails ? "KV SELFTEST: %d FAILED\n" : "KV SELFTEST: all passed\n", fails);
    return fails != 0;
}
