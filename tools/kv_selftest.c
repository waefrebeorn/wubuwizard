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

    in2_kv_free(&kv); in2_kv_free(&fk);
    printf(fails ? "KV SELFTEST: %d FAILED\n" : "KV SELFTEST: all passed\n", fails);
    return fails != 0;
}
