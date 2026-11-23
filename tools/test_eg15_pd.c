/* test_eg15_pd.c -- EG-15 gate: disaggregated prefill/decode handoff.
 * Links the three owning modules directly: pd_serve policy, pd_split cost
 * model and the real localhost KV transfer buffer. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wubu_pd_serve.h"
#include "wubu_pd_split.h"
#include "wubu_kv_transfer.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    int pt = -1, dt = -1;
    wubu_pd_split_t *s;
    wubu_kv_transfer_t *x;
    char path[512];
    const char *tmp = getenv("TMPDIR");
    unsigned char src[64], dst[64];

    /* AB01-AB06 policy gates. */
    CHECK(wubu_pd_split(2, 3) == 1);
    CHECK(wubu_pd_split(0, 3) == 0);
    CHECK(wubu_kv_handoff_ready(127, 128) == 0);
    CHECK(wubu_kv_handoff_ready(128, 128) == 1);
    CHECK(wubu_pull_route(3, 4) == 1);
    CHECK(wubu_pull_route(4, 4) == 0);
    wubu_hetero_map(&pt, &dt);
    CHECK(pt == 0 && dt == 1);
    CHECK(wubu_kv_xfer_fits(1e6, 1e9, 0.01) == 1);
    CHECK(wubu_kv_xfer_fits(1e6, 1e9, 1e-9) == 0);
    CHECK(wubu_prefix_reuse(0xabc123u, 0xabc123u) == 1);
    CHECK(wubu_prefix_reuse(0xabc123u, 0xabc124u) == 0);

    /* AC01-AC03 mixture-of-depths gates. */
    CHECK(wubu_mod_execute(0.8f, 0.5f) == 1);
    CHECK(wubu_mod_execute(0.2f, 0.5f) == 0);
    CHECK(wubu_mod_capacity(12, 4) == 4);
    CHECK(wubu_mod_capacity(2, 4) == 2);
    CHECK(wubu_early_exit(0.9f, 0.8f) == 1);
    CHECK(wubu_early_exit(0.5f, 0.8f) == 0);

    /* Transfer cost model and routing. */
    s = wubu_pd_split_create(2, 4, 0.05);
    CHECK(s != NULL);
    if (s) {
        double ms = wubu_pd_kv_transfer_ms(s, 1024, 32, 8, 128, 16);
        int q[4] = { 5, 2, 7, 9 };
        printf("EG-15 kv_transfer_ms=%.6f route=%d mode=%d\n",
               ms, wubu_pd_route_decode(s, q),
               wubu_pd_transfer_mode(s, 0.2, 0.9));
        CHECK(ms > 0.0);
        CHECK(wubu_pd_route_decode(s, q) == 1);
        CHECK(wubu_pd_transfer_mode(s, 0.2, 0.9) == 1);
        CHECK(wubu_pd_transfer_mode(s, 0.9, 0.2) == 0);
        wubu_pd_split_free(s);
    }

    /* Real localhost KV handoff: prefill writes, decode reads the same slot. */
    snprintf(path, sizeof(path), "%s/wubu_eg15_kv.bin",
             tmp && *tmp ? tmp : ".");
    x = wubu_kv_transfer_create(path, 4096);
    CHECK(x != NULL);
    if (x) {
        for (int i = 0; i < 64; i++) src[i] = (unsigned char)(i * 7 + 3);
        memset(dst, 0, sizeof(dst));
        CHECK(wubu_kv_transfer_put(x, 0, src, sizeof(src)) == 0);
        CHECK(wubu_kv_transfer_used(x) == 64);
        CHECK(wubu_kv_transfer_get(x, 0, dst, sizeof(dst)) == 0);
        CHECK(memcmp(src, dst, sizeof(src)) == 0);
        CHECK(wubu_kv_transfer_put(x, 4000, src, sizeof(src)) == -1);
        wubu_kv_transfer_free(x);
    }

    if (failures) {
        fprintf(stderr, "test_eg15_pd: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg15_pd: PASS\n");
    return 0;
}
