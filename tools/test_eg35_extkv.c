/* test_eg35_extkv.c -- EG-35 gate: UMA-corrected post-warmup KV sizing. */
#include <stdio.h>

#include "wubu_ext_kv.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define GiB(x) ((size_t)(x) * 1024u * 1024u * 1024u)

int main(void) {
    wubu_ext_kv_config_t cfg = {
        GiB(6), GiB(4), GiB(256) / 1024, GiB(1), 1, 80, GiB(128) / 1024
    };
    size_t samples[3] = { GiB(2), (size_t)(GiB(9) / 5), GiB(19) / 10 };
    wubu_ext_kv_t uma, plain, shortfall, warm_min;

    CHECK(wubu_ext_kv_plan(&cfg, samples, 3, &uma) == 0);
    printf("EG-35 UMA free=%.2fGiB kv=%.2fGiB accepted=%d\n",
           (double)uma.effective_free / GiB(1),
           (double)uma.kv_bytes / GiB(1), uma.accepted);
    CHECK(uma.accepted == 1);

    cfg.uma = 0;
    CHECK(wubu_ext_kv_plan(&cfg, samples, 3, &plain) == 0);
    printf("EG-35 plain free=%.2fGiB kv=%.2fGiB accepted=%d\n",
           (double)plain.effective_free / GiB(1),
           (double)plain.kv_bytes / GiB(1), plain.accepted);
    CHECK(plain.accepted == 1 && plain.kv_bytes > uma.kv_bytes);

    /* Page cache alone consumes the whole UMA free pool -> refuse. */
    cfg.uma = 1;
    cfg.page_cache = GiB(2);
    CHECK(wubu_ext_kv_plan(&cfg, samples, 3, &shortfall) == 0);
    printf("EG-35 shortfall accepted=%d reason=%d\n",
           shortfall.accepted, shortfall.reason);
    CHECK(shortfall.accepted == 0 && shortfall.reason == 3);

    /* A low warmup sample caps the budget even if later samples look better. */
    cfg.page_cache = GiB(1);
    cfg.uma = 0;
    samples[0] = GiB(3) / 10;
    CHECK(wubu_ext_kv_plan(&cfg, samples, 3, &warm_min) == 0);
    printf("EG-35 warmup-min free=%.3fGiB reason=%d\n",
           (double)warm_min.effective_free / GiB(1), warm_min.reason);
    CHECK(warm_min.effective_free < plain.effective_free / 4);
    CHECK(warm_min.accepted == 0 && warm_min.reason == 2);

    CHECK(wubu_ext_kv_plan(NULL, samples, 3, &uma) == -1);
    CHECK(wubu_ext_kv_plan(&cfg, NULL, 3, &uma) == -1);
    CHECK(wubu_ext_kv_plan(&cfg, samples, 0, &uma) == -1);

    if (failures) {
        fprintf(stderr, "test_eg35_extkv: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg35_extkv: PASS\n");
    return 0;
}
