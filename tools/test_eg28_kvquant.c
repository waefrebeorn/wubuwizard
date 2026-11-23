/* test_eg28_kvquant.c -- EG-28 gate: lazy per-layer KV quantization. */
#include <stdio.h>

#include "wubu_kv_lazy.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    wubu_kv_layer_t layers[4] = {
        { 100, 25, 10.0f, 0 },
        { 100, 25,  1.0f, 0 },
        { 200, 50, 30.0f, 0 },
        { 100, 25, 50.0f, 0 }
    };
    wubu_kv_lazy_result_t r;
    wubu_kv_lazy_t *k;

    k = wubu_kv_lazy_create(layers, 4, 350);
    CHECK(k != NULL);
    if (!k) return 1;
    CHECK(wubu_kv_lazy_apply(k, &r) == 0);
    printf("EG-28 converted=%d saved=%llu after=%llu quality=%.1f\n",
           r.converted, (unsigned long long)r.bytes_saved,
           (unsigned long long)r.bytes_after, r.quality_cost);
    CHECK(r.converted == 2 && r.bytes_saved == 150);
    CHECK(r.bytes_after == 350 && r.within_budget == 1);
    CHECK(r.quality_cost == 11.0f);
    CHECK(wubu_kv_lazy_layer(k, 1)->quantized == 1);
    CHECK(wubu_kv_lazy_layer(k, 0)->quantized == 1);
    CHECK(wubu_kv_lazy_layer(k, 2)->quantized == 0);
    CHECK(wubu_kv_lazy_layer(k, 3)->quantized == 0);

    /* Second call is a no-op once the budget is satisfied. */
    CHECK(wubu_kv_lazy_apply(k, &r) == 0);
    CHECK(r.converted == 0 && r.bytes_saved == 0);

    /* Severe budget converts everything and reports the shortfall honestly. */
    {
        wubu_kv_lazy_t *tight = wubu_kv_lazy_create(layers, 4, 100);
        CHECK(tight != NULL);
        if (tight) {
            CHECK(wubu_kv_lazy_apply(tight, &r) == 0);
            CHECK(r.converted == 4 && r.within_budget == 0);
            printf("EG-28 tight after=%llu within=%d\n",
                   (unsigned long long)r.bytes_after, r.within_budget);
            wubu_kv_lazy_free(tight);
        }
    }
    /* Under budget: no conversion at all. */
    {
        wubu_kv_lazy_t *roomy = wubu_kv_lazy_create(layers, 4, 600);
        CHECK(roomy != NULL);
        if (roomy) {
            CHECK(wubu_kv_lazy_apply(roomy, &r) == 0);
            CHECK(r.converted == 0 && r.within_budget == 1);
            wubu_kv_lazy_free(roomy);
        }
    }

    CHECK(wubu_kv_lazy_create(NULL, 4, 350) == NULL);
    CHECK(wubu_kv_lazy_layer(k, 9) == NULL);
    CHECK(wubu_kv_lazy_apply(NULL, &r) == -1);

    wubu_kv_lazy_free(k);
    if (failures) {
        fprintf(stderr, "test_eg28_kvquant: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg28_kvquant: PASS\n");
    return 0;
}
