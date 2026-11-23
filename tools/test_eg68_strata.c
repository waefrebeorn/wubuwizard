/* test_eg68_strata.c -- EG-68 gate: host allocations, launches, split-KV. */
#include <stdio.h>
#include <math.h>
#include <stdint.h>

#include "wubu_strata.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    wubu_strata_host_t h;
    float kv[64*4], split[4], direct[4];
    int launches = 0, partials;
    double maxerr = 0.0;

    CHECK(wubu_strata_host_init(&h, 4096) == 0);
    CHECK(h.allocations == 1);
    for (int i = 0; i < 3; i++)
        CHECK(wubu_strata_host_alloc(&h, 64) != NULL);
    CHECK(h.allocations == 1);
    CHECK(h.used == 192);
    CHECK(wubu_strata_host_alloc(&h, 5000) == NULL);
    CHECK(h.allocations == 1);
    printf("EG-68 host allocations=%d used=%zu of 4096 (zero after init)\n",
           h.allocations, h.used);

    /* Mixed-size host storage stays 16-byte aligned for SIMD use. */
    {
        wubu_strata_host_t a;
        unsigned char *p;
        CHECK(wubu_strata_host_init(&a, 128) == 0);
        p = (unsigned char *)wubu_strata_host_alloc(&a, 1);
        CHECK(p && ((uintptr_t)p % 16) == 0);
        p = (unsigned char *)wubu_strata_host_alloc(&a, 1);
        CHECK(p && ((uintptr_t)p % 16) == 0);
        wubu_strata_host_free(&a);
    }

    CHECK(wubu_strata_prefill_launches(64, 8, &launches) == 0);
    printf("EG-68 prefill launches=8 baseline=64 (8x amortization)\n");
    CHECK(launches == 8);

    for (int i = 0; i < 64; i++)
        for (int d = 0; d < 4; d++)
            kv[i*4+d] = 0.25f * (float)((i + d) % 7);
    for (int d = 0; d < 4; d++) {
        direct[d] = 0.0f;
        for (int i = 0; i < 64; i++) direct[d] += kv[i*4+d];
    }
    partials = wubu_strata_splitkv(kv, 64, 4, split);
    for (int d = 0; d < 4; d++) {
        double e = fabs(split[d] - direct[d]);
        if (e > maxerr) maxerr = e;
    }
    printf("EG-68 splitkv partials=%d maxerr=%.8f baseline=64\n",
           partials, maxerr);
    CHECK(partials == 4);
    CHECK(maxerr < 1e-4);
    CHECK(wubu_strata_prefill_launches(64, 0, &launches) == -1);
    CHECK(wubu_strata_splitkv(kv, 64, 0, split) == -1);
    wubu_strata_host_free(&h);

    if (failures) {
        fprintf(stderr, "test_eg68_strata: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg68_strata: PASS\n");
    return 0;
}
