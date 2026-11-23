/* test_eg97_reclaim_margin.c -- EG-97 gate: overflow-class memory margin. */
#include <stdio.h>
#include <math.h>

#include "wubu_reclaim_margin.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define GiB (1024ull * 1024ull * 1024ull)
#define MiB (1024ull * 1024ull)

int main(void) {
    wubu_reclaim_margin_t *p = wubu_reclaim_margin_create(
        0.10, 0.20, 0.25, 0.15, 64 * MiB);
    CHECK(p != NULL);

    /* Margin follows what losing memory costs on this kernel. */
    CHECK(wubu_reclaim_margin_bytes(p, GiB, WUBU_OVERFLOW_COMPRESS) == (size_t)(0.10 * GiB));
    CHECK(wubu_reclaim_margin_bytes(p, GiB, WUBU_OVERFLOW_SWAP) == (size_t)(0.20 * GiB));
    CHECK(wubu_reclaim_margin_bytes(p, GiB, WUBU_OVERFLOW_KILL) == 256 * MiB);
    CHECK(wubu_reclaim_margin_bytes(p, GiB, WUBU_OVERFLOW_UNKNOWN) == (size_t)(0.15 * GiB));
    CHECK(wubu_reclaim_margin_bytes(p, GiB, WUBU_OVERFLOW_NONE) == 0);
    printf("EG-97 margins compress=%zu swap=%zu kill=%zu unknown=%zu none=0 MiB\n",
           wubu_reclaim_margin_bytes(p, GiB, WUBU_OVERFLOW_COMPRESS) / MiB,
           wubu_reclaim_margin_bytes(p, GiB, WUBU_OVERFLOW_SWAP) / MiB,
           wubu_reclaim_margin_bytes(p, GiB, WUBU_OVERFLOW_KILL) / MiB,
           wubu_reclaim_margin_bytes(p, GiB, WUBU_OVERFLOW_UNKNOWN) / MiB);

    CHECK(wubu_reclaim_residency(p, GiB, WUBU_OVERFLOW_KILL) == GiB - 256 * MiB);

    /* The minimum floor wins on small budgets. */
    CHECK(wubu_reclaim_margin_bytes(p, 100 * MiB, WUBU_OVERFLOW_COMPRESS) == 64 * MiB);
    printf("EG-97 min floor: 100 MiB budget -> %zu MiB margin\n",
           wubu_reclaim_margin_bytes(p, 100 * MiB, WUBU_OVERFLOW_COMPRESS) / MiB);

    /* Only routine reclaim is cheap; Unknown never inherits a verdict. */
    CHECK(wubu_reclaim_is_cheap(WUBU_OVERFLOW_COMPRESS) == 1);
    CHECK(wubu_reclaim_is_cheap(WUBU_OVERFLOW_REFAULT) == 1);
    CHECK(wubu_reclaim_is_cheap(WUBU_OVERFLOW_SWAP) == 0);
    CHECK(wubu_reclaim_is_cheap(WUBU_OVERFLOW_KILL) == 0);
    CHECK(wubu_reclaim_is_cheap(WUBU_OVERFLOW_UNKNOWN) == 0);
    CHECK(wubu_reclaim_is_cheap(WUBU_OVERFLOW_NONE) == 0);
    wubu_reclaim_margin_set_refault(p, 0.05);
    CHECK(wubu_reclaim_margin_bytes(p, 2 * GiB, WUBU_OVERFLOW_REFAULT) == (size_t)(0.05 * 2 * GiB));
    CHECK(wubu_reclaim_residency(p, 32 * MiB, WUBU_OVERFLOW_KILL) == 0);
    printf("EG-97 cheap={compress,refault}; refault override=%.2f; tiny budget residency=0\n", 0.05);

    /* Out-of-range and NaN fractions are rejected by status. */
    CHECK(wubu_reclaim_margin_create(1.5, 0.2, 0.25, 0.15, 0) == NULL);
    CHECK(wubu_reclaim_margin_create(0.1, -0.2, 0.25, 0.15, 0) == NULL);
    CHECK(wubu_reclaim_margin_create(NAN, 0.2, 0.25, 0.15, 0) == NULL);
    CHECK(wubu_reclaim_margin_bytes(NULL, GiB, WUBU_OVERFLOW_SWAP) == 0);

    wubu_reclaim_margin_free(p);
    if (failures) { fprintf(stderr, "test_eg97_reclaim_margin: FAIL (%d)\n", failures); return 1; }
    printf("test_eg97_reclaim_margin: PASS\n");
    return 0;
}
