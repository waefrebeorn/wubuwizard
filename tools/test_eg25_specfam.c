/* test_eg25_specfam.c -- EG-25a gate: Sequoia-style draft-length sizing. */
#include <stdio.h>
#include <math.h>

#include "wubu_spec_size.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    wubu_spec_size_t lo, hi, perfect, costly;
    CHECK(wubu_spec_size(8, 0.2, 0.1, 1.0, &lo) == 0);
    CHECK(wubu_spec_size(8, 0.9, 0.1, 1.0, &hi) == 0);
    CHECK(wubu_spec_size(8, 1.0, 0.1, 1.0, &perfect) == 0);
    CHECK(wubu_spec_size(8, 0.9, 2.0, 1.0, &costly) == 0);

    printf("EG-25 p=0.2 k=%d (%.3f tok/unit)\n", lo.k, lo.throughput);
    printf("EG-25 p=0.9 k=%d (%.3f tok/unit)\n", hi.k, hi.throughput);
    printf("EG-25 p=1.0 k=%d (%.3f tok/unit)\n", perfect.k, perfect.throughput);
    printf("EG-25 expensive draft k=%d (%.3f tok/unit)\n",
           costly.k, costly.throughput);

    CHECK(lo.k >= 1 && lo.k <= 8);
    CHECK(hi.k > lo.k);
    CHECK(perfect.k == 8);
    CHECK(costly.k < hi.k);
    CHECK(lo.throughput > 0.0 && hi.throughput > lo.throughput);

    /* Reported expected tokens must match the truncated geometric formula. */
    {
        double e = (1.0 - pow(0.9, hi.k)) / 0.1;
        CHECK(fabs(e - hi.expected) < 1e-9);
    }

    CHECK(wubu_spec_size(0, 0.9, 0.1, 1.0, &lo) == -1);
    CHECK(wubu_spec_size(8, 0.0, 0.1, 1.0, &lo) == -1);
    CHECK(wubu_spec_size(8, 1.1, 0.1, 1.0, &lo) == -1);
    CHECK(wubu_spec_size(8, 0.9, 0.1, 0.0, &lo) == -1);
    CHECK(wubu_spec_size(8, 0.9, 0.1, 1.0, NULL) == -1);

    if (failures) {
        fprintf(stderr, "test_eg25_specfam: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg25_specfam: PASS\n");
    return 0;
}
