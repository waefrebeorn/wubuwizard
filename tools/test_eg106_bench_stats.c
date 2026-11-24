/* test_eg106_bench_stats.c -- EG-106 gate: benchmark statistics + regression. */
#include <stdio.h>
#include <math.h>

#include "wubu_bench_stats.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    double base[5] = {126.0, 125.0, 127.0, 126.0, 126.0};
    double fast[5] = {196.0, 195.0, 197.0, 196.0, 196.0};
    double slow[5] = {100.0, 99.0, 101.0, 100.0, 100.0};
    double noise[5] = {126.0, 124.0, 128.0, 126.0, 126.0};
    CHECK(fabs(wubu_bench_mean(base, 5) - 126.0) < 1e-9);
    CHECK(fabs(wubu_bench_sd(base, 5) - sqrt(0.5)) < 1e-9);
    CHECK(fabs(wubu_bench_cv(base, 5) - sqrt(0.5) / 126.0) < 1e-9);
    CHECK(wubu_bench_regressed(base, fast, 5) == 0);
    CHECK(wubu_bench_regressed(base, slow, 5) == 1);
    CHECK(wubu_bench_regressed(base, noise, 5) == 0);
    printf("EG-106 base=%.2f sd=%.4f cv=%.5f fast_regressed=%d slow=%d noise=%d\n",
           wubu_bench_mean(base, 5), wubu_bench_sd(base, 5), wubu_bench_cv(base, 5),
           wubu_bench_regressed(base, fast, 5),
           wubu_bench_regressed(base, slow, 5),
           wubu_bench_regressed(base, noise, 5));
    CHECK(wubu_bench_mean(NULL, 5) < 0.0);
    CHECK(wubu_bench_sd(base, 1) < 0.0);
    CHECK(wubu_bench_cv(base, 0) < 0.0);
    CHECK(wubu_bench_regressed(NULL, fast, 5) == -1);
    CHECK(wubu_bench_regressed(base, fast, 0) == -1);
    if (failures) { fprintf(stderr, "test_eg106_bench_stats: FAIL (%d)\n", failures); return 1; }
    printf("test_eg106_bench_stats: PASS\n");
    return 0;
}
