/* test_eg105_loop_place.c -- EG-105 gate: loop-aware expert placement. */
#include <stdio.h>

#include "wubu_loop_place.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define MiB (1024ULL * 1024ULL)

int main(void) {
    wubu_loop_cfg_t cfg = {24, 6, 17, 3, 4608ULL * MiB, 64ULL * MiB};
    int plan[24], baseline[24];
    uint64_t opt_reads, base_reads;
    CHECK(wubu_loop_place_plan(&cfg, plan) == 0);
    for (int i = 6; i <= 17; ++i) CHECK(plan[i] == 0);       /* loop core on GPU */
    for (int i = 0; i < 6; ++i) CHECK(plan[i] == 1);          /* non-loop CPU */
    for (int i = 18; i < 24; ++i) CHECK(plan[i] == 1);
    CHECK(wubu_loop_place_loop_layers(&cfg) == 12);

    opt_reads = wubu_loop_place_cpu_reads(&cfg, plan);
    for (int i = 0; i < 24; ++i) baseline[i] = i < 12 ? 1 : 0; /* -ncmoe 12 */
    base_reads = wubu_loop_place_cpu_reads(&cfg, baseline);
    CHECK(opt_reads == 12ULL * 64ULL * MiB);          /* non-loop CPU once each */
    CHECK(base_reads == (6ULL + 6ULL * 3ULL) * 64ULL * MiB);
    printf("EG-105 loop_layers=12 baseline_cpu_reads=%llu MiB optimized=%llu MiB "
           "(loop re-reads eliminated=%llu MiB)\n",
           (unsigned long long)(base_reads / MiB),
           (unsigned long long)(opt_reads / MiB),
           (unsigned long long)((base_reads - opt_reads) / MiB));

    /* Loop core larger than the GPU hot budget must be refused. */
    {
        wubu_loop_cfg_t tight = cfg;
        tight.gpu_budget = 512ULL * MiB;
        CHECK(wubu_loop_place_plan(&tight, plan) == -1);
    }
    CHECK(wubu_loop_place_plan(NULL, plan) == -1);
    CHECK(wubu_loop_place_plan(&cfg, NULL) == -1);
    CHECK(wubu_loop_place_cpu_reads(&cfg, NULL) == 0);

    if (failures) { fprintf(stderr, "test_eg105_loop_place: FAIL (%d)\n", failures); return 1; }
    printf("test_eg105_loop_place: PASS\n");
    return 0;
}
