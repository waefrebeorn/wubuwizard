/* test_eg110_residency.c -- EG-110 gate: DRAM/Flash residency planner. */
#include <stdio.h>

#include "wubu_residency.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define MiB (1024ULL * 1024ULL)

int main(void) {
    wubu_residency_layer_t layers[4] = {
        {1, 128ULL * MiB}, {0, 128ULL * MiB}, {1, 64ULL * MiB}, {0, 64ULL * MiB}
    };
    int plan[4];
    uint64_t dram;
    /* hot 192 MiB + 128 MiB staging buffer for the largest streamed layer. */
    CHECK(wubu_residency_plan(layers, 4, 320ULL * MiB, plan) == 0);
    CHECK(plan[0] == 1 && plan[1] == 0 && plan[2] == 1 && plan[3] == 0);
    dram = wubu_residency_dram_bytes(layers, plan, 4);
    CHECK(dram == 192ULL * MiB);
    CHECK(wubu_residency_flash_bytes(layers, plan, 4) == 192ULL * MiB);
    printf("EG-110 plan=[%d %d %d %d] dram=%llu MiB flash=%llu MiB\n",
           plan[0], plan[1], plan[2], plan[3],
           (unsigned long long)(dram / MiB),
           (unsigned long long)(wubu_residency_flash_bytes(layers, plan, 4) / MiB));

    /* Tight DRAM budget must fail closed rather than silently exceed it. */
    {
        wubu_residency_layer_t tiny[1] = {{0, 128ULL * MiB}};
        CHECK(wubu_residency_plan(tiny, 1, 64ULL * MiB, plan) == -1);
    }
    /* 256 MiB cannot stage a 128 MiB streamed layer beside 192 MiB of hot weights. */
    CHECK(wubu_residency_plan(layers, 4, 256ULL * MiB, plan) == -1);
    CHECK(wubu_residency_plan(NULL, 4, 320ULL * MiB, plan) == -1);
    CHECK(wubu_residency_plan(layers, 4, 320ULL * MiB, NULL) == -1);
    CHECK(wubu_residency_dram_bytes(layers, NULL, 4) == 0);

    if (failures) { fprintf(stderr, "test_eg110_residency: FAIL (%d)\n", failures); return 1; }
    printf("test_eg110_residency: PASS\n");
    return 0;
}
