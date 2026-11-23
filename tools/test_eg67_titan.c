/* test_eg67_titan.c -- EG-67 gate: tiered experts under a 6 GB cap. */
#include <stdio.h>

#include "wubu_titan.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define GB (size_t)1 << 30

int main(void) {
    wubu_titan_expert_t e[8];
    size_t peak = 0, naive = 0;
    int streamed = 0;
    const int tokens[6] = { 1, 2, 1, 3, 2, 1 };
    const int tiers[6] = { 0, 0, 0, 1, 1, 0 };
    int hits[6];

    for (int i = 0; i < 8; i++) {
        e[i].id = i;
        e[i].bytes = GB;
        e[i].tier = i < 4 ? 0 : (i < 6 ? 1 : 2);
        naive += e[i].bytes;
    }
    CHECK(wubu_titan_plan(e, 8, 6 * GB, &peak, &streamed) == 0);
    printf("EG-67 experts=8 naive=%zu GB peak=%zu GB streamed=%d cap=6 GB\n",
           naive >> 30, peak >> 30, streamed);
    CHECK(naive > 6 * GB);
    CHECK(peak <= 6 * GB);
    CHECK(streamed == 4);
    CHECK(wubu_titan_prefix(tokens, tiers, 6, hits) == 2);

    CHECK(wubu_titan_plan(e, 8, 3 * GB, &peak, &streamed) == -1);
    CHECK(wubu_titan_plan(NULL, 8, 6 * GB, &peak, &streamed) == -1);
    CHECK(wubu_titan_prefix(tokens, tiers, 0, hits) == -1);

    if (failures) {
        fprintf(stderr, "test_eg67_titan: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg67_titan: PASS\n");
    return 0;
}
