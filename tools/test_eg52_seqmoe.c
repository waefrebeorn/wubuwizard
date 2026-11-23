/* test_eg52_seqmoe.c -- EG-52 gate: sequence-model expert hit rate. */
#include <stdio.h>

#include "wubu_seqmoe.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define EXPERTS 8
#define CAPACITY 3
#define CYCLES 8
#define TEST_CYCLES 3

int main(void) {
    int train[CYCLES * 6], test[TEST_CYCLES * 6];
    wubu_seqmoe_t m;
    int top[3], seq, lru, stat;

    for (int c = 0; c < CYCLES; c++)
        for (int j = 0; j < 6; j++) train[c * 6 + j] = j + 1;
    for (int c = 0; c < TEST_CYCLES; c++)
        for (int j = 0; j < 6; j++) test[c * 6 + j] = j + 1;

    CHECK(wubu_seqmoe_init(&m, EXPERTS, CAPACITY) == 0);
    CHECK(wubu_seqmoe_observe(&m, train, CYCLES * 6) == 0);
    CHECK(wubu_seqmoe_plan(&m, 5, 6, top) == CAPACITY);
    printf("EG-52 next experts after [5 6]: [%d %d %d]\n", top[0], top[1], top[2]);
    CHECK(top[0] == 1);

    seq = wubu_seqmoe_hits(&m, test, TEST_CYCLES * 6, 5, 6);
    lru = wubu_lru_hits(EXPERTS, CAPACITY, test, TEST_CYCLES * 6);
    stat = wubu_static_hits(EXPERTS, CAPACITY, test, TEST_CYCLES * 6);
    printf("EG-52 hits seq=%d/18 lru=%d/18 static=%d/18\n", seq, lru, stat);
    CHECK(seq == TEST_CYCLES * 6);
    CHECK(seq > lru);
    CHECK(seq >= stat + 6);

    CHECK(wubu_seqmoe_init(&m, 8, 8) == -1);
    CHECK(wubu_seqmoe_observe(&m, train, 2) == -1);
    CHECK(wubu_seqmoe_plan(&m, -1, 6, top) == -1);
    CHECK(wubu_seqmoe_hits(&m, test, 1, -1, 6) == -1);
    CHECK(wubu_lru_hits(8, 3, test, 0) == -1);
    CHECK(wubu_static_hits(8, 3, test, 0) == -1);
    CHECK(wubu_seqmoe_free(&m) == 0);

    if (failures) {
        fprintf(stderr, "test_eg52_seqmoe: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg52_seqmoe: PASS\n");
    return 0;
}
