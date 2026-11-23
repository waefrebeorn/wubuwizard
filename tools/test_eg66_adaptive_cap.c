/* test_eg66_adaptive_cap.c -- EG-66b gate: adaptive draft cap + KV budget. */
#include <stdio.h>

#include "wubu_adaptive_cap.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    int cap;
    CHECK(wubu_adaptive_cap_update(4, 4, 4, 16, 1, 8) == 5);
    CHECK(wubu_adaptive_cap_update(4, 0, 8, 16, 1, 8) == 3);
    CHECK(wubu_adaptive_cap_update(4, 4, 8, 16, 1, 8) == 4);
    CHECK(wubu_adaptive_cap_update(4, 4, 4, 3, 1, 8) == 3);
    CHECK(wubu_adaptive_cap_update(8, 4, 4, 16, 1, 8) == 8);
    cap = wubu_adaptive_cap_update(1, 0, 8, 16, 1, 8);
    CHECK(cap == 1);
    printf("EG-66b cap 4->5 (full accept), ->3 (reject), held at 4 (half), "
           "clamped by kv=3 and max=8\n");

    CHECK(wubu_adaptive_cap_update(4, 4, 4, 0, 1, 8) == -1);
    CHECK(wubu_adaptive_cap_update(4, 5, 4, 16, 1, 8) == -1);
    CHECK(wubu_adaptive_cap_update(4, 4, 0, 16, 1, 8) == -1);
    CHECK(wubu_adaptive_cap_update(4, 4, 4, 16, 0, 8) == -1);
    CHECK(wubu_adaptive_cap_update(4, 4, 4, 16, 1, 0) == -1);

    if (failures) { fprintf(stderr, "test_eg66_adaptive_cap: FAIL (%d)\n", failures); return 1; }
    printf("test_eg66_adaptive_cap: PASS\n");
    return 0;
}
