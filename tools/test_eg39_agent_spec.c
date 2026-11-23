/* test_eg39_agent_spec.c -- EG-39c gate: adaptive agent speculation. */
#include <stdio.h>

#include "wubu_agent_spec.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    int d;
    CHECK(wubu_agent_spec_depth(1, 1, 4, 16, 4) == 3);
    CHECK(wubu_agent_spec_depth(1, 1, 0, 16, 4) == 1);
    CHECK(wubu_agent_spec_depth(2, 1, 4, 16, 4) == 1);  /* turn reset */
    CHECK(wubu_agent_spec_depth(1, 1, 4, 2, 4) == 2);   /* token bound */
    CHECK(wubu_agent_spec_depth(1, 1, 4, 16, 2) == 2);  /* configured max */
    d = wubu_agent_spec_depth(1, 1, 2, 16, 4);
    CHECK(d == 2);
    printf("EG-39c depth streak4=%d streak0=1 turn-reset=1 token-clamp=2\n", d);

    CHECK(wubu_agent_spec_depth(0, 1, 4, 16, 4) == -1);
    CHECK(wubu_agent_spec_depth(1, 1, -1, 16, 4) == -1);
    CHECK(wubu_agent_spec_depth(1, 1, 4, -1, 4) == -1);
    CHECK(wubu_agent_spec_depth(1, 1, 4, 16, 0) == -1);

    if (failures) { fprintf(stderr, "test_eg39_agent_spec: FAIL (%d)\n", failures); return 1; }
    printf("test_eg39_agent_spec: PASS\n");
    return 0;
}
