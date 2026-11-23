/* test_eg25_relaxed.c -- EG-25d gate: relaxed speculative verification. */
#include <stdio.h>

#include "wubu_relaxed_verify.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    double p[3] = {0.90, 0.80, 0.70};
    double q[3] = {0.85, 0.70, 0.65};
    int exact[3], relaxed[3];
    int ne, nr, extra;
    CHECK(wubu_relaxed_verify_chain(p, q, 3, 0.0, exact) == 0);
    CHECK(wubu_relaxed_verify_chain(p, q, 3, 0.1, relaxed) == 0);
    CHECK(exact[0] == 0);                    /* 0.85 < 0.90 */
    CHECK(relaxed[0] == 1);                  /* 0.944 >= 0.90 */
    CHECK(relaxed[1] == 0);                  /* 0.875 < 0.90 */
    ne = wubu_relaxed_accepted(exact, 3);
    nr = wubu_relaxed_accepted(relaxed, 3);
    extra = wubu_relaxed_extra(exact, relaxed, 3);
    CHECK(ne == 0 && nr == 1 && extra == 1);
    printf("EG-25d exact=%d relaxed=%d extra=%d delta=0.10\n", ne, nr, extra);
    CHECK(wubu_relaxed_accept(0.0, 0.5, 0.1) == -1);
    CHECK(wubu_relaxed_verify_chain(p, q, 3, -0.1, relaxed) == -1);
    CHECK(wubu_relaxed_accept(0.5, 0.5, 0.0) == 1);
    CHECK(wubu_relaxed_accept(0.5, 0.4, 0.0) == 0);
    CHECK(wubu_relaxed_accept(0.5, 0.4, 0.2) == 1);
    CHECK(wubu_relaxed_accept(0.5, 0.4, 1.1) == -1);
    CHECK(wubu_relaxed_verify_chain(p, q, 0, 0.1, relaxed) == -1);
    CHECK(wubu_relaxed_verify_chain(NULL, q, 3, 0.1, relaxed) == -1);
    CHECK(wubu_relaxed_accepted(NULL, 3) == -1);
    CHECK(wubu_relaxed_extra(exact, NULL, 3) == -1);
    printf("EG-25d boundary and invalid-input checks pass\n");

    if (failures) { fprintf(stderr, "test_eg25_relaxed: FAIL (%d)\n", failures); return 1; }
    printf("test_eg25_relaxed: PASS\n");
    return 0;
}
