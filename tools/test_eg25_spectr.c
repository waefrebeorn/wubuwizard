/* test_eg25_spectr.c -- EG-25e gate: SpecTr OT candidate selection. */
#include <stdio.h>
#include <math.h>

#include "wubu_spectr_select.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    double p[6] = {0.05, 0.25, 0.10, 0.40, 0.15, 0.05};
    int out[2];
    double cost, best = 2.0;
    CHECK(wubu_spectr_select(p, 6, 2, out) == 2);
    CHECK(out[0] == 3 && out[1] == 1);
    cost = wubu_spectr_transport_cost(p, out, 2);
    CHECK(fabs(cost - 0.35) < 1e-9);
    printf("EG-25e selected=[%d %d] cost=%.2f\n", out[0], out[1], cost);

    for (int i = 0; i < 6; ++i)
        for (int j = i + 1; j < 6; ++j) {
            int pair[2] = {i, j};
            double c = wubu_spectr_transport_cost(p, pair, 2);
            if (c < best) best = c;
        }
    CHECK(fabs(best - cost) < 1e-9);
    printf("EG-25e brute-force best pair cost=%.2f (matches selection)\n", best);

    CHECK(wubu_spectr_select(NULL, 6, 2, out) == -1);
    CHECK(wubu_spectr_select(p, 6, 0, out) == -1);
    CHECK(wubu_spectr_select(p, 6, 7, out) == -1);
    CHECK(wubu_spectr_select(p, 6, 2, NULL) == -1);
    CHECK(wubu_spectr_transport_cost(p, NULL, 2) < 0.0);

    if (failures) { fprintf(stderr, "test_eg25_spectr: FAIL (%d)\n", failures); return 1; }
    printf("test_eg25_spectr: PASS\n");
    return 0;
}
