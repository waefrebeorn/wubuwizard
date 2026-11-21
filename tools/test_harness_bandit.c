/* test_harness_bandit.c -- gate for LP-04 (repair-arm bandit). */
#include <stdio.h>
#include <math.h>
#include "wubu_harness_bandit.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)
#define WUBU_NEAR(a, b) (fabs((a) - (b)) < 1e-9)

int main(void)
{
    wubu_harness_bandit_t *b = wubu_harness_bandit_create(2);
    CHECK(b != NULL);

    int a0 = wubu_harness_bandit_add(b, 0.9, 0.8, 0.7, 0.1); /* value 2.3 */
    int a1 = wubu_harness_bandit_add(b, 0.2, 0.2, 0.2, 0.1); /* value 0.5 */
    int a2 = wubu_harness_bandit_add(b, 0.3, 0.1, 0.1, 0.4); /* value 0.1 */
    CHECK(a0 == 0 && a1 == 1 && a2 == 2);
    CHECK(wubu_harness_bandit_count(b) == 3);

    /* all four scores feed the value: sev + fix + breadth - side_effect */
    CHECK(WUBU_NEAR(wubu_harness_bandit_value(b, a0), 2.3));
    CHECK(WUBU_NEAR(wubu_harness_bandit_value(b, a1), 0.5));
    CHECK(WUBU_NEAR(wubu_harness_bandit_value(b, a2), 0.1));

    /* exploration cap = 2: first two selects explore round-robin 0,1 */
    CHECK(wubu_harness_bandit_select(b) == 0);
    CHECK(wubu_harness_bandit_select(b) == 1);
    CHECK(wubu_harness_bandit_explorations(b) == 2);
    /* now greedy -> best arm */
    CHECK(wubu_harness_bandit_select(b) == a0);
    CHECK(wubu_harness_bandit_explorations(b) == 2);

    /* reward signal can flip the greedy choice */
    for (int i = 0; i < 50; i++) wubu_harness_bandit_reward(b, a2, 1);
    CHECK(wubu_harness_bandit_select(b) == a2);

    /* unsafe revision rejected; safe revision accepted */
    double before = wubu_harness_bandit_value(b, a0);
    CHECK(wubu_harness_bandit_revise(b, a0, 0.9, 0.9, 0.9, 0.9) == -1);
    CHECK(WUBU_NEAR(wubu_harness_bandit_value(b, a0), before));  /* unchanged */
    CHECK(wubu_harness_bandit_revise(b, a0, 0.9, 0.9, 0.9, 0.1) == 0);
    CHECK(wubu_harness_bandit_value(b, a0) > before);

    CHECK(wubu_harness_bandit_create(-1) == NULL);
    wubu_harness_bandit_free(b);

    if (failures == 0) printf("test_harness_bandit: PASS (16 gates)\n");
    return failures ? 1 : 0;
}
