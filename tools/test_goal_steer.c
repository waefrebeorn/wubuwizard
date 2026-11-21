/* test_goal_steer.c -- gate for LP-02 (goal-drift steering). */
#include <stdio.h>
#include "wubu_goal_steer.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void)
{
    wubu_goal_steer_t *s = wubu_goal_steer_create(0.5);
    CHECK(s != NULL);
    CHECK(wubu_goal_steer_tau(s) == 0.5);

    /* below tau */
    CHECK(wubu_goal_steer_observe(s, 0.4) == 0);
    CHECK(wubu_goal_steer_drift_observed(s) == 0);

    /* above tau (magnitude, both signs) */
    CHECK(wubu_goal_steer_observe(s, 0.6) == 1);
    CHECK(wubu_goal_steer_drift_observed(s) == 1);
    CHECK(wubu_goal_steer_observe(s, -0.7) == 1);

    /* no veto + observed drift -> steer forces CONTINUE */
    wubu_goal_verdict_t d = WUBU_GOAL_FINISHED;
    CHECK(wubu_goal_steer_apply(s, &d) == WUBU_GOAL_STEER_APPLIED);
    CHECK(d == WUBU_GOAL_CONTINUE);

    /* veto preserves the existing judge decision */
    CHECK(wubu_goal_steer_set_veto(s, 1) == 0);
    CHECK(wubu_goal_steer_veto(s) == 1);
    d = WUBU_GOAL_FINISHED;
    CHECK(wubu_goal_steer_apply(s, &d) == WUBU_GOAL_STEER_VETOED);
    CHECK(d == WUBU_GOAL_FINISHED);   /* preserved */

    /* below tau + no veto -> OK, unchanged */
    CHECK(wubu_goal_steer_set_veto(s, 0) == 0);
    CHECK(wubu_goal_steer_observe(s, 0.1) == 0);
    d = WUBU_GOAL_IS_BLOCKED;
    CHECK(wubu_goal_steer_apply(s, &d) == WUBU_GOAL_STEER_OK);
    CHECK(d == WUBU_GOAL_IS_BLOCKED);

    /* invalid arguments */
    CHECK(wubu_goal_steer_apply(s, NULL) == WUBU_GOAL_STEER_INVALID);
    CHECK(wubu_goal_steer_create(-1.0) == NULL);
    wubu_goal_steer_free(s);

    if (failures == 0) printf("test_goal_steer: PASS (13 gates)\n");
    return failures ? 1 : 0;
}
