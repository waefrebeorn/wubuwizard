/* test_goal_resample.c -- gate for LP-03 (bounded candidate resampling). */
#include <stdio.h>
#include "wubu_goal_resample.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void)
{
    /* bounded attempts: 3-attempt cap -> ACCEPT, ACCEPT, STOP */
    wubu_goal_resample_t *r = wubu_goal_resample_create(10, 3);
    CHECK(r != NULL);
    CHECK(wubu_goal_resample_begin(r) == 0);
    CHECK(wubu_goal_resample_submit(r, 0.1, 1) == WUBU_RESAMPLE_ACCEPT);
    CHECK(wubu_goal_resample_submit(r, 0.2, 1) == WUBU_RESAMPLE_ACCEPT);
    CHECK(wubu_goal_resample_submit(r, 0.05, 1) == WUBU_RESAMPLE_STOP);
    CHECK(wubu_goal_resample_attempts(r) == 3);
    CHECK(wubu_goal_resample_submit(r, 0.9, 1) == WUBU_RESAMPLE_STOP);
    CHECK(wubu_goal_resample_best_score(r) == 0.2);
    wubu_goal_resample_free(r);

    /* verifier rejection, early stop, snapshot rollback */
    r = wubu_goal_resample_create(8, 8);
    CHECK(wubu_goal_resample_begin(r) == 0);
    CHECK(wubu_goal_resample_submit(r, 0.5, 1) == WUBU_RESAMPLE_ACCEPT);
    CHECK(wubu_goal_resample_submit(r, 0.9, 0) == WUBU_RESAMPLE_REJECT);
    CHECK(wubu_goal_resample_best_score(r) == 0.5);

    wubu_goal_resample_snapshot_t *snap = wubu_goal_resample_snapshot(r);
    CHECK(snap != NULL);
    CHECK(wubu_goal_resample_submit(r, 0.7, 1) == WUBU_RESAMPLE_ACCEPT);
    CHECK(wubu_goal_resample_best_score(r) == 0.7);

    CHECK(wubu_goal_resample_restore(r, snap) == 0);       /* rollback */
    CHECK(wubu_goal_resample_best_score(r) == 0.5);
    CHECK(wubu_goal_resample_attempts(r) == 2);

    CHECK(wubu_goal_resample_stop(r) == 0);                /* early stop */
    CHECK(wubu_goal_resample_submit(r, 0.99, 1) == WUBU_RESAMPLE_STOP);
    CHECK(wubu_goal_resample_attempts(r) == 2);            /* not consumed */

    wubu_goal_resample_snapshot_free(snap);
    wubu_goal_resample_free(r);

    CHECK(wubu_goal_resample_create(0, 3) == NULL);
    if (failures == 0) printf("test_goal_resample: PASS (17 gates)\n");
    return failures ? 1 : 0;
}
