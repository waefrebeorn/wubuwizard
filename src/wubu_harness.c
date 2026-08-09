/*
 * wubu_harness.c -- the SUSTAINED AUTONOMY TASK HARNESS (see the
 * header). Fixed suite, deterministic scorers, traj-cell outcomes,
 * loopguard-bounded, suite score as first-class fitness.
 */
#include "wubu_harness.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wubu_trajcell.h"

int wubu_harness_init(wubu_harness_t *h, wubu_hive_t *tissue,
                      long deadline_ns, long max_steps)
{
    if (!h || !tissue) return -1;
    memset(h, 0, sizeof(*h));
    h->tissue = tissue;
    h->guard.deadline_ns = deadline_ns;
    h->guard.max_steps = max_steps;
    /* the fixed task suite: 3 code edits, 2 transforms, 2 tool-use */
    wubu_task_t suite[7] = {
        { WUBU_TASK_EDIT,      11, 0, 3, 0.0f, 0.0f, 0 },
        { WUBU_TASK_EDIT,      12, 0, 4, 0.0f, 0.0f, 0 },
        { WUBU_TASK_EDIT,      13, 0, 3, 0.0f, 0.0f, 0 },
        { WUBU_TASK_TRANSFORM, 21, 0, 5, 0.0f, 0.0f, 0 },
        { WUBU_TASK_TRANSFORM, 22, 0, 4, 0.0f, 0.0f, 0 },
        { WUBU_TASK_TOOLUSE,   31, 0, 3, 0.0f, 0.0f, 0 },
        { WUBU_TASK_TOOLUSE,   32, 0, 2, 0.0f, 0.0f, 0 },
    };
    memcpy(h->suite, suite, sizeof(suite));
    h->n_tasks = 7;
    return 0;
}

int wubu_harness_run_task(wubu_harness_t *h, int task_idx,
                          int steps_done, float correctness, float cost)
{
    if (!h || task_idx < 0 || task_idx >= h->n_tasks) return -1;
    wubu_task_t *t = &h->suite[task_idx];
    t->steps_done = steps_done;
    t->correctness = correctness < 0 ? 0 : (correctness > 1 ? 1 : correctness);
    t->cost = cost;
    t->deadline_ok = (steps_done <= t->steps_required + 2) ? 1 : 0;
    /* the deterministic pass: enough steps + correct enough + in time */
    int passed = (steps_done >= t->steps_required &&
                  t->correctness >= 0.7f && t->deadline_ok) ? 1 : 0;
    /* the outcome is a TRAJ CELL (goal, outcome, cost, provenance) */
    wubu_traj_tracker_t tt;
    wubu_traj_init(&tt, h->tissue);
    uint32_t steps[8];
    for (int i = 0; i < steps_done && i < 8; i++) steps[i] = (uint32_t)(task_idx * 100 + i);
    wubu_traj_record(&tt, t->goal_token, steps, steps_done,
                     passed, passed ? (1.0f - 0.1f * cost) : t->correctness,
                     passed ? 1.0f : t->correctness * 0.5f, cost,
                     (uint64_t)h->step);
    h->step++;
    if (passed) h->n_passed++; else h->n_failed++;
    h->suite_score = wubu_harness_suite_score(h);
    return passed;
}

int wubu_harness_may_continue(wubu_harness_t *h)
{
    if (!h) return 0;
    return wubu_loop_may_continue(&h->guard, h->step, h->now_ns);
}

float wubu_harness_suite_score(wubu_harness_t *h)
{
    if (!h || h->n_tasks == 0) return 0.0f;
    float sum = 0;
    for (int i = 0; i < h->n_tasks; i++) {
        wubu_task_t *t = &h->suite[i];
        if (t->steps_done > 0) {
            /* per-task score: correctness weighted by the deadline */
            sum += t->correctness * (t->deadline_ok ? 1.0f : 0.5f);
        }
    }
    return sum / (float)h->n_tasks;
}

void wubu_harness_stats(const wubu_harness_t *h, char *buf, size_t cap)
{
    if (!h || !buf || cap == 0) return;
    snprintf(buf, cap,
             "tasks=%d passed=%llu failed=%llu suite_score=%.3f step=%ld",
             h->n_tasks,
             (unsigned long long)h->n_passed,
             (unsigned long long)h->n_failed,
             h->suite_score, h->step);
}
