/* test_eg09_overlap.c -- EG-09 gate: one-step deferred result queue,
 * round-robin batch admission and no-starvation accounting.
 * Prints: test_eg09_overlap: PASS. */
#include <stdio.h>
#include <string.h>

#include "wubu_overlap_sched.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    enum { NREQ = 6, TOTAL = 27 };
    const int lens[NREQ] = { 3, 5, 2, 7, 4, 6 };
    wubu_overlap_result_t out[32];
    int seen[NREQ] = {0};
    int last_step[NREQ];
    for (int i = 0; i < NREQ; i++) last_step[i] = -1;

    wubu_overlap_sched_t *s = wubu_overlap_sched_create(NREQ, 32, 4);
    CHECK(s != NULL);
    if (!s) return 1;
    for (int i = 0; i < NREQ; i++)
        CHECK(wubu_overlap_sched_add(s, i, lens[i]) == 0);

    int steps = 0, first_drain = -1;
    while (steps < 64) {
        int nd = -1;
        int active = wubu_overlap_sched_step(s, out, 16, &nd);
        CHECK(active >= 0);
        if (steps == 0) {
            CHECK(nd == 0); /* results are deferred by exactly one step */
            first_drain = 0;
        } else if (nd > 0) {
            CHECK(out[0].step == steps - 1);
        }
        for (int j = 0; j < nd; j++) {
            int id = out[j].req_id;
            CHECK(id >= 0 && id < NREQ);
            CHECK(out[j].token == 1000 * id + out[j].step);
            CHECK(out[j].step > last_step[id]); /* strictly increasing */
            last_step[id] = out[j].step;
            CHECK(seen[id] < lens[id]);         /* no extra tokens */
            seen[id]++;
        }
        steps++;
        if (active == 0 && wubu_overlap_sched_pending(s) == 0) break;
    }

    printf("EG-09 steps=%d produced=%llu drained=%llu max_wait=%d\n",
           steps, (unsigned long long)wubu_overlap_sched_produced(s),
           (unsigned long long)wubu_overlap_sched_drained(s),
           wubu_overlap_sched_max_wait(s));
    CHECK(first_drain == 0);
    CHECK(wubu_overlap_sched_produced(s) == TOTAL);
    CHECK(wubu_overlap_sched_drained(s) == TOTAL);
    for (int i = 0; i < NREQ; i++) {
        CHECK(seen[i] == lens[i]);
    }
    CHECK(wubu_overlap_sched_step_count(s) == steps);
    CHECK(steps > 7);   /* batch-limited, not a single 27-step serial loop */
    CHECK(steps <= 12); /* but bounded: no starvation */
    CHECK(wubu_overlap_sched_max_wait(s) > 0);
    CHECK(wubu_overlap_sched_max_wait(s) <= 4); /* bounded, no starvation */
    CHECK(wubu_overlap_sched_active(s) == 0);
    CHECK(wubu_overlap_sched_pending(s) == 0);
    printf("EG-09 serial baseline=%d steps (%.2fx fewer)\n",
           TOTAL, (double)TOTAL / (double)steps);

    /* Drain cap smaller than the pending set: no token may be lost. */
    wubu_overlap_sched_t *t = wubu_overlap_sched_create(4, 32, 4);
    CHECK(t != NULL);
    if (t) {
        for (int i = 0; i < 4; i++)
            CHECK(wubu_overlap_sched_add(t, i, 3) == 0);
        for (int k = 0; k < 32; k++) {
            int nd = -1, active = wubu_overlap_sched_step(t, out, 1, &nd);
            CHECK(nd <= 1);
            if (active == 0 && wubu_overlap_sched_pending(t) == 0) break;
        }
        printf("EG-09 cap=1 produced=%llu drained=%llu\n",
               (unsigned long long)wubu_overlap_sched_produced(t),
               (unsigned long long)wubu_overlap_sched_drained(t));
        CHECK(wubu_overlap_sched_produced(t) == 12);
        CHECK(wubu_overlap_sched_drained(t) == 12);
    }

    CHECK(wubu_overlap_sched_create(0, 4, 2) == NULL);
    CHECK(wubu_overlap_sched_create(4, 0, 2) == NULL);
    CHECK(wubu_overlap_sched_create(4, 4, 0) == NULL);
    CHECK(wubu_overlap_sched_add(s, 0, 1) == -1); /* duplicate id */
    {
        int nd = 0;
        CHECK(wubu_overlap_sched_step(NULL, out, 4, &nd) == -1);
    }

    wubu_overlap_sched_free(s);
    wubu_overlap_sched_free(t);
    if (failures) {
        fprintf(stderr, "test_eg09_overlap: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg09_overlap: PASS\n");
    return 0;
}
