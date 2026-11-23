/*
 * wubu_memspec.c -- EG-64 memory-aware draft scheduler. Pure C11.
 */
#include "wubu_memspec.h"

#include <stdlib.h>

typedef struct {
    int left;
    size_t bytes;
    int running;
} active_t;

static int wubu_memspec_run(const wubu_memstep_t *steps, int n, size_t cap,
                            int aware, size_t *peak) {
    active_t *act;
    int *order, *started;
    int done = 0, tick = 0, rc = -1;
    size_t used = 0, top = 0;

    if (!steps || !peak || n <= 0 || cap == 0) return -1;
    for (int i = 0; i < n; i++)
        if (steps[i].bytes == 0 || steps[i].bytes > cap ||
            steps[i].ticks <= 0) return -1;
    act = (active_t *)calloc((size_t)n, sizeof(active_t));
    order = (int *)malloc((size_t)n * sizeof(int));
    started = (int *)calloc((size_t)n, sizeof(int));
    if (!act || !order || !started) goto out;
    for (int i = 0; i < n; i++) order[i] = i;
    /* Both policies keep the step order. The memory-aware one backfills
     * smaller later steps around a step that does not currently fit; FIFO
     * stalls at that step (head-of-line blocking). */

    while (done < n) {
        /* Start what fits. FIFO stops at the first non-fitting step;
         * memory-aware backfills smaller steps around it. */
        for (int k = 0; k < n; k++) {
            int t = order[k];
            if (started[t]) continue;
            if (steps[t].bytes > cap - used) {
                if (!aware) break;
                continue;
            }
            act[t].running = 1;
            act[t].left = steps[t].ticks;
            act[t].bytes = steps[t].bytes;
            used += steps[t].bytes;
            started[t] = 1;
        }
        for (int i = 0; i < n; i++)
            if (act[i].running && act[i].left > 0) act[i].left--;
        for (int i = 0; i < n; i++)
            if (act[i].running && act[i].left == 0) {
                act[i].running = 0;
                used -= act[i].bytes;
                done++;
            }
        if (used > top) top = used;
        tick++;
        if (tick > n * (64 + n)) goto out;
    }
    *peak = top;
    rc = tick;

out:
    free(act);
    free(order);
    free(started);
    return rc;
}

int wubu_memspec_schedule(const wubu_memstep_t *steps, int n, size_t cap,
                          int policy, size_t *peak) {
    if (policy != 0 && policy != 1) return -1;
    return wubu_memspec_run(steps, n, cap, policy == 0, peak);
}
