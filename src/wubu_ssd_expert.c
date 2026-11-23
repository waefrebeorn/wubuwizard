/*
 * wubu_ssd_expert.c -- EG-50 concurrent SSD expert read / async H2D gate.
 * Pure C11, deterministic tick simulation.
 */
#include "wubu_ssd_expert.h"

#include <stdlib.h>

typedef struct {
    int task;     /* -1 = idle */
    int left;
} lane_t;

int wubu_ssd_expert_run(const wubu_ssd_task_t *tasks, int n,
                        int read_lanes, int h2d_lanes,
                        wubu_ssd_result_t *out, int cap,
                        int *max_overlap, int *total_ticks) {
    lane_t *reads = NULL, *h2ds = NULL;
    int *pending = NULL, npend = 0, next = 0, done = 0, tick = 0;
    int overlap_max = 0, rc = -1;

    if (!tasks || !out || !max_overlap || !total_ticks || n <= 0 ||
        cap < n || read_lanes <= 0 || h2d_lanes <= 0) return -1;
    for (int i = 0; i < n; i++)
        if (tasks[i].read_ticks <= 0 || tasks[i].h2d_ticks <= 0 ||
            tasks[i].sym < 0 || tasks[i].sym > 3) return -1;

    reads = (lane_t *)calloc((size_t)read_lanes, sizeof(lane_t));
    h2ds = (lane_t *)calloc((size_t)h2d_lanes, sizeof(lane_t));
    pending = (int *)malloc((size_t)n * sizeof(int));
    if (!reads || !h2ds || !pending) goto out;
    for (int i = 0; i < read_lanes; i++) reads[i].task = -1;
    for (int i = 0; i < h2d_lanes; i++) h2ds[i].task = -1;

    while (done < n || next < n || npend > 0) {
        int busy = 0;

        /* Start async H2D on completed reads, FIFO. */
        for (int l = 0; l < h2d_lanes && npend > 0; l++) {
            if (h2ds[l].task == -1) {
                int t = pending[0];
                for (int k = 1; k < npend; k++) pending[k - 1] = pending[k];
                npend--;
                h2ds[l].task = t;
                h2ds[l].left = tasks[t].h2d_ticks;
            }
        }

        /* Start new SSD reads on idle lanes. */
        for (int l = 0; l < read_lanes && next < n; l++)
            if (reads[l].task == -1) {
                reads[l].task = next;
                reads[l].left = tasks[next].read_ticks;
                next++;
            }

        /* Advance one tick; completed reads queue for H2D, completed H2D
         * decode locally into the result buffer. */
        for (int l = 0; l < read_lanes; l++)
            if (reads[l].task != -1 && --reads[l].left <= 0) {
                pending[npend++] = reads[l].task;
                reads[l].task = -1;
            }
        for (int l = 0; l < h2d_lanes; l++)
            if (h2ds[l].task != -1 && --h2ds[l].left <= 0) {
                int t = h2ds[l].task;
                out[done].expert = tasks[t].expert;
                out[done].done_tick = tick;
                /* Tiny rANS-style pair: encode state=(sym+1)&3, decode inverse. */
                out[done].decoded = ((((tasks[t].sym + 1) & 3) + 3) & 3);
                done++;
                h2ds[l].task = -1;
            }

        for (int l = 0; l < read_lanes; l++) if (reads[l].task != -1) busy++;
        for (int l = 0; l < h2d_lanes; l++) if (h2ds[l].task != -1) busy++;
        if (busy > overlap_max) overlap_max = busy;
        tick++;
        if (tick > n * (64 + n)) goto out; /* guard against bad durations */
    }

    *max_overlap = overlap_max;
    *total_ticks = tick;
    rc = 0;

out:
    free(reads);
    free(h2ds);
    free(pending);
    return rc;
}
