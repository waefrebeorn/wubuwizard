#ifndef WUBU_SSD_EXPERT_H
#define WUBU_SSD_EXPERT_H

/*
 * wubu_ssd_expert.h -- EG-50: concurrent per-expert SSD reads with async H2D
 * on completion and VRAM-local rANS-style decode (SSD-LLaMA lineage),
 * reimplemented as a deterministic C11 scheduling gate.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int expert;      /* expert id, 0..255 */
    int read_ticks;  /* SSD read duration */
    int h2d_ticks;   /* async host-to-device duration */
    int sym;         /* 2-bit rANS-style payload, 0..3 */
} wubu_ssd_task_t;

typedef struct {
    int expert;
    int done_tick;
    int decoded;
} wubu_ssd_result_t;

/* Deterministic tick scheduler. Returns 0 or -1 on bad arguments.
 * max_overlap and total_ticks receive measured concurrency and makespan. */
int wubu_ssd_expert_run(const wubu_ssd_task_t *tasks, int n,
                        int read_lanes, int h2d_lanes,
                        wubu_ssd_result_t *out, int cap,
                        int *max_overlap, int *total_ticks);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_SSD_EXPERT_H */
