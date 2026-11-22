#ifndef WUBU_OVERLAP_SCHED_H
#define WUBU_OVERLAP_SCHED_H

/*
 * wubu_overlap_sched.h -- EG-09: overlapping scheduler with a deferred
 * result queue (SGLang event_loop_overlap lineage), reimplemented in C11.
 *
 * Results produced by decode step t are drained at step t+1, so result
 * handling overlaps the next decode step. Every active request advances
 * once per step: no request starves while others drain.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int req_id;
    int token;
    int step;          /* decode step that produced the token */
} wubu_overlap_result_t;

typedef struct wubu_overlap_sched wubu_overlap_sched_t;

wubu_overlap_sched_t *wubu_overlap_sched_create(int max_requests,
                                                 int queue_capacity,
                                                 int max_batch);
void wubu_overlap_sched_free(wubu_overlap_sched_t *s);

/* Register a request that needs `n_tokens` generated tokens. */
int wubu_overlap_sched_add(wubu_overlap_sched_t *s, int req_id, int n_tokens);

/* Advance one step. Drains results produced by the previous step into
 * out[0..cap), writing the count to n_drained, then decodes every active
 * request and defers its result to the next step. Returns active count. */
int wubu_overlap_sched_step(wubu_overlap_sched_t *s,
                            wubu_overlap_result_t *out, int cap,
                            int *n_drained);

uint64_t wubu_overlap_sched_produced(const wubu_overlap_sched_t *s);
uint64_t wubu_overlap_sched_drained(const wubu_overlap_sched_t *s);
int wubu_overlap_sched_step_count(const wubu_overlap_sched_t *s);
int wubu_overlap_sched_pending(const wubu_overlap_sched_t *s);
int wubu_overlap_sched_active(const wubu_overlap_sched_t *s);
int wubu_overlap_sched_max_wait(const wubu_overlap_sched_t *s);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_OVERLAP_SCHED_H */
