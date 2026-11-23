#ifndef WUBU_CAPACITY_SCHED_H
#define WUBU_CAPACITY_SCHED_H

/*
 * wubu_capacity_sched.h -- EG-32a: capacity + micro-batch scheduler
 * (depends on EG-09/EG-10), reimplemented in C11.
 *
 * Packs admitted requests into micro-batches under a per-cycle token capacity
 * and a per-batch token ceiling. Priority is honoured first; requests that do
 * not fit are rejected explicitly instead of being silently deferred.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_CAP_MAX_REQS 32
#define WUBU_CAP_MAX_BATCHES 16

typedef struct {
    int id;
    int tokens;
    int priority;
} wubu_cap_req_t;

typedef struct {
    int accepted[WUBU_CAP_MAX_REQS];   /* 1 if admitted this cycle */
    int batch_of[WUBU_CAP_MAX_REQS];   /* batch index, -1 if rejected */
    int batch_tokens[WUBU_CAP_MAX_BATCHES];
    int n_batches;
    int tokens_admitted;
    int tokens_rejected;
} wubu_cap_plan_t;

/* Returns 0 or -1 on bad arguments. */
int wubu_capacity_plan(const wubu_cap_req_t *reqs, int n,
                       int capacity_tokens, int max_batch_tokens,
                       wubu_cap_plan_t *out);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_CAPACITY_SCHED_H */
