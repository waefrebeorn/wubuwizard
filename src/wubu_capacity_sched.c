/*
 * wubu_capacity_sched.c -- EG-32a capacity scheduler. Pure C11.
 */
#include "wubu_capacity_sched.h"

#include <string.h>

int wubu_capacity_plan(const wubu_cap_req_t *reqs, int n,
                       int capacity_tokens, int max_batch_tokens,
                       wubu_cap_plan_t *out) {
    int order[WUBU_CAP_MAX_REQS];
    if (!reqs || !out || n <= 0 || n > WUBU_CAP_MAX_REQS ||
        capacity_tokens <= 0 || max_batch_tokens <= 0) return -1;

    memset(out, 0, sizeof(*out));
    for (int i = 0; i < n; i++) {
        out->accepted[i] = 0;
        out->batch_of[i] = -1;
        order[i] = i;
    }
    for (int i = 0; i < WUBU_CAP_MAX_BATCHES; i++) out->batch_tokens[i] = 0;

    /* Insertion sort by descending priority, then ascending token count. */
    for (int i = 1; i < n; i++) {
        int v = order[i], j = i - 1;
        while (j >= 0 &&
               (reqs[order[j]].priority < reqs[v].priority ||
                (reqs[order[j]].priority == reqs[v].priority &&
                 reqs[order[j]].tokens > reqs[v].tokens))) {
            order[j + 1] = order[j]; j--;
        }
        order[j + 1] = v;
    }

    for (int k = 0; k < n; k++) {
        int i = order[k], placed = 0;
        if (reqs[i].tokens <= 0 || reqs[i].tokens > max_batch_tokens) {
            out->tokens_rejected += reqs[i].tokens > 0 ? reqs[i].tokens : 0;
            continue;
        }
        if (out->tokens_admitted + reqs[i].tokens > capacity_tokens) {
            out->tokens_rejected += reqs[i].tokens;
            continue;
        }
        for (int b = 0; b < out->n_batches && !placed; b++)
            if (out->batch_tokens[b] + reqs[i].tokens <= max_batch_tokens) {
                out->batch_tokens[b] += reqs[i].tokens;
                out->batch_of[i] = b;
                placed = 1;
            }
        if (!placed && out->n_batches < WUBU_CAP_MAX_BATCHES) {
            int b = out->n_batches++;
            out->batch_tokens[b] = reqs[i].tokens;
            out->batch_of[i] = b;
            placed = 1;
        }
        if (placed) {
            out->accepted[i] = 1;
            out->tokens_admitted += reqs[i].tokens;
        } else {
            out->tokens_rejected += reqs[i].tokens;
        }
    }
    return 0;
}
