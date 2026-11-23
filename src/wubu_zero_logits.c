/*
 * wubu_zero_logits.c -- EG-103 zero-logits handoff. Pure C11.
 */
#include "wubu_zero_logits.h"

#include <math.h>

int wubu_zero_logits_plan(int vocab, int n,
                          wubu_zero_logits_plan_t *out) {
    if (!out || vocab <= 0 || n <= 0) return -1;
    out->full_bytes = (size_t)vocab * (size_t)n * sizeof(float);
    out->handoff_bytes = (size_t)n * (sizeof(int) + sizeof(float));
    out->reduction = out->full_bytes > 0
        ? (double)out->handoff_bytes / (double)out->full_bytes : 0.0;
    return 0;
}

int wubu_zero_logits_verify(const float *logits, int vocab,
                            const int *ids, const float *logprobs,
                            int n, float tol) {
    if (!logits || !ids || !logprobs || vocab <= 0 || n <= 0 ||
        !(tol >= 0.0f)) return -1;
    for (int i = 0; i < n; i++) {
        const float *row = logits + (size_t)i * vocab;
        int id = ids[i];
        float mx = row[0], denom = 0.0f, lp;
        if (id < 0 || id >= vocab) return -1;
        for (int v = 1; v < vocab; v++) if (row[v] > mx) mx = row[v];
        for (int v = 0; v < vocab; v++) denom += expf(row[v] - mx);
        lp = row[id] - mx - logf(denom);
        if (fabsf(lp - logprobs[i]) > tol) return -1;
    }
    return 0;
}
