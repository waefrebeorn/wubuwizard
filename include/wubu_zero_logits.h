#ifndef WUBU_ZERO_LOGITS_H
#define WUBU_ZERO_LOGITS_H

/*
 * wubu_zero_logits.h -- EG-103: zero-logits device-to-host handoff
 * (Strata Inkling lineage). Only sampled/verified token ids and their log
 * probabilities cross the bus; full vocabulary logits stay on the device.
 * Pure C11.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t full_bytes;
    size_t handoff_bytes;
    double reduction;
} wubu_zero_logits_plan_t;

/* Bytes for n rows of vocab logits vs n (id, logprob) pairs. */
int wubu_zero_logits_plan(int vocab, int n,
                          wubu_zero_logits_plan_t *out);

/* Verify the returned ids/logprobs against the full logits row.
 * Returns 0 when consistent (or -1 on bad arguments / mismatch). */
int wubu_zero_logits_verify(const float *logits, int vocab,
                            const int *ids, const float *logprobs,
                            int n, float tol);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_ZERO_LOGITS_H */
