#ifndef WUBU_FRSPEC_H
#define WUBU_FRSPEC_H

/*
 * wubu_frspec.h -- EG-57: frequency-ranked draft LM head (FR-Spec lineage).
 * Drafting uses only the K highest-frequency tokens; verification still uses
 * the full vocabulary. One gate: LM-head memory cut versus target-mass coverage.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Pick the K most frequent token ids, descending (stable on ties). */
int wubu_frspec_select(const unsigned long long *freq, int vocab, int k,
                       int *out);

/* Probability mass the shortlist covers under the full target distribution. */
double wubu_frspec_coverage(const float *target, int vocab,
                            const int *subset, int k);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_FRSPEC_H */
