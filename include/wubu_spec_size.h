#ifndef WUBU_SPEC_SIZE_H
#define WUBU_SPEC_SIZE_H

/*
 * wubu_spec_size.h -- EG-25a: Sequoia-style draft-length sizing, reimplemented
 * in C11. Picks the draft length k that maximises expected accepted tokens per
 * unit time, given per-token acceptance probability and draft/verify costs.
 */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int k;                 /* chosen draft length */
    double expected;       /* expected accepted tokens for that k */
    double throughput;     /* expected tokens per unit time */
} wubu_spec_size_t;

/* Choose k in [1, kmax]. Returns 0 or -1 on bad arguments. */
int wubu_spec_size(int kmax, double accept_p,
                   double draft_cost, double verify_cost,
                   wubu_spec_size_t *out);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_SPEC_SIZE_H */
