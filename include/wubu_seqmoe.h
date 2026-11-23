#ifndef WUBU_SEQMOE_H
#define WUBU_SEQMOE_H

/*
 * wubu_seqmoe.h -- EG-52: expert activation as a sequence model with deadline
 * sequencing and probabilistic Belady residency (SeqMoE lineage), reimplemented
 * in C11. One technique, one gate: hit rate at fixed expert residency.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_SEQMOE_ORDER 2

typedef struct {
    int experts;
    int capacity;
    unsigned long long *trans; /* (prev2*E + prev1)*E + next */
    unsigned long long *freq;
} wubu_seqmoe_t;

int wubu_seqmoe_init(wubu_seqmoe_t *m, int experts, int capacity);
int wubu_seqmoe_free(wubu_seqmoe_t *m);
int wubu_seqmoe_observe(wubu_seqmoe_t *m, const int *trace, int n);

/* Rank experts for the next step from the last two experts. Writes up to
 * capacity expert ids in descending predicted probability; returns count. */
int wubu_seqmoe_plan(const wubu_seqmoe_t *m, int prev2, int prev1, int *out);

/* Residency simulation. Returns hits, or -1 on bad input. */
int wubu_seqmoe_hits(const wubu_seqmoe_t *m, const int *trace, int n,
                     int prev2, int prev1);

/* Baseline comparators at the same capacity. */
int wubu_lru_hits(int experts, int capacity, const int *trace, int n);
int wubu_static_hits(int experts, int capacity, const int *trace, int n);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_SEQMOE_H */
