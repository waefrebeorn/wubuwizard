#ifndef WUBU_TITAN_H
#define WUBU_TITAN_H

/*
 * wubu_titan.h -- EG-67: tiered experts + prefill expert streaming + hybrid
 * prefix cache on a 6 GB-class GPU (titan-engine lineage), reimplemented as a
 * C11 planning gate. Bytes are modeled, not allocated.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int id;
    size_t bytes;
    int tier; /* 0 hot/VRAM, 1 warm/host, 2 cold/disk */
} wubu_titan_expert_t;

/* Plan streaming under a VRAM cap. Returns 0 or -1. peak receives the modeled
 * peak VRAM; streamed receives the number of non-hot experts paged in. */
int wubu_titan_plan(const wubu_titan_expert_t *experts, int n, size_t cap,
                    size_t *peak, int *streamed);

/* Hybrid prefix cache hit count: for each query token, report whether the
 * (token, tier) pair was seen in the retained prefix table. */
int wubu_titan_prefix(const int *tokens, const int *tiers, int n,
                      int *hits);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_TITAN_H */
