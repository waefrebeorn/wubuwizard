#ifndef WUBU_SPEC_SEARCH_H
#define WUBU_SPEC_SEARCH_H

/*
 * wubu_spec_search.h -- EG-41: hardware/workload/memory-aware speculative
 * configuration search (SpecSearch lineage). Refines EG-25's single-size
 * decision by searching a candidate set under a VRAM constraint. Pure C11.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int id;
    int k;               /* draft length */
    int heads;           /* number of draft heads */
    size_t draft_bytes;  /* VRAM needed by the drafter */
    float accept_p;      /* measured/predicted per-token acceptance */
} wubu_spec_candidate_t;

typedef struct {
    int best_id;
    int n_feasible;
    double throughput;
} wubu_spec_choice_t;

/* Returns 0 or -1. Candidates that exceed vram_free are skipped. */
int wubu_spec_search(const wubu_spec_candidate_t *cands, int n,
                     double draft_cost, double verify_cost,
                     size_t vram_free, wubu_spec_choice_t *out);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_SPEC_SEARCH_H */
