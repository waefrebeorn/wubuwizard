#ifndef WUBU_CANDIDATE_SELECTOR_H
#define WUBU_CANDIDATE_SELECTOR_H

/*
 * wubu_candidate_selector.h -- EG-74: candidate selector with p-min cut.
 * DFlash2/LiliCorr: candidates below a probability floor are dropped before
 * the hidden-state-conditioned scorer ranks the survivors. Pure index work.
 */

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_SELECT_MAX 64

typedef struct wubu_candidate_selector wubu_candidate_selector_t;

/* max_candidates in [1,64]. NULL on bad input. */
wubu_candidate_selector_t *wubu_candidate_selector_create(int max_candidates);
void wubu_candidate_selector_free(wubu_candidate_selector_t *p);

/* Keep probs[i] >= p_min, rank by probs[i]+(scores?scores[i]:0) descending
 * (ties by ascending index), write kept indices. Returns count, -1 bad input. */
int wubu_candidate_selector_select(wubu_candidate_selector_t *p,
                                   const float *probs, const double *scores,
                                   int n, float p_min, int *out_idx);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_CANDIDATE_SELECTOR_H */
