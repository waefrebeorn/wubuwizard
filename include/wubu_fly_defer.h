#ifndef WUBU_FLY_DEFER_H
#define WUBU_FLY_DEFER_H

/*
 * wubu_fly_defer.h -- EG-80: FLy entropy-gated rejection deferral.
 * A native rejection at a high-entropy position is deferred when the next
 * window of native decisions all align. Entropy is normalized by log(k) into
 * [0,1]. Intentionally lossy: opt-in only, never used as an exactness claim.
 */

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_FLY_MAX_K 64

typedef struct wubu_fly_defer wubu_fly_defer_t;

/* threshold in [0,1], window >= 1, enabled in {0,1}. NULL on bad input. */
wubu_fly_defer_t *wubu_fly_defer_create(double threshold, int window,
                                        int enabled);
void wubu_fly_defer_free(wubu_fly_defer_t *p);

/* Shannon entropy of top-k scores normalized by log(k); -1 on bad input. */
double wubu_fly_normalized_entropy(const double *scores, int k);

/* Decide whether to defer a rejection; 1 defer, 0 keep rejection, -1 bad. */
int wubu_fly_defer_decide(wubu_fly_defer_t *p, int rejected_now,
                          double norm_entropy, const unsigned char *align_next,
                          int window_n);

int wubu_fly_defer_deferred(const wubu_fly_defer_t *p);
int wubu_fly_defer_rejections(const wubu_fly_defer_t *p);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_FLY_DEFER_H */
