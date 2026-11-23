#ifndef WUBU_ADAPT_LEN_H
#define WUBU_ADAPT_LEN_H

/*
 * wubu_adapt_len.h -- EG-88: adaptive candidate length, AdaFlash M10.
 *
 * Always draft the full block, then truncate at the first position where
 * 1 - prod(acceptance probabilities from position 1) exceeds a fixed scalar
 * threshold. Clamp to [2, block_size]. Only the retained prefix is sent to
 * the target model; no trained length head is used in this gate.
 */

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_ADAPT_LEN_MIN 2

typedef struct wubu_adapt_len wubu_adapt_len_t;

/* block_size >= 2, threshold in [0,1]. NULL on invalid input. */
wubu_adapt_len_t *wubu_adapt_len_create(int block_size, float threshold);
void wubu_adapt_len_free(wubu_adapt_len_t *p);

/* Chosen candidate length in [2, min(block_size,n)]; -1 on bad input.
 * probs[0] is the anchor (always retained); cumprod starts at probs[1]. */
int wubu_adapt_len_choose(const wubu_adapt_len_t *p, const float *probs, int n);

/* Consecutive greedy accepts after the anchor: draft[i] vs target[i-1] for
 * i = 1..len-1. Returns 0..len-1, or -1 on bad input. */
int wubu_adapt_len_accept(const int *draft, const int *target, int len);

/* Target KV rows kept after cropping to accepted+1 (0 on bad input). */
int wubu_adapt_len_kv_rows(int accepted);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_ADAPT_LEN_H */
