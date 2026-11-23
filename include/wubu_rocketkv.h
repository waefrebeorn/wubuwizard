#ifndef WUBU_ROCKETKV_H
#define WUBU_ROCKETKV_H

/*
 * wubu_rocketkv.h -- EG-53: training-free prompt-KV page eviction from
 * calibration attention mass, followed by page-level Top-K decode
 * (RocketKV lineage), reimplemented in C11.
 *
 * One technique, one gate: prompt KV bytes retained versus attention output
 * error, against a naive first-pages baseline at the same byte budget.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Accumulate mean attention mass per page for calibration queries.
 * keys layout: n_pages * page_size * dim floats. Returns 0 or -1. */
int wubu_rocketkv_importance(const float *keys, int n_pages, int page_size,
                             int dim, const float *q, int n_q, float *score);

/* Keep the highest-importance pages within retain_frac (0..1).
 * Writes page indices into keep (capacity n_pages). Returns kept count. */
int wubu_rocketkv_select(const float *score, int n_pages, float retain_frac,
                         int *keep);

/* Attention over only the kept pages; values are the key vectors themselves. */
int wubu_rocketkv_decode(const float *keys, const int *keep, int n_keep,
                         int page_size, int dim, const float *q, float *out);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_ROCKETKV_H */
