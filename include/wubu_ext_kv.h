#ifndef WUBU_EXT_KV_H
#define WUBU_EXT_KV_H

/*
 * wubu_ext_kv.h -- EG-35: extensible KV sized after warmup, with UMA
 * page-cache-corrected free memory (refines EG-03). Pure C11.
 *
 * Free memory is sampled during warmup; the minimum sample is used so a later
 * allocation cannot be based on an optimistic observation. On UMA hardware
 * part of "free" memory is reclaimable page cache and must be discounted.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_EXT_KV_MAX_SAMPLES 64

typedef struct {
    size_t total_vram;
    size_t used;
    size_t reserved;
    size_t page_cache;
    int uma;
    size_t kv_fraction_percent;
    size_t min_kv_bytes;
} wubu_ext_kv_config_t;

typedef struct {
    size_t effective_free;
    size_t kv_bytes;
    int accepted;
    int reason; /* 0 ok, 1 no samples, 2 below minimum, 3 UMA shortfall */
} wubu_ext_kv_t;

int wubu_ext_kv_plan(const wubu_ext_kv_config_t *cfg,
                     const size_t *samples, int n_samples,
                     wubu_ext_kv_t *out);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_EXT_KV_H */
