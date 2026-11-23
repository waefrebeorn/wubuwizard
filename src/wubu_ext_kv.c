/*
 * wubu_ext_kv.c -- EG-35 UMA-corrected KV sizing. Pure C11.
 */
#include "wubu_ext_kv.h"

int wubu_ext_kv_plan(const wubu_ext_kv_config_t *cfg,
                     const size_t *samples, int n_samples,
                     wubu_ext_kv_t *out) {
    size_t min_free, effective;
    if (!cfg || !out || !samples || n_samples <= 0 ||
        n_samples > WUBU_EXT_KV_MAX_SAMPLES ||
        cfg->kv_fraction_percent == 0 || cfg->kv_fraction_percent > 100)
        return -1;

    min_free = samples[0];
    for (int i = 1; i < n_samples; i++)
        if (samples[i] < min_free) min_free = samples[i];

    size_t phys_free = cfg->total_vram > cfg->used
        ? cfg->total_vram - cfg->used : 0;
    if (min_free > phys_free) min_free = phys_free;
    effective = min_free > cfg->reserved ? min_free - cfg->reserved : 0;
    if (cfg->uma) {
        /* Page cache is reclaimable; discount the part counted as free. */
        size_t discount = cfg->page_cache < effective ? cfg->page_cache : effective;
        effective -= discount;
    }

    out->effective_free = effective;
    out->kv_bytes = effective * cfg->kv_fraction_percent / 100;
    out->accepted = 0;
    out->reason = 0;

    if (cfg->uma && effective < cfg->min_kv_bytes) {
        out->reason = 3;
    } else if (out->kv_bytes < cfg->min_kv_bytes) {
        out->reason = 2;
    } else {
        out->accepted = 1;
    }
    return 0;
}
