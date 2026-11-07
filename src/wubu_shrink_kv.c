/* wubu_shrink_kv.c — KV-space amoeba shrink operator
 *
 * Classifies /kv/in/ regions as DEAD when they are both low-utilization
 * (little attention mass) AND low-coherence (not understood), then prunes
 * them from the namespace so the blocks return to the freelist.
 *
 * Design: docs/wubu1-hive-mind-plan.md §1 Track B, module #6 (AN22);
 * research/060 / INDEX AN04 (block importance).
 * WaefreBeorn Umbrella License v3.0
 */
#include "wubu_shrink_kv.h"
#include <stdlib.h>
#include <string.h>

/* One analyzed region. */
typedef struct {
    char   path[512];   /* /kv/in/<file> namespace key */
    float  utilization;
    float  coherence;
    float  entropy;
    int    pruneable;
} sk_region_t;

struct wubu_shrink_kv {
    wubu_kv_embedding_t *kv;
    wubu_kvfs_t         *fs;
    double  util_thr;
    double  coh_thr;
    uint32_t block_size;
    sk_region_t *regions;
    size_t      n_regions;
    size_t      cap_regions;
    unsigned long total_pruned;
};

wubu_shrink_kv_cfg_t wubu_shrink_kv_default_cfg(void) {
    wubu_shrink_kv_cfg_t c;
    c.utilization_threshold = 0.2;
    c.coherence_threshold   = 0.4;
    c.block_size            = 256;
    return c;
}

static int sk_ensure_cap(wubu_shrink_kv_t *s) {
    if (s->n_regions < s->cap_regions) return 0;
    size_t newcap = s->cap_regions ? s->cap_regions * 2 : 32;
    sk_region_t *r = (sk_region_t *)realloc(s->regions, newcap * sizeof(*r));
    if (!r) return -1;
    s->regions = r;
    s->cap_regions = newcap;
    return 0;
}

wubu_shrink_kv_t *wubu_shrink_kv_create(wubu_kv_embedding_t *kv,
                                        const wubu_shrink_kv_cfg_t *cfg) {
    if (!kv) return NULL;
    wubu_shrink_kv_t *s = (wubu_shrink_kv_t *)calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->kv = kv;
    s->fs = wubu_kv_embedding_get_fs(kv);
    s->util_thr = cfg ? cfg->utilization_threshold : 0.2;
    s->coh_thr  = cfg ? cfg->coherence_threshold   : 0.4;
    s->block_size = cfg ? cfg->block_size : 256;
    return s;
}

int wubu_shrink_kv_analyze(wubu_shrink_kv_t *s,
                           const char **paths,
                           const float *utilizations,
                           const float *coherences,
                           const float *entropies,
                           int n_files) {
    if (!s || !paths || !utilizations || !coherences || n_files < 0) return -1;
    s->n_regions = 0;
    int pruneable = 0;
    for (int i = 0; i < n_files; i++) {
        if (!paths[i]) continue;
        if (sk_ensure_cap(s) != 0) return -1;
        sk_region_t *r = &s->regions[s->n_regions++];
        strncpy(r->path, paths[i], sizeof(r->path) - 1);
        r->path[sizeof(r->path) - 1] = '\0';
        r->utilization = utilizations[i];
        r->coherence   = coherences[i];
        r->entropy     = entropies ? entropies[i] : 0.0f;
        /* DEAD region: neither attended to nor understood. */
        r->pruneable = (utilizations[i] < s->util_thr &&
                        coherences[i]   < s->coh_thr);
        if (r->pruneable) pruneable++;
    }
    return pruneable;
}

int wubu_shrink_kv_prunable(const wubu_shrink_kv_t *s,
                            const char **out_paths, int cap) {
    if (!s || !out_paths) return 0;
    int n = 0;
    for (size_t i = 0; i < s->n_regions && n < cap; i++)
        if (s->regions[i].pruneable) out_paths[n++] = s->regions[i].path;
    return n;
}

int wubu_shrink_kv_shrink(wubu_shrink_kv_t *s, int max_prune) {
    if (!s || max_prune < 0) return -1;
    int pruned = 0;
    for (size_t i = 0; i < s->n_regions && pruned < max_prune; i++) {
        sk_region_t *r = &s->regions[i];
        if (!r->pruneable) continue;
        if (wubu_kvfs_unmount(s->fs, r->path) != 0) continue; /* already gone */
        r->pruneable = 0;
        s->total_pruned++;
        pruned++;
    }
    return pruned;
}

unsigned long wubu_shrink_kv_total_pruned(const wubu_shrink_kv_t *s) {
    return s ? s->total_pruned : 0;
}

void wubu_shrink_kv_free(wubu_shrink_kv_t *s) {
    if (!s) return;
    free(s->regions);
    free(s);
}
