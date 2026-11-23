/* wubu_early_expert.c -- EG-96 lossless cross-layer expert pread gate. C11. */
#include "wubu_early_expert.h"

#include <stdlib.h>

struct wubu_early_expert {
    int reads, slots, staged_n, adopted, wasted;
    int ids[WUBU_EE_MAX_EXPERTS];
    double scores[WUBU_EE_MAX_EXPERTS];
};

wubu_early_expert_t *wubu_early_expert_create(int reads, int staging_slots)
{
    wubu_early_expert_t *p;
    if (reads < 1 || reads > 8 || staging_slots < 1 ||
        staging_slots > WUBU_EE_MAX_EXPERTS) return NULL;
    p = (wubu_early_expert_t *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->reads = reads;
    p->slots = staging_slots;
    return p;
}

void wubu_early_expert_free(wubu_early_expert_t *p) { free(p); }

int wubu_early_expert_stage(wubu_early_expert_t *p,
                            const int *ids, const double *scores, int n,
                            const unsigned char *cached, int cached_n)
{
    int limit;
    if (!p || !ids || !scores || !cached || n < 0 || n > WUBU_EE_MAX_EXPERTS ||
        cached_n < 0) return -1;
    p->staged_n = 0;
    limit = p->reads < p->slots ? p->reads : p->slots;
    for (int i = 0; i < n; ++i) {
        int id = ids[i], insert = p->staged_n;
        if (id < 0 || (id < cached_n && cached[id])) continue;
        for (int k = 0; k < p->staged_n; ++k) if (p->ids[k] == id) insert = -1;
        if (insert < 0) continue;
        if (p->staged_n == limit) {
            /* Replace the lowest score if this candidate is better. */
            int worst = 0;
            for (int k = 1; k < p->staged_n; ++k)
                if (p->scores[k] < p->scores[worst]) worst = k;
            if (scores[i] <= p->scores[worst]) continue;
            insert = worst;
        } else {
            ++p->staged_n;
        }
        p->ids[insert] = id;
        p->scores[insert] = scores[i];
    }
    return p->staged_n;
}

int wubu_early_expert_is_staged(const wubu_early_expert_t *p, int expert_id)
{
    if (!p) return -1;
    for (int i = 0; i < p->staged_n; ++i)
        if (p->ids[i] == expert_id) return 1;
    return 0;
}

int wubu_early_expert_staged(const wubu_early_expert_t *p)
{
    return p ? p->staged_n : -1;
}

int wubu_early_expert_adopt(wubu_early_expert_t *p, const int *plan_ids, int n,
                            int *out_ids, unsigned char *used_staged)
{
    unsigned char taken[WUBU_EE_MAX_EXPERTS] = {0};
    int used = 0;
    if (!p || !plan_ids || !out_ids || !used_staged ||
        n < 0 || n > WUBU_EE_MAX_EXPERTS) return -1;
    p->adopted = 0;
    for (int i = 0; i < n; ++i) {
        out_ids[i] = plan_ids[i];       /* lossless: plan never changes */
        used_staged[i] = 0;
        for (int k = 0; k < p->staged_n; ++k) {
            if (p->ids[k] == plan_ids[i] && !taken[k]) {
                taken[k] = 1;
                used_staged[i] = 1;
                ++p->adopted;
                break;
            }
        }
    }
    for (int k = 0; k < p->staged_n; ++k) if (taken[k]) ++used;
    p->wasted = p->staged_n - used;
    return p->adopted;
}

int wubu_early_expert_adopted(const wubu_early_expert_t *p)
{
    return p ? p->adopted : -1;
}

int wubu_early_expert_wasted(const wubu_early_expert_t *p)
{
    return p ? p->wasted : -1;
}
