/* wubu_aot_pack.c -- EG-66c AOT kernel pack selector. Strict C11. */
#include "wubu_aot_pack.h"

#include <stdlib.h>

struct wubu_aot_pack {
    int n;
    wubu_aot_pack_entry_t entry[WUBU_AOT_MAX_PACKS];
};

wubu_aot_pack_t *wubu_aot_pack_create(const wubu_aot_pack_entry_t *entries, int n)
{
    wubu_aot_pack_t *p;
    if (!entries || n < 1 || n > WUBU_AOT_MAX_PACKS) return NULL;
    for (int i = 0; i < n; ++i) {
        if (entries[i].dtype < 0 || entries[i].variant < 0 ||
            entries[i].min_n < 1 || entries[i].max_n < entries[i].min_n) return NULL;
        for (int j = i + 1; j < n; ++j) {
            if (entries[i].dtype != entries[j].dtype) continue;
            if (entries[i].min_n <= entries[j].max_n &&
                entries[j].min_n <= entries[i].max_n) return NULL;
        }
    }
    p = (wubu_aot_pack_t *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->n = n;
    for (int i = 0; i < n; ++i) p->entry[i] = entries[i];
    return p;
}

void wubu_aot_pack_free(wubu_aot_pack_t *p) { free(p); }

int wubu_aot_pack_count(const wubu_aot_pack_t *p)
{
    return p ? p->n : -1;
}

int wubu_aot_pack_select(const wubu_aot_pack_t *p, int dtype, int n,
                         int *variant_out)
{
    if (!p || !variant_out || dtype < 0 || n < 1) return -1;
    for (int i = 0; i < p->n; ++i) {
        const wubu_aot_pack_entry_t *e = &p->entry[i];
        if (e->dtype == dtype && n >= e->min_n && n <= e->max_n) {
            *variant_out = e->variant;
            return 0;
        }
    }
    return -1;
}
