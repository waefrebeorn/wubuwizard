/* wubu_expert_cache.c -- EG-36b LRFU/ClassLRU gate. Strict C11. */
#include "wubu_expert_cache.h"

#include <stdlib.h>

typedef struct {
    int expert_id, class, used, freq;
    uint64_t last;
} expert_slot_t;

struct wubu_expert_cache {
    int slots, protected_class, n;
    int evictions, protected_evictions, pending;
    expert_slot_t slot[WUBU_EXPERT_MAX_SLOTS];
};

wubu_expert_cache_t *wubu_expert_cache_create(int slots, int protected_class)
{
    wubu_expert_cache_t *p;
    if (slots < 1 || slots > WUBU_EXPERT_MAX_SLOTS ||
        protected_class < 0) return NULL;
    p = (wubu_expert_cache_t *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->slots = slots;
    p->protected_class = protected_class;
    return p;
}

void wubu_expert_cache_free(wubu_expert_cache_t *p) { free(p); }

static double lrfu(const expert_slot_t *s, uint64_t now)
{
    return (double)s->freq / (double)(now - s->last + 1u);
}

int wubu_expert_cache_access(wubu_expert_cache_t *p, int expert_id,
                             int class, uint64_t now)
{
    int victim = -1;
    if (!p || expert_id < 0 || class < 0) return -1;
    for (int i = 0; i < p->slots; ++i) {
        expert_slot_t *s = &p->slot[i];
        if (s->used && s->expert_id == expert_id) {
            ++s->freq;
            s->last = now;
            s->class = class;
            return 1;
        }
    }
    ++p->pending;
    for (int i = 0; i < p->slots; ++i) {
        if (!p->slot[i].used) {
            victim = i;
            break;
        }
    }
    if (victim < 0) {
        int has_unprotected = 0;
        for (int i = 0; i < p->slots; ++i)
            if (p->slot[i].class != p->protected_class) has_unprotected = 1;
        for (int i = 0; i < p->slots; ++i) {
            expert_slot_t *s = &p->slot[i];
            int prot = s->class == p->protected_class;
            if (has_unprotected && prot) continue;
            if (victim < 0 || lrfu(s, now) < lrfu(&p->slot[victim], now))
                victim = i;
        }
        if (!has_unprotected) ++p->protected_evictions;
        ++p->evictions;
    } else {
        ++p->n;
    }
    p->slot[victim].expert_id = expert_id;
    p->slot[victim].class = class;
    p->slot[victim].freq = 1;
    p->slot[victim].last = now;
    p->slot[victim].used = 1;
    return 0;
}

int wubu_expert_cache_find(const wubu_expert_cache_t *p, int expert_id)
{
    if (!p || expert_id < 0) return -1;
    for (int i = 0; i < p->slots; ++i)
        if (p->slot[i].used && p->slot[i].expert_id == expert_id) return 1;
    return 0;
}

int wubu_expert_cache_count(const wubu_expert_cache_t *p)
{
    return p ? p->n : -1;
}

int wubu_expert_cache_evictions(const wubu_expert_cache_t *p)
{
    return p ? p->evictions : -1;
}

int wubu_expert_cache_protected_evictions(const wubu_expert_cache_t *p)
{
    return p ? p->protected_evictions : -1;
}

int wubu_expert_cache_wait_units(const wubu_expert_cache_t *p)
{
    return p ? 0 : -1;
}

int wubu_expert_cache_pending_fetches(const wubu_expert_cache_t *p)
{
    return p ? p->pending : -1;
}
