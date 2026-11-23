/* wubu_kv_rollback.c -- EG-93 bounded-checkpoint sliding KV. Strict C11. */
#include "wubu_kv_rollback.h"

#include <stdlib.h>
#include <string.h>

struct kv_checkpoint {
    int position, valid, idx;
    int ring[WUBU_KV_MAX_WINDOW];
};

struct wubu_kv_rollback {
    int window, position, valid, idx;
    int hist_n, cp_n;
    int ring[WUBU_KV_MAX_WINDOW];
    int history[WUBU_KV_MAX_HISTORY];
    struct kv_checkpoint cp[WUBU_KV_MAX_CP];
};

wubu_kv_rollback_t *wubu_kv_rollback_create(int window)
{
    wubu_kv_rollback_t *p;
    if (window < 1 || window > WUBU_KV_MAX_WINDOW) return NULL;
    p = (wubu_kv_rollback_t *)calloc(1, sizeof(*p));
    if (p) p->window = window;
    return p;
}

void wubu_kv_rollback_free(wubu_kv_rollback_t *p) { free(p); }

int wubu_kv_rollback_prefill(wubu_kv_rollback_t *p, const int *tokens, int n)
{
    if (!p || !tokens || n < 0 || n > WUBU_KV_MAX_HISTORY - p->hist_n) return -1;
    for (int i = 0; i < n; ++i) {
        p->ring[p->position % p->window] = tokens[i];
        p->valid = p->position + 1 < p->window ? p->position + 1 : p->window;
        p->idx = (p->position + 1) % p->window;
        p->history[p->hist_n++] = tokens[i];
        ++p->position;
    }
    return 0;
}

int wubu_kv_rollback_checkpoint(wubu_kv_rollback_t *p)
{
    struct kv_checkpoint *c;
    if (!p) return -1;
    if (p->cp_n == WUBU_KV_MAX_CP) {
        p->cp[0] = p->cp[1];
        p->cp_n = 1;
    }
    c = &p->cp[p->cp_n++];
    c->position = p->position;
    c->valid = p->valid;
    c->idx = p->idx;
    for (int i = 0; i < WUBU_KV_MAX_WINDOW; ++i) c->ring[i] = p->ring[i];
    return 0;
}

int wubu_kv_rollback_checkpoint_count(const wubu_kv_rollback_t *p)
{
    return p ? p->cp_n : -1;
}

int wubu_kv_rollback_position(const wubu_kv_rollback_t *p)
{
    return p ? p->position : -1;
}

uint64_t wubu_kv_rollback_ring_hash(const wubu_kv_rollback_t *p)
{
    uint64_t h = 1469598103934665603ULL;
    if (!p) return 0;
    for (int i = 0; i < p->window; ++i) {
        h ^= (uint64_t)(unsigned)p->ring[i];
        h *= 1099511628211ULL;
    }
    h ^= (uint64_t)(unsigned)p->valid; h *= 1099511628211ULL;
    h ^= (uint64_t)(unsigned)p->idx; h *= 1099511628211ULL;
    return h;
}

int wubu_kv_rollback_to_prompt(wubu_kv_rollback_t *p,
                               const int *new_tokens, int n)
{
    int common = 0, sel = -1, chosen_position;
    if (!p || !new_tokens || n < 0 || n > WUBU_KV_MAX_HISTORY) return -1;
    while (common < p->hist_n && common < n &&
           p->history[common] == new_tokens[common]) ++common;
    /* Checkpoints are stored in increasing position order. */
    for (int i = p->cp_n - 1; i >= 0; --i) {
        if (p->cp[i].position <= common) { sel = i; break; }
    }
    if (sel < 0) return -1;
    chosen_position = p->cp[sel].position;
    p->position = chosen_position;
    p->valid = p->cp[sel].valid;
    p->idx = p->cp[sel].idx;
    memcpy(p->ring, p->cp[sel].ring, sizeof(p->ring));
    p->hist_n = chosen_position;
    p->cp_n = sel + 1;                      /* discard later checkpoints */
    /* Replay the suffix through prefill only; decode KV is never staged. */
    if (wubu_kv_rollback_prefill(p, new_tokens + chosen_position,
                                 n - chosen_position) != 0) return -1;
    return chosen_position;
}
