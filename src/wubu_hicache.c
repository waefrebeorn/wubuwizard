/*
 * wubu_hicache.c -- EG-18 three-tier cache. Pure C11.
 */
#include "wubu_hicache.h"

#include <stdlib.h>
#include <string.h>

#define WRITEBACK_MS 10ULL
#define PROMOTE_MS   50ULL

typedef struct {
    uint64_t key;
    unsigned char data[WUBU_HICACHE_ENTRY_BYTES];
    unsigned long long stamp;
} entry_t;

typedef struct {
    uint64_t key;
    unsigned char data[WUBU_HICACHE_ENTRY_BYTES];
    int active;
    int from_tier;      /* 2 or 3 */
    int to_tier;        /* 1 for prefetch, 2 for write-back */
    unsigned long long ready_ms;
} pending_t;

struct wubu_hicache {
    entry_t *tier[3];
    int cap[3];
    int count[3];
    pending_t *pending;
    int pending_cap;
    unsigned long long now, stamp;
    wubu_hicache_stats_t st;
};

wubu_hicache_t *wubu_hicache_create(int n1, int n2, int n3) {
    wubu_hicache_t *h = (wubu_hicache_t *)calloc(1, sizeof(*h));
    int i;
    if (!h) return NULL;
    h->cap[0] = n1; h->cap[1] = n2; h->cap[2] = n3;
    h->pending_cap = (n1 + n2 + n3) > 4 ? (n1 + n2 + n3) : 4;
    for (i = 0; i < 3; i++) {
        if (h->cap[i] <= 0) { wubu_hicache_free(h); return NULL; }
        h->tier[i] = (entry_t *)calloc((size_t)h->cap[i], sizeof(entry_t));
        if (!h->tier[i]) { wubu_hicache_free(h); return NULL; }
    }
    h->pending = (pending_t *)calloc((size_t)h->pending_cap, sizeof(pending_t));
    if (!h->pending) { wubu_hicache_free(h); return NULL; }
    return h;
}

void wubu_hicache_free(wubu_hicache_t *h) {
    int i;
    if (!h) return;
    for (i = 0; i < 3; i++) free(h->tier[i]);
    free(h->pending);
    free(h);
}

static entry_t *find_slot(wubu_hicache_t *h, int tier, uint64_t key, int *idx) {
    int i;
    for (i = 0; i < h->cap[tier]; i++) {
        if (h->tier[tier][i].stamp && h->tier[tier][i].key == key) {
            if (idx) *idx = i;
            return &h->tier[tier][i];
        }
    }
    return NULL;
}

static int find_lru(wubu_hicache_t *h, int tier) {
    int i, best = 0;
    unsigned long long oldest = h->tier[tier][0].stamp;
    for (i = 1; i < h->cap[tier]; i++) {
        if (h->tier[tier][i].stamp < oldest) {
            oldest = h->tier[tier][i].stamp;
            best = i;
        }
    }
    return best;
}

static void copy_in(entry_t *e, uint64_t key, const void *data) {
    e->key = key;
    memcpy(e->data, data, WUBU_HICACHE_ENTRY_BYTES);
    e->stamp = 0; /* caller assigns a nonzero stamp */
}

static int queue(wubu_hicache_t *h, const entry_t *e, int from, int to,
                 unsigned long long ready) {
    int i;
    for (i = 0; i < h->pending_cap; i++) {
        pending_t *p = &h->pending[i];
        if (!p->active) {
            p->active = 1;
            p->key = e->key;
            memcpy(p->data, e->data, WUBU_HICACHE_ENTRY_BYTES);
            p->from_tier = from;
            p->to_tier = to;
            p->ready_ms = ready;
            h->st.pending++;
            return 0;
        }
    }
    return -1; /* queue full: caller keeps the entry in place */
}

static void apply(wubu_hicache_t *h, const pending_t *p) {
    int idx;
    entry_t *slot;
    if (p->from_tier >= 0 && p->from_tier < 3) {
        slot = find_slot(h, p->from_tier, p->key, &idx);
        if (slot) { memset(slot, 0, sizeof(*slot)); h->count[p->from_tier]--; }
    }
    if (p->to_tier >= 0 && p->to_tier < 3) {
        int t = p->to_tier;
        if (find_slot(h, t, p->key, NULL)) { h->st.moves++; return; }
        if (h->count[t] == h->cap[t]) {
            int lru = find_lru(h, t);
            entry_t evicted = h->tier[t][lru];
            if (t == 1 && queue(h, &evicted, 1, 2, h->now + WRITEBACK_MS) != 0) {
                /* queue full: drop the evicted entry */
            }
            memset(&h->tier[t][lru], 0, sizeof(entry_t));
            h->count[t]--;
        }
        slot = &h->tier[t][find_lru(h, t)];
        copy_in(slot, p->key, p->data);
        h->stamp++;
        slot->stamp = h->stamp;
        h->count[t]++;
        h->st.moves++;
    }
}

static void insert_l1(wubu_hicache_t *h, uint64_t key, const void *data) {
    int idx;
    entry_t *slot = find_slot(h, 0, key, &idx);
    if (slot) {
        memcpy(slot->data, data, WUBU_HICACHE_ENTRY_BYTES);
        h->stamp++;
        slot->stamp = h->stamp;
        return;
    }
    if (h->count[0] == h->cap[0]) {
        int lru = find_lru(h, 0);
        entry_t evicted = h->tier[0][lru];
        if (queue(h, &evicted, 0, 1, h->now + WRITEBACK_MS) != 0)
            h->st.moves++; /* dropped */
        memset(&h->tier[0][lru], 0, sizeof(entry_t));
        h->count[0]--;
    }
    slot = &h->tier[0][find_lru(h, 0)];
    copy_in(slot, key, data);
    h->stamp++;
    slot->stamp = h->stamp;
    h->count[0]++;
}

int wubu_hicache_put(wubu_hicache_t *h, uint64_t key, const void *data) {
    if (!h || !data) return -1;
    insert_l1(h, key, data);
    return 0;
}

int wubu_hicache_get(wubu_hicache_t *h, uint64_t key, void *out) {
    int i, idx;
    entry_t *e;
    if (!h || !out) return -1;
    for (i = 0; i < 3; i++) {
        e = find_slot(h, i, key, &idx);
        if (!e) continue;
        memcpy(out, e->data, WUBU_HICACHE_ENTRY_BYTES);
        h->st.hits[i]++;
        h->stamp++;
        e->stamp = h->stamp;
        if (i > 0) {
            unsigned char tmp[WUBU_HICACHE_ENTRY_BYTES];
            memcpy(tmp, e->data, sizeof(tmp));
            memset(e, 0, sizeof(*e));
            h->count[i]--;
            insert_l1(h, key, tmp);
            h->st.moves++;
            return i + 1;
        }
        return 1;
    }
    h->st.misses++;
    return -1;
}

int wubu_hicache_prefetch(wubu_hicache_t *h, uint64_t key,
                          unsigned long long deadline_ms) {
    int i, idx;
    entry_t *e;
    if (!h) return -1;
    if (find_slot(h, 0, key, NULL)) return 0;
    for (i = 1; i < 3; i++) {
        e = find_slot(h, i, key, &idx);
        if (e) break;
    }
    if (i == 3) return -1;
    if (h->now + PROMOTE_MS > deadline_ms) return 0;
    return queue(h, e, i, 0, h->now + PROMOTE_MS) == 0 ? 1 : -1;
}

int wubu_hicache_tick(wubu_hicache_t *h, unsigned long long now_ms) {
    int i, done = 0;
    if (!h || now_ms < h->now) return -1;
    h->now = now_ms;
    for (i = 0; i < h->pending_cap; i++) {
        pending_t p = h->pending[i];
        if (!p.active || p.ready_ms > now_ms) continue;
        h->pending[i].active = 0;
        h->st.pending--;
        apply(h, &p);
        done++;
    }
    return done;
}

void wubu_hicache_stats(const wubu_hicache_t *h, wubu_hicache_stats_t *out) {
    if (!h || !out) return;
    *out = h->st;
    out->l1 = h->count[0];
    out->l2 = h->count[1];
    out->l3 = h->count[2];
    out->pending = h->st.pending;
}
