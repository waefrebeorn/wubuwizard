/*
 * wubu_ctxvm.c -- AGI-OS context virtual-memory hierarchy (AF08-AF10). C11.
 *
 * Convergence (LLM context as virtual memory / demand paging / 4-level 7-hop):
 *   - AF08 4-level context hierarchy: L1 gen window, L2 session, L3 long-term,
 *          L4 cross-session. Classify a memory by tier + promote/demote.
 *   - AF09 demand-paging eviction over a HIVE-backed ring: linked fixed
 *          blocks + skipfield + freelist (stable slot pointers, O(1) erase,
 *          cache-friendly iteration).  Eviction is importance-aware -- the
 *          lowest-importance slot is dropped, oldest wins ties -- replacing
 *          the old dense FIFO array, which could only drop by age.
 *   - AF10 semantic cache reuse across agents: two prompts within cosine sim
 *          threshold share a cached result (vector similarity).
 *
 * Pure C11, deterministic, testable.
 */
#include "wubu_ctxvm.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* AF08 tier classify by importance + ttl (mirrors agentic_mem tiers). */
int wubu_ctx_tier(float importance, long ttl) {
    if (importance >= 0.8f && ttl > 1000) return WUBU_CTX_L4;
    if (importance >= 0.4f && ttl > 10)   return WUBU_CTX_L3;
    if (ttl > 0)                          return WUBU_CTX_L2;
    return WUBU_CTX_L1;
}

int wubu_ctxring_init(wubu_ctxring_t *r, size_t capacity) {
    if (!r) return -1;
    if (wubu_hive_init(&r->hive) != 0) return -1;
    r->capacity = capacity;
    r->seq = 0;
    r->size = 0;
    return 0;
}

void wubu_ctxring_free(wubu_ctxring_t *r) {
    if (!r) return;
    /* free every owned slot payload, then the hive blocks */
    wubu_hive_clear_with(&r->hive, free);
    r->size = 0;
}

/* --- eviction: pick the lowest-importance slot (oldest wins ties) --- */
struct evict_scan {
    wubu_ctx_slot_t *victim;
    float best_imp;
    uint64_t best_seq;
    int found;
};

static int evict_scan_cb(void *ptr, void *user) {
    wubu_ctx_slot_t *s = (wubu_ctx_slot_t *)ptr;
    struct evict_scan *sc = (struct evict_scan *)user;
    if (!sc->found || s->importance < sc->best_imp ||
        (s->importance == sc->best_imp && s->seq < sc->best_seq)) {
        sc->victim = s;
        sc->best_imp = s->importance;
        sc->best_seq = s->seq;
        sc->found = 1;
    }
    return 0;
}

/* AF09 hive-backed demand-paging insert.  Evicts the lowest-importance
 * resident slot when over capacity; returns 1 if an eviction occurred. */
int wubu_ctx_evict_fifo(wubu_ctxring_t *r, long tok) {
    if (!r) return 0;
    int evicted = 0;
    if (r->capacity == 0) return 0;
    if (r->size >= r->capacity) {
        struct evict_scan sc;
        memset(&sc, 0, sizeof sc);
        wubu_hive_foreach(&r->hive, evict_scan_cb, &sc);
        if (sc.found && sc.victim) {
            wubu_hive_erase(&r->hive, sc.victim);   /* O(1) mark + freelist */
            free(sc.victim);
            r->size--;
            evicted = 1;
        }
    }
    wubu_ctx_slot_t *s = (wubu_ctx_slot_t *)malloc(sizeof(*s));
    if (!s) return evicted;
    s->tok = tok;
    s->seq = r->seq++;
    s->importance = 0.0f;
    s->ttl = 0;
    if (wubu_hive_insert(&r->hive, s) != 0) { free(s); return evicted; }
    r->size++;
    return evicted;
}

/* set a resident slot's importance + ttl (drives eviction) */
struct imp_scan { long tok; float importance; long ttl; int found; };

static int imp_cb(void *ptr, void *user) {
    wubu_ctx_slot_t *s = (wubu_ctx_slot_t *)ptr;
    struct imp_scan *sc = (struct imp_scan *)user;
    if (s->tok == sc->tok) {
        s->importance = sc->importance;
        s->ttl = sc->ttl;
        sc->found = 1;
        return 1;                     /* stop */
    }
    return 0;
}

int wubu_ctx_importance(wubu_ctxring_t *r, long tok, float importance, long ttl) {
    if (!r) return -1;
    struct imp_scan sc;
    sc.tok = tok; sc.importance = importance; sc.ttl = ttl; sc.found = 0;
    wubu_hive_foreach(&r->hive, imp_cb, &sc);
    return sc.found ? 0 : -1;
}

/* working-set resident check: is `tok` among the `ws` most-recent insertions? */
struct resident_scan {
    long tok;
    uint64_t tok_seq;
    int found;
    size_t newer;      /* slots inserted after the token */
    size_t checked;
};

static int resident_cb(void *ptr, void *user) {
    wubu_ctx_slot_t *s = (wubu_ctx_slot_t *)ptr;
    struct resident_scan *rs = (struct resident_scan *)user;
    rs->checked++;
    if (s->tok == rs->tok) { rs->found = 1; rs->tok_seq = s->seq; }
    return 0;
}

static int resident_count_cb(void *ptr, void *user) {
    wubu_ctx_slot_t *s = (wubu_ctx_slot_t *)ptr;
    struct resident_scan *rs = (struct resident_scan *)user;
    if (rs->found && s->seq > rs->tok_seq) rs->newer++;
    return 0;
}

int wubu_ctx_resident(const wubu_ctxring_t *r, long tok, int ws) {
    if (!r || ws <= 0) return 0;
    struct resident_scan rs;
    memset(&rs, 0, sizeof rs);
    rs.tok = tok;
    wubu_hive_foreach(&r->hive, resident_cb, &rs);
    if (!rs.found) return 0;
    wubu_hive_foreach(&r->hive, resident_count_cb, &rs);
    return rs.newer < (size_t)ws ? 1 : 0;
}

/* AF10 cosine similarity of two equal-length vectors (for semantic cache). */
float wubu_cosine(const float *a, const float *b, int n) {
    if (!a || !b || n <= 0) return 0.0f;
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    if (na <= 0 || nb <= 0) return 0.0f;
    return (float)(dot / (sqrt(na) * sqrt(nb)));
}

/* AF10: share cache if cosine >= thr. */
int wubu_sem_cache_hit(const float *q, const float *cached, int n, float thr) {
    return wubu_cosine(q, cached, n) >= thr ? 1 : 0;
}
