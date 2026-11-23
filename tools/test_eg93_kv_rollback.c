/* test_eg93_kv_rollback.c -- EG-93 gate: bounded-checkpoint KV prefix rollback. */
#include <stdio.h>
#include <stdint.h>

#include "wubu_kv_rollback.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

static uint64_t ref_hash(const int *ring, int w, int valid, int idx) {
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < w; ++i) { h ^= (uint64_t)(unsigned)ring[i]; h *= 1099511628211ULL; }
    h ^= (uint64_t)(unsigned)valid; h *= 1099511628211ULL;
    h ^= (uint64_t)(unsigned)idx; h *= 1099511628211ULL;
    return h;
}

#define WINDOW 8

static void prefill_run(int n, const int *tokens, int *ring, int *valid, int *idx) {
    for (int i = 0; i < n; ++i) {
        ring[(i) % WINDOW] = tokens[i];
        *valid = i + 1 < WINDOW ? i + 1 : WINDOW;
        *idx = (i + 1) % WINDOW;
    }
}

int main(void) {
    wubu_kv_rollback_t *p = wubu_kv_rollback_create(WINDOW);
    int tokens[20];
    for (int i = 0; i < 20; ++i) tokens[i] = 10 + i;
    CHECK(p != NULL);
    CHECK(wubu_kv_rollback_position(p) == 0);

    CHECK(wubu_kv_rollback_prefill(p, tokens, 10) == 0);
    CHECK(wubu_kv_rollback_position(p) == 10);
    CHECK(wubu_kv_rollback_checkpoint(p) == 0);
    CHECK(wubu_kv_rollback_prefill(p, tokens + 10, 10) == 0);
    CHECK(wubu_kv_rollback_checkpoint(p) == 0);
    CHECK(wubu_kv_rollback_checkpoint(p) == 0);
    CHECK(wubu_kv_rollback_checkpoint_count(p) == 2);
    printf("EG-93 checkpoints capped at %d (position=%d)\n",
           wubu_kv_rollback_checkpoint_count(p), wubu_kv_rollback_position(p));

    {
        int ring[WINDOW], valid = 0, idx = 0;
        prefill_run(20, tokens, ring, &valid, &idx);
        CHECK(wubu_kv_rollback_ring_hash(p) == ref_hash(ring, WINDOW, valid, idx));
        printf("EG-93 ring hash=0x%016llx matches independent prefill\n",
               (unsigned long long)wubu_kv_rollback_ring_hash(p));
    }

    {
        wubu_kv_rollback_t *q = wubu_kv_rollback_create(WINDOW);
        wubu_kv_rollback_t *fresh = wubu_kv_rollback_create(WINDOW);
        wubu_kv_rollback_t *r = wubu_kv_rollback_create(WINDOW);
        int newp[17], shortp[3] = {1, 2, 3}, sel;
        for (int i = 0; i < 15; ++i) newp[i] = 10 + i;
        newp[15] = 99; newp[16] = 98;
        CHECK(q && fresh && r);
        CHECK(wubu_kv_rollback_prefill(q, tokens, 10) == 0);
        CHECK(wubu_kv_rollback_checkpoint(q) == 0);
        CHECK(wubu_kv_rollback_prefill(q, tokens + 10, 10) == 0);
        CHECK(wubu_kv_rollback_checkpoint(q) == 0);
        sel = wubu_kv_rollback_to_prompt(q, newp, 17);
        CHECK(sel == 10);                       /* newest checkpoint <= common=15 */
        CHECK(wubu_kv_rollback_position(q) == 17);
        CHECK(wubu_kv_rollback_checkpoint_count(q) == 1); /* later cp discarded */
        CHECK(wubu_kv_rollback_prefill(fresh, newp, 17) == 0);
        CHECK(wubu_kv_rollback_ring_hash(q) == wubu_kv_rollback_ring_hash(fresh));
        printf("EG-93 rollback cp=%d position=%d ring matches fresh prefill\n",
               sel, wubu_kv_rollback_position(q));
        CHECK(wubu_kv_rollback_prefill(r, tokens, 10) == 0);
        CHECK(wubu_kv_rollback_checkpoint(r) == 0);
        CHECK(wubu_kv_rollback_to_prompt(r, shortp, 3) == -1);
        CHECK(wubu_kv_rollback_to_prompt(NULL, newp, 17) == -1);
        wubu_kv_rollback_free(q);
        wubu_kv_rollback_free(fresh);
        wubu_kv_rollback_free(r);
    }

    wubu_kv_rollback_free(p);
    if (failures) { fprintf(stderr, "test_eg93_kv_rollback: FAIL (%d)\n", failures); return 1; }
    printf("test_eg93_kv_rollback: PASS\n");
    return 0;
}
