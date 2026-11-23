/* test_eg82_engram.c -- EG-82 gate: Engram conditional-memory lookup. */
#include <stdio.h>

#include "wubu_cond_mem.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    int primes[2] = {1009, 1013};
    wubu_cond_mem_t *p = wubu_cond_mem_create(2, primes, 3, 4);
    int toks[3] = {10, 20, 30}, b1[2], b2[2];
    float vec[4] = {1.0f, 2.0f, 3.0f, 4.0f}, out[4];
    CHECK(p != NULL);

    wubu_cond_mem_hash(p, toks, 3, b1);
    wubu_cond_mem_hash(p, toks, 3, b2);
    CHECK(b1[0] == b2[0] && b1[1] == b2[1]);
    CHECK(b1[0] >= 0 && b1[0] < 1009);
    CHECK(b1[1] >= 1009 && b1[1] < 1009 + 1013);
    CHECK(wubu_cond_mem_store(p, toks, 3, vec) == 0);
    CHECK(wubu_cond_mem_lookup(p, toks, 3, out) == 1);
    CHECK(out[0] == 1.0f && out[3] == 4.0f);
    printf("EG-82 buckets=[%d %d] disjoint ranges; store/lookup round-trip\n", b1[0], b1[1]);

    /* NFKC-style compression collapses surface variants to one lookup. */
    {
        int va[3] = {0x101, 20, 30}, vb[3] = {0x201, 20, 30};
        int ha[2], hb[2];
        float hidden[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        float mem[4] = {0.5f, 0.5f, 0.5f, 0.5f}, fused[4];
        CHECK(wubu_cond_mem_normalize(0x101) == wubu_cond_mem_normalize(0x201));
        wubu_cond_mem_hash(p, va, 3, ha);
        wubu_cond_mem_hash(p, vb, 3, hb);
        CHECK(ha[0] == hb[0] && ha[1] == hb[1]);
        CHECK(wubu_cond_mem_fuse(p, hidden, mem, 0.5f, fused) == 0);
        CHECK(fused[0] == 1.25f && fused[3] == 4.25f);
        printf("EG-82 normalized variants share buckets; gated fuse=%.2f\n", fused[0]);
        CHECK(wubu_cond_mem_normalize(-1) == -1);
        CHECK(wubu_cond_mem_store(p, NULL, 3, vec) == -1);
        CHECK(wubu_cond_mem_lookup(p, va, 3, out) == 0);
        {
            int dup[2] = {1009, 1009};
            CHECK(wubu_cond_mem_create(2, dup, 3, 4) == NULL);
        }
        CHECK(wubu_cond_mem_fuse(NULL, hidden, mem, 0.5f, fused) == -1);
    }

    wubu_cond_mem_free(p);
    if (failures) { fprintf(stderr, "test_eg82_engram: FAIL (%d)\n", failures); return 1; }
    printf("test_eg82_engram: PASS\n");
    return 0;
}
