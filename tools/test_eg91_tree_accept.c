/* test_eg91_tree_accept.c -- EG-91 gate: pointer-jumping tree accept + KV perm. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "wubu_tree_accept.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    int parent[8] = {-1, 0, 0, 1, 1, 2, 2, 3};
    int match[8]  = {1, 1, 0, 1, 1, 1, 0, 1};
    int path[8];
    int len = wubu_tree_accept_jump(parent, match, 8, path);

    CHECK(len == 4);
    CHECK(path[0] == 0 && path[1] == 1 && path[2] == 3 && path[3] == 7);
    printf("EG-91 tree n=8 accepted_path=[%d %d %d %d] len=%d\n",
           path[0], path[1], path[2], path[3], len);

    /* ceil(log2 n) doubling rounds, independently recomputed. */
    for (int n = 1; n <= 64; n++) {
        int want = 0;
        while ((1 << want) < n) ++want;
        CHECK(wubu_tree_rounds(n) == want);
    }
    printf("EG-91 rounds n=8 -> %d (ceil log2), n=32 -> %d\n",
           wubu_tree_rounds(8), wubu_tree_rounds(32));

    /* KV slots: accepted path first, remainder ascending and complete. */
    {
        int slots[8], expect[8] = {0, 1, 3, 7, 2, 4, 5, 6};
        CHECK(wubu_tree_kv_permute(path, len, 8, slots) == 8);
        CHECK(memcmp(slots, expect, sizeof(expect)) == 0);
        printf("EG-91 kv_slots=[%d %d %d %d %d %d %d %d]\n",
               slots[0], slots[1], slots[2], slots[3],
               slots[4], slots[5], slots[6], slots[7]);
    }

    /* 200 hashed-random trees: doubling result equals a linear path walk. */
    {
        uint64_t seed = 0x9e3779b97f4a7c15ULL;
        int mismatches = 0;
        for (int trial = 0; trial < 200; trial++) {
            int rp[32], rm[32], got[32], want_path[32];
            int n, glen, wlen, best = 0, best_depth = 0;
            seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
            n = 2 + (int)((seed >> 33) % 31u);
            rp[0] = -1; rm[0] = 1;
            for (int i = 1; i < n; i++) {
                seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
                rp[i] = (int)((seed >> 33) % (unsigned)i);
                rm[i] = (int)((seed >> 62) & 1u);
            }
            glen = wubu_tree_accept_jump(rp, rm, n, got);
            for (int i = 1; i < n; i++) {
                int ok = 1, depth = 0, at = i;
                while (at > 0) {
                    if (!rm[at]) { ok = 0; break; }
                    ++depth; at = rp[at];
                }
                if (ok && depth > best_depth) { best = i; best_depth = depth; }
            }
            wlen = 0;
            for (int at = best; at >= 0; at = rp[at]) want_path[wlen++] = at;
            for (int i = 0; i < wlen / 2; i++) {
                int t = want_path[i];
                want_path[i] = want_path[wlen - 1 - i];
                want_path[wlen - 1 - i] = t;
            }
            if (glen != wlen || memcmp(got, want_path, (size_t)wlen * sizeof(int))) mismatches++;
        }
        CHECK(mismatches == 0);
        printf("EG-91 random_trees=200 mismatches=%d\n", mismatches);
    }

    /* Bad input is rejected by status. */
    {
        int badp[2] = {-1, 5}, bm[2] = {1, 1};
        CHECK(wubu_tree_rounds(0) == -1);
        CHECK(wubu_tree_accept_jump(badp, bm, 2, path) == -1);
        CHECK(wubu_tree_accept_jump(NULL, bm, 8, path) == -1);
        CHECK(wubu_tree_accept_jump(parent, match, 8, NULL) == -1);
        CHECK(wubu_tree_kv_permute(path, 9, 8, path) == -1);
    }

    if (failures) { fprintf(stderr, "test_eg91_tree_accept: FAIL (%d)\n", failures); return 1; }
    printf("test_eg91_tree_accept: PASS\n");
    return 0;
}
