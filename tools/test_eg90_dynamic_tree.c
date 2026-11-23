/* test_eg90_dynamic_tree.c -- EG-90 gate: EAGLE-2 confidence-shaped tree. */
#include <stdio.h>
#include <stdint.h>

#include "wubu_dynamic_tree.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    wubu_dyn_tree_t *p = wubu_dyn_tree_create(4, 2);
    int parent[6] = {-1, 0, 0, 1, 1, 2};
    double logp[6] = {0.0, -0.1, -0.8, -0.1, -0.2, -0.1};
    int selected[8], par_pos[8], depth[8], n;
    CHECK(p != NULL);

    /* Cumulative log-prob selects 0(0), 1(-0.1), 3(-0.2), 4(-0.3). */
    n = wubu_dyn_tree_build(p, parent, logp, 6, selected, par_pos, depth);
    CHECK(n == 4);
    CHECK(selected[0] == 0 && selected[1] == 1 &&
          selected[2] == 3 && selected[3] == 4);
    CHECK(par_pos[0] == -1 && par_pos[1] == 0 && par_pos[2] == 1 && par_pos[3] == 1);
    CHECK(depth[0] == 0 && depth[1] == 1 && depth[2] == 2 && depth[3] == 2);
    printf("EG-90 selected=[%d %d %d %d] parent_pos=[%d %d %d %d] depth=[%d %d %d %d]\n",
           selected[0], selected[1], selected[2], selected[3],
           par_pos[0], par_pos[1], par_pos[2], par_pos[3],
           depth[0], depth[1], depth[2], depth[3]);

    /* Confidence reshapes the tree: a better sibling displaces old branches. */
    {
        double shaped[6] = {0.0, -0.1, -0.05, -0.1, -0.2, -0.1};
        CHECK(wubu_dyn_tree_build(p, parent, shaped, 6,
                                  selected, par_pos, depth) == 4);
        CHECK(selected[1] == 2);
        printf("EG-90 shaped selected=[%d %d %d %d] (top-scoring sibling first)\n",
               selected[0], selected[1], selected[2], selected[3]);
    }

    /* Random trees: parent positions and depths match an independent walk. */
    {
        uint64_t seed = 0x9e3779b97f4a7c15ULL;
        int mismatches = 0;
        wubu_dyn_tree_t *q = wubu_dyn_tree_create(16, 3);
        CHECK(q != NULL);
        for (int trial = 0; trial < 100; ++trial) {
            int pr[32], sel[32], pp[32], dp[32], got;
            double lp[32];
            seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
            int n = 2 + (int)((seed >> 33) % 31u);
            pr[0] = -1; lp[0] = 0.0;
            for (int i = 1; i < n; ++i) {
                seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
                pr[i] = (int)((seed >> 33) % (unsigned)i);
                lp[i] = -0.05 * (double)(1 + (seed >> 60) % 17u);
            }
            got = wubu_dyn_tree_build(q, pr, lp, n, sel, pp, dp);
            if (got < 1 || got > 16) { ++mismatches; continue; }
            for (int i = 0; i < got; ++i) {
                int par = pr[sel[i]], want_pos = -1, want_depth = 0;
                for (int j = 0; j < got; ++j)
                    if (sel[j] == par) { want_pos = j; want_depth = dp[j] + 1; break; }
                if (par == -1) { want_pos = -1; want_depth = 0; }
                if (pp[i] != want_pos || dp[i] != want_depth) ++mismatches;
            }
        }
        CHECK(mismatches == 0);
        printf("EG-90 random_trees=100 parent/depth mismatches=%d\n", mismatches);
        wubu_dyn_tree_free(q);
    }

    /* Invalid parent, logp, buffers and capacities are rejected. */
    {
        int badp[2] = {-1, 5};
        double badl[2] = {0.0, 0.5};
        CHECK(wubu_dyn_tree_build(p, badp, logp, 2, selected, par_pos, depth) == -1);
        CHECK(wubu_dyn_tree_build(p, parent, badl, 2, selected, par_pos, depth) == -1);
        CHECK(wubu_dyn_tree_build(p, parent, logp, 6, NULL, par_pos, depth) == -1);
        CHECK(wubu_dyn_tree_create(0, 2) == NULL);
        CHECK(wubu_dyn_tree_create(4, 0) == NULL);
    }

    wubu_dyn_tree_free(p);
    if (failures) { fprintf(stderr, "test_eg90_dynamic_tree: FAIL (%d)\n", failures); return 1; }
    printf("test_eg90_dynamic_tree: PASS\n");
    return 0;
}
