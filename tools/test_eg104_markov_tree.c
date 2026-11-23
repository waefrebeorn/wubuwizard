/* test_eg104_markov_tree.c -- EG-104 gate: W=4 Markov branch prior + EG-21. */
#include <stdio.h>
#include <math.h>

#include "wubu_markov_w4.h"
#include "wubu_tree_verify.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    wubu_markov_w4_t m;
    int seq[4 * 100], k = 0, r;
    const int reps[5] = { 50, 30, 15, 4, 1 };
    const int nxt[5] = { 4, 5, 6, 7, 8 };
    const int hist[3] = { 1, 2, 3 };
    const int cand[6] = { 9, 8, 7, 6, 5, 4 };
    int pick[4];
    wubu_tree_node_t tree[5];
    float u[4] = { 0.5f, 0.5f, 0.5f, 0.5f };
    int out[8], n, i;

    for (i = 0; i < 5; i++)
        for (r = 0; r < reps[i]; r++) {
            seq[k++] = 1; seq[k++] = 2; seq[k++] = 3; seq[k++] = nxt[i];
        }
    CHECK(k == 400);
    CHECK(wubu_markov_w4_init(&m, 16) == 0);
    CHECK(wubu_markov_w4_observe(&m, seq, k) == 0);

    printf("EG-104 p4=%.4f p5=%.4f p6=%.4f p7=%.4f p8=%.4f\n",
           wubu_markov_w4_prob(&m, hist, 3, 4),
           wubu_markov_w4_prob(&m, hist, 3, 5),
           wubu_markov_w4_prob(&m, hist, 3, 6),
           wubu_markov_w4_prob(&m, hist, 3, 7),
           wubu_markov_w4_prob(&m, hist, 3, 8));
    CHECK(fabs(wubu_markov_w4_prob(&m, hist, 3, 4) - 0.50) < 1e-9);
    CHECK(fabs(wubu_markov_w4_prob(&m, hist, 3, 5) - 0.30) < 1e-9);
    CHECK(fabs(wubu_markov_w4_prob(&m, hist, 3, 9)) < 1e-12);
    {
        const int unseen[3] = { 1, 2, 9 };
        CHECK(fabs(wubu_markov_w4_prob(&m, unseen, 3, 4) - 1.0 / 16.0) < 1e-12);
    }

    CHECK(wubu_markov_w4_select(&m, hist, 3, cand, 6, 4, pick, 4) == 4);
    printf("EG-104 selected tokens=[%d %d %d %d] (from shuffled candidates)\n",
           cand[pick[0]], cand[pick[1]], cand[pick[2]], cand[pick[3]]);
    CHECK(cand[pick[0]] == 4 && cand[pick[1]] == 5 &&
          cand[pick[2]] == 6 && cand[pick[3]] == 7);
    printf("EG-104 nodes full=7 beam=5 (1 root + 4 branches)\n");

    tree[0] = (wubu_tree_node_t){ -1, 0, 1.0f, 0 };
    for (i = 0; i < 4; i++)
        tree[i + 1] = (wubu_tree_node_t){ i, cand[pick[i]], 1.0f, 100 + i };
    n = wubu_tree_verify(tree, 5, u, 4, out, 8);
    printf("EG-21 accepted=%d tokens=[%d %d %d %d]\n",
           n, out[0], out[1], out[2], out[3]);
    CHECK(n == 4);
    CHECK(out[0] == 4 && out[1] == 5 && out[2] == 6 && out[3] == 7);

    CHECK(wubu_markov_w4_select(&m, hist, 3, cand, 6, 4, pick, 3) == -1);
    CHECK(wubu_markov_w4_select(&m, hist, 3, cand, 6, 5, pick, 4) == -1);
    CHECK(wubu_markov_w4_select(&m, hist, 3, cand, 0, 4, pick, 4) == -1);
    CHECK(wubu_markov_w4_observe(&m, seq, 3) == -1);
    CHECK(wubu_markov_w4_init(&m, 1) == -1);
    CHECK(wubu_markov_w4_free(&m) == 0);

    if (failures) {
        fprintf(stderr, "test_eg104_markov_tree: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg104_markov_tree: PASS\n");
    return 0;
}
