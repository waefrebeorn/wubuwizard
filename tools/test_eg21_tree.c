/* test_eg21_tree.c -- EG-21 gate: tree attention mask + longest-chain verify. */
#include <stdio.h>
#include <string.h>

#include "wubu_tree_verify.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

static int row_sum(const unsigned char *m, int stride, int row, int n) {
    int s = 0;
    for (int j = 0; j < n; j++) s += m[row * stride + j];
    return s;
}

int main(void) {
    wubu_tree_node_t t[6] = {
        { -1,  0, 1.0f,   0 },
        {  0, 11, 1.0f, 101 },
        {  1, 12, 1.0f, 102 },
        {  2, 13, 1.0f, 103 },
        {  0, 14, 1.0f, 104 },
        {  4, 15, 0.0f, 105 }
    };
    float u[4] = { 0.5f, 0.5f, 0.5f, 0.5f };
    unsigned char mask[36];
    int out[8], n;

    CHECK(wubu_tree_mask(t, 6, mask, 6) == 0);
    CHECK(row_sum(mask, 6, 3, 6) == 4); /* root,1,2,3 */
    CHECK(row_sum(mask, 6, 5, 6) == 3); /* root,4,5 */
    CHECK(mask[3 * 6 + 2] == 1 && mask[3 * 6 + 4] == 0);

    n = wubu_tree_verify(t, 6, u, 4, out, 8);
    printf("EG-21 accepted=%d tokens=[%d %d %d]\n", n, out[0], out[1], out[2]);
    CHECK(n == 3);
    CHECK(out[0] == 11 && out[1] == 12 && out[2] == 13);

    /* Single rejected chain: emit the target's bonus token. */
    {
        wubu_tree_node_t b[3] = {
            { -1, 0, 1.0f, 0 },
            {  0, 14, 1.0f, 104 },
            {  1, 15, 0.0f, 105 }
        };
        float ur[1] = { 0.1f };
        n = wubu_tree_verify(b, 3, ur, 1, out, 8);
        printf("EG-21 rejected bonus=%d (n=%d)\n", out[0], n);
        CHECK(n == 2 && out[0] == 14 && out[1] == 105);

        b[2].accept_p = 0.9f;
        float ua[2] = { 0.1f, 0.1f };
        n = wubu_tree_verify(b, 3, ua, 2, out, 8);
        CHECK(n == 2 && out[0] == 14 && out[1] == 15);
    }

    CHECK(wubu_tree_verify(t, 6, u, 4, out, 1) == -1); /* too small */
    CHECK(wubu_tree_verify(t, 6, u, 4, NULL, 8) == -1);
    CHECK(wubu_tree_mask(t, 6, mask, 2) == -1);

    {
        wubu_tree_node_t bad[2] = { { -1, 0, 1.0f, 0 }, { 5, 1, 1.0f, 0 } };
        CHECK(wubu_tree_verify(bad, 2, u, 4, out, 8) == -1);
    }

    if (failures) {
        fprintf(stderr, "test_eg21_tree: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg21_tree: PASS\n");
    return 0;
}
