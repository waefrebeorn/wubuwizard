/* test_eg75_sparse_retrieve.c -- EG-75 gate: training-free sparse-KV retrieval. */
#include <stdio.h>
#include <string.h>

#include "wubu_sparse_retrieve.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    wubu_sparse_retrieve_t *p = wubu_sparse_retrieve_create(4, 16, 2);
    float keys[8][4] = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1},
                        {2,0,0,0},{0,2,0,0},{0,0,2,0},{0,0,0,2}};
    float q[4] = {1.5f, 0.0f, 0.0f, 0.0f};
    int pages[4];
    double scores[4];
    int n;
    CHECK(p != NULL);
    for (int i = 0; i < 8; ++i)
        CHECK(wubu_sparse_retrieve_update(p, keys[i], 4) == 0);
    CHECK(wubu_sparse_retrieve_page_count(p) == 8);

    n = wubu_sparse_retrieve_query(p, q, 3, pages, scores);
    CHECK(n == 3);
    /* page 4 wins on score; the two most recent pages (6,7) are retained */
    CHECK(pages[0] == 4 && pages[1] == 6 && pages[2] == 7);
    CHECK(scores[0] > scores[1]);
    printf("EG-75 top3=[%d %d %d] scores=[%.1f %.1f %.1f] recent=2\n",
           pages[0], pages[1], pages[2], scores[0], scores[1], scores[2]);
    CHECK(wubu_sparse_retrieve_query(p, NULL, 3, pages, scores) == -1);
    CHECK(wubu_sparse_retrieve_update(p, NULL, 4) == -1);

    wubu_sparse_retrieve_free(p);
    if (failures) { fprintf(stderr, "test_eg75_sparse_retrieve: FAIL (%d)\n", failures); return 1; }
    printf("test_eg75_sparse_retrieve: PASS\n");
    return 0;
}
