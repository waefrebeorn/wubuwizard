/* test_eg44_gqa_index.c -- EG-44b gate: GQA grouping for sparse index. */
#include <stdio.h>

#include "wubu_gqa_index.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    wubu_gqa_index_t *p = wubu_gqa_index_create(4, 2, 6);
    /* scores[query_head * 6 + token] */
    float scores[4][6] = {{1, 5, 2, 8, 3, 4},
                          {3, 1, 6, 2, 7, 0},
                          {0, 2, 9, 1, 4, 3},
                          {2, 4, 1, 6, 0, 5}};
    int out[4];
    int n, work, naive;
    CHECK(p != NULL);
    CHECK(wubu_gqa_index_group_size(p) == 2);

    /* kv head 0 groups query heads 0 and 1: averaged scores peak at tokens
       3 (6.5) and 4 (5.0), so the shared index selects [3 4]. */
    n = wubu_gqa_index_select(p, &scores[0][0], 6, 0, 2, out);
    CHECK(n == 2);
    CHECK(out[0] == 3 && out[1] == 4);
    work = wubu_gqa_index_work(p, 2);
    naive = wubu_gqa_index_work_naive(p, 2);
    CHECK(work == 4 && naive == 8);
    printf("EG-44b G=2 shared index=[%d %d] work=%d vs naive=%d\n",
           out[0], out[1], work, naive);

    CHECK(wubu_gqa_index_create(3, 2, 6) == NULL);
    CHECK(wubu_gqa_index_select(p, NULL, 6, 0, 2, out) == -1);
    CHECK(wubu_gqa_index_select(p, &scores[0][0], 6, 2, 2, out) == -1);
    CHECK(wubu_gqa_index_select(p, &scores[0][0], 6, 0, 0, out) == -1);
    CHECK(wubu_gqa_index_group_size(NULL) == -1);
    {
        int shared[2];
        CHECK(wubu_gqa_index_select(p, &scores[0][0], 6, 1, 2, shared) == 2);
        CHECK(shared[0] == 2 && shared[1] == 5);
        CHECK(wubu_gqa_index_select(p, &scores[0][0], 5, 0, 2, shared) == -1);
        printf("EG-44b kv head 1 shared index=[%d %d]\n", shared[0], shared[1]);
    }

    wubu_gqa_index_free(p);
    if (failures) { fprintf(stderr, "test_eg44_gqa_index: FAIL (%d)\n", failures); return 1; }
    printf("test_eg44_gqa_index: PASS\n");
    return 0;
}
