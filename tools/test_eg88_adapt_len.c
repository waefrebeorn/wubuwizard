/* test_eg88_adapt_len.c -- EG-88 gate: adaptive candidate length. */
#include <stdio.h>

#include "wubu_adapt_len.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    wubu_adapt_len_t *p = wubu_adapt_len_create(8, 0.5f);
    float p05[8] = {1.0f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    float p10[8] = {1.0f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f};
    float one[8] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    CHECK(p != NULL);

    /* cumprod from position 1: 0.5 -> 0.25, 1-cum=0.75 > 0.5 at k=2. */
    CHECK(wubu_adapt_len_choose(p, p05, 8) == 3);
    printf("EG-88 thresh=0.5 p=0.5 length=3 (baseline 8, 5 target rows saved)\n");

    /* Early rejection clamps to len_min=2; all-accept keeps the full block. */
    CHECK(wubu_adapt_len_choose(p, p10, 8) == 2);
    CHECK(wubu_adapt_len_choose(p, one, 8) == 8);
    printf("EG-88 early-reject length=2, all-accept length=8\n");

    /* Target rows are only evaluated on the retained prefix. */
    {
        int draft[8]  = {10, 20, 30, 40, 50, 60, 70, 80};
        int target[8] = {20, 99, 40, 50, 60, 70, 80, 90};
        int full[8]   = {20, 30, 40, 50, 60, 70, 80, 90};
        int L = wubu_adapt_len_choose(p, p05, 8);
        CHECK(L == 3);
        CHECK(wubu_adapt_len_accept(draft, target, L) == 1);
        CHECK(wubu_adapt_len_kv_rows(1) == 2);
        CHECK(wubu_adapt_len_accept(draft, full, 5) == 4);
        CHECK(wubu_adapt_len_kv_rows(4) == 5);
        printf("EG-88 prefix_len=3 accepted=1 kv_rows=2; full_len=5 accepted=4 kv_rows=5\n");
    }

    /* Invalid length, NULL pointers and bad threshold are rejected. */
    CHECK(wubu_adapt_len_create(1, 0.5f) == NULL);
    CHECK(wubu_adapt_len_create(8, 1.5f) == NULL);
    CHECK(wubu_adapt_len_choose(NULL, p05, 8) == -1);
    CHECK(wubu_adapt_len_choose(p, p05, 1) == -1);
    CHECK(wubu_adapt_len_accept(NULL, NULL, 3) == -1);
    CHECK(wubu_adapt_len_kv_rows(-1) == 0);

    wubu_adapt_len_free(p);
    if (failures) { fprintf(stderr, "test_eg88_adapt_len: FAIL (%d)\n", failures); return 1; }
    printf("test_eg88_adapt_len: PASS\n");
    return 0;
}
