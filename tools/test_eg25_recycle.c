/* test_eg25_recycle.c -- EG-25b gate: token recycling from verified chains. */
#include <stdio.h>

#include "wubu_token_recycle.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    wubu_token_recycle_t *p = wubu_token_recycle_create(8, 16);
    int chain[5] = {7, 20, 30, 40, 50};
    int out[16];
    int n;
    CHECK(p != NULL);
    CHECK(wubu_token_recycle_record(p, chain, 5) == 0);
    CHECK(wubu_token_recycle_count(p) == 1);

    n = wubu_token_recycle_fetch(p, 7, out, 16);
    CHECK(n == 4);
    CHECK(out[0] == 20 && out[1] == 30 && out[2] == 40 && out[3] == 50);
    printf("EG-25b recycled anchor=7 chain=[%d %d %d %d] count=%d\n",
           out[0], out[1], out[2], out[3], n);

    /* Anchor id -1 is a legal token id, not an empty-slot sentinel. */
    {
        int neg[3] = {-1, 60, 70}, older[4] = {7, 21, 22, 23};
        CHECK(wubu_token_recycle_fetch(p, 8, out, 16) == 0);
        CHECK(wubu_token_recycle_record(p, neg, 3) == 0);
        CHECK(wubu_token_recycle_fetch(p, -1, out, 16) == 2);
        CHECK(out[0] == 60 && out[1] == 70);
        CHECK(wubu_token_recycle_fetch(p, -1, out, 1) == 1);
        CHECK(out[0] == 60);
        CHECK(wubu_token_recycle_record(p, older, 4) == 0);
        CHECK(wubu_token_recycle_fetch(p, 7, out, 16) == 3);
        CHECK(out[0] == 21 && out[1] == 22 && out[2] == 23);
        printf("EG-25b newest anchor wins; cap truncates; no match=0\n");
        CHECK(wubu_token_recycle_record(p, NULL, 3) == -1);
        CHECK(wubu_token_recycle_record(p, neg, 1) == -1);
        CHECK(wubu_token_recycle_fetch(p, 7, NULL, 4) == -1);
        CHECK(wubu_token_recycle_fetch(p, 7, out, 0) == -1);
        CHECK(wubu_token_recycle_create(0, 16) == NULL);
        CHECK(wubu_token_recycle_create(8, 0) == NULL);
        CHECK(wubu_token_recycle_record(NULL, neg, 3) == -1);
    }

    wubu_token_recycle_free(p);
    if (failures) { fprintf(stderr, "test_eg25_recycle: FAIL (%d)\n", failures); return 1; }
    printf("test_eg25_recycle: PASS\n");
    return 0;
}
