/* test_eg73_suffix_spec.c -- EG-73 gate: suffix decoding from a corpus. */
#include <stdio.h>
#include <string.h>

#include "wubu_suffix_spec.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    wubu_suffix_t *p = wubu_suffix_create(4, 16);
    int finished[7] = {10, 20, 30, 40, 50, 60, 70};
    int prompt[6] = {99, 98, 40, 50, 60, 70};
    int out[6];
    int n;
    CHECK(p != NULL);

    CHECK(wubu_suffix_ingest(p, finished, 7) == 0);
    CHECK(wubu_suffix_corpus_count(p) == 1);
    n = wubu_suffix_match(p, prompt, 6, out, 6);
    CHECK(n == 4);                 /* longest suffix [40 50 60 70] */
    CHECK(out[0] == 40 && out[1] == 50 && out[2] == 60 && out[3] == 70);
    printf("EG-73 match_len=%d suffix=[%d %d %d %d]\n",
           n, out[0], out[1], out[2], out[3]);

    /* Unmatched slots repeat the last sampled token; oldest corpus ages out. */
    {
        int draft[6], none[6] = {1, 2, 3, 4, 5, 6};
        wubu_suffix_t *q = wubu_suffix_create(2, 16);
        int a[4] = {1, 2, 3, 4}, b[4] = {5, 6, 7, 8}, c[4] = {9, 10, 11, 12};
        int old[4] = {7, 8, 1, 2}, fresh[4] = {7, 8, 11, 12};
        CHECK(q != NULL);
        CHECK(wubu_suffix_fill(draft, 4, 6, 70) == 0);
        CHECK(draft[4] == 70 && draft[5] == 70);
        CHECK(wubu_suffix_match(p, none, 6, out, 6) == 0);
        CHECK(wubu_suffix_ingest(q, a, 4) == 0);
        CHECK(wubu_suffix_ingest(q, b, 4) == 0);
        CHECK(wubu_suffix_ingest(q, c, 4) == 0);
        CHECK(wubu_suffix_corpus_count(q) == 2);
        CHECK(wubu_suffix_match(q, old, 4, out, 4) == 0);
        CHECK(wubu_suffix_match(q, fresh, 4, out, 4) == 2);
        printf("EG-73 fill=2 repeats, oldest evicted, newest suffix matches\n");
        CHECK(wubu_suffix_create(0, 16) == NULL);
        CHECK(wubu_suffix_create(2, 1) == NULL);
        CHECK(wubu_suffix_ingest(NULL, a, 4) == -1);
        CHECK(wubu_suffix_match(NULL, fresh, 4, out, 4) == -1);
        wubu_suffix_free(q);
    }

    wubu_suffix_free(p);
    if (failures) { fprintf(stderr, "test_eg73_suffix_spec: FAIL (%d)\n", failures); return 1; }
    printf("test_eg73_suffix_spec: PASS\n");
    return 0;
}
