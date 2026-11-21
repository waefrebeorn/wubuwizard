/* test_swa_reclaim.c -- gate for LP-06 (multi-owner SWA reclaim). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wubu_swa_reclaim.h"

static int checks = 0;
static int failures = 0;
#define CHECK(c) do { ++checks; if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++failures; \
} } while (0)

typedef struct {
    unsigned char bytes[8192];
    size_t n;
} snap_t;

static snap_t grab(const wubu_swa_reclaim_t *c)
{
    snap_t s;
    memset(&s, 0, sizeof(s));
    s.n = wubu_swa_reclaim_snapshot_size(c);
    CHECK(s.n <= sizeof(s.bytes));
    if (s.n > sizeof(s.bytes)) { s.n = 0; return s; }
    wubu_swa_reclaim_snapshot(c, s.bytes);
    return s;
}

static int same(snap_t x, snap_t y)
{
    return x.n == y.n && memcmp(x.bytes, y.bytes, x.n) == 0;
}

/* An unsafe reuse must be refused by BOTH entry points and change nothing. */
static void expect_unsafe(wubu_swa_reclaim_t *c, int cell, int owner,
                          int64_t pos)
{
    snap_t before = grab(c);
    CHECK(wubu_swa_reclaim_can_reuse(c, cell, owner, pos) == 0);
    CHECK(wubu_swa_reclaim_reuse(c, cell, owner, pos) == WUBU_SWA_ERR_UNSAFE);
    CHECK(same(before, grab(c)));
}

int main(void)
{
    wubu_swa_reclaim_t *c;

    c = wubu_swa_reclaim_create(2, 2);
    CHECK(c != NULL);
    if (!c) { fprintf(stderr, "test_swa_reclaim: no cache\n"); return 1; }

    /* Two owners with deliberately divergent bounds. */
    CHECK(wubu_swa_reclaim_set_bound(c, 0, 5) == WUBU_SWA_OK);
    CHECK(wubu_swa_reclaim_set_bound(c, 1, 20) == WUBU_SWA_OK);
    CHECK(wubu_swa_reclaim_bound(c, 0) == 5);
    CHECK(wubu_swa_reclaim_bound(c, 1) == 20);

    /* Empty cell is reusable by owner 0 at a fresh-enough position. */
    CHECK(wubu_swa_reclaim_can_reuse(c, 0, 0, 8) == 1);
    CHECK(wubu_swa_reclaim_reuse(c, 0, 0, 8) == WUBU_SWA_OK);
    CHECK(wubu_swa_reclaim_cell_pos(c, 0) == 8);
    CHECK(wubu_swa_reclaim_cell_owner(c, 0) == 0);

    /* Unsafe: pos 8 is expired for owner 1 (bound 20) but NOT for owner 0
     * (bound 5). Owner 0's live window vetoes owner 1's reuse. */
    expect_unsafe(c, 0, 1, 22);

    /* A position stale for the requester is refused too. */
    CHECK(wubu_swa_reclaim_can_reuse(c, 0, 0, 3) == 0);
    CHECK(wubu_swa_reclaim_can_reuse(c, -1, 0, 8) == 0);
    CHECK(wubu_swa_reclaim_can_reuse(c, 0, 9, 8) == 0);

    /* Advance ONLY owner 0 past 8: now the content is masked for both. */
    CHECK(wubu_swa_reclaim_set_bound(c, 0, 9) == WUBU_SWA_OK);
    CHECK(wubu_swa_reclaim_can_reuse(c, 0, 1, 22) == 1);
    CHECK(wubu_swa_reclaim_reuse(c, 0, 1, 22) == WUBU_SWA_OK);
    CHECK(wubu_swa_reclaim_cell_pos(c, 0) == 22);
    CHECK(wubu_swa_reclaim_cell_owner(c, 0) == 1);
    /* Overwritten position 8 is logged against its displaced owner (0). */
    CHECK(wubu_swa_reclaim_overwritten_count(c, 0) == 1);
    CHECK(wubu_swa_reclaim_overwritten_at(c, 0, 0) == 8);
    CHECK(wubu_swa_reclaim_overwritten_count(c, 1) == 0);

    /* Cell now holds 22; unsafe until BOTH bounds pass 22. */
    expect_unsafe(c, 0, 0, 30);
    expect_unsafe(c, 0, 1, 30);
    CHECK(wubu_swa_reclaim_set_bound(c, 0, 23) == WUBU_SWA_OK);
    expect_unsafe(c, 0, 0, 30); /* owner 1 (bound 20) still needs 22 */
    CHECK(wubu_swa_reclaim_set_bound(c, 1, 23) == WUBU_SWA_OK);
    CHECK(wubu_swa_reclaim_can_reuse(c, 0, 0, 30) == 1);
    CHECK(wubu_swa_reclaim_reuse(c, 0, 0, 30) == WUBU_SWA_OK);
    CHECK(wubu_swa_reclaim_overwritten_count(c, 1) == 1);
    CHECK(wubu_swa_reclaim_overwritten_at(c, 1, 0) == 22);
    CHECK(wubu_swa_reclaim_overwritten_count(c, 0) == 1);
    CHECK(wubu_swa_reclaim_overwritten_at(c, 0, 0) == 8);

    /* Distinct error codes, each non-mutating. */
    {
        snap_t before = grab(c);
        CHECK(wubu_swa_reclaim_reuse(c, -1, 0, 30) == WUBU_SWA_ERR_CELL);
        CHECK(wubu_swa_reclaim_reuse(c, 0, 9, 30) == WUBU_SWA_ERR_OWNER);
        CHECK(wubu_swa_reclaim_reuse(c, 0, 0, -1) == WUBU_SWA_ERR_POS);
        CHECK(wubu_swa_reclaim_set_bound(c, 9, 1) == WUBU_SWA_ERR_OWNER);
        CHECK(wubu_swa_reclaim_set_bound(c, 0, -1) == WUBU_SWA_ERR_POS);
        CHECK(same(before, grab(c)));
    }
    wubu_swa_reclaim_free(c);

    /* A cell written by owner 1 stays protected for owner 0, and vice versa:
     * advancing only the requester's bound is not enough. */
    c = wubu_swa_reclaim_create(2, 2);
    CHECK(c != NULL);
    if (!c) return 1;
    CHECK(wubu_swa_reclaim_set_bound(c, 0, 5) == WUBU_SWA_OK);
    CHECK(wubu_swa_reclaim_set_bound(c, 1, 20) == WUBU_SWA_OK);
    CHECK(wubu_swa_reclaim_reuse(c, 1, 1, 30) == WUBU_SWA_OK);
    expect_unsafe(c, 1, 0, 40);          /* owner 0 (bound 5) needs 30 */
    CHECK(wubu_swa_reclaim_set_bound(c, 0, 31) == WUBU_SWA_OK);
    expect_unsafe(c, 1, 0, 40);          /* owner 1 (bound 20) needs 30 */
    CHECK(wubu_swa_reclaim_set_bound(c, 1, 31) == WUBU_SWA_OK);
    CHECK(wubu_swa_reclaim_can_reuse(c, 1, 0, 40) == 1);
    CHECK(wubu_swa_reclaim_reuse(c, 1, 0, 40) == WUBU_SWA_OK);
    CHECK(wubu_swa_reclaim_overwritten_count(c, 1) == 1);
    CHECK(wubu_swa_reclaim_overwritten_at(c, 1, 0) == 30);
    wubu_swa_reclaim_free(c);

    if (failures == 0)
        printf("test_swa_reclaim: PASS\n");
    else
        fprintf(stderr, "test_swa_reclaim: FAIL (%d/%d checks)\n",
                failures, checks);
    return failures ? 1 : 0;
}
