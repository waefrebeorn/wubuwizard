/* test_kv_alloc_exact.c -- gate for LP-05 (atomic exact-ID KV allocation). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wubu_kv_exact.h"

static int checks = 0;
static int failures = 0;
#define CHECK(c) do { ++checks; if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++failures; \
} } while (0)

/* Full observable state, captured for byte-identical assertions. */
typedef struct {
    unsigned char bytes[8192];
    size_t n;
} snap_t;

static snap_t grab(const wubu_kv_exact_t *a)
{
    snap_t s;
    memset(&s, 0, sizeof(s));
    s.n = wubu_kv_exact_snapshot_size(a);
    CHECK(s.n <= sizeof(s.bytes));
    if (s.n > sizeof(s.bytes)) { s.n = 0; return s; }
    wubu_kv_exact_snapshot(a, s.bytes);
    return s;
}

static int same(snap_t x, snap_t y)
{
    return x.n == y.n && memcmp(x.bytes, y.bytes, x.n) == 0;
}

/* Run a failing call and assert both the status and byte-identity. */
static void expect_failure(wubu_kv_exact_t *a, const int *ids, int n,
                           wubu_kv_exact_status_t want, const char *label)
{
    snap_t before = grab(a);
    wubu_kv_exact_status_t st = wubu_kv_exact_alloc_exact(a, ids, n);
    snap_t after = grab(a);
    (void)label;
    CHECK(st == want);
    CHECK(same(before, after));
}

int main(void)
{
    wubu_kv_exact_t *a;
    const int valid3[]   = { 1, 2, 3 };
    const int neg[]      = { -1, 2 };
    const int dup[]      = { 1, 1, 2 };
    const int oor[]      = { 8 };
    const int quota5[]   = { 0, 1, 2, 3, 4 };

    /* --- create / introspection --- */
    a = wubu_kv_exact_create(8, 4);
    CHECK(a != NULL);
    if (!a) { fprintf(stderr, "test_kv_alloc_exact: no allocator\n"); return 1; }
    CHECK(wubu_kv_exact_capacity(a) == 8);
    CHECK(wubu_kv_exact_quota(a) == 4);
    CHECK(wubu_kv_exact_free_count(a) == 8);

    /* --- can_alloc_exact is pure --- */
    {
        snap_t before = grab(a);
        CHECK(wubu_kv_exact_can_alloc_exact(a, valid3, 3) == WUBU_KV_EXACT_OK);
        CHECK(same(before, grab(a)));
    }
    {
        snap_t before = grab(a);
        CHECK(wubu_kv_exact_can_alloc_exact(a, neg, 2) == WUBU_KV_EXACT_ERR_BAD_ID);
        CHECK(wubu_kv_exact_can_alloc_exact(a, dup, 3) == WUBU_KV_EXACT_ERR_DUP_ID);
        CHECK(wubu_kv_exact_can_alloc_exact(a, oor, 1) == WUBU_KV_EXACT_ERR_BAD_ID);
        CHECK(wubu_kv_exact_can_alloc_exact(a, quota5, 5) == WUBU_KV_EXACT_ERR_QUOTA);
        CHECK(same(before, grab(a)));
    }

    /* --- every alloc failure leaves the object byte-identical --- */
    expect_failure(a, neg,    2, WUBU_KV_EXACT_ERR_BAD_ID, "negative id");
    expect_failure(a, dup,    3, WUBU_KV_EXACT_ERR_DUP_ID, "duplicate id");
    expect_failure(a, oor,    1, WUBU_KV_EXACT_ERR_BAD_ID, "out-of-range id");
    expect_failure(a, quota5, 5, WUBU_KV_EXACT_ERR_QUOTA,  "quota");
    {
        snap_t before = grab(a);
        CHECK(wubu_kv_exact_alloc_exact(a, NULL, 1) == WUBU_KV_EXACT_ERR_ARG);
        CHECK(wubu_kv_exact_alloc_exact(a, valid3, -1) == WUBU_KV_EXACT_ERR_ARG);
        CHECK(same(before, grab(a)));
    }
    wubu_kv_exact_free(a);

    /* --- successful exact allocation + state-dependent failures --- */
    a = wubu_kv_exact_create(8, 8);
    CHECK(a != NULL);
    if (!a) return 1;
    {
        const int want[] = { 2, 3, 5 };
        CHECK(wubu_kv_exact_alloc_exact(a, want, 3) == WUBU_KV_EXACT_OK);
        CHECK(wubu_kv_exact_free_count(a) == 5);
        CHECK(wubu_kv_exact_is_free(a, 2) == 0);
        CHECK(wubu_kv_exact_is_free(a, 3) == 0);
        CHECK(wubu_kv_exact_is_free(a, 5) == 0);
        CHECK(wubu_kv_exact_is_free(a, 0) == 1);
        CHECK(wubu_kv_exact_is_free(a, 4) == 1);
    }
    /* named slot already in use -> NOT_FREE (n fits in free_count) */
    {
        const int mixed[] = { 2, 4 };
        snap_t before = grab(a);
        CHECK(wubu_kv_exact_can_alloc_exact(a, mixed, 2) == WUBU_KV_EXACT_ERR_NOT_FREE);
        CHECK(same(before, grab(a)));
        expect_failure(a, mixed, 2, WUBU_KV_EXACT_ERR_NOT_FREE, "not free");
    }
    /* n exceeds free_count -> NO_CAPACITY (quota is not the blocker) */
    {
        const int too_many[] = { 0, 1, 4, 6, 7 };
        const int six[]      = { 0, 1, 4, 6, 7, 2 };
        CHECK(wubu_kv_exact_alloc_exact(a, too_many, 5) == WUBU_KV_EXACT_OK);
        CHECK(wubu_kv_exact_free_count(a) == 0);
        expect_failure(a, six, 6, WUBU_KV_EXACT_ERR_NO_CAPACITY, "no capacity");
        {
            const int one[] = { 0 };
            expect_failure(a, one, 1, WUBU_KV_EXACT_ERR_NO_CAPACITY, "full page");
        }
    }
    /* atomic release restores slots, then they can be re-allocated exactly */
    {
        const int back[]  = { 2, 3, 5 };
        const int one[]   = { 2 };
        snap_t before;
        CHECK(wubu_kv_exact_release(a, back, 3) == WUBU_KV_EXACT_OK);
        CHECK(wubu_kv_exact_free_count(a) == 3);
        CHECK(wubu_kv_exact_is_free(a, 2) == 1);
        /* releasing an already-free slot is refused without mutation */
        before = grab(a);
        CHECK(wubu_kv_exact_release(a, one, 1) == WUBU_KV_EXACT_ERR_NOT_IN_USE);
        CHECK(same(before, grab(a)));
        /* the restored exact slots are allocatable again */
        CHECK(wubu_kv_exact_alloc_exact(a, back, 3) == WUBU_KV_EXACT_OK);
        CHECK(wubu_kv_exact_free_count(a) == 0);
        CHECK(wubu_kv_exact_is_free(a, 5) == 0);
    }
    wubu_kv_exact_free(a);

    /* --- quota is checked independently of page capacity --- */
    a = wubu_kv_exact_create(4, 2);
    CHECK(a != NULL);
    if (!a) return 1;
    {
        const int three[] = { 0, 1, 2 };
        const int two[]   = { 0, 1 };
        CHECK(wubu_kv_exact_can_alloc_exact(a, three, 3) == WUBU_KV_EXACT_ERR_QUOTA);
        expect_failure(a, three, 3, WUBU_KV_EXACT_ERR_QUOTA, "small quota");
        CHECK(wubu_kv_exact_alloc_exact(a, two, 2) == WUBU_KV_EXACT_OK);
        CHECK(wubu_kv_exact_free_count(a) == 2);
    }
    wubu_kv_exact_free(a);

    if (failures == 0)
        printf("test_kv_alloc_exact: PASS\n");
    else
        fprintf(stderr, "test_kv_alloc_exact: FAIL (%d/%d checks)\n",
                failures, checks);
    return failures ? 1 : 0;
}
