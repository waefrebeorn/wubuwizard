/* test_eg38_pages.c -- EG-38 gate: page-owning radix + draft rollback. */
#include <stdio.h>

#include "wubu_page_radix.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    int a[4] = { 1, 2, 3, 4 }, pa[2] = { 0, 1 };
    int b[4] = { 1, 2, 5, 6 }, pb[2] = { 2, 3 };
    int q[4] = { 1, 2, 3, 9 }, r[4] = { 1, 2, 5, 9 };
    int c[4] = { 1, 2, 3, 7 }, pc[2] = { 4, 5 };
    int pages[8], npages = 0, matched = 0, ea, eb, ec;
    wubu_page_radix_t *x = wubu_page_radix_create(2, 8);
    CHECK(x != NULL);
    if (!x) return 1;

    ea = wubu_page_radix_insert(x, a, 4, pa, 2);
    eb = wubu_page_radix_insert(x, b, 4, pb, 2);
    CHECK(ea == 0 && eb == 1);

    CHECK(wubu_page_radix_match(x, q, 4, &matched, pages, &npages) == ea);
    printf("EG-38 match q prefix=%d pages=%d [%d %d]\n",
           matched, npages, pages[0], pages[1]);
    CHECK(matched == 3 && npages == 2 && pages[0] == 0 && pages[1] == 1);
    CHECK(wubu_page_radix_refcount(x, 0) == 1);
    CHECK(wubu_page_radix_refcount(x, 1) == 1);

    CHECK(wubu_page_radix_match(x, r, 4, &matched, pages, &npages) == eb);
    printf("EG-38 match r prefix=%d pages=%d\n", matched, npages);
    CHECK(matched == 3 && npages == 2 && pages[0] == 2 && pages[1] == 3);

    /* Speculative branch borrows pages, then rolls back on rejection. */
    wubu_page_radix_snapshot(x);
    ec = wubu_page_radix_insert(x, c, 4, pc, 2);
    CHECK(ec == 2);
    CHECK(wubu_page_radix_match(x, c, 4, &matched, pages, &npages) == ec);
    CHECK(matched == 4 && npages == 2 && pages[0] == 4);
    CHECK(wubu_page_radix_refcount(x, 4) == 1);
    wubu_page_radix_restore(x);
    printf("EG-38 rollback refcounts p0=%d p2=%d p4=%d\n",
           wubu_page_radix_refcount(x, 0),
           wubu_page_radix_refcount(x, 2),
           wubu_page_radix_refcount(x, 4));
    CHECK(wubu_page_radix_refcount(x, 0) == 1);
    CHECK(wubu_page_radix_refcount(x, 2) == 1);
    CHECK(wubu_page_radix_refcount(x, 4) == 0);

    CHECK(wubu_page_radix_release(x, ea) == 0);
    CHECK(wubu_page_radix_release(x, eb) == 0);
    CHECK(wubu_page_radix_refcount(x, 0) == 0);
    CHECK(wubu_page_radix_refcount(x, 1) == 0);
    CHECK(wubu_page_radix_refcount(x, 2) == 0);
    CHECK(wubu_page_radix_refcount(x, 4) == 0);

    CHECK(wubu_page_radix_create(0, 8) == NULL);
    CHECK(wubu_page_radix_insert(NULL, a, 4, pa, 2) == -1);
    CHECK(wubu_page_radix_match(NULL, a, 4, NULL, NULL, NULL) == -1);
    CHECK(wubu_page_radix_release(x, 99) == -1);
    CHECK(wubu_page_radix_refcount(x, 99) == -1);

    wubu_page_radix_free(x);
    if (failures) {
        fprintf(stderr, "test_eg38_pages: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg38_pages: PASS\n");
    return 0;
}
