/* test_eg107_virt_kv.c -- EG-107 gate: contiguous virtual KV, demand physical. */
#include <stdio.h>
#include <stdlib.h>

#include "wubu_virt_kv.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    wubu_virt_kv_t *p = wubu_virt_kv_create(4, 16, 4, 8);
    int seq[4] = {5, 0, 9, 0};
    int allocated[4] = {-1, -1, -1, -1};
    size_t mapped = 0;
    CHECK(p != NULL);
    CHECK(wubu_virt_kv_virtual_tokens(p) == 64);       /* batch * max_len */
    CHECK(wubu_virt_kv_alloc_req(p, 101) == 0);
    CHECK(wubu_virt_kv_alloc_req(p, 102) == 1);
    CHECK(wubu_virt_kv_offset(p, 0, 7) == 7);          /* contiguous per req */
    CHECK(wubu_virt_kv_offset(p, 1, 7) == 23);
    CHECK(wubu_virt_kv_step(p, seq, 4) == 0);
    mapped = wubu_virt_kv_mapped_pages(p);
    CHECK(mapped == 5);                                /* ceil(5/4)+ceil(9/4) */
    printf("EG-107 virtual_tokens=64 mapped_pages=%zu offsets=[%d %d]\n",
           mapped, (int)wubu_virt_kv_offset(p, 0, 7),
           (int)wubu_virt_kv_offset(p, 1, 7));

    /* Physical exhaustion and invalid inputs fail closed. */
    {
        wubu_virt_kv_t *q = wubu_virt_kv_create(2, 8, 4, 3);
        int part[2] = {8, 4}, big[2] = {8, 8};
        CHECK(q != NULL);
        CHECK(wubu_virt_kv_alloc_req(q, 1) == 0);
        CHECK(wubu_virt_kv_alloc_req(q, 2) == 1);
        CHECK(wubu_virt_kv_step(q, part, 2) == 0);       /* 3 pages */
        CHECK(wubu_virt_kv_step(q, big, 2) == -1);       /* would need 4 */
        CHECK(wubu_virt_kv_mapped_pages(q) == 3);
        printf("EG-107 demand mapping holds at 4 pages; overcommit refused\n");
        CHECK(wubu_virt_kv_create(0, 8, 4, 3) == NULL);
        CHECK(wubu_virt_kv_alloc_req(q, -1) == -1);
        CHECK(wubu_virt_kv_step(q, NULL, 2) == -1);
        CHECK(wubu_virt_kv_offset(q, 0, 8) == -1);
        wubu_virt_kv_free(q);
    }
    (void)allocated;

    wubu_virt_kv_free(p);
    if (failures) { fprintf(stderr, "test_eg107_virt_kv: FAIL (%d)\n", failures); return 1; }
    printf("test_eg107_virt_kv: PASS\n");
    return 0;
}
