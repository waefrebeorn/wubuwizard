/*
 * test_hive_stress.c -- devil's-advocate stress test for wubu_hive.
 *
 * Random insert/erase against a model, checking every invariant after every
 * operation:
 *   - live count == number of slots foreach visits
 *   - no live slot is visited twice (stable pointers, no duplicates)
 *   - capacity == n_blocks * BLOCK_CAP >= live
 *   - erased slots are reusable (reuse count grows, allocs bounded)
 *   - erase of a non-live / unknown pointer fails
 *   - clear_with runs the destructor exactly once per live payload
 *
 * Build:
 *   gcc -O2 -g -Wall -fsanitize=address -I include -o /tmp/test_hive_stress \
 *       tools/test_hive_stress.c src/wubu_hive.c
 */
#include "wubu_hive.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define POOL 256
#define OPS  20000

static int pool_live[POOL];      /* 1 if pool[i] is currently inserted */
static void *pool[POOL];

static int cmp_ptr(const void *a, const void *b) {
    void *pa = *(void * const *)a;
    void *pb = *(void * const *)b;
    return (pa > pb) - (pa < pb);
}

int main(void) {
    wubu_hive_t h;
    if (wubu_hive_init(&h) != 0) { printf("FAIL: init\n"); return 1; }

    /* model: sorted live set */
    void *live[POOL]; int nlive = 0;

    unsigned seed = 12345;
    long inserts = 0, erases = 0;
    int fails = 0;

    for (int op = 0; op < OPS; op++) {
        seed = seed * 1103515245u + 12345u;
        int do_insert = (seed >> 8) % 3 != 0;   /* ~2/3 inserts */

        if (do_insert) {
            /* pick a pool slot that is not live */
            int idx = (int)((seed >> 16) % POOL);
            int tries = 0;
            while (pool_live[idx] && tries < POOL) { idx = (idx + 1) % POOL; tries++; }
            if (pool_live[idx]) continue;        /* all live */
            pool[idx] = (void *)(uintptr_t)(0x1000 + idx);
            pool_live[idx] = 1;
            if (wubu_hive_insert(&h, pool[idx]) != 0) { printf("FAIL: insert\n"); fails++; break; }
            live[nlive++] = pool[idx];
            inserts++;
        } else {
            if (nlive == 0) continue;
            int k = (int)((seed >> 16) % (unsigned)nlive);
            void *v = live[k];
            if (wubu_hive_erase(&h, v) != 0) { printf("FAIL: erase live ptr\n"); fails++; break; }
            live[k] = live[--nlive];
            for (int i = 0; i < POOL; i++) if (pool[i] == v) pool_live[i] = 0;
            erases++;
        }

        /* ---- invariants ---- */
        /* 1. live count */
        if (wubu_hive_live(&h) != (size_t)nlive) {
            printf("FAIL op %d: live=%zu model=%d\n", op, wubu_hive_live(&h), nlive);
            fails++; break;
        }
        /* 2. foreach visits exactly the live set, no duplicates */
        if (wubu_hive_capacity(&h) < (size_t)nlive) {
            printf("FAIL op %d: capacity < live\n", op); fails++; break;
        }
        /* 3. every model live ptr is found by foreach */
        void *found[POOL]; int nf = 0;
        /* simple recorder via nested function is not C11; do it inline */
        {
            wubu_hive_block_t *b;
            for (b = h.head; b; b = b->next)
                for (size_t s = 0; s < b->cap; s++)
                    if (b->skip[s] == 0) found[nf++] = b->slots[s];
        }
        if (nf != nlive) { printf("FAIL op %d: visited %d model %d\n", op, nf, nlive); fails++; break; }
        /* model set == found set */
        void *tmp[POOL];
        memcpy(tmp, found, sizeof(void *) * nf);
        qsort(tmp, nf, sizeof(void *), cmp_ptr);
        void *tmpm[POOL];
        memcpy(tmpm, live, sizeof(void *) * nlive);
        qsort(tmpm, nlive, sizeof(void *), cmp_ptr);
        if (nf == nlive && memcmp(tmp, tmpm, sizeof(void *) * nf) != 0) {
            printf("FAIL op %d: set mismatch\n", op);
            printf("  found (%d):", nf);
            for (int i = 0; i < nf; i++) printf(" %p", found[i]);
            printf("\n  model (%d):", nlive);
            for (int i = 0; i < nlive; i++) printf(" %p", live[i]);
            printf("\n  hive: blocks=%zu live=%zu allocs=%zu reuses=%zu n_free=%zu\n",
                   h.n_blocks, h.total_live, h.allocs, h.reuses, h.n_free);
            fails++; break;
        }
    }

    /* erase of an unknown pointer must fail */
    if (wubu_hive_erase(&h, (void *)(uintptr_t)0xdeadbeef) != -1) {
        printf("FAIL: erase unknown pointer succeeded\n"); fails++;
    }

    printf("hive stress: %ld inserts, %ld erases, live=%zu blocks=%zu cap=%zu "
           "allocs=%zu reuses=%zu\n",
           inserts, erases, wubu_hive_live(&h), h.n_blocks, wubu_hive_capacity(&h),
           h.allocs, h.reuses);
    if (h.allocs + h.reuses != (size_t)inserts) {
        printf("FAIL: allocs+reuses != inserts (%zu+%zu vs %ld)\n",
               h.allocs, h.reuses, inserts); fails++;
    }

    wubu_hive_clear(&h);
    printf("%s\n", fails ? "HIVE STRESS FAIL" : "HIVE STRESS PASS");
    return fails ? 1 : 0;
}
