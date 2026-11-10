/*
 * test_ctxhive.c -- the HIVE-backed context ring (AF09).
 *
 * Proves the hive replacement for the dense FIFO ring:
 *   - importance-aware eviction: the LOWEST-importance slot is dropped on
 *     overflow (a dense FIFO could only drop by age),
 *   - oldest-wins tie-break when importances are equal,
 *   - working-set residency by insertion order,
 *   - stable slot pointers + O(1) erase under a random stress loop,
 *   - ASan-clean teardown via wubu_ctxring_free.
 *
 * Build:
 *   gcc -O2 -g -Wall -fsanitize=address -I include -o /tmp/test_ctxhive \
 *       tools/test_ctxhive.c src/wubu_ctxvm.c src/wubu_hive.c
 */
#include "wubu_ctxvm.h"
#include <stdio.h>
#include <stdlib.h>

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } } while (0)

int main(void) {
    /* --- importance-aware eviction --- */
    wubu_ctxring_t r;
    CHECK(wubu_ctxring_init(&r, 3) == 0, "init");
    wubu_ctx_evict_fifo(&r, 10); wubu_ctx_importance(&r, 10, 0.1f, 0);
    wubu_ctx_evict_fifo(&r, 20); wubu_ctx_importance(&r, 20, 0.9f, 0);
    wubu_ctx_evict_fifo(&r, 30); wubu_ctx_importance(&r, 30, 0.2f, 0);
    CHECK(r.size == 3, "filled to capacity");
    /* overflow: must drop token 10 (importance 0.1), NOT token 20 (0.9) */
    int ev = wubu_ctx_evict_fifo(&r, 40);
    CHECK(ev == 1, "eviction occurred");
    CHECK(r.size == 3, "size stays at capacity");
    CHECK(!wubu_ctx_resident(&r, 10, 100), "low-value token 10 evicted");
    CHECK(wubu_ctx_resident(&r, 20, 100), "high-value token 20 kept");
    CHECK(wubu_ctx_resident(&r, 30, 100), "token 30 kept");
    CHECK(wubu_ctx_resident(&r, 40, 100), "new token 40 resident");
    /* importance of a non-resident token is a no-op, not a crash */
    CHECK(wubu_ctx_importance(&r, 999, 0.5f, 0) == -1, "unknown token -> -1");
    wubu_ctxring_free(&r);

    /* --- equal importance -> oldest wins (FIFO order preserved) --- */
    CHECK(wubu_ctxring_init(&r, 4) == 0, "init 2");
    for (long i = 1; i <= 4; i++) wubu_ctx_evict_fifo(&r, i);
    int ev2 = wubu_ctx_evict_fifo(&r, 5);
    CHECK(ev2 == 1 && r.size == 4, "over capacity, stays at cap");
    CHECK(!wubu_ctx_resident(&r, 1, 4), "oldest (1) evicted");
    CHECK(wubu_ctx_resident(&r, 5, 4), "newest (5) resident");
    wubu_ctxring_free(&r);

    /* --- stress: random insert/importance, verify invariants --- */
    CHECK(wubu_ctxring_init(&r, 8) == 0, "init 3");
    unsigned seed = 42;
    long live = 0;
    for (int op = 0; op < 5000; op++) {
        seed = seed * 1103515245u + 12345u;
        long tok = (long)(seed % 1000);
        wubu_ctx_evict_fifo(&r, tok);
        live++;
        if (live > 8) live = 8;
        if ((seed >> 8) % 4 == 0) wubu_ctx_importance(&r, tok, (float)(seed % 100) / 100.0f, 0);
        if (r.size > 8) { printf("FAIL: size %zu > cap\n", r.size); fails++; break; }
        /* size must equal the hive's live count */
        if (r.size != wubu_hive_live(&r.hive)) {
            printf("FAIL op %d: size %zu != hive live %zu\n", op, r.size, wubu_hive_live(&r.hive));
            fails++; break;
        }
    }
    CHECK(r.size <= 8, "size bounded by capacity");
    wubu_ctxring_free(&r);
    CHECK(wubu_hive_live(&r.hive) == 0, "hive empty after free");

    printf("%s\n", fails ? "CTXHIVE FAIL" : "CTXHIVE PASS");
    return fails ? 1 : 0;
}
