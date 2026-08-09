/*
 * test_colonel.c -- the LIVE COLONEL bridge gate (the user directive
 * #5: "the Brain requests Body actions through the Live Colonel
 * within the hive trust boundary").
 *
 * Asserts:
 *   1. requests with a valid 9P capability subtree enqueue; requests
 *      OUTSIDE the subtree are DENIED at the trust boundary
 *   2. the executor pulls by priority (critical first)
 *   3. outcomes: success counts done; failure counts failed + retries
 *   4. the queue is durable (save -> load restores the requests)
 */
#include <stdio.h>
#include <string.h>

#include "wubu_colonel.h"

#define FAIL(...) do { printf("  FAIL: " __VA_ARGS__); printf("\n"); return 1; } while (0)

int main(void)
{
    printf("=== test_colonel (the Brain->Body channel) ===\n");

    wubu_colonel_t col;
    if (wubu_colonel_init(&col, 8, 1000) != 0) FAIL("colonel init");

    /* 1. the trust boundary: a valid request enqueues */
    int64_t rid = wubu_colonel_request(&col, WUBU_ACT_READ,
                                       "/brain/memory/kv",
                                       "/brain", 42, 0, 1,
                                       100, 64, 1024);
    if (rid <= 0) FAIL("valid request denied (id %lld)", (long long)rid);
    printf("  request %lld enqueued (read /brain/memory/kv)\n", (long long)rid);

    /* a request OUTSIDE the subtree must be denied */
    int64_t bad = wubu_colonel_request(&col, WUBU_ACT_WRITE,
                                       "/kernel/ring0",     /* outside /brain */
                                       "/brain", 42, 0, 2,
                                       100, 64, 1024);
    if (bad >= 0) FAIL("out-of-subtree request was NOT denied (id %lld)", (long long)bad);
    printf("  out-of-boundary request denied (the trust boundary holds)\n");

    /* 2. priority pull: enqueue a low + a critical, pull critical first */
    wubu_colonel_request(&col, WUBU_ACT_RUN, "/brain/tasks/slow",
                         "/brain", 7, 1, 0, 500, 256, 2048);
    wubu_colonel_request(&col, WUBU_ACT_KILL, "/brain/tasks/runaway",
                         "/brain", 7, 1, 3, 10, 16, 128);
    int first = wubu_colonel_pull(&col);
    if (first < 0) FAIL("nothing to pull");
    printf("  executor pulled request %d (kind %d)", first, col.queue[first].kind);
    if (col.queue[first].kind != WUBU_ACT_KILL) FAIL("critical was not pulled first");
    printf(" — critical first ✓\n");

    /* 3. outcomes: report success + failure */
    wubu_colonel_report(&col, first, 1);
    if (col.n_done != 1) FAIL("success not counted");
    int second = wubu_colonel_pull(&col);
    if (second < 0) FAIL("second pull failed");
    wubu_colonel_report(&col, second, 0);
    if (col.n_failed != 1) FAIL("failure not counted");
    if (col.queue[second].attempt != 1) FAIL("retry attempt not recorded");
    printf("  outcomes: %llu done, %llu failed (retry armed)\n",
           (unsigned long long)col.n_done, (unsigned long long)col.n_failed);

    /* 4. durability: save -> load restores */
    static char ck[8192];
    long n = wubu_colonel_save(&col, ck, (long)sizeof(ck));
    if (n <= 0) FAIL("save failed");
    wubu_colonel_t col2;
    wubu_colonel_init(&col2, 8, 1000);
    int restored = wubu_colonel_load(&col2, ck, n);
    if (restored != col.q_n) FAIL("restore count mismatch (%d vs %d)", restored, col.q_n);
    printf("  durable: %d requests survived the checkpoint (%ld bytes)\n",
           restored, n);
    if (col2.queue[0].req_id != col.queue[0].req_id)
        FAIL("restored request id mismatch");

    char stats[256];
    wubu_colonel_stats(&col, stats, sizeof(stats));
    printf("  stats: %s\n", stats);

    wubu_colonel_free(&col);
    wubu_colonel_free(&col2);
    printf("=== ALL COLONEL TESTS PASSED (the Brain speaks to the Body) ===\n");
    return 0;
}
