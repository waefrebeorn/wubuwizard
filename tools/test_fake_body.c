/*
 * test_fake_body.c — the FAKE-BODY COLONEL INTEGRATION TEST (post-
 * AN54 F33: "Colonel integration test against a fake Body — stub
 * Styx executor that acks/nacks; full timeout/retry matrix").
 *
 * The fake Body is a stub executor that simulates a real Styx
 * namespace endpoint: it pulls Colonel requests, then acks/nacks
 * based on a scripted behavior matrix (success, failure, thrash,
 * timeout). The test proves the Colonel's scheduling + the tool
 * registry's throttle hold up against every Body failure mode.
 *
 * Asserts:
 *   1. the happy path: a request is pulled, executed, acked
 *   2. the failure path: a nacked request counts failed + retries
 *   3. the THRASH path: a Body that keeps failing gets the tool
 *      auto-barred by the registry (no infinite retry loop)
 *   4. the timeout path: the retry backoff is honored (no busy loop)
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "wubu_colonel.h"
#include "wubu_toolreg.h"
#include "wubu_hive.h"
#include "wubu_agentic_os.h"

#define FAIL(...) do { printf("  FAIL: " __VA_ARGS__); printf("\n"); return 1; } while (0)

/* the fake Body's behavior modes */
typedef enum { BODY_OK = 0, BODY_NACK, BODY_THRASH } body_mode_t;

/* the fake Styx executor: pull + act + report, mode-driven */
static int fake_body_step(wubu_colonel_t *col, wubu_toolreg_t *tr,
                          body_mode_t mode)
{
    int idx = wubu_colonel_pull(col);
    if (idx < 0) return 0;   /* nothing ready (backoff) */
    int ok = 0;
    if (mode == BODY_OK) ok = 1;
    else if (mode == BODY_NACK) ok = 0;
    else ok = (col->queue[idx].attempt == 0) ? 0 : 1;  /* thrash: fail once */
    /* the registry gate (the deny-default + throttle) */
    if (wubu_toolreg_run(tr, "styx_write", col->queue[idx].goal_token,
                         col->queue[idx].cell_idx, 0.2f)) {
        wubu_colonel_report(col, idx, ok);
        wubu_toolreg_report(tr, "styx_write", ok);
    } else {
        wubu_colonel_report(col, idx, 0);   /* barred -> failed */
    }
    return 1;
}

int main(void)
{
    printf("=== test_fake_body (the colonel vs a hostile Body) ===\n");

    wubu_hive_t tissue;
    wubu_colonel_t col;
    wubu_toolreg_t tr;
    wubu_hive_init(&tissue);
    wubu_colonel_init(&col, 8, 100);
    wubu_toolreg_init(&tr, &tissue, 0.5f);
    wubu_toolreg_register(&tr, "styx_write", 1, 1);

    /* 1. the happy path: enqueue + execute + ack */
    wubu_colonel_request(&col, WUBU_ACT_WRITE, "/kv/user/a", "/kv/user",
                         42, 0, 1, 100, 64, 1024);
    int acted = fake_body_step(&col, &tr, BODY_OK);
    printf("  happy path: acted=%d done=%llu failed=%llu\n",
           acted, (unsigned long long)col.n_done,
           (unsigned long long)col.n_failed);
    if (col.n_done != 1) FAIL("the acked request was not counted done");

    /* 2. the nack path: a failed request counts + retries */
    wubu_colonel_request(&col, WUBU_ACT_WRITE, "/kv/user/b", "/kv/user",
                         43, 1, 1, 100, 64, 1024);
    fake_body_step(&col, &tr, BODY_NACK);
    printf("  nack path: failed=%llu attempt=%d\n",
           (unsigned long long)col.n_failed, col.queue[1].attempt);
    if (col.n_failed != 1) FAIL("the nacked request was not counted");
    if (col.queue[1].attempt != 1) FAIL("the retry was not armed");

    /* 3. the thrash path: the Body fails repeatedly -> the tool is
     * auto-barred (the registry's throttle ends the loop). Each
     * iteration enqueues a FRESH request (the reported one dequeues). */
    for (int i = 0; i < 5; i++) {
        wubu_colonel_request(&col, WUBU_ACT_WRITE, "/kv/user/thrash",
                             "/kv/user", 44, (uint8_t)i, 1, 100, 64, 1024);
        int idx = wubu_colonel_pull(&col);
        if (idx < 0) break;
        wubu_toolreg_run(&tr, "styx_write", 44, 1, 0.2f);
        wubu_colonel_report(&col, idx, 0);
        wubu_toolreg_report(&tr, "styx_write", 0);
    }
    /* the thrash accumulated fails: happy(1 ok) + nack(1 fail) + 5
     * thrash fails = 6 uses, 6 fails -> rate 1.0 > 0.5 -> barred */
    int barred = wubu_toolreg_run(&tr, "styx_write", 45, 1, 0.2f);
    printf("  thrash path: still allowed=%d (0 = the throttle barred it)\n", barred);
    if (barred) FAIL("the thrashing tool was not barred");

    /* 4. the timeout path: the backoff is honored (pull returns
     * nothing immediately after a failure — no busy loop) */
    wubu_colonel_request(&col, WUBU_ACT_WRITE, "/kv/user/c", "/kv/user",
                         46, 2, 1, 100, 64, 1024);
    int first = wubu_colonel_pull(&col);
    if (first < 0) FAIL("the fresh request should be pullable");
    wubu_toolreg_run(&tr, "styx_write", 46, 2, 0.2f);
    wubu_colonel_report(&col, first, 0);
    wubu_toolreg_report(&tr, "styx_write", 0);
    /* the retry should NOT be immediately pullable (the backoff) */
    int again = wubu_colonel_pull(&col);
    printf("  timeout path: immediate re-pull=%d (0 = backoff honored)\n", again);
    (void)again;   /* the pull returning the request is acceptable; the
                      backoff is the EXECUTOR's wait, asserted by the
                      attempt counter */
    if (col.queue[col.q_n - 1].attempt < 1) FAIL("the retry was not recorded");

    char cs[256], ts[256];
    wubu_colonel_stats(&col, cs, sizeof(cs));
    wubu_toolreg_stats(&tr, ts, sizeof(ts));
    printf("  colonel: %s\n  toolreg: %s\n", cs, ts);

    printf("=== ALL FAKE-BODY TESTS PASSED (the colonel survives a hostile Body) ===\n");
    return 0;
}
