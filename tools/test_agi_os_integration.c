/*
 * test_agi_os_integration.c -- AF05-AF13 verification (latency / ctx-vm / safety).
 */
#include "wubu_latency.h"
#include "wubu_ctxvm.h"
#include "wubu_safekern.h"
#include <stdio.h>
#include <math.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_agi_os_integration (AF05-AF13) ===\n");

    /* AF05 EDF ordering */
    wubu_task_t t[3] = { {1,300,0},{2,100,0},{3,200,0} };
    wubu_edf_order(t, 3);
    CHECK(t[0].id == 2 && t[1].id == 3 && t[2].id == 1, "EDF orders by deadline asc");

    /* AF06 WCET + jitter + deadline miss */
    long s[4] = { 8, 12, 9, 11 };
    wubu_wcet_t w; wubu_wcet_account(s, 4, &w);
    CHECK(w.wcet_ms == 12, "wcet = max sample");
    CHECK(fabs(w.mean_ms - 10.0) < 1e-9, "mean correct");
    CHECK(w.jitter_ms == 2.0, "jitter = max deviation");
    CHECK(wubu_deadline_miss(&w, 10) == 1, "wcet>budget -> miss");
    CHECK(wubu_deadline_miss(&w, 20) == 0, "wcet<=budget -> ok");

    /* AF07 SLO check per class */
    wubu_slo_meas_t hrt = { 15, 18, 3, 0 };
    CHECK(wubu_slo_check(WUBU_LC_HRT, &hrt) == 0, "HRT within SLO");
    wubu_slo_meas_t hrt2 = { 25, 18, 3, 0 };
    CHECK(wubu_slo_check(WUBU_LC_HRT, &hrt2) & 1, "HRT TTFT>20 -> fail");
    wubu_slo_meas_t srt = { 250, 1000, 100, 0 };
    CHECK(wubu_slo_check(WUBU_LC_SRT, &srt) == 0, "SRT within SLO");
    wubu_slo_meas_t dt = { 0, 0, 0, 50.0 };
    CHECK(wubu_slo_check(WUBU_LC_DT, &dt) == 0, "DT throughput ok");

    /* AF08 context tier */
    CHECK(wubu_ctx_tier(0.9f, 5000) == WUBU_CTX_L4, "high imp+ttl -> L4");
    CHECK(wubu_ctx_tier(0.5f, 100)  == WUBU_CTX_L3, "mid -> L3");
    CHECK(wubu_ctx_tier(0.1f, 5)    == WUBU_CTX_L2, "low ttl -> L2");

    /* AF09 HIVE-backed eviction + working-set residency */
    wubu_ctxring_t r;
    CHECK(wubu_ctxring_init(&r, 4) == 0, "ring init");
    for (long i = 1; i <= 4; i++) wubu_ctx_evict_fifo(&r, i);
    CHECK(r.size == 4, "filled to capacity");
    int ev = wubu_ctx_evict_fifo(&r, 5); /* evicts 1 (oldest, all equal importance) */
    CHECK(ev == 1 && r.size == 4, "over capacity -> evict 1, stays at cap");
    CHECK(wubu_ctx_resident(&r, 5, 4) == 1, "token 5 resident in WS");
    CHECK(wubu_ctx_resident(&r, 1, 4) == 0, "evicted token 1 not resident");
    wubu_ctxring_free(&r);

    /* AF10 cosine + semantic cache */
    float a[3] = {1,0,0}, b[3] = {1,0,0}, c[3] = {0,1,0};
    CHECK(fabsf(wubu_cosine(a, b, 3) - 1.0f) < 1e-5f, "identical -> cos 1");
    CHECK(fabsf(wubu_cosine(a, c, 3)) < 1e-5f, "orthogonal -> cos 0");
    CHECK(wubu_sem_cache_hit(a, b, 3, 0.9f) == 1, "cos>=thr -> cache hit");
    CHECK(wubu_sem_cache_hit(a, c, 3, 0.9f) == 0, "cos<thr -> miss");

    /* AF11 non-tamperable stop */
    wubu_safekern_t k; k.stop_flag = 1; k.oom_ceiling = WUBU_OOM_CEILING; k.gate_enabled = 1;
    CHECK(wubu_stop_honored(&k) == 1, "kernel stop honored");
    k.stop_flag = 0;
    CHECK(wubu_stop_honored(&k) == 0, "no stop -> not honored");
    /* reasoner has no setter for stop_flag (kernel-owned) -- invariant enforced by API */

    /* AF12 graduated containment */
    CHECK(wubu_containment_level(0.1f) == WUBU_CONT_NONE,     "low sev -> none");
    CHECK(wubu_containment_level(0.3f) == WUBU_CONT_WARN,     "0.3 -> warn");
    CHECK(wubu_containment_level(0.5f) == WUBU_CONT_THROTTLE, "0.5 -> throttle");
    CHECK(wubu_containment_level(0.7f) == WUBU_CONT_SUSPEND,  "0.7 -> suspend");
    CHECK(wubu_containment_level(0.9f) == WUBU_CONT_STOP,     "0.9 -> stop");
    CHECK(wubu_containment_reversible(WUBU_CONT_THROTTLE) == 1, "throttle reversible");
    CHECK(wubu_containment_reversible(WUBU_CONT_STOP) == 0,     "stop not reversible");

    /* AF13 stability-plasticity guard */
    CHECK(wubu_rsi_mutation_ok(&k, 524288, 1) == 1, "same ceiling + gate -> ok");
    CHECK(wubu_rsi_mutation_ok(&k, 1024,   1) == 0, "lower ceiling -> REJECTED");
    CHECK(wubu_rsi_mutation_ok(&k, 524288, 0) == 0, "disable gate -> REJECTED");

    if (failures == 0) { printf("ALL AGI-OS-INTEGRATION TESTS PASSED\n"); return 0; }
    printf("%d AGI-OS-INTEGRATION TEST(S) FAILED\n", failures);
    return 1;
}
