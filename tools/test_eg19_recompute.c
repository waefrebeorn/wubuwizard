/* test_eg19_recompute.c -- EG-19 gate: restore-vs-recompute cost scheduler. */
#include <stdio.h>

#include "wubu_recompute_sched.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    wubu_recompute_config_t cfg = { 1e9, 2e6, 1u << 30, 1.0 };
    wubu_recompute_sched_t *s = wubu_recompute_sched_create(cfg);
    wubu_recompute_decision_t d;
    size_t moved = 0;
    int tokens = 0, restores = 0, recomputes = 0;

    CHECK(s != NULL);
    if (!s) return 1;

    /* Short prefix: recompute is cheaper than an SSD restore. */
    CHECK(wubu_recompute_plan(s, 512, 32, 16u << 20, &d) == 0);
    printf("EG-19 short: restore=%.2fms recompute=%.2fms choice=%s\n",
           d.restore_ms, d.recompute_ms, d.restore ? "restore" : "recompute");
    CHECK(d.restore == 0 && d.bytes_moved == 0 && d.tokens_recomputed == 512);

    /* Long prefix: restore beats recomputation. */
    CHECK(wubu_recompute_plan(s, 4096, 32, 16u << 20, &d) == 0);
    printf("EG-19 long: restore=%.2fms recompute=%.2fms choice=%s\n",
           d.restore_ms, d.recompute_ms, d.restore ? "restore" : "recompute");
    CHECK(d.restore == 1 && d.bytes_moved == (16u << 20));
    CHECK(d.tokens_recomputed == 0);

    wubu_recompute_stats(s, &moved, &tokens, &restores, &recomputes);
    printf("EG-19 stats moved=%llu tokens=%d restores=%d recomputes=%d\n",
           (unsigned long long)moved, tokens, restores, recomputes);
    CHECK(moved == (16u << 20) && tokens == 512);
    CHECK(restores == 1 && recomputes == 1);

    /* Margin flips a near-tie toward restore. */
    {
        wubu_recompute_config_t c2 = { 1e9, 5e5, 1u << 30, 1.0 };
        wubu_recompute_config_t c3 = c2;
        wubu_recompute_sched_t *a = wubu_recompute_sched_create(c2);
        wubu_recompute_sched_t *b;
        c3.restore_margin = 1.1;
        b = wubu_recompute_sched_create(c3);
        CHECK(a && b);
        if (a && b) {
            CHECK(wubu_recompute_plan(a, 384, 32, 24u << 20, &d) == 0);
            CHECK(d.restore == 0);
            CHECK(wubu_recompute_plan(b, 384, 32, 24u << 20, &d) == 0);
            CHECK(d.restore == 1);
            printf("EG-19 margin flip: restore=%.2fms recompute=%.2fms\n",
                   d.restore_ms, d.recompute_ms);
        }
        wubu_recompute_sched_free(a);
        wubu_recompute_sched_free(b);
    }

    /* KV larger than the device budget forces recompute even when cheaper. */
    {
        wubu_recompute_config_t c4 = { 1e12, 2e6, 1u << 30, 1.0 };
        wubu_recompute_sched_t *big = wubu_recompute_sched_create(c4);
        CHECK(big != NULL);
        if (big) {
            CHECK(wubu_recompute_plan(big, 256, 32,
                                      2ULL << 30, &d) == 0);
            CHECK(d.restore == 0 && d.forced_by_budget == 1);
            printf("EG-19 budget forced: restore=%.2fms recompute=%.2fms\n",
                   d.restore_ms, d.recompute_ms);
        }
        wubu_recompute_sched_free(big);
    }

    CHECK(wubu_recompute_plan(s, 0, 32, 16u << 20, &d) == -1);
    CHECK(wubu_recompute_plan(s, 16, 0, 16u << 20, &d) == -1);
    CHECK(wubu_recompute_plan(NULL, 16, 32, 16u << 20, &d) == -1);
    CHECK(wubu_recompute_sched_create(cfg) != NULL);
    CHECK(wubu_recompute_sched_create((wubu_recompute_config_t){0}) == NULL);

    wubu_recompute_sched_free(s);
    if (failures) {
        fprintf(stderr, "test_eg19_recompute: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg19_recompute: PASS\n");
    return 0;
}
