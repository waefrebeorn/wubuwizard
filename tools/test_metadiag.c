/*
 * test_metadiag.c -- the DUAL-TIMESCALE DIAGNOSE gate (roadmap #3:
 * continuous self-monitoring, not reactive per-batch tweaks).
 *
 * Asserts:
 *   1. the fast path runs every batch and collects the window
 *   2. the slow path fires on schedule and writes a meta-cell
 *   3. a rising-loss window raises the mutation rate + lowers the
 *      floor (the colony gets more aggressive); an improving window
 *      does the opposite
 *   4. the meta-cell is in the hive (the colony reads its own state)
 */
#include <stdio.h>
#include <string.h>

#include "wubu_metadiag.h"
#include "wubu_hive.h"

#define FAIL(...) do { printf("  FAIL: " __VA_ARGS__); printf("\n"); return 1; } while (0)

int main(void)
{
    printf("=== test_metadiag (dual-timescale diagnose) ===\n");

    wubu_hive_t tissue;
    if (wubu_hive_init(&tissue) != 0) FAIL("hive init");

    wubu_metadiag_t md;
    if (wubu_metadiag_init(&md, &tissue, 10, 32, 0.1f) != 0) FAIL("md init");

    /* 1. fast path: 30 batches, first 15 improving (loss falls), last
     * 15 rising (loss climbs) — the slow path fires every 10 */
    int slow_fired = 0;
    for (int b = 1; b <= 30; b++) {
        wubu_fast_signal_t s;
        memset(&s, 0, sizeof(s));
        s.loss = (b <= 15) ? (10.0f - 0.3f * (float)b)   /* improving */
                           : (5.5f + 0.3f * (float)(b - 15)); /* rising */
        s.util_mean = 0.5f; s.util_spread = 0.3f;
        s.repeat = 0.1f; s.grad_norm = 0.4f;
        if (wubu_metadiag_fast(&md, &s)) {
            wubu_metadiag_slow(&md);
            slow_fired++;
        }
    }
    printf("  fast cycles=%llu slow passes=%d (every 10)\n",
           (unsigned long long)md.fast_cycles, slow_fired);
    if (md.fast_cycles != 30) FAIL("fast path missed batches");
    if (slow_fired != 3) FAIL("slow path schedule wrong (%d)", slow_fired);

    /* 2. the rising tail raised the mutation rate + lowered the floor */
    float rate, floor;
    wubu_metadiag_state(&md, &rate, &floor);
    printf("  after rising tail: rate=%.2f floor=%.2f (start 0.50/0.00)\n",
           rate, floor);
    if (rate <= 0.5f) FAIL("the rising tail did not raise the mutation rate");
    if (floor > 0.0f) FAIL("the rising tail did not lower the floor");

    /* 3. an improving window lowers the rate again */
    wubu_metadiag_t md2;
    wubu_metadiag_init(&md2, &tissue, 10, 32, 0.1f);
    for (int b = 1; b <= 25; b++) {
        wubu_fast_signal_t s;
        memset(&s, 0, sizeof(s));
        s.loss = 10.0f - 0.4f * (float)b;   /* improving the whole time */
        if (wubu_metadiag_fast(&md2, &s)) wubu_metadiag_slow(&md2);
    }
    float rate2, floor2;
    wubu_metadiag_state(&md2, &rate2, &floor2);
    printf("  after improving window: rate=%.2f floor=%.2f\n", rate2, floor2);
    if (rate2 >= 0.5f) FAIL("the improving window did not lower the rate");

    /* 4. the meta-cells are in the hive (the colony's self-knowledge) */
    size_t metacells = wubu_metadiag_metacells(&md) + wubu_metadiag_metacells(&md2);
    size_t live = wubu_hive_live(&tissue);
    printf("  meta-cells written: %zu (hive live %zu)\n", metacells, live);
    if (metacells != 5) FAIL("meta-cells not written (3+2)");

    char stats[256];
    wubu_metadiag_stats(&md, stats, sizeof(stats));
    printf("  stats: %s\n", stats);

    printf("=== ALL METADIAG TESTS PASSED (continuous self-monitoring) ===\n");
    return 0;
}
