/*
 * test_diag.c -- THE HIVE DIAGNOSTIC SYSTEM test (research/056, AN08).
 *
 * The 6 DA oracles from the design doc:
 *   1. ring-bounded: inserting > capacity recycles the oldest (foreach
 *      sees exactly capacity live cells; the freelist recycles)
 *   2. z-score: an injected 10x outlier has |z| > 2.5; a normal value < 1
 *   3. trend: a cell whose grad rises across 20 steps classifies GROW
 *   4. dead colony: all grads below the floor classifies SHRINK (the DA bug)
 *   5. the walker: build a trace with a normal window, inject an anomaly at
 *      step N, inject a fitness drop at N+50; the walker reports kind/cell/N
 *   6. honest failure: a drop with no anomaly reports "unexplained"
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "wubu_diag.h"
#include "wubu_hive.h"
#include "wubu.h"
#include "wubu_train.h"
#include "wubu_runtime_dims.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } else { printf("  ok: %s\n", m); } } while (0)

/* fill the window with a baseline so the anomaly stands out */
static void seed_window(wubu_diag_t *d, wubu_diag_kind kind, int cell,
                        float base, int n, float noise)
{
    for (int i = 0; i < n; i++) {
        float v = base + noise * ((float)(i % 7) - 3.0f);
        wubu_diag_record(d, kind, cell, v, 0.0f);
    }
}

int main(int argc, char **argv)
{
    const char *model_path = (argc > 1) ? argv[1] : "models/wubu/model.safetensors";
    printf("=== test_diag (the hive diagnostic system, AN08) ===\n");

    /* --- oracle 1: ring-bounded --- */
    {
        printf("[oracle 1] ring-bounded trace\n");
        wubu_hive_t hive;
        wubu_hive_init(&hive);
        wubu_diag_t *d = wubu_diag_init(&hive, 0);
        wubu_diag_set_capacity(d, 64);
        for (int i = 0; i < 200; i++)
            wubu_diag_record(d, WUBU_DIAG_LOSS, -1, 1.0f + 0.01f * (i % 5), 0.0f);
        CHECK(wubu_diag_live(d) == 64, "inserting > capacity keeps exactly 64 live");
        CHECK(hive.reuses > 0, "the freelist recycled (reuses > 0)");
        CHECK(wubu_hive_capacity(&hive) >= 64, "capacity >= live");
        /* the trace kept the NEWEST 64: oldest step == 136 */
        int64_t min_step = INT64_MAX, max_step = 0;
        for (wubu_hive_block_t *blk = hive.head; blk; blk = blk->next)
            for (size_t i = 0; i < blk->cap; i++) {
                if (blk->skip[i]) continue;
                wubu_diag_cell *c = (wubu_diag_cell *)blk->slots[i];
                if (c->step < min_step) min_step = c->step;
                if (c->step > max_step) max_step = c->step;
            }
        CHECK(min_step == 136 && max_step == 199,
              "oldest recycled (min step 136), newest kept (step 199)");
        wubu_diag_free(d);
        wubu_hive_clear(&hive);
    }

    /* --- oracle 2: z-score --- */
    {
        printf("[oracle 2] z-score detection\n");
        wubu_hive_t hive;
        wubu_hive_init(&hive);
        wubu_diag_t *d = wubu_diag_init(&hive, 0);
        seed_window(d, WUBU_DIAG_LOSS, -1, 1.0f, 20, 0.01f);
        float z_normal = wubu_diag_zscore(d, WUBU_DIAG_LOSS, 1.0f);
        float z_outlier = wubu_diag_zscore(d, WUBU_DIAG_LOSS, 10.0f);
        CHECK(fabsf(z_normal) < 1.0f, "normal value |z| < 1");
        CHECK(fabsf(z_outlier) > 2.5f, "10x outlier |z| > 2.5");
        printf("  (z_normal=%.2f z_outlier=%.2f)\n", z_normal, z_outlier);
        wubu_diag_free(d);
        wubu_hive_clear(&hive);
    }

    /* --- oracle 3: rising grad trend classifies GROW --- */
    {
        printf("[oracle 3] rising grad trend -> GROW\n");
        wubu_hive_t hive;
        wubu_hive_init(&hive);
        wubu_diag_t *d = wubu_diag_init(&hive, 0);
        /* cell 7: a long steady baseline, then a sharp rising tail (the
         * overworked cell -- most of the window is normal, the END
         * climbs out of family) */
        for (int i = 0; i < 15; i++)
            wubu_diag_record(d, WUBU_DIAG_GRAD, 7, 0.05f, 0.0f);
        for (int i = 0; i < 5; i++)
            wubu_diag_record(d, WUBU_DIAG_GRAD, 7, 0.05f + 0.25f * (float)(i + 1), 0.0f);
        /* cell 3: the healthy sibling, steady the whole window */
        for (int i = 0; i < 20; i++)
            wubu_diag_record(d, WUBU_DIAG_GRAD, 3, 0.05f, 0.0f);
        float grow = -1, shrink = -1;
        wubu_diag_classify(d, &grow, &shrink);
        CHECK(grow >= 1.0f, "the climbing cell classifies GROW");
        CHECK(shrink == 0.0f, "no shrink candidates in a healthy window");
        printf("  (grow=%.0f shrink=%.0f)\n", grow, shrink);
        wubu_diag_free(d);
        wubu_hive_clear(&hive);
    }

    /* --- oracle 4: dead colony classifies SHRINK (the DA bug guard) --- */
    {
        printf("[oracle 4] dead colony -> SHRINK (absolute floor)\n");
        wubu_hive_t hive;
        wubu_hive_init(&hive);
        wubu_diag_t *d = wubu_diag_init(&hive, 0);
        /* every grad below the 1e-4 floor the whole window */
        for (int i = 0; i < 20; i++)
            wubu_diag_record(d, WUBU_DIAG_GRAD, 4, 1e-6f, 0.0f);
        float grow = -1, shrink = -1;
        wubu_diag_classify(d, &grow, &shrink);
        CHECK(shrink >= 1.0f, "all-dead cell classifies SHRINK (the DA bug)");
        CHECK(grow == 0.0f, "no grow candidates in a dead colony");
        printf("  (grow=%.0f shrink=%.0f)\n", grow, shrink);
        wubu_diag_free(d);
        wubu_hive_clear(&hive);
    }

    /* --- oracle 5: the causal walker finds the root cause --- */
    {
        printf("[oracle 5] the causal walker\n");
        wubu_hive_t hive;
        wubu_hive_init(&hive);
        wubu_diag_t *d = wubu_diag_init(&hive, 0);
        /* a normal window of LOSS + steady grads */
        seed_window(d, WUBU_DIAG_LOSS, -1, 1.0f, 100, 0.02f);
        for (int i = 0; i < 100; i++)
            wubu_diag_record(d, WUBU_DIAG_GRAD, 7, 0.05f, 0.0f);
        /* inject the anomaly: cell 7's grad explodes across 10 records */
        for (int i = 0; i < 10; i++)
            wubu_diag_record(d, WUBU_DIAG_GRAD, 7, 0.05f + 5.0f * (float)i, 0.0f);
        /* the fitness drop: a LOSS cell far above the window baseline.
         * Its step id is the current clock -- strictly after all the
         * anomalous GRAD cells. */
        int64_t drop_id = -1;
        wubu_diag_record(d, WUBU_DIAG_LOSS, -1, 3.5f, 0.0f);
        for (wubu_hive_block_t *blk = hive.head; blk; blk = blk->next)
            for (size_t i = 0; i < blk->cap; i++) {
                if (blk->skip[i]) continue;
                wubu_diag_cell *c = (wubu_diag_cell *)blk->slots[i];
                if (c->kind == WUBU_DIAG_LOSS && c->value > 3.0f) drop_id = c->step;
            }
        CHECK(drop_id >= 0, "the drop was recorded");
        char report[256];
        int rc = wubu_diag_walk(d, drop_id, report, sizeof(report));
        CHECK(rc == 1, "walker found a cause");
        CHECK(strstr(report, "GRAD") != NULL, "cause kind = GRAD");
        CHECK(strstr(report, "cell=7") != NULL, "cause cell = 7");
        printf("  walker: %s\n", report);
        wubu_diag_free(d);
        wubu_hive_clear(&hive);
    }

    /* --- oracle 6: honest failure --- */
    {
        printf("[oracle 6] honest 'unexplained' fallback\n");
        wubu_hive_t hive;
        wubu_hive_init(&hive);
        wubu_diag_t *d = wubu_diag_init(&hive, 0);
        /* a perfectly normal window, then a drop with NO preceding anomaly */
        seed_window(d, WUBU_DIAG_LOSS, -1, 1.0f, 100, 0.01f);
        for (int i = 0; i < 100; i++)
            wubu_diag_record(d, WUBU_DIAG_GRAD, 7, 0.05f, 0.0f);
        /* record the drop, then find its step id */
        wubu_diag_record(d, WUBU_DIAG_LOSS, -1, 3.5f, 0.0f);
        int64_t drop_id = -1;
        for (wubu_hive_block_t *blk = hive.head; blk; blk = blk->next)
            for (size_t i = 0; i < blk->cap; i++) {
                if (blk->skip[i]) continue;
                wubu_diag_cell *c = (wubu_diag_cell *)blk->slots[i];
                if (c->kind == WUBU_DIAG_LOSS && c->value > 3.0f) drop_id = c->step;
            }
        CHECK(drop_id >= 0, "the drop was recorded");
        char report[256];
        int rc = wubu_diag_walk(d, drop_id, report, sizeof(report));
        CHECK(rc == 0, "walker reports NO cause");
        CHECK(strstr(report, "unexplained") != NULL, "honest 'unexplained' text");
        printf("  walker: %s\n", report);

        /* snapshot round-trip */
        CHECK(wubu_diag_snapshot(d, "/tmp/diag_snapshot.json") == 0,
              "snapshot written");
        wubu_diag_free(d);
        wubu_hive_clear(&hive);
        FILE *sf = fopen("/tmp/diag_snapshot.json", "r");
        CHECK(sf != NULL, "snapshot file exists");
        if (sf) {
            char buf[4096];
            size_t got = fread(buf, 1, sizeof(buf) - 1, sf);
            buf[got] = 0;
            fclose(sf);
            CHECK(strstr(buf, "\"LOSS\"") != NULL, "snapshot has LOSS aggregate");
            CHECK(strstr(buf, "\"cells\"") != NULL, "snapshot has cells");
        }
    }

    /* --- oracle 7: THE REAL-GRAD BRIDGE (milestone 2) ---
     * the trainer's ACTUAL per-layer gradients become GRAD cells.
     * cell = layer index; the diagnostic sees the REAL training signal. */
    {
        printf("[oracle 7] real backprop grads -> GRAD cells\n");
        wubu_model_t m;
        /* probe + set the runtime dims BEFORE load (revolver doctrine:
         * wubu_load reads the global WUBU_RUNTIME_DIMS, not compile-time) */
        wubu_runtime_dims_t d7;
        if (wubu_runtime_dims_probe(model_path, &d7) != 0) {
            printf("  SKIP: no 512-aligned checkpoint at %s (oracle 7 "
                   "needs one; the 448-dim SD seed predates the aligned "
                   "rewrite)\n", model_path);
        } else {
            wubu_runtime_dims_set(&d7);
        if (wubu_load(&m, model_path) != 0) {
            printf("  SKIP: checkpoint %s is not 512-aligned (448-dim "
                   "seed predates Theory/08; oracle 7 runs when a WuBu1 "
                   "checkpoint exists)\n", model_path);
        } else {
            wubu_buf_t b;
            wubu_hive_t hive;
            wubu_hive_init(&hive);
            wubu_diag_t *d = wubu_diag_init(&hive, 0);
            CHECK(wubu_buf_alloc(&b, 64) == 0, "buf alloc");
            wubu_train_t tr;
            CHECK(wubu_train_init(&tr, &m) == 0, "train init");
            wubu_train_zero_grad(&tr);
            /* one real microbatch: a short synthetic sequence */
            uint16_t tok[48];
            for (int i = 0; i < 48; i++) tok[i] = (uint16_t)(10 + (i * 7) % 60);
            float loss = wubu_train_microbatch(&m, &tr, &b, tok, 48);
            printf("  (microbatch loss=%.4f)\n", loss);
            CHECK(isfinite(loss), "loss finite");
            /* the bridge: real grads -> GRAD cells */
            int n_rec = wubu_diag_record_grads(d, &tr);
            CHECK(n_rec == WUBU_LAYERS + 1, "one GRAD cell per layer + embedding");
            printf("  (recorded %d GRAD cells: layers 0..%d + emb)\n",
                   n_rec, WUBU_LAYERS - 1);
            /* the cells are populated with REAL norms (nonzero, finite) */
            int live = 0, finite_ok = 1;
            for (wubu_hive_block_t *blk = hive.head; blk; blk = blk->next)
                for (size_t i = 0; i < blk->cap; i++) {
                    if (blk->skip[i]) continue;
                    wubu_diag_cell *c = (wubu_diag_cell *)blk->slots[i];
                    if (c->kind == WUBU_DIAG_GRAD) {
                        live++;
                        if (!isfinite(c->value) || c->value < 0) finite_ok = 0;
                    }
                }
            CHECK(live == WUBU_LAYERS + 1, "all grad cells live");
            CHECK(finite_ok, "all grad norms finite and non-negative");
            /* classify runs over real grads without crashing */
            float grow = -1, shrink = -1;
            CHECK(wubu_diag_classify(d, &grow, &shrink) == 0, "classify over real grads");
            printf("  (classify: grow=%.0f shrink=%.0f)\n", grow, shrink);
            wubu_diag_free(d);
            wubu_hive_clear(&hive);
            wubu_train_zero_grad(&tr);
            wubu_free(&m, &b);
        }
        }   /* probe-else */
    }

    printf("\n%s (%d failures)\n",
           failures == 0 ? "=== test_diag PASSED ===" : "=== test_diag FAILED ===",
           failures);
    return failures == 0 ? 0 : 1;
}
