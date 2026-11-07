/* test_kv_coherence_diag.c — the KV-FS diagnose→grow/shrink loop (Phase 3/7)
 *
 * Closes the metabolism loop: forward → attention → coherence →
 * grow/shrink → next forward (docs/wubu1-hive-mind-plan Track B).
 *
 * Two encoded files: low.txt (uniform attention -> score 0.4, below the
 * 0.5 threshold) and high.txt (focused -> score 0.9). The cycle must:
 *   T1 measure() reports the true per-file scores
 *   T2 cycle() marks n_under=1 (low.txt only)
 *   T3 grow runs toward it -> n_grown=1, region /kv/in/low.txt/grow0 exists
 *   T4 shrink finds nothing cold on one pass -> n_pruned=0
 *   T5 summary mean == reward mean
 *
 * WaefreBeorn Umbrella License v3.0
 */
#include "wubu_kv_coherence_diag.h"
#include "wubu_kv_embedding.h"
#include "wubu_grow_kv.h"
#include "wubu_kv_shrink.h"
#include "wubu_kvfs.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static int fail = 0;
static void chk(const char *label, int cond) {
    printf("  [%s] %s\n", cond ? "PASS" : "FAIL", label);
    if (!cond) fail++;
}
static int close_f(float a, float b, float t) { return fabsf(a - b) <= t; }

int main(void) {
    wubu_kvfs_t *fs = wubu_kvfs_create(256, 128);
    wubu_kv_embedding_t *kv = wubu_kv_embedding_create(fs, 256);
    size_t nt;
    wubu_kv_embedding_encode(kv, "low.txt", "this file is poorly understood", 30, &nt);
    wubu_kv_embedding_encode(kv, "high.txt", "well understood file", 20, &nt);

    /* grow: threshold 0.5 -> low.txt (0.4) under, high.txt (0.9) not */
    wubu_grow_kv_cfg_t gc = wubu_grow_kv_default_cfg();
    gc.coherence_threshold = 0.5;
    gc.block_size = 256;
    wubu_grow_kv_t *grow = wubu_grow_kv_create(kv, &gc);

    /* shrink: cold_iters=1 so a truly-cold region would prune */
    wubu_kv_shrink_cfg_t sc = wubu_kv_shrink_default_cfg();
    sc.cold_iters = 1;
    wubu_kv_shrink_t *shrink = wubu_kv_shrink_create(fs, &sc);

    wubu_kv_diag_t *diag = wubu_kv_diag_create(kv, grow, shrink);
    chk("diag_create returns handle", diag != NULL);

    /* attention: n_context=8, whole context is the file region [0,8) */
    float low[8]  = { 0.125f,0.125f,0.125f,0.125f,0.125f,0.125f,0.125f,0.125f };
    float high[8] = { 0.5f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    const char *paths[2] = { "/kv/in/low.txt", "/kv/in/high.txt" };
    const float *attn[2] = { low, high };
    const size_t nq[2] = { 1, 1 }, nc[2] = { 8, 8 };
    const size_t cs[2] = { 0, 0 }, cl[2] = { 8, 8 };
    const size_t qs[2] = { 0, 0 }, ql[2] = { 1, 1 };

    printf("[T1] measure() per-file scores\n");
    wubu_coherence_t scores[2];
    chk("measure returns 0", wubu_kv_diag_measure(diag, paths, attn,
        nq, nc, cs, cl, qs, ql, 2, scores) == 0);
    chk("low score == 0.4 (uniform)", close_f(scores[0].score, 0.4f, 1e-4f));
    chk("high score == 0.9 (focused)", close_f(scores[1].score, 0.9f, 1e-4f));

    printf("[T2-T5] full cycle\n");
    wubu_reward_result_t reward;
    wubu_kv_diag_summary_t sum;
    int rc = wubu_kv_diag_cycle(diag, paths, attn, nq, nc, cs, cl, qs, ql,
                                2, &reward, &sum);
    chk("cycle returns 0", rc == 0);
    chk("T2 n_under == 1 (low.txt only)", sum.n_under == 1);
    chk("T3 n_grown == 1", sum.n_grown == 1);
    chk("T4 n_pruned == 0 (both regions attended this pass)",
        sum.n_pruned == 0);
    chk("T5 mean_score == 0.65", close_f(sum.mean_score, 0.65f, 1e-4f));
    chk("summary mean == reward mean",
        close_f(sum.mean_score, reward.mean_score, 1e-6f));

    /* the grow mount landed toward low.txt */
    uint32_t blk; size_t off, nf;
    chk("grow0 region toward low.txt exists",
        wubu_kv_embedding_region(kv, "/kv/in/low.txt/grow0", &blk, &off, &nf) == 0);
    chk("grow0 is 1 float", nf == 1);

    wubu_reward_result_free(&reward);
    wubu_kv_diag_free(diag);
    wubu_grow_kv_free(grow);
    wubu_kv_shrink_free(shrink);
    wubu_kv_embedding_free(kv);
    wubu_kvfs_free(fs);

    printf("test_kv_coherence_diag: %s (%d failures)\n", fail ? "FAIL" : "PASS", fail);
    return fail ? 1 : 0;
}
