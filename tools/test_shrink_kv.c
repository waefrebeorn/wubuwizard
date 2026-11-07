/* test_shrink_kv.c — KV-space amoeba shrink operator (AN22 #6)
 *
 * Verifies:
 *   T1 analyze classifies dead regions (low util AND low coherence)
 *   T2 prunable list exposes exactly the pruneable files
 *   T3 shrink(max) prunes up to max, unmounts from the namespace
 *   T4 live regions survive the prune
 *   T5 cumulative pruned counter
 *
 * WaefreBeorn Umbrella License v3.0
 */
#include "wubu_shrink_kv.h"
#include "wubu_kv_embedding.h"
#include "wubu_kvfs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail = 0;
static void chk(const char *label, int cond) {
    printf("  [%s] %s\n", cond ? "PASS" : "FAIL", label);
    if (!cond) fail++;
}

int main(void) {
    wubu_kvfs_t *fs = wubu_kvfs_create(256, 256);
    wubu_kv_embedding_t *kv = wubu_kv_embedding_create(fs, 256);
    wubu_shrink_kv_cfg_t cfg = wubu_shrink_kv_default_cfg();
    cfg.block_size = 256;
    wubu_shrink_kv_t *s = wubu_shrink_kv_create(kv, &cfg);
    chk("shrink_kv_create returns handle", s != NULL);

    /* encode three files */
    size_t nt;
    wubu_kv_embedding_encode(kv, "f1.txt", "dead region one", 15, &nt);
    wubu_kv_embedding_encode(kv, "f2.txt", "lived region two", 16, &nt);
    wubu_kv_embedding_encode(kv, "f3.txt", "dead region three", 17, &nt);

    const char *paths[] = { "/kv/in/f1.txt", "/kv/in/f2.txt", "/kv/in/f3.txt" };
    float util[] = { 0.10f, 0.50f, 0.05f };  /* f1, f3 idle */
    float coh[]  = { 0.20f, 0.90f, 0.30f };  /* f1, f3 not understood */
    float ent[]  = { 3.00f, 0.50f, 2.50f };

    printf("[T1] analyze classifies dead regions\n");
    int pruneable = wubu_shrink_kv_analyze(s, paths, util, coh, ent, 3);
    chk("analyze identifies 2 pruneable", pruneable == 2);

    printf("[T2] prunable list\n");
    const char *got[8];
    int n = wubu_shrink_kv_prunable(s, got, 8);
    chk("prunable returns 2", n == 2);
    chk("prunable[0] == f1", strcmp(got[0], "/kv/in/f1.txt") == 0);
    chk("prunable[1] == f3", strcmp(got[1], "/kv/in/f3.txt") == 0);

    printf("[T3] shrink respects max\n");
    int p = wubu_shrink_kv_shrink(s, 1);
    chk("shrink(1) prunes 1", p == 1);
    chk("total_pruned == 1", wubu_shrink_kv_total_pruned(s) == 1);

    printf("[T4] namespace state after prune\n");
    uint32_t blk; size_t off;
    chk("f1 unmounted (gone)",
        wubu_kvfs_lookup(fs, "/kv/in/f1.txt", &blk, &off) == -1);
    chk("f2 still mounted (live)",
        wubu_kvfs_lookup(fs, "/kv/in/f2.txt", &blk, &off) == 0);
    chk("f3 still mounted (not yet pruned)",
        wubu_kvfs_lookup(fs, "/kv/in/f3.txt", &blk, &off) == 0);

    printf("[T5] shrink drains the rest\n");
    p = wubu_shrink_kv_shrink(s, 10);
    chk("shrink(10) prunes the remaining 1", p == 1);
    chk("total_pruned == 2", wubu_shrink_kv_total_pruned(s) == 2);
    chk("f3 now unmounted",
        wubu_kvfs_lookup(fs, "/kv/in/f3.txt", &blk, &off) == -1);
    chk("f2 survives the drain",
        wubu_kvfs_lookup(fs, "/kv/in/f2.txt", &blk, &off) == 0);

    wubu_shrink_kv_free(s);
    wubu_kv_embedding_free(kv);
    wubu_kvfs_free(fs);

    printf("test_shrink_kv: %s (%d failures)\n", fail ? "FAIL" : "PASS", fail);
    return fail ? 1 : 0;
}
