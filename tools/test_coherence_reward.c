/* test_coherence_reward.c — batched coherence reward for KV-FS training
 * (AN22 Phase 2 / module #2)
 *
 *   T1 a two-file batch: focused vs uniform attention -> correct mean
 *     and clamped reward
 *   T2 per-file entries carry their own coherence scores
 *   T3 an UNKNOWN path (not encoded) is zero-scored, not a crash, and
 *     pulls the mean down
 *   T4 free() leaves the result inert
 *
 * WaefreBeorn Umbrella License v3.0
 */
#include "wubu_coherence_reward.h"
#include "wubu_kv_embedding.h"
#include "wubu_kvfs.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

static int fail = 0;
static void chk(const char *label, int cond) {
    printf("  [%s] %s\n", cond ? "PASS" : "FAIL", label);
    if (!cond) fail++;
}
static int close_f(float a, float b, float t) { return fabsf(a - b) <= t; }

int main(void) {
    wubu_kvfs_t *fs = wubu_kvfs_create(256, 256);
    wubu_kv_embedding_t *kv = wubu_kv_embedding_create(fs, 256);
    size_t nt;
    wubu_kv_embedding_encode(kv, "a.txt", "hello world", 11, &nt);
    wubu_kv_embedding_encode(kv, "b.txt", "kv cache", 8, &nt);

    /* n_query=1, n_context=4, file region = whole context [0,4) */
    float attn_a[4] = { 0.5f, 0.5f, 0.0f, 0.0f };   /* focused -> score 0.85 */
    float attn_b[4] = { 0.25f, 0.25f, 0.25f, 0.25f }; /* uniform -> 0.4 */
    const char *paths[2] = { "/kv/in/a.txt", "/kv/in/b.txt" };
    const float *attn[2] = { attn_a, attn_b };
    const size_t nq[2] = { 1, 1 };
    const size_t nc[2] = { 4, 4 };
    const size_t cs[2] = { 0, 0 };
    const size_t cl[2] = { 4, 4 };
    const size_t qs[2] = { 0, 0 };
    const size_t ql[2] = { 1, 1 };

    wubu_reward_result_t res;
    printf("[T1] batched reward\n");
    int rc = wubu_coherence_reward_compute(kv, paths, attn, nq, nc, cs, cl,
                                           qs, ql, 2, &res);
    chk("compute returns 0", rc == 0);
    chk("n_entries == 2", res.n_entries == 2);
    chk("mean_score == 0.625", close_f(res.mean_score, 0.625f, 1e-4f));
    chk("reward == clamped mean 0.625", close_f(res.reward, 0.625f, 1e-4f));

    printf("[T2] per-file scores\n");
    chk("entry[0] score == 0.85 (focused)",
        close_f(res.entries[0].coh.score, 0.85f, 1e-4f));
    chk("entry[1] score == 0.4 (uniform)",
        close_f(res.entries[1].coh.score, 0.4f, 1e-4f));
    chk("entry[0] mass == 1.0",
        close_f(res.entries[0].coh.attention_mass, 1.0f, 1e-5f));
    wubu_reward_result_free(&res);

    printf("[T3] unknown path -> zero-scored, no crash\n");
    const char *paths3[3] = { "/kv/in/a.txt", "/kv/in/b.txt", "/kv/in/nope.txt" };
    const float *attn3[3] = { attn_a, attn_b, attn_b };
    const size_t nq3[3] = { 1, 1, 1 };
    const size_t nc3[3] = { 4, 4, 4 };
    const size_t cs3[3] = { 0, 0, 0 };
    const size_t cl3[3] = { 4, 4, 4 };
    const size_t qs3[3] = { 0, 0, 0 };
    const size_t ql3[3] = { 1, 1, 1 };
    rc = wubu_coherence_reward_compute(kv, paths3, attn3, nq3, nc3, cs3, cl3,
                                       qs3, ql3, 3, &res);
    chk("compute returns 0", rc == 0);
    chk("n_entries == 3", res.n_entries == 3);
    chk("unknown entry score == 0", close_f(res.entries[2].coh.score, 0.0f, 1e-5f));
    chk("mean == (0.85+0.4+0)/3 = 0.4167",
        close_f(res.mean_score, 0.4167f, 1e-4f));
    chk("reward tracks the mean", close_f(res.reward, res.mean_score, 1e-6f));

    printf("[T4] free leaves the result inert\n");
    wubu_reward_result_free(&res);
    chk("after free, n_entries == 0", res.n_entries == 0);
    chk("after free, reward == 0", res.reward == 0.0f);

    wubu_kv_embedding_free(kv);
    wubu_kvfs_free(fs);

    printf("test_coherence_reward: %s (%d failures)\n", fail ? "FAIL" : "PASS", fail);
    return fail ? 1 : 0;
}
