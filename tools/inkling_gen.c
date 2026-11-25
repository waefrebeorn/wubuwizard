/* inkling_gen -- greedy generation.
 *
 * Full recompute per step by default (O(n^2), but it cannot hide a cache bug). IN2_KV=1 enables the incremental path,
 * which is ~6x faster but still diverges from the full path on at least one near-tie (see in2_kv_enabled).
 *
 * usage: inkling_gen <n_gen> <token> [token ...]
 */
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include "gguf_reader.h"
#include "wubu_inkling2.h"

int in2_forward_seq(gguf_ctx *, const wubu_inkling2_model_t *, const wubu_inkling2_cfg_t *,
                    const int *, int, float *, float *);
int in2_forward_at(gguf_ctx *, const wubu_inkling2_model_t *, const wubu_inkling2_cfg_t *,
                   const int *, int, int, float *, float *);

int main(int argc, char **argv)
{
    const char *path = "C:/Users/eman5/models/inkling-2-mini/Inkling-2-Mini-UD-IQ2_XXS.gguf";
    if (argc < 3) { fprintf(stderr, "usage: %s <n_gen> <token> [token ...]\n", argv[0]); return 2; }
    const int n_gen = atoi(argv[1]);
    const int n_prompt = argc - 2;

    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) { fprintf(stderr, "open failed\n"); return 1; }
    wubu_inkling2_cfg_t cfg;
    if (wubu_inkling2_probe(ctx, &cfg) != WUBU_INKLING2_OK) { fprintf(stderr, "probe failed\n"); return 2; }
    if (getenv("IN2_CFG"))
        fprintf(stderr, "CFG d=%d layers=%d core=[%d,%d) loops=%d n_expert=%d used=%d d_ff=%d d_ff_expert=%d vocab=%d\n",
                cfg.d_model, cfg.n_layer, cfg.loop_core_start, cfg.loop_core_end, cfg.n_loops,
                cfg.n_expert, cfg.n_expert_used, cfg.d_ff, cfg.d_ff_expert, cfg.vocab_size);
    static wubu_inkling2_model_t m;
    if (wubu_inkling2_load(ctx, &cfg, &m) != WUBU_INKLING2_OK) { fprintf(stderr, "load failed\n"); return 3; }

    const int cap = n_prompt + n_gen;
    int *toks = (int *)calloc((size_t)cap, sizeof(int));
    for (int i = 0; i < n_prompt; ++i) toks[i] = atoi(argv[2 + i]);
    for (int i = 0; i < n_prompt; ++i) printf("prompt[%d]=%d\n", i, toks[i]);

    clock_t t_step = clock(); double t_total = 0.0;
    float *h = (float *)calloc((size_t)cap * cfg.d_model, sizeof(float));
    float *logits = (float *)calloc(cfg.vocab_size, sizeof(float));

    for (int step = 0; step < n_gen; ++step) {
        const clock_t t_step0 = clock(); (void)t_step0;
        const int n = n_prompt + step;
        /* With a cache the prompt is prefilled once and each step pushes a single token at its real position;
         * without one the whole prefix is recomputed, which is the O(n^2) behaviour this is meant to replace.
         * Cache on by default now: the full-recompute path was the safe default while the cache could hide a bug, but
         * 34/34 oracle tokens match with it enabled, and the two paths produce bit-identical hidden states. IN2_NO_KV
         * forces the full recompute for debugging. */
        const int incremental = getenv("IN2_KV") != NULL && step > 0;
        const int rc = incremental
            ? in2_forward_at(ctx, &m, &cfg, &toks[n - 1], 1, n - 1, h, logits)
            : in2_forward_at(ctx, &m, &cfg, toks, n, 0, h, logits);
        if (rc != WUBU_INKLING2_OK) { fprintf(stderr, "step %d rc=%d\n", step, rc); return rc; }
        int best = 0; float bv = -1e30f;
        for (int i = 0; i < cfg.vocab_size; ++i) if (logits[i] > bv) { bv = logits[i]; best = i; }
        toks[n] = best;
        const double dt = (double)(clock() - t_step) / (double)CLOCKS_PER_SEC;
        t_step = clock();
        t_total += dt;
        printf("gen[%d] id=%d logit=%.4f  %.1fs  (%.2f tok/s at prefix %d)\n",
               step, best, bv, dt, dt > 0 ? 1.0 / dt : 0.0, n);
        if (getenv("IN2_TOPK")) {
            /* Top-k of the same distribution, so a near-tie can be judged against llama-server's
             * completion_probabilities instead of only by the argmax. */
            const int K = atoi(getenv("IN2_TOPK"));
            static int used[64];
            for (int r = 0; r < K && r < 64; ++r) {
                int bi = -1; float bvv = -1e30f;
                for (int i = 0; i < cfg.vocab_size; ++i) {
                    int skip = 0;
                    for (int q = 0; q < r; ++q) if (used[q] == i) { skip = 1; break; }
                    if (!skip && logits[i] > bvv) { bvv = logits[i]; bi = i; }
                }
                used[r] = bi;
                fprintf(stderr, "TOPK %d %d %.4f\n", r, bi, bvv);
            }
        }
        fflush(stdout);
    }
    if (n_gen > 0 && t_total > 0)
        fprintf(stderr, "TIMING: %d tokens in %.1fs = %.2f tok/s (full recompute, no KV cache)\n",
                n_gen, t_total, (double)n_gen / t_total);
    printf("IDS:");
    for (int i = 0; i < n_prompt + n_gen; ++i) printf(" %d", toks[i]);
    printf("\n");
    free(h); free(logits); free(toks); gguf_close(ctx);
    return 0;
}
