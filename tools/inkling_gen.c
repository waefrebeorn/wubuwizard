/* inkling_gen -- greedy generation with full recompute per step.
 *
 * No KV cache yet: every step re-runs the whole prefix. That is slow (seconds per token) but it cannot
 * hide a cache bug, which is the point of getting output this way first.
 *
 * usage: inkling_gen <n_gen> <token> [token ...]
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gguf_reader.h"
#include "wubu_inkling2.h"

int in2_forward_seq(gguf_ctx *, const wubu_inkling2_model_t *, const wubu_inkling2_cfg_t *,
                    const int *, int, float *, float *);

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
    static wubu_inkling2_model_t m;
    if (wubu_inkling2_load(ctx, &cfg, &m) != WUBU_INKLING2_OK) { fprintf(stderr, "load failed\n"); return 3; }

    const int cap = n_prompt + n_gen;
    int *toks = (int *)calloc((size_t)cap, sizeof(int));
    for (int i = 0; i < n_prompt; ++i) toks[i] = atoi(argv[2 + i]);
    for (int i = 0; i < n_prompt; ++i) printf("prompt[%d]=%d\n", i, toks[i]);

    float *h = (float *)calloc((size_t)cap * cfg.d_model, sizeof(float));
    float *logits = (float *)calloc(cfg.vocab_size, sizeof(float));

    for (int step = 0; step < n_gen; ++step) {
        const int n = n_prompt + step;
        const int rc = in2_forward_seq(ctx, &m, &cfg, toks, n, h, logits);
        if (rc != WUBU_INKLING2_OK) { fprintf(stderr, "step %d rc=%d\n", step, rc); return rc; }
        int best = 0; float bv = -1e30f;
        for (int i = 0; i < cfg.vocab_size; ++i) if (logits[i] > bv) { bv = logits[i]; best = i; }
        toks[n] = best;
        printf("gen[%d] id=%d logit=%.4f\n", step, best, bv);
        fflush(stdout);
    }
    printf("IDS:");
    for (int i = 0; i < n_prompt + n_gen; ++i) printf(" %d", toks[i]);
    printf("\n");
    free(h); free(logits); free(toks); gguf_close(ctx);
    return 0;
}
