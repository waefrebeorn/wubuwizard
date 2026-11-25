/* inkling_logits -- run the depth path for one token and report the logit distribution. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gguf_reader.h"
#include "wubu_inkling2.h"

int in2_forward_token(gguf_ctx *, const wubu_inkling2_model_t *, const wubu_inkling2_cfg_t *,
                      int, float *, float *);

int main(int argc, char **argv)
{
    const char *path = "C:/Users/eman5/models/inkling-2-mini/Inkling-2-Mini-UD-IQ2_XXS.gguf";
    const int token = argc > 1 ? atoi(argv[1]) : 1234;

    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) { fprintf(stderr, "open failed\n"); return 1; }
    wubu_inkling2_cfg_t cfg;
    if (wubu_inkling2_probe(ctx, &cfg) != WUBU_INKLING2_OK) { fprintf(stderr, "probe failed\n"); return 2; }
    static wubu_inkling2_model_t m;
    if (wubu_inkling2_load(ctx, &cfg, &m) != WUBU_INKLING2_OK) { fprintf(stderr, "load failed\n"); return 3; }

    float *h = (float *)calloc(cfg.d_model, sizeof(float));
    float *logits = (float *)calloc(cfg.vocab_size, sizeof(float));
    const int rc = in2_forward_token(ctx, &m, &cfg, token, h, logits);
    printf("forward rc=%d  token=%d  d=%d vocab=%d\n", rc, token, cfg.d_model, cfg.vocab_size);
    if (rc != 0) return rc;

    double mx = -1e30, sum = 0; int bad = 0;
    for (int i = 0; i < cfg.vocab_size; ++i) {
        if (!(logits[i] == logits[i])) ++bad;
        if (logits[i] > mx) mx = logits[i];
    }
    for (int i = 0; i < cfg.vocab_size; ++i) sum += exp(logits[i] - mx);
    printf("logits: max %.4f  min %.4f  mean-logit %.4f  nan=%d  logZ %.4f\n",
           mx, logits[0], 0.0, bad, log(sum) + mx);
    /* top-5 */
    for (int k = 0; k < 5; ++k) {
        int best = -1; double bv = -1e30;
        for (int i = 0; i < cfg.vocab_size; ++i) {
            int taken = 0;
            for (int j = 0; j < k; ++j) { }
            if (logits[i] > bv) { bv = logits[i]; best = i; }
        }
        printf("  top%d id=%d logit=%.4f z=%.2f\n", k + 1, best, bv, (bv - mx));
        logits[best] = -1e30;
    }
    free(h); free(logits); gguf_close(ctx);
    return 0;
}
