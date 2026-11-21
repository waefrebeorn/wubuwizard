/* infer_any.c — architecture-driven inference runner.
 *
 * Probes the GGUF, then dispatches to the appropriate engine family instead
 * of requiring the caller to know whether a model is Gemma 4, LFM2, MoE, or
 * a generic transformer. Unknown architectures still use the generic path
 * with an explicit warning.
 */
#include "wubu_model_arch.h"
#include "wubu_gemma4.h"
#include "wubu_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_top5(const float *logits, int vocab)
{
    int best[5] = { -1, -1, -1, -1, -1 };
    float bv[5] = { -1e30f, -1e30f, -1e30f, -1e30f, -1e30f };
    for (int i = 0; i < vocab; i++) {
        float v = logits[i];
        for (int k = 0; k < 5; k++) {
            if (v > bv[k]) {
                for (int j = 4; j > k; j--) { bv[j] = bv[j - 1]; best[j] = best[j - 1]; }
                bv[k] = v; best[k] = i;
                break;
            }
        }
    }
    printf("top5:");
    for (int k = 0; k < 5; k++) printf(" (%d,%.2f)", best[k], bv[k]);
    printf("\n");
}

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]); return 2; }
    const char *path = argv[1];
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) { fprintf(stderr, "cannot open %s\n", path); return 1; }
    wubu_arch_spec_t spec;
    if (!wubu_arch_probe(ctx, &spec)) { fprintf(stderr, "probe failed\n"); gguf_close(ctx); return 1; }
    wubu_arch_print(&spec, stdout);
    gguf_close(ctx);

    int vocab = spec.vocab_size > 0 ? spec.vocab_size : 32000;
    int T = 5;
    float *logits = (float *)calloc((size_t)vocab * (size_t)T, sizeof(float));
    if (!logits) return 1;
    int tokens[5] = { 107, 105, 106, 4368, 107 };
    for (int i = 0; i < T; i++) if (tokens[i] < 0 || tokens[i] >= vocab) tokens[i] = vocab - 1;
    const char *runner = wubu_runner_name(spec.runner);

    if (spec.runner == WUBU_RUNNER_GEMMA4) {
        g4_model_t g4;
        if (!g4_model_init(&g4, path)) { fprintf(stderr, "g4 init failed\n"); free(logits); return 1; }
        g4_model_forward_from_tokens(&g4, tokens, 1, T, logits);
        g4_model_destroy(&g4);
    } else if (spec.runner == WUBU_RUNNER_MOE || spec.runner == WUBU_RUNNER_SSM) {
        fprintf(stderr, "runner=%s not wired into infer_any; refusing generic dispatch\n", runner);
        free(logits);
        return 3;
    } else if (spec.runner == WUBU_RUNNER_UNKNOWN) {
        fprintf(stderr, "unknown architecture %s; refusing generic dispatch\n", spec.arch_name);
        free(logits);
        return 3;
    } else {
        wubu_model_t m;
        if (!wubu_model_init(&m, path)) { fprintf(stderr, "generic init failed\n"); free(logits); return 1; }
        wubu_model_forward(&m, tokens, 1, T, logits);
        wubu_model_free(&m);
    }
    printf("runner=%s\n", runner);
    print_top5(logits + (size_t)(T - 1) * vocab, vocab);
    free(logits);
    return 0;
}
