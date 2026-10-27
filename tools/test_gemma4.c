/* test_gemma4.c — Gemma 4 12B dual-head-dim ISWA test.
 *
 * CPU only: make test_gemma4
 * GPU: make test_gemma4_gpu  (links with gpu_gemma4.o, includes GPU path)
 */

#include "wubu_gemma4.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#ifdef GPU_SUPPORT
#include "gpu_gemma4.h"
#endif

static double wall_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#ifdef GPU_SUPPORT
/* GPU context — global so GPU-forward functions can access it */
g4_gpu_ctx_t *g_gpu_ctx = NULL;
#endif

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    g4_model_t model;
    double t0 = wall_time();

    if (!g4_model_init(&model, argv[1])) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    double t1 = wall_time();
    printf("[TEST] Model loaded in %.2f s (data blob: %zu MB, token_embd.data=%p)\n",
           t1 - t0, model.data_blob_size / 1048576,
           (const void*)model.token_embd.data);

#ifdef GPU_SUPPORT
    /* Initialize GPU context */
    extern g4_gpu_ctx_t *g_gpu_ctx;
    g_gpu_ctx = g4_gpu_init(16, 4096, G4_N_LAYERS);
    if (g_gpu_ctx) {
        printf("[TEST] GPU context initialized (sm_89 + sm_120 supported)\n");
    } else {
        printf("[TEST] GPU init failed, running CPU only\n");
    }
#endif

    /* Quick prefill test with 8 dummy tokens */
    int prompt_len = 8;
    int *tokens = (int *)calloc((size_t)prompt_len, sizeof(int));
    for (int i = 0; i < prompt_len; i++) tokens[i] = 1;

    float *logits = (float *)malloc((size_t)prompt_len * G4_VOCAB * sizeof(float));

#ifdef GPU_SUPPORT
    /* GPU forward: prefill — do embedding lookup, then GPU forward */
    {
        float *embd = (float*)malloc((size_t)prompt_len * G4_HIDDEN * sizeof(float));
        int row_bytes = (int)(model.token_embd.raw_bytes / G4_VOCAB);
        for (int i = 0; i < prompt_len; i++) {
            int tok = tokens[i];
            if (tok < 0 || tok >= G4_VOCAB) tok = 0;
            gguf_dequantize(model.token_embd.data + (size_t)tok * row_bytes,
                           model.token_embd.ggml_type, G4_HIDDEN,
                           embd + (size_t)i * G4_HIDDEN);
        }
        float scale = sqrtf((float)G4_HIDDEN);
        for (int i = 0; i < prompt_len * G4_HIDDEN; i++) embd[i] *= scale;

        printf("[TEST] Prefill %d tokens (GPU mode)...\n", prompt_len);
        t0 = wall_time();
        g4_model_forward_gpu(g_gpu_ctx, &model, embd, 1, prompt_len, logits);
        t1 = wall_time();
        printf("[TEST] Prefill done: %.2f s (%.1f tok/s)\n", t1 - t0, prompt_len / (t1 - t0));
        free(embd);
    }
#else
    printf("[TEST] Prefill %d tokens (CPU)...\n", prompt_len);
    t0 = wall_time();
    g4_model_forward_from_tokens(&model, tokens, 1, prompt_len, logits);
    t1 = wall_time();
    printf("[TEST] Prefill done: %.2f s (%.1f tok/s)\n", t1 - t0, prompt_len / (t1 - t0));
#endif

    /* Top token from last position */
    int last_idx = (prompt_len - 1) * G4_VOCAB;
    int top = 0; float top_s = -1e30f;
    for (int i = 0; i < G4_VOCAB; i++) {
        if (logits[last_idx + i] > top_s) { top_s = logits[last_idx + i]; top = i; }
    }
    printf("[TEST] Top token: %d (score=%.4f)\n", top, top_s);

    /* Verify architecture */
    printf("[TEST] Architecture:\n");
    int full_count = 0;
    for (int i = 0; i < G4_N_LAYERS; i++) {
        if (g4_layer_is_full(i)) {
            full_count++;
            g4_layer_t *l = &model.layers[i];
            printf("  Layer %2d: FULL (Q=%d, KV=%d, hd=%d, kv_heads=%d, rot=%d, base=%.0f, share=%d)\n",
                   i, l->q_dim, l->kv_dim, l->head_dim, l->kv_heads, l->n_rot, l->rope_base, l->share_kv);
        }
    }
    printf("  Full attention layers: %d, KV sharing layers: 40-47, Tied output: %s\n",
           full_count, model.tied_output ? "yes" : "no");

    /* Decode 3 tokens */
    printf("[TEST] Decode 3 tokens...\n");
    t0 = wall_time();
    for (int i = 0; i < 3; i++) {
#ifdef GPU_SUPPORT
        g4_model_decode_gpu(g_gpu_ctx, &model, top, logits);
#else
        g4_model_decode(&model, top, logits);
#endif
        top = 0; top_s = -1e30f;
        for (int j = 0; j < G4_VOCAB; j++) {
            if (logits[j] > top_s) { top_s = logits[j]; top = j; }
        }
        printf("  tok %d: %d (%.4f)\n", i, top, top_s);
    }
    t1 = wall_time();
    printf("[TEST] Decode 3: %.2f s (%.1f tok/s)\n", t1 - t0, 3.0 / (t1 - t0));

#ifdef GPU_SUPPORT
    if (g_gpu_ctx) g4_gpu_destroy(g_gpu_ctx);
#endif
    g4_model_destroy(&model);
    free(tokens);
    free(logits);
    printf("[TEST] Done.\n");
    return 0;
}
