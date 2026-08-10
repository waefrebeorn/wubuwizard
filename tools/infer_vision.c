/**
 * infer_vision.c — Vision encoder inference engine
 * Loads an mmproj GGUF, processes an image, outputs embeddings.
 *
 * Auto-detects the encoder architecture from the mmproj tensors:
 *   - SigLIP (SmolVLM2): v.patch_embd.weight + v.blk.* (wubu_siglip)
 *   - Qwen3.6 3D-ViT:    vision_encoder.* (wubu_vision_encoder)
 * Both encoders coexist; the device's SmolVLM2 mmproj is SigLIP.
 */
#include "wubu_siglip.h"
#include "wubu_vision.h"
#include "wubu_vision_encoder.h"
#include "wubu_core_dumps.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Does the GGUF carry SigLIP tensors (v.patch_embd / v.blk.*)? */
static int is_siglip_mmproj(const char *path) {
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) return 0;
    int sig = (gguf_find_tensor(ctx, "v.patch_embd.weight") != NULL) &&
              (gguf_find_tensor(ctx, "v.post_ln.weight") != NULL) &&
              (gguf_find_tensor(ctx, "mm.model.fc.weight") != NULL);
    gguf_close(ctx);
    return sig;
}

int main(int argc, char **argv) {
    wubu_disable_core_dumps();
    const char *path = argc > 1 ? argv[1] : "models/mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf";
    int H = argc > 2 ? atoi(argv[2]) : 378;
    int W = argc > 3 ? atoi(argv[3]) : 378;
    const int C = 3;

    printf("=== Vision Inference Engine ===\n");
    printf("Model: %s\nImage: %dx%d\n", path, H, W);

    if (is_siglip_mmproj(path)) {
        /* ---- SigLIP (SmolVLM2) path ---- */
        double t0 = now_sec();
        wubu_siglip_t enc;
        if (!wubu_siglip_init(&enc, path)) return 1;
        printf("  Architecture: SigLIP (SmolVLM2)\n");
        printf("  Loaded in %.2fs\n", now_sec() - t0);

        float *pixels = (float *)malloc((size_t)C * H * W * sizeof(float));
        for (int c = 0; c < C; c++)
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                    pixels[c * H * W + y * W + x] = ((x + y) % 2 == 0) ? 1.0f : 0.0f;

        float *output = (float *)malloc((size_t)SIGLIP_N_TOK * SIGLIP_PROJ * sizeof(float));
        double t1 = now_sec();
        wubu_siglip_forward(&enc, pixels, 1, C, H, W, output);
        double t_vision = now_sec() - t1;
        printf("  Forward: %.3f ms\n", t_vision * 1000);

        int nan_c = 0;
        for (int i = 0; i < SIGLIP_N_TOK * SIGLIP_PROJ; i++)
            if (isnan(output[i])) nan_c++;
        printf("  Output[0:8]:");
        for (int i = 0; i < 8; i++) printf(" %+.4f", output[i]);
        printf("\n  NaN: %d | dim=%dx%d\n", nan_c, SIGLIP_N_TOK, SIGLIP_PROJ);

        free(pixels);
        free(output);
        wubu_siglip_free(&enc);
    } else {
        /* ---- Qwen3.6 3D-ViT path (recovered encoder) ---- */
        double t0 = now_sec();
        vision_encoder_t enc;
        if (!vision_encoder_init(&enc, path)) return 1;
        printf("  Architecture: Qwen3.6 3D-ViT\n");
        printf("  Loaded in %.2fs\n", now_sec() - t0);

        int Cv = 3;
        float *pixels = (float *)malloc(Cv * H * W * sizeof(float));
        for (int c = 0; c < Cv; c++)
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                    pixels[c * H * W + y * W + x] = ((x / 16 + y / 16) % 2) * 0.8f + 0.1f;

        int n_merged = (H / 16 / 2) * (W / 16 / 2) * V_TEMP_PATCH;
        int out_dim = n_merged * V_HIDDEN;
        if (n_merged == 4 && enc.mm0_weight) out_dim = V_OUT_HIDDEN;
        float *output = (float *)malloc(out_dim * sizeof(float));
        double t1 = now_sec();
        vision_encoder_forward(&enc, pixels, 1, Cv, H, W, output);
        double t_vision = now_sec() - t1;
        printf("  Forward: %.3f ms\n", t_vision * 1000);

        float min_v = 1e30f, max_v = -1e30f;
        int nan_c = 0;
        for (int i = 0; i < out_dim; i++) {
            if (output[i] < min_v) min_v = output[i];
            if (output[i] > max_v) max_v = output[i];
            if (isnan(output[i])) nan_c++;
        }
        printf("  Output[0:8]:");
        for (int i = 0; i < 8 && i < out_dim; i++) printf(" %+.4f", output[i]);
        printf("\n  Range: [%.4f, %.4f] | NaN: %d | dim=%d\n", min_v, max_v, nan_c, out_dim);

        free(pixels);
        free(output);
        vision_encoder_free(&enc);
    }
    printf("=== Vision Inference PASS ===\n");
    return 0;
}
