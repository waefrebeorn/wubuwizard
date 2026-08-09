/**
 * infer_vision_text.c — Vision→Text pipeline integration (S5)
 *
 * Loads vision encoder + text model, runs vision→40-layer pipeline.
 * Usage: ./infer_vision_text [model.gguf] [mmproj.gguf]
 */
#include "wubu_model.h"
#include "wubu_vision.h"
#include "wubu_vision_encoder.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "wubu_core_dumps.h"

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    wubu_disable_core_dumps();
    const char *model_path = argc > 1 ? argv[1] : "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    const char *vision_path = argc > 2 ? argv[2] : "/mnt/wslg/distro/models/qwen3.6-35b-mmproj-F16.gguf";
    int H = argc > 3 ? atoi(argv[3]) : 256;
    int W = argc > 4 ? atoi(argv[4]) : 256;

    printf("=== Vision→Text Pipeline Integration (S5) ===\n");

    double t0 = now_sec();

    // 1. Load vision encoder
    printf("\n--- Vision Encoder ---\n");
    double tv0 = now_sec();
    vision_encoder_t enc;
    if (!vision_encoder_init(&enc, vision_path)) { fprintf(stderr, "Failed to load vision\n"); return 1; }
    printf("  Loaded in %.2fs\n", now_sec() - tv0);

    // 2. Load text model
    printf("\n--- Text Model ---\n");
    double tm0 = now_sec();
    wubu_model_t model;
    if (!wubu_model_init(&model, model_path)) { fprintf(stderr, "Failed to load model\n"); return 1; }
    printf("  Loaded in %.2fs\n", now_sec() - tm0);

    // 3. Create synthetic image
    printf("\n--- Vision Forward ---\n");
    int C = 3;
    float *pixels = (float *)malloc(C * H * W * sizeof(float));
    for (int c = 0; c < C; c++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                pixels[c * H * W + y * W + x] = ((x / 16 + y / 16) % 2) * 0.8f + 0.1f;

    int n_merged = (H/16/2) * (W/16/2) * V_TEMP_PATCH;
    float *vision_embd = (float *)malloc(n_merged * V_OUT_HIDDEN * sizeof(float));

    double tv1 = now_sec();
    vision_encoder_forward(&enc, pixels, 1, C, H, W, vision_embd);
    double t_vision = now_sec() - tv1;
    printf("  Forward: %.3f ms\n", t_vision * 1000);
    printf("  Tokens: %d x %d\n", n_merged, V_OUT_HIDDEN);

    // Vision output stats
    float min_v = 1e30, max_v = -1e30; int nan_c = 0;
    for (int i = 0; i < n_merged * V_OUT_HIDDEN; i++) {
        if (vision_embd[i] < min_v) min_v = vision_embd[i];
        if (vision_embd[i] > max_v) max_v = vision_embd[i];
        if (isnan(vision_embd[i])) nan_c++;
    }
    printf("  Range: [%.4f, %.4f] | NaN: %d\n", min_v, max_v, nan_c);
    printf("  First token[0:4]: %.4f %.4f %.4f %.4f\n",
           vision_embd[0], vision_embd[1], vision_embd[2], vision_embd[3]);

    // 4. Feed through text model
    printf("\n--- Text Model Forward ---\n");
    int B = 1, T = n_merged;
    float *logits = (float *)malloc(B * T * model.vocab_size * sizeof(float));

    double tf0 = now_sec();
    wubu_model_forward_from_embd(&model, vision_embd, B, T, logits);
    double t_text = now_sec() - tf0;
    printf("  Forward: %.3f s (%d layers, %d tokens)\n", t_text, model.n_layers, T);

    // Logit stats
    float min_l = 1e30, max_l = -1e30; int nan_l = 0;
    for (int i = 0; i < B * T * model.vocab_size; i++) {
        if (logits[i] < min_l) min_l = logits[i];
        if (logits[i] > max_l) max_l = logits[i];
        if (isnan(logits[i])) nan_l++;
    }
    printf("  Logit range: [%.4f, %.4f] | NaN: %d\n", min_l, max_l, nan_l);
    printf("  First logit[0:4]: %.4f %.4f %.4f %.4f\n",
           logits[0], logits[1], logits[2], logits[3]);

    // 5. Final summary
    double total = now_sec() - t0;
    printf("\n=== Results ===\n");
    printf("  Vision: %.3f ms | Text: %.3f s | Total: %.3f s\n",
           t_vision * 1000, t_text, total);
    printf("  Vision tokens: %d | Model layers: %d\n", T, model.n_layers);
    printf("  Logits valid: %s\n", nan_l == 0 ? "YES ✓" : "NO ✗");

    // Cleanup
    vision_encoder_free(&enc);
    wubu_model_free(&model);
    free(pixels);
    free(vision_embd);
    free(logits);

    return nan_l > 0 ? 1 : 0;
}
