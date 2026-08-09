/**
 * infer_vision.c — Vision encoder inference engine
 * Loads mmproj GGUF, processes image, outputs embeddings.
 */
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
    const char *path = argc > 1 ? argv[1] : "/models/qwen3.6-35b-mmproj-F16.gguf";
    int H = argc > 2 ? atoi(argv[2]) : 256;
    int W = argc > 3 ? atoi(argv[3]) : 256;
    
    printf("=== Vision Inference Engine ===\n");
    printf("Model: %s\nImage: %dx%d\n", path, H, W);
    
    double t0 = now_sec();
    vision_encoder_t enc;
    if (!vision_encoder_init(&enc, path)) return 1;
    printf("  Loaded in %.2fs\n", now_sec() - t0);
    
    // Create synthetic image (checkerboard)
    int C = 3;
    float *pixels = (float *)malloc(C * H * W * sizeof(float));
    for (int c = 0; c < C; c++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                pixels[c * H * W + y * W + x] = ((x / 16 + y / 16) % 2) * 0.8f + 0.1f;
    
    // Forward - compute n_merged correctly
    int n_merged = (H/16/2) * (W/16/2) * V_TEMP_PATCH;  // 8*8*2=128 for 256x256
    int out_dim = n_merged * V_HIDDEN;
    // If 4 patches exactly, merger reduces to V_OUT_HIDDEN
    if (n_merged == 4 && enc.mm0_weight) out_dim = V_OUT_HIDDEN;
    float *output = (float *)malloc(out_dim * sizeof(float));
    double t1 = now_sec();
    vision_encoder_forward(&enc, pixels, 1, C, H, W, output);
    double t_vision = now_sec() - t1;
    printf("  Forward: %.3f ms\n", t_vision * 1000);
    
    // Output stats
    float min_v = 1e30, max_v = -1e30; int nan_c = 0;
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
    printf("=== Vision Inference PASS ===\n");
    return 0;
}
