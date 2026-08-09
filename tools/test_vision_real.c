/**
 * test_vision_real.c — Vision encoder with real screenshot image
 * Loads raw float pixels from file, runs vision encoder.
 */
#include "wubu_vision.h"
#include "wubu_vision_encoder.h"
#include "wubu_model.h"
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

int main(int argc, char **argv) {
    const char *vision_path = argc > 1 ? argv[1] : "/mnt/wslg/distro/models/qwen3.6-35b-mmproj-F16.gguf";
    const char *pixel_file  = argc > 2 ? argv[2] : "/tmp/screen_vision_input.bin";
    int H = argc > 3 ? atoi(argv[3]) : 256;
    int W = argc > 4 ? atoi(argv[4]) : 256;
    const char *model_path = argc > 5 ? argv[5] : "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";

    printf("=== Vision Encoder — Real Image Test ===\n");
    printf("Image: %dx%d\n", H, W);

    // Load vision encoder
    double t0 = now_sec();
    vision_encoder_t enc;
    if (!vision_encoder_init(&enc, vision_path)) return 1;
    printf("Vision loaded in %.2fs\n", now_sec() - t0);

    // Load pixels from file ([C,H,W] f32, normalized [0,1])
    FILE *f = fopen(pixel_file, "rb");
    if (!f) { perror("fopen"); return 1; }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    rewind(f);
    float *pixels = (float *)malloc(fsize);
    fread(pixels, 1, fsize, f);
    fclose(f);
    printf("Pixels loaded: %ld bytes (%ld floats)\n", fsize, fsize/sizeof(float));

    // Run vision encoder
    int n_merged = (H/16/2) * (W/16/2) * V_TEMP_PATCH;
    int out_size = n_merged * V_OUT_HIDDEN;
    float *output = (float *)malloc(out_size * sizeof(float));

    double t1 = now_sec();
    vision_encoder_forward(&enc, pixels, 1, 3, H, W, output);
    double t_vision = now_sec() - t1;
    printf("Vision forward: %.3f ms\n", t_vision * 1000);
    printf("Output tokens: %d x %d\n", n_merged, V_OUT_HIDDEN);

    // Output stats
    float min_v = 1e30, max_v = -1e30; int nan_c = 0, inf_c = 0;
    for (int i = 0; i < out_size; i++) {
        if (output[i] < min_v) min_v = output[i];
        if (output[i] > max_v) max_v = output[i];
        if (isnan(output[i])) nan_c++;
        if (!isfinite(output[i]) && !isnan(output[i])) inf_c++;
    }
    printf("Vision output range: [%.4f, %.4f]\n", min_v, max_v);
    printf("NaN: %d | Inf: %d\n", nan_c, inf_c);
    printf("First token[0:8]:");
    for (int i = 0; i < 8 && i < V_OUT_HIDDEN; i++)
        printf(" %+.4f", output[i]);
    printf("\n");

    // Now feed through text model (optional, only if model file provided)
    if (argc > 5) {
        printf("\n--- Text Model Forward ---\n");
        wubu_model_t model;
        if (!wubu_model_init(&model, model_path)) return 1;

        int B = 1, T = n_merged;
        float *logits = (float *)malloc(B * T * model.vocab_size * sizeof(float));

        double t2 = now_sec();
        wubu_model_forward_from_embd(&model, output, B, T, logits);
        double t_text = now_sec() - t2;
        printf("Text forward: %.3f s (%d layers, %d tokens)\n", t_text, model.n_layers, T);

        float min_l = 1e30, max_l = -1e30; int nan_l = 0;
        for (int i = 0; i < B * T * model.vocab_size; i++) {
            if (logits[i] < min_l) min_l = logits[i];
            if (logits[i] > max_l) max_l = logits[i];
            if (isnan(logits[i])) nan_l++;
        }
        printf("Logit range: [%.4f, %.4f] | NaN: %d\n", min_l, max_l, nan_l);
        printf("First logit[0:4]: %.4f %.4f %.4f %.4f\n",
               logits[0], logits[1], logits[2], logits[3]);

        wubu_model_free(&model);
        free(logits);
    }

    vision_encoder_free(&enc);
    free(pixels);
    free(output);

    printf("\n=== DONE ===\n");
    return (nan_c + inf_c) > 0;
}
