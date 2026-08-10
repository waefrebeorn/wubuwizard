/* test_siglip.c — load the SmolVLM2 mmproj, run a synthetic image,
 * check output is finite and deterministic. */
#include "wubu_siglip.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "models/mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf";
    printf("=== SigLIP test (%s) ===\n", path);
    wubu_siglip_t enc;
    if (!wubu_siglip_init(&enc, path)) return 1;
    printf("  loaded\n");

    const int H = 378, W = 378, C = 3;
    float *img = (float *)malloc(C * H * W * sizeof(float));
    /* mtmd-debug "cb" pattern: alternate 1.0f and 0.0f per pixel,
     * same for all channels — matches llama.cpp's checkerboard. */
    for (int c = 0; c < C; c++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                img[c * H * W + y * W + x] = ((x + y) % 2 == 0) ? 1.0f : 0.0f;

    float *out = (float *)malloc(SIGLIP_N_TOK * SIGLIP_PROJ * sizeof(float));
    memset(out, 0, SIGLIP_N_TOK * SIGLIP_PROJ * sizeof(float));
    wubu_siglip_forward(&enc, img, 1, C, H, W, out);

    int nan = 0; double rms = 0, mx = -1e30, mn = 1e30;
    for (int i = 0; i < SIGLIP_N_TOK * SIGLIP_PROJ; i++) {
        if (isnan(out[i]) || isinf(out[i])) nan++;
        rms += (double)out[i] * out[i];
        if (out[i] > mx) mx = out[i];
        if (out[i] < mn) mn = out[i];
    }
    printf("  out[0:6]  = %.4f %.4f %.4f %.4f %.4f %.4f\n", out[0], out[1], out[2], out[3], out[4], out[5]);
    printf("  out[-6:]  = %.4f %.4f %.4f %.4f %.4f %.4f\n",
           out[SIGLIP_N_TOK*SIGLIP_PROJ-6], out[SIGLIP_N_TOK*SIGLIP_PROJ-5],
           out[SIGLIP_N_TOK*SIGLIP_PROJ-4], out[SIGLIP_N_TOK*SIGLIP_PROJ-3],
           out[SIGLIP_N_TOK*SIGLIP_PROJ-2], out[SIGLIP_N_TOK*SIGLIP_PROJ-1]);
    printf("  NaN/Inf: %d, rms=%.4f, min=%.4f max=%.4f\n", nan, sqrt(rms / (SIGLIP_N_TOK*SIGLIP_PROJ)), mn, mx);

    /* deterministic check: run twice, compare */
    float *out2 = (float *)malloc(SIGLIP_N_TOK * SIGLIP_PROJ * sizeof(float));
    wubu_siglip_forward(&enc, img, 1, C, H, W, out2);
    double maxdiff = 0;
    for (int i = 0; i < SIGLIP_N_TOK * SIGLIP_PROJ; i++) {
        double d = fabs((double)out[i] - out2[i]);
        if (d > maxdiff) maxdiff = d;
    }
    printf("  deterministic maxdiff: %e\n", maxdiff);

    free(img); free(out); free(out2);
    wubu_siglip_free(&enc);
    printf("=== done ===\n");
    return nan ? 1 : 0;
}
