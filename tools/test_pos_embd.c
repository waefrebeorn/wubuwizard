/**
 * test_pos_embd.c — dump vision position embedding to find NaN source
 */
#include "wubu_vision.h"
#include "wubu_vision_encoder.h"
#include <stdio.h>

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/mnt/wslg/distro/models/qwen3.6-35b-mmproj-F16.gguf";
    vision_encoder_t enc;
    if (!vision_encoder_init(&enc, path)) return 1;

    float *w = enc.pos_embd_weight;
    if (!w) { printf("No pos_embd_weight\n"); return 1; }

    // Check if any NaN in full weight
    int first_nan = -1;
    for (int i = 0; i < 1152 * 2304; i++) {
        if (isnan(w[i])) { first_nan = i; break; }
    }
    printf("pos_embd_weight total=%d first_nan=%d\n", 1152*2304, first_nan);

    // Check each position 0..511 for NaN
    for (int pos = 0; pos < 512; pos++) {
        for (int f = 0; f < 1152; f++) {
            float val = w[f * 2304 + pos];  // same access as current code
            if (isnan(val)) {
                printf("  NaN at [f=%d, pos=%d] (linear_idx=%d)\n", f, pos, f*2304+pos);
                goto found;
            }
        }
    }
    found:
    // Try reverse access pattern
    printf("\nAlternative access pattern:\n");
    for (int pos = 0; pos < 512; pos++) {
        for (int f = 0; f < 1152; f++) {
            float val = w[pos * 1152 + f];  // position-major access
            if (isnan(val)) {
                printf("  ALTERNATIVE NaN at [pos=%d, f=%d] (linear_idx=%d)\n", pos, f, pos*1152+f);
                goto found2;
            }
        }
    }
    found2:
    printf("  No NaN with alternative pattern up to pos=511\n");

    vision_encoder_free(&enc);
    return 0;
}
