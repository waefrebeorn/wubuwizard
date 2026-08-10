/* test_siglip_parity.c — run the same cb checkerboard through our
 * SigLIP encoder and print stats for oracle comparison:
 *   - pixel_shuffle stage sum (before fc)
 *   - final output sum (after fc)
 *   - first/last few values of the 81x2048 output
 * The oracle (llama-mtmd-debug -p encode -n 378 --image cb):
 *   pixel_shuffle sum = -652.302490
 *   node_860 (final)  sum = 4342.329102
 */
#include "wubu_siglip.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "models/mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf";
    wubu_siglip_t enc;
    if (!wubu_siglip_init(&enc, path)) return 1;

    const int H = 378, W = 378, C = 3;
    float *img = (float *)malloc(C * H * W * sizeof(float));
    for (int c = 0; c < C; c++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                img[c * H * W + y * W + x] = ((x + y) % 2 == 0) ? 1.0f : 0.0f;

    float *out = (float *)calloc(SIGLIP_N_TOK * SIGLIP_PROJ, sizeof(float));
    wubu_siglip_forward(&enc, img, 1, C, H, W, out);

    double sum = 0, rms = 0;
    for (int i = 0; i < SIGLIP_N_TOK * SIGLIP_PROJ; i++) {
        sum += out[i];
        rms += (double)out[i] * out[i];
    }
    printf("final output sum    = %.6f   (oracle 4342.329102)\n", sum);
    printf("final output rms    = %.6f\n", sqrt(rms / (SIGLIP_N_TOK * SIGLIP_PROJ)));
    printf("final out[0:6]      = %.4f %.4f %.4f %.4f %.4f %.4f\n",
           out[0], out[1], out[2], out[3], out[4], out[5]);
    printf("final out tok0[-6:] = %.4f %.4f %.4f %.4f %.4f %.4f\n",
           out[SIGLIP_PROJ-6], out[SIGLIP_PROJ-5], out[SIGLIP_PROJ-4],
           out[SIGLIP_PROJ-3], out[SIGLIP_PROJ-2], out[SIGLIP_PROJ-1]);

    free(img); free(out);
    wubu_siglip_free(&enc);
    return 0;
}
