/* vae_sanity.c -- decode a ZERO latent (and a scaled-ones latent).
 * Correct VAE: zeros -> flat gray (mean ~0, tiny spread).
 * Broken VAE: zeros -> noise or huge values. */
#include "wubu_sd_vae.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    gguf_ctx *g = gguf_open(argv[1]);
    if (!g) { fprintf(stderr, "open failed\n"); return 1; }
    wubu_sd_vae_t *v = wubu_sd_vae_load(g);
    if (!v) { fprintf(stderr, "vae load failed\n"); return 1; }

    float latent[4 * 64 * 64];
    float img[3 * 512 * 512];
    memset(latent, 0, sizeof latent);
    fprintf(stderr, "[vae] decoding zeros...\n");
    if (wubu_sd_vae_decode(v, latent, 64, 64, img) != 0) {
        fprintf(stderr, "decode failed\n"); return 1;
    }
    double mean = 0, var = 0, mx = -1e30, mn = 1e30;
    for (int i = 0; i < 3 * 512 * 512; i++) {
        double f = img[i];
        mean += f; var += f * f;
        if (f > mx) mx = f; if (f < mn) mn = f;
    }
    mean /= 3 * 512 * 512;
    var = var / (3 * 512 * 512) - mean * mean;
    printf("zeros: mean=%+.6f var=%.6f min=%.6f max=%.6f\n", mean, var, mn, mx);
    printf("verdict: %s\n", (fabs(mean) < 0.2 && var < 0.5) ? "VAE OK (gray)" :
           (var > 2.0) ? "VAE BROKEN (noise)" : "VAE SUSPICIOUS");
    wubu_sd_vae_free(v);
    return 0;
}
