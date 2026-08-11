#ifndef WUBU_SD_TAESD_H
#define WUBU_SD_TAESD_H

/* wubu_sd_taesd.h -- Tiny AutoEncoder for Stable Diffusion (TAESD)
 * decoder: madebyollin/taesd, 1.2M-param distilled VAE decoder.
 * Decodes a 4x(H/8)x(W/8) latent to 3xHxW in [0,1] at ~40x less cost
 * than the full SD VAE decoder. Preview-quality ("fudges fine details").
 * Weights: taesd_decoder.safetensors (F32, [C_out][C_in][KH][KW]). */

typedef struct wubu_sd_taesd wubu_sd_taesd_t;

/* Load decoder weights from a safetensors file. Returns NULL on failure. */
wubu_sd_taesd_t *wubu_sd_taesd_load(const char *safetensors_path);

/* Decode latent [4][H][W] (H,W multiples of 8) -> img [3][8H][8W] in [0,1].
 * Returns 0 on success, non-zero on failure. */
int wubu_sd_taesd_decode(wubu_sd_taesd_t *t, const float *latent,
                         int H, int W, float *img);

void wubu_sd_taesd_free(wubu_sd_taesd_t *t);

#endif
