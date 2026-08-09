/* wubu_sd_vae.h -- LDM AutoencoderKL decoder for the WuBu stable-diffusion
 * engine (SD 1.5 / Anything-V5 VAE: ch=128, ch_mult [1,2,4,4], z=4).
 * C11, opaque struct.
 */
#ifndef WUBU_SD_VAE_H
#define WUBU_SD_VAE_H

#include <stdint.h>

typedef struct wubu_sd_vae wubu_sd_vae_t;

/* load from an open GGUF context */
wubu_sd_vae_t *wubu_sd_vae_load(void *gguf_ctx);
void wubu_sd_vae_free(wubu_sd_vae_t *v);

/* release weight cache — VAE runs after UNet; free UNet cache first */
void wubu_sd_vae_clear_cache(wubu_sd_vae_t *v);

/* decode latent [4][H][W] (H,W = 64 for 512x512) -> RGB [3][8H][8W]
 * (8x spatial upsample: 64 -> 512). out must be 3*8H*8W floats. */
int wubu_sd_vae_decode(wubu_sd_vae_t *v, const float *latent, int H, int W,
                       float *out);

#endif
