/* wubu_sd_unet.h -- LDM UNet for the WuBu stable-diffusion engine
 * (SD 1.5 topology: 320/640/1280 channels, 12+12 blocks, cross-attn
 * against the CLIP 77x768 context). C11, opaque struct.
 */
#ifndef WUBU_SD_UNET_H
#define WUBU_SD_UNET_H

#include <stddef.h>

typedef struct wubu_sd_unet wubu_sd_unet_t;

/* Load the UNet from a GGUF ctx (weights stay quantized; F32 materialized
 * lazily per tensor). Returns NULL on failure. */
wubu_sd_unet_t *wubu_sd_unet_load(void *gguf_ctx);

void wubu_sd_unet_free(wubu_sd_unet_t *u);

/* release weight cache — call between UNet phase and VAE phase */
void wubu_sd_unet_clear_cache(wubu_sd_unet_t *u);

/* Denoise one step:
 *   latent  [4][64][64] (noisy latent at timestep t)
 *   t       diffusion timestep (int, e.g. from the sampler's schedule)
 *   ctx     [77][768] CLIP text embedding (the conditioning)
 *   out     [4][64][64] predicted noise
 * Returns 0 on success. */
int wubu_sd_unet_forward(wubu_sd_unet_t *u,
                         const float *latent, int t,
                         const float *ctx,
                         float *out);

#endif /* WUBU_SD_UNET_H */
