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

/* DeepCache cadence: set the denoising STEP index (0-based) before the
 * forward calls of that step. Full passes run when step % interval == 0
 * (or the first call after enable); cached passes reuse the deep feature. */
void wubu_sd_unet_set_step(wubu_sd_unet_t *u, int step);
/* DeepCache CFG pass selector: 0 = cond, 1 = uncond (per forward call). */
void wubu_sd_unet_set_pass(wubu_sd_unet_t *u, int pass);
/* DeepCache schedule: per-step full-pass map (1 = full, 0 = cached),
 * mirroring the paper's non-uniform quad-center schedule. */
void wubu_sd_unet_set_dc_schedule(wubu_sd_unet_t *u, const int *full_map, int n);

/* Set the latent spatial resolution (HxW, both multiples of 8). Smaller
 * latent = proportionally fewer UNet FLOPs (e.g. 30x52 for the 240x416
 * display vs the 64x64 default). 0 on success, -1 if dims invalid. */
int wubu_sd_unet_set_resolution(wubu_sd_unet_t *u, int h, int w);

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

/* per-layer perf counters (0=conv 1=linear 2=gn 3=silu 4=attn 5=ffn
 * 6=lookup 7=other); g_timing=1 enables per-layer stderr timing. */
extern double g_t[8];
extern int g_timing;

#endif /* WUBU_SD_UNET_H */
