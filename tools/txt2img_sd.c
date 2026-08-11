/* txt2img_sd.c -- end-to-end SD 1.5 txt2img on the wizard C11 engine.
 * CLIP text encode -> DDIM (12 steps) -> VAE decode -> PPM out.
 * Usage: txt2img_sd model.gguf "prompt" steps seed out.ppm
 */
#include "wubu_sd_clip.h"
#include "wubu_sd_unet.h"
#include "wubu_sd_vae.h"
#include "wubu_sd_taesd.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* xorshift64 PRNG for reproducible init noise */
static uint64_t rng_state;
static double rng_rand(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (double)(rng_state & 0xFFFFFFFFu) / 4294967296.0;
}
static double rng_gauss(void) {
    double u = rng_rand(), v = rng_rand();
    if (u < 1e-12) u = 1e-12;
    return sqrt(-2.0 * log(u)) * cos(2.0 * M_PI * v);
}

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s model.gguf \"prompt\" steps seed out.ppm [noise.bin]\n", argv[0]);
        return 1;
    }
    const char *model = argv[1];
    const char *prompt = argv[2];
    int steps = atoi(argv[3]);
    uint64_t seed = strtoull(argv[4], NULL, 10);
    const char *outpath = argv[5];
    const char *noise_path = argc > 6 ? argv[6] : NULL;

    gguf_ctx *g = gguf_open(model);
    if (!g) return 1;
    if (getenv("SD_TIMING")) g_timing = 1;
    double t0 = (double)clock() / CLOCKS_PER_SEC;
    wubu_sd_clip_t *clip = wubu_sd_clip_load(g);
    double t1 = (double)clock() / CLOCKS_PER_SEC;
    wubu_sd_unet_t *unet = wubu_sd_unet_load(g);
    double t2 = (double)clock() / CLOCKS_PER_SEC;
    wubu_sd_vae_t *vae = wubu_sd_vae_load(g);
    double t3 = (double)clock() / CLOCKS_PER_SEC;
    fprintf(stderr, "[load] clip %.2fs unet %.2fs vae %.2fs\n", t1 - t0, t2 - t1, t3 - t2);
    if (!clip || !unet || !vae) { fprintf(stderr, "module load failed\n"); return 1; }

    /* CLIP encode */
    float ctx[77 * 768];
    float pooled[768];
    if (wubu_sd_clip_encode(clip, prompt, ctx, pooled) != 0) {
        fprintf(stderr, "clip encode failed\n"); return 1;
    }
    fprintf(stderr, "[txt2img] clip done, pooled[0]=%.4f\n", pooled[0]);
    /* phase separation: CLIP weights are done — free before the UNet
     * (2.9GB F32 UNet cache + CLIP + VAE exceeds the 5.8GB box) */
    wubu_sd_clip_clear_cache(clip);

    /* init latent: gaussian noise, 4x64x64 (or load sd.cpp's EXACT noise) */
    float *x = (float *)malloc(4 * 64 * 64 * sizeof(float));
    rng_state = seed;
    if (noise_path) {
        FILE *nf = fopen(noise_path, "rb");
        if (!nf || fread(x, sizeof(float), 4 * 64 * 64, nf) != (size_t)(4 * 64 * 64)) {
            fprintf(stderr, "noise file read failed\n"); return 1;
        }
        fclose(nf);
        fprintf(stderr, "[txt2img] loaded sd.cpp noise from %s\n", noise_path);
    } else {
        for (int i = 0; i < 4 * 64 * 64; i++) x[i] = (float)rng_gauss();
    }

    /* SD1.5 sampler matching stable-diffusion.cpp exactly:
     * - scaled-linear beta schedule: beta_t = (sqrt(beta_start) +
     *   (sqrt(beta_end)-sqrt(beta_start))*t/999)^2
     * - sigma_t = sqrt((1-ab_t)/ab_t), CompVisDenoiser c_in = 1/sqrt(sigma^2+1)
     * - initial latent x = noise * sigma_max (noise_scaling: latent + noise*sigma
     *   with empty init_latent) — the UNet input x*c_in then has rms ~1.
     *   WITHOUT this the UNet sees near-zeros -> bias-only response -> DC drift.
     * - UNet input is x * c_in (eps-prediction, c_skip=1, c_out=-sigma)
     * - update (eta=0, Euler on sigmas): x = (s_next/s)*x + (1-s_next/s)*denoised
     *   where denoised = x - sigma*eps_cfg. This INTERPOLATES (no DC drift —
     *   the alpha_bar DDIM form can blow up with CFG). */
    float beta[1000], alpha_bar[1000];
    const float bs = sqrtf(0.00085f), be = sqrtf(0.012f);
    for (int t = 0; t < 1000; t++) {
        float b = bs + (be - bs) * (float)t / 999.0f;
        beta[t] = b * b;
        alpha_bar[t] = t == 0 ? (1.0f - beta[0]) : alpha_bar[t - 1] * (1.0f - beta[t]);
    }
    float sigmas[1000];
    for (int t = 0; t < 1000; t++) sigmas[t] = sqrtf((1.0f - alpha_bar[t]) / alpha_bar[t]);
    /* scale initial noise to sigma_max (noise_scaling) */
    const float sigma_max = sigmas[999];
    for (int i = 0; i < 4 * 64 * 64; i++) x[i] *= sigma_max;
    /* even t-spacing (LinearScheduler): t = 999 - step*999/(steps-1), then 0 */
    float sched[13]; /* steps+1 */
    for (int s = 0; s < steps; s++)
        sched[s] = sigmas[(int)lrintf(999.0f - 999.0f * (float)s / (float)(steps - 1))];
    sched[steps] = 0.0f;
    /* uncond guidance: eps = uncond + g*(cond - uncond), g=7.5 */
    {
        /* uncond context = REAL CLIP embedding of the empty prompt "" —
         * encoded ONCE (hoisted out of the step loop; was 12 wasted CLIP
         * forwards per image). */
        float empty_ctx[77 * 768];
        float empty_pool[768];
        if (wubu_sd_clip_encode(clip, "", empty_ctx, empty_pool) != 0) return 1;
        /* cfg_every: run the uncond only every N steps and reuse the last
         * uncond eps for the skipped steps (standard CFG-skip trick, ~25%
         * off the UNet budget at N=2). Default 1 = full CFG, unchanged. */
        int cfg_every = 1;
        if (argc > 7) cfg_every = atoi(argv[7]);
        if (cfg_every < 1) cfg_every = 1;
        float *nc = (float *)malloc(4 * 64 * 64 * sizeof(float));
        if (!nc) return 1;
        for (int s = 0; s < steps; s++) {
            float sigma = sched[s], sigma_to = sched[s + 1];
            int t = (int)lrintf(999.0f - 999.0f * (float)s / (float)(steps - 1));
            float c_in = 1.0f / sqrtf(sigma * sigma + 1.0f);
            float *noise = (float *)malloc(4 * 64 * 64 * sizeof(float));
            float *xscaled = (float *)malloc(4 * 64 * 64 * sizeof(float));
            for (int i = 0; i < 4 * 64 * 64; i++) xscaled[i] = x[i] * c_in;
            if (wubu_sd_unet_forward(unet, xscaled, t, ctx, noise) != 0) {
                fprintf(stderr, "unet step %d failed\n", s); return 1;
            }
            if (s % cfg_every == 0) {
                if (wubu_sd_unet_forward(unet, xscaled, t, empty_ctx, nc) != 0) return 1;
            }
            for (int i = 0; i < 4 * 64 * 64; i++) noise[i] = nc[i] + 7.5f * (noise[i] - nc[i]);
            /* denoised = x - sigma*eps; x = (sigma_to/sigma)*x + (1-sigma_to/sigma)*denoised */
            float ratio = (sigma_to > 0.0f) ? sigma_to / sigma : 0.0f;
            #pragma omp parallel for
            for (int i = 0; i < 4 * 64 * 64; i++) {
                float denoised = x[i] - sigma * noise[i];
                x[i] = ratio * x[i] + (1.0f - ratio) * denoised;
            }
            free(noise); free(xscaled);
            double rms = 0.0;
            for (int i = 0; i < 4 * 64 * 64; i++) rms += (double)x[i] * x[i];
            rms = sqrt(rms / (4.0 * 64.0 * 64.0));
            double means[4] = {0};
        for (int c = 0; c < 4; c++) {
            double m = 0.0;
            for (int i = 0; i < 64 * 64; i++) m += x[c * 64 * 64 + i];
            means[c] = m / (64.0 * 64.0);
        }
        fprintf(stderr, "[txt2img] step %d/%d (t=%d sigma=%.4f): rms=%.4f ch-means=[%.3f %.3f %.3f %.3f]\n",
                s + 1, steps, t, sigma, rms, means[0], means[1], means[2], means[3]);
        if (g_timing)
            fprintf(stderr, "[unet-time] conv=%.1fs lin=%.1fs gn=%.1fs silu=%.1fs attn=%.1fs ffn=%.1fs\n",
                    g_t[0], g_t[1], g_t[2], g_t[3], g_t[4], g_t[5]);
        }
        free(nc);
    }

    /* VAE decode -> 512x512x3 (free UNet cache first: phase separation).
     * SD1.5 latent scale: x must be multiplied by 1/0.18215 before decode
     * (the vae_scale_factor — missing this produces garbage/noise). */
    wubu_sd_unet_clear_cache(unet);
    float *img = (float *)malloc(3 * 512 * 512 * sizeof(float));
    if (getenv("SD_DUMP_LATENT")) {
        /* debug: dump the pre-scale latent for reference comparison */
        FILE *df = fopen(getenv("SD_DUMP_LATENT"), "wb");
        if (df) { fwrite(x, sizeof(float), 4 * 64 * 64, df); fclose(df); }
    }
    if (getenv("SD_TAESD")) {
        /* Tiny AutoEncoder fast decode (preview quality, [0,1] output).
         * TAESD's latent scale_factor is 1 (no 1/0.18215 multiply). */
        const char *tap = getenv("SD_TAESD_MODEL");
        if (!tap) tap = "taesd_decoder.safetensors";
        wubu_sd_taesd_t *taesd = wubu_sd_taesd_load(tap);
        if (!taesd) { fprintf(stderr, "taesd load failed\n"); free(img); return 1; }
        if (wubu_sd_taesd_decode(taesd, x, 64, 64, img) != 0) {
            fprintf(stderr, "taesd decode failed\n");
            wubu_sd_taesd_free(taesd); free(img); return 1;
        }
        wubu_sd_taesd_free(taesd);
        fprintf(stderr, "[txt2img] taesd done\n");
        /* PPM from [0,1] */
        FILE *f = fopen(outpath, "wb");
        if (!f) { fprintf(stderr, "cannot write %s\n", outpath); free(img); return 1; }
        fprintf(f, "P6\n512 512\n255\n");
        for (int p = 0; p < 512 * 512; p++) {
            float r = img[p], gr = img[512 * 512 + p], b = img[2 * 512 * 512 + p];
            unsigned char pr = (unsigned char)fminf(255.0f, fmaxf(0.0f, r * 255.0f));
            unsigned char pg = (unsigned char)fminf(255.0f, fmaxf(0.0f, gr * 255.0f));
            unsigned char pb = (unsigned char)fminf(255.0f, fmaxf(0.0f, b * 255.0f));
            fputc(pr, f); fputc(pg, f); fputc(pb, f);
        }
        fclose(f);
        fprintf(stderr, "[txt2img] wrote %s (taesd)\n", outpath);
        wubu_sd_clip_free(clip);
        wubu_sd_unet_free(unet);
        wubu_sd_vae_free(vae);
        free(img);
        return 0;
    }
    const float vae_scale = 1.0f / 0.18215f;
    #pragma omp parallel for
    for (int i = 0; i < 4 * 64 * 64; i++) x[i] *= vae_scale;
    if (wubu_sd_vae_decode(vae, x, 64, 64, img) != 0) {
        fprintf(stderr, "vae decode failed\n"); return 1;
    }
    fprintf(stderr, "[txt2img] vae done\n");

    /* write PPM (P6) */
    FILE *f = fopen(outpath, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", outpath); return 1; }
    fprintf(f, "P6\n512 512\n255\n");
    for (int p = 0; p < 512 * 512; p++) {
        float r = img[p], gr = img[512 * 512 + p], b = img[2 * 512 * 512 + p];
        /* SD VAE output is in [-1,1] -> [0,255] */
        unsigned char pr = (unsigned char)fminf(255.0f, fmaxf(0.0f, (r + 1.0f) * 127.5f));
        unsigned char pg = (unsigned char)fminf(255.0f, fmaxf(0.0f, (gr + 1.0f) * 127.5f));
        unsigned char pb = (unsigned char)fminf(255.0f, fmaxf(0.0f, (b + 1.0f) * 127.5f));
        fputc(pr, f); fputc(pg, f); fputc(pb, f);
    }
    fclose(f);
    fprintf(stderr, "[txt2img] wrote %s\n", outpath);
    wubu_sd_clip_free(clip);
    wubu_sd_unet_free(unet);
    wubu_sd_vae_free(vae);
    return 0;
}
