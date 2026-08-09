/* rng_match.c -- replicate std::default_random_engine (libstdc++ =
 * minstd_rand0, LCG a=16807 m=2147483647) + std::normal_distribution
 * (libstdc++ Marsaglia polar method) to match sd.cpp's initial noise. */
#define _GNU_SOURCE
#include "wubu_sd_clip.h"
#include "wubu_sd_unet.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* minstd_rand0: x_{n+1} = (16807 * x_n) mod 2147483647 */
static uint64_t lcg_state;
static uint64_t lcg_next(void) {
    lcg_state = (uint64_t)((int64_t)lcg_state * 16807 % 2147483647);
    return lcg_state;
}
/* libstdc++ normal_distribution<float>: Marsaglia polar, consumes 2
 * uniforms (from minstd_rand0 / max) per pair, cached second value. */
static float cache_val = 0.0f;
static int has_cache = 0;
static float mt_gauss(void) {
    if (has_cache) { has_cache = 0; return cache_val; }
    float x, y, r2;
    do {
        /* minstd_rand0 returns [1, 2147483646]; libstdc++ maps to (0,1) via /max */
        x = 2.0f * ((float)lcg_next() / 2147483647.0f) - 1.0f;
        y = 2.0f * ((float)lcg_next() / 2147483647.0f) - 1.0f;
        r2 = x*x + y*y;
    } while (r2 >= 1.0f || r2 == 0.0f);
    float f = sqrtf(-2.0f * logf(r2) / r2);
    cache_val = y * f; has_cache = 1;
    return x * f;
}

int main(int argc, char **argv) {
    gguf_ctx *g = gguf_open(argv[1]);
    wubu_sd_clip_t *clip = wubu_sd_clip_load(g);
    wubu_sd_unet_t *unet = wubu_sd_unet_load(g);
    float ctx[77*768], pooled[768], ectx[77*768], epool[768];
    wubu_sd_clip_encode(clip, "a serene anime landscape, soft morning light, detailed", ctx, pooled);
    wubu_sd_clip_encode(clip, "", ectx, epool);
    /* seed 42: sd.cpp seeds with (unsigned int)42 */
    lcg_state = 42;
    has_cache = 0;
    float x[4*64*64];
    for (int i = 0; i < 4*64*64; i++) x[i] = mt_gauss();
    /* save for verification */
    FILE *f = fopen("/tmp/sdcpp_noise.bin", "wb");
    fwrite(x, 4, 4*64*64, f); fclose(f);
    double m = 0, v = 0;
    for (int i = 0; i < 4*64*64; i++) { m += x[i]; v += x[i]*x[i]; }
    m /= 4*64*64; v = v/(4*64*64) - m*m;
    printf("replicated sd.cpp noise: mean=%+.5f rms=%+.5f\n", m, sqrt(v));
    printf("first 8: %+.4f %+.4f %+.4f %+.4f %+.4f %+.4f %+.4f %+.4f\n",
           x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]);
    printf("saved /tmp/sdcpp_noise.bin (feed to sd.cpp? no — sd.cpp gens its own)\n");
    return 0;
}
