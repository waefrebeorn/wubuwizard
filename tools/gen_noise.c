/* gen_noise.c — generate 4x64x64 gaussian noise with the SAME xorshift64
 * + Box-Muller as tools/txt2img_sd.c, seed 42. */
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <stdlib.h>

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
    const char *out = argc > 1 ? argv[1] : "/tmp/noise42.bin";
    uint64_t seed = argc > 2 ? strtoull(argv[2], NULL, 10) : 42;
    FILE *f = fopen(out, "wb");
    if (!f) return 1;
    rng_state = seed;
    for (int i = 0; i < 4 * 64 * 64; i++) {
        float v = (float)rng_gauss();
        fwrite(&v, 4, 1, f);
    }
    fclose(f);
    printf("wrote %s seed=%llu\n", out, (unsigned long long)seed);
    return 0;
}
