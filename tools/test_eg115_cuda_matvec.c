/* test_eg115_cuda_matvec.c -- EG-115 gate: device quantized matvec vs the CPU kernels.
 *
 * Loads wubu_cuda.dll, uploads the IQ grids, and compares a device matvec against
 * the exact CPU ports (which EG-111/113/114 already prove against the dequantiser
 * reference) for IQ2_S, IQ3_S, Q3_K and Q4_0. It also times a 15360 x 3840 IQ2_S
 * matvec (the FFN shape of the 12B model) on CPU and GPU.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include <time.h>

#include "gguf_reader.h"
#include "wubu_vecdot_iq.h"
#include "wubu_vecdot_q3_k.h"
#include "wubu_vecdot_q4_0.h"

extern const void *wubu_vecdot_iq2s_grid(void);
extern const void *wubu_vecdot_iq3s_grid(void);

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define QK_K 256

typedef int (*fn_tab_t)(const void *, const void *);
typedef int (*fn_mv_t)(const void *, int, const void *, float *, int, int, long long);
typedef int (*fn_avail_t)(void);

static uint64_t rng_state = 0xA11CE5EEDULL;
static uint32_t rng_u32(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state >> 32);
}

static int close_enough(float a, float b)
{
    const float d = fabsf(a - b);
    const float s = fabsf(a) > fabsf(b) ? fabsf(a) : fabsf(b);
    return d <= 1e-2f + 1e-3f * s;
}

static double now_s(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}

/* CPU reference for one column, dispatching to the exact scalar ports. */
static int cpu_col_dot(int type, const void *w, const block_q8_K *q8, int K, float *out)
{
    switch (type) {
    case 22: return wubu_vec_dot_iq2_s_q8_k_scalar(K, out, w, q8);
    case 21: return wubu_vec_dot_iq3_s_q8_k_scalar(K, out, w, q8);
    case 11: return wubu_vec_dot_q3_k_q8_k_scalar(K, out, w, q8);
    case 2:  return wubu_vec_dot_q4_0_q8_k_scalar(K, out, w, q8);
    default: return -1;
    }
}

static int block_bytes(int type)
{
    switch (type) {
    case 22: return WUBU_IQ2S_BLOCK_SIZE;
    case 21: return WUBU_IQ3S_BLOCK_SIZE;
    case 11: return WUBU_Q3_K_BLOCK_SIZE;
    case 2:  return 8 * WUBU_Q4_0_BLOCK_SIZE;   /* 8 sub-blocks per 256 elems */
    default: return 0;
    }
}

static void fill_block(int type, uint8_t *b)
{
    const int n = block_bytes(type);
    for (int i = 0; i < n; ++i) b[i] = (uint8_t)rng_u32();
    if (type == 11) {                       /* Q3_K: fp16 scale at the END */
        const uint16_t d16 = (uint16_t)(0x2C00u + (rng_u32() & 0x3FFu));
        memcpy(b + n - 2, &d16, 2);
    } else if (type == 22 || type == 21) {  /* IQ2_S/IQ3_S: fp16 scale first */
        const uint16_t d16 = (uint16_t)(0x2C00u + (rng_u32() & 0x3FFu));
        memcpy(b, &d16, 2);
    } else {                                /* Q4_0: fp16 scale per 32-elem block */
        for (int s = 0; s < 8; ++s) {
            const uint16_t d16 = (uint16_t)(0x2C00u + (rng_u32() & 0x3FFu));
            memcpy(b + s * WUBU_Q4_0_BLOCK_SIZE, &d16, 2);
        }
    }
}

int main(int argc, char **argv)
{
    const char *dll = argc > 1 ? argv[1] : "wubu_cuda.dll";
    HMODULE h = LoadLibraryA(dll);
    if (!h) { fprintf(stderr, "test_eg115: cannot load %s (%lu)\n", dll, GetLastError()); return 2; }

    fn_tab_t tab = (fn_tab_t)(void *)GetProcAddress(h, "wubu_cuda_upload_tables");
    fn_mv_t mv = (fn_mv_t)(void *)GetProcAddress(h, "wubu_cuda_quant_matvec_host");
    fn_avail_t avail = (fn_avail_t)(void *)GetProcAddress(h, "wubu_cuda_available");
    CHECK(tab && mv && avail);
    if (!tab || !mv || !avail) { FreeLibrary(h); return 1; }
    if (!avail()) { printf("test_eg115: SKIP (no CUDA device)\n"); FreeLibrary(h); return 0; }
    CHECK(tab(wubu_vecdot_iq2s_grid(), wubu_vecdot_iq3s_grid()) == 0);

    const int types[4] = { 22, 21, 11, 2 };
    const char *names[4] = { "IQ2_S", "IQ3_S", "Q3_K", "Q4_0" };
    enum { K = 512, NCOL = 64 };   /* 2 blocks per column */

    for (int ti = 0; ti < 4; ++ti) {
        const int type = types[ti];
        const int cbytes = block_bytes(type) * (K / QK_K);
        static uint8_t w[64 * 2 * 8 * 110];
        static float x[K];
        static block_q8_K q8;
        static float gpu[NCOL], cpu[NCOL];

        for (int c = 0; c < NCOL; ++c)
            for (int b = 0; b < (K / QK_K); ++b)
                fill_block(type, w + c * cbytes + b * block_bytes(type));
        for (int j = 0; j < K; ++j)
            x[j] = ((float)(int32_t)(rng_u32() & 0xFFFF) - 32768.0f) / 4096.0f;
        quantize_row_q8_K(x, &q8, K);

        for (int c = 0; c < NCOL; ++c) {
            CHECK(cpu_col_dot(type, w + c * cbytes, &q8, K, &cpu[c]) == 0);
        }
        const int rc = mv(w, type, &q8, gpu, K, NCOL, cbytes);
        CHECK(rc == 0);
        int bad = 0;
        for (int c = 0; c < NCOL; ++c) if (!close_enough(gpu[c], cpu[c])) ++bad;
        CHECK(bad == 0);
        printf("EG-115 %-5s columns=%d mismatches=%d\n", names[ti], NCOL, bad);
    }

    /* Bench: FFN-shaped IQ2_S matvec (15360 columns x 3840). */
    {
        enum { BK = 3840, BN = 15360 };
        const int cbytes = WUBU_IQ2S_BLOCK_SIZE * (BK / QK_K);
        uint8_t *w = (uint8_t *)malloc((size_t)BN * cbytes);
        float *x = (float *)malloc(sizeof(float) * BK);
        float *out = (float *)malloc(sizeof(float) * BN);
        static block_q8_K q8b[16];
        CHECK(w && x && out);
        if (w && x && out) {
            for (size_t i = 0; i < (size_t)BN * cbytes; ++i) w[i] = (uint8_t)rng_u32();
            for (int j = 0; j < BK; ++j) x[j] = ((float)(int32_t)(rng_u32() & 0xFFFF) - 32768.0f) / 4096.0f;
            quantize_row_q8_K(x, q8b, BK);

            const double t0 = now_s();
            for (int c = 0; c < BN; ++c) {
                float v;
                wubu_vec_dot_iq2_s_q8_k_scalar(BK, &v, w + (size_t)c * cbytes, q8b);
                out[c] = v;
            }
            const double t_cpu = now_s() - t0;

            const double t1 = now_s();
            CHECK(mv(w, 22, q8b, out, BK, BN, cbytes) == 0);
            const double t_gpu = now_s() - t1;

            printf("EG-115 bench iq2_s %dx%d: cpu %.1f ms  gpu %.1f ms  speedup %.2fx\n",
                   BN, BK, t_cpu * 1000.0, t_gpu * 1000.0, t_cpu / (t_gpu > 1e-9 ? t_gpu : 1e-9));
        }
        free(w); free(x); free(out);
    }

    FreeLibrary(h);
    if (failures) { fprintf(stderr, "test_eg115_cuda_matvec: FAIL (%d)\n", failures); return 1; }
    printf("test_eg115_cuda_matvec: PASS\n");
    return 0;
}
