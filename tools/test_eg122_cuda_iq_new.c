/* test_eg122_cuda_iq_new.c -- EG-122: device IQ3_XXS / IQ4_XS kernels vs CPU ports.
 *
 * The engine's greedy tokens changed when the IQ4_XS device kernel was first wired,
 * so a numeric comparison per column is the only honest way to accept it. Random
 * blocks are legal for both formats (every bit pattern is a valid block), and the CPU
 * ports are the ones gated against the dequantiser by EG-120/EG-121.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>

#include "gguf_reader.h"
#include "wubu_vecdot_iq3_xxs.h"
#include "wubu_vecdot_iq4_xs.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define QK_K 256

typedef int (*fn_tab2_t)(const void *, const void *);
typedef int (*fn_tab1_t)(const void *);
typedef int (*fn_mv_t)(const void *, int, const void *, float *, int, int, long long);
typedef int (*fn_avail_t)(void);

static uint64_t rng_state = 0xC0FFEE1234ULL;
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

int main(int argc, char **argv)
{
    const char *dll = argc > 1 ? argv[1] : "wubu_cuda.dll";
    HMODULE h = LoadLibraryA(dll);
    if (!h) { printf("test_eg122: SKIP (no %s)\n", dll); return 0; }

    fn_avail_t avail = (fn_avail_t)(void *)GetProcAddress(h, "wubu_cuda_available");
    fn_tab2_t tab3 = (fn_tab2_t)(void *)GetProcAddress(h, "wubu_cuda_upload_tables_iq3xxs");
    fn_tab1_t tab4 = (fn_tab1_t)(void *)GetProcAddress(h, "wubu_cuda_upload_table_iq4nl");
    /* Host entry point: uploads, dispatches by type (18 / 23 included), downloads. */
    fn_mv_t mv18 = (fn_mv_t)(void *)GetProcAddress(h, "wubu_cuda_quant_matvec_host");
    fn_mv_t mv23 = mv18;
    CHECK(avail && tab3 && tab4 && mv18 && mv23);
    if (!avail || !tab3 || !tab4 || !mv18 || !mv23) { FreeLibrary(h); return 1; }
    if (!avail()) { printf("test_eg122: SKIP (no CUDA device)\n"); FreeLibrary(h); return 0; }

    CHECK(tab3(wubu_vecdot_iq3_xxs_grid(), wubu_vecdot_iq3_xxs_signs()) == 0);
    CHECK(tab4(wubu_vecdot_iq4_xs_table()) == 0);
    /* IQ2_S/IQ3_S grids too: without them c_iq2s_grid is all zeros and every dot
     * silently returns 0, which looks like a kernel that "ran fine". */
    {
        extern const void *wubu_vecdot_iq2s_grid(void);
        extern const void *wubu_vecdot_iq3s_grid(void);
        fn_tab2_t tab = (fn_tab2_t)(void *)GetProcAddress(h, "wubu_cuda_upload_tables");
        CHECK(tab != NULL);
        if (tab) CHECK(tab(wubu_vecdot_iq2s_grid(), wubu_vecdot_iq3s_grid()) == 0);
    }

    enum { K = 1024, NCOL = 96 };
    static uint8_t w18[NCOL * 4 * 98];
    static uint8_t w23[NCOL * 4 * 136];
    static float x[K];
    static block_q8_K q8;
    static float gpu[NCOL], cpu[NCOL];

    for (int j = 0; j < K; ++j)
        x[j] = ((float)(int32_t)(rng_u32() & 0xFFFF) - 32768.0f) / 3000.0f;
    quantize_row_q8_K(x, &q8, K);

    /* IQ3_XXS: fp16 d first in every 98-byte block (0x2Cxx keeps it sane). */
    for (int c = 0; c < NCOL; ++c)
        for (int b = 0; b < K / QK_K; ++b) {
            uint8_t *blk = w18 + c * 4 * 98 + b * 98;
            for (int i = 0; i < 98; ++i) blk[i] = (uint8_t)rng_u32();
            const uint16_t d16 = (uint16_t)(0x2C00u + (rng_u32() & 0x3FFu));
            memcpy(blk, &d16, 2);
        }
    for (int c = 0; c < NCOL; ++c)
        cpu[c] = wubu_vec_dot_iq3_xxs_q8_K(w18 + c * 4 * 98, &q8, K);
    CHECK(mv18(w18, 18, &q8, gpu, K, NCOL, 4 * 98) == 0);
    int bad18 = 0;
    for (int c = 0; c < NCOL; ++c) if (!close_enough(gpu[c], cpu[c])) {
        if (bad18 < 3) printf("EG-122 iq3_xxs col %d gpu=%g cpu=%g\n", c, gpu[c], cpu[c]);
        ++bad18;
    }
    printf("EG-122 iq3_xxs columns=%d mismatches=%d\n", NCOL, bad18);
    CHECK(bad18 == 0);

    /* IQ4_XS: fp16 d first, then scales_h, scales_l[4], qs[128]. */
    for (int c = 0; c < NCOL; ++c)
        for (int b = 0; b < K / QK_K; ++b) {
            uint8_t *blk = w23 + c * 4 * 136 + b * 136;
            for (int i = 0; i < 136; ++i) blk[i] = (uint8_t)rng_u32();
            const uint16_t d16 = (uint16_t)(0x2C00u + (rng_u32() & 0x3FFu));
            memcpy(blk, &d16, 2);
        }
    for (int c = 0; c < NCOL; ++c)
        cpu[c] = wubu_vec_dot_iq4_xs_q8_K(w23 + c * 4 * 136, &q8, K);
    CHECK(mv23(w23, 23, &q8, gpu, K, NCOL, 4 * 136) == 0);
    int bad23 = 0;
    for (int c = 0; c < NCOL; ++c) if (!close_enough(gpu[c], cpu[c])) {
        if (bad23 < 3) printf("EG-122 iq4_xs col %d gpu=%g cpu=%g\n", c, gpu[c], cpu[c]);
        ++bad23;
    }
    printf("EG-122 iq4_xs columns=%d mismatches=%d\n", NCOL, bad23);
    CHECK(bad23 == 0);

    /* IQ2_S narrow-width path (the q/k/v shapes): the warp-per-column kernel must
     * match the CPU port at 2048 columns too, not only at FFN widths. */
    {
        extern int wubu_vec_dot_iq2_s_q8_k_scalar(int n, float *s, const void *vx, const void *vy);
        enum { KW = 2048, NCW = 96 };
        static uint8_t w22[NCW * (KW / QK_K) * 82];
        static float xw[KW];
        static block_q8_K q8w;
        static float gw[NCW], cw[NCW];
        unsigned s3 = 777u;
        for (int c = 0; c < NCW; ++c)
            for (int b = 0; b < KW / QK_K; ++b) {
                uint8_t *blk = w22 + ((size_t)c * (KW / QK_K) + b) * 82;
                for (int i = 0; i < 82; ++i) blk[i] = (uint8_t)(s3 = s3 * 1103515245u + 12345u) >> 16;
                const uint16_t d16 = (uint16_t)(0x2C00u + (s3 & 0x3FFu));
                memcpy(blk, &d16, 2);
            }
        for (int j = 0; j < KW; ++j) xw[j] = (float)((int)((s3 = s3 * 1103515245u + 12345u) >> 16 & 0x3FF) - 512) / 250.0f;
        quantize_row_q8_K(xw, &q8w, KW);
        for (int c = 0; c < NCW; ++c)
            CHECK(wubu_vec_dot_iq2_s_q8_k_scalar(KW, &cw[c], w22 + (size_t)c * (KW / QK_K) * 82, &q8w) == 0);
        CHECK(mv18(w22, 22, &q8w, gw, KW, NCW, (KW / QK_K) * 82) == 0);
        int bad = 0;
        for (int c = 0; c < NCW; ++c) if (!close_enough(gw[c], cw[c])) {
            if (bad < 3) printf("EG-122 iq2_s narrow col %d gpu=%g cpu=%g\n", c, gw[c], cw[c]);
            ++bad;
        }
        printf("EG-122 iq2_s narrow columns=%d mismatches=%d\n", NCW, bad);
        CHECK(bad == 0);
    }

    FreeLibrary(h);
    if (failures) { fprintf(stderr, "test_eg122_cuda_iq_new: FAIL (%d)\n", failures); return 1; }
    printf("test_eg122_cuda_iq_new: PASS\n");
    return 0;
}
