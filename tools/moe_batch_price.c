/* Price the batched expert launch against 8 per-expert launches, and check the two agree.
 * No engine, no OpenMP, no cache - just the two device paths on identical bytes.
 * argv[1]/argv[2] = iq2xxs grid and ksigns dumps (256 x u64, 128 x u8) taken from gguf_reader.c.
 * Without those tables the kernel legitimately returns zeros and every number here would be meaningless. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <windows.h>

static double now(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double) c.QuadPart / (double) f.QuadPart;
}

typedef int (*upload_sig)(const void *, const void *);
typedef int (*res_sig)(int, int64_t, const void *, size_t, const void *, int, float *, int, int, int64_t);
typedef int (*batch_sig)(int, const void *const *, int, size_t, const void *, int, float *, int, int);

int main(int argc, char **argv)
{
    const char *gp = argc > 1 ? argv[1] : "iq2xxs_grid.bin";
    const char *kp = argc > 2 ? argv[2] : "iq2xxs_ksigns.bin";

    HMODULE h = LoadLibraryA("wubu_cuda.dll");
    if (!h) { printf("no dll\n"); return 1; }
    upload_sig up = (upload_sig) (void *) GetProcAddress(h, "wubu_cuda_upload_iq2xxs");
    res_sig    rs = (res_sig)    (void *) GetProcAddress(h, "wubu_cuda_resident_matvec_id");
    batch_sig  bt = (batch_sig)  (void *) GetProcAddress(h, "wubu_cuda_moe_batch");
    if (!rs || !bt) { printf("missing exports rs=%p bt=%p\n", (void *) rs, (void *) bt); return 1; }

    static uint64_t grid[256];
    static unsigned char ks[128];
    FILE *f = fopen(gp, "rb");
    if (!f || fread(grid, sizeof(uint64_t), 256, f) != 256) { printf("cannot read %s\n", gp); return 1; }
    fclose(f);
    f = fopen(kp, "rb");
    if (!f || fread(ks, 1, 128, f) != 128) { printf("cannot read %s\n", kp); return 1; }
    fclose(f);
    if (up) printf("upload tables rc=%d\n", up(grid, ks));

    const int K = 2304, NEXP = 8, NROWS = 2304;      /* gate_up: 2304 rows x 2304 elements per expert */
    const size_t RB = (size_t) (K / 256) * 66;       /* IQ2_XXS: 66 bytes per 256 elements */
    const size_t WB = RB * (size_t) NROWS;
    const int QKB = (K / 256) * (4 + 256 + 32);      /* q8_K blocks */
    printf("per-expert slab %.2f MB, 8 experts %.2f MB, act %d B\n",
           WB / 1048576.0, 8.0 * WB / 1048576.0, QKB);

    unsigned char **slab = (unsigned char **) malloc(NEXP * sizeof(void *));
    float *act = (float *) malloc((size_t) K * sizeof(float));
    unsigned char *aq = (unsigned char *) calloc(1, (size_t) QKB);
    float *y1 = (float *) malloc((size_t) NROWS * sizeof(float));
    float *ref = (float *) malloc((size_t) NEXP * NROWS * sizeof(float));
    float *y8 = (float *) malloc((size_t) NEXP * NROWS * sizeof(float));
    for (int e = 0; e < NEXP; ++e) {
        slab[e] = (unsigned char *) malloc(WB);
        for (size_t i = 0; i < WB; ++i) slab[e][i] = (unsigned char) ((i * 2654435761u + (unsigned) e * 7919u) >> 13);
        for (size_t b = 0; b + 66 <= WB; b += 66) { slab[e][b] = 0x00; slab[e][b + 1] = 0x3C; }
    }
    for (int i = 0; i < K; ++i) act[i] = (float) ((i % 29) - 14) / 14.0f;
    for (int b = 0; b < K / 256; ++b) {
        unsigned char *blk = aq + (size_t) b * 292;
        ((unsigned short *) blk)[0] = 0x3C00;
        /* The block scale was being left unset, so every product came out ~4e-37 and the "reference"
         * magnitude printed as if the two paths agreed on real numbers. Set it so the comparison is
         * about values as well as timing. */
        { const float dsc = 0.25f; memcpy(blk, &dsc, 4); }
        memset(blk + 4, 3, 256);
        memset(blk + 4 + 256, 0, 32);
    }

    for (int e = 0; e < NEXP; ++e) rs(16, 1000 + e, slab[e], WB, aq, QKB, y1, K, NROWS, (int64_t) RB);

    int rc8 = 0;
    for (int e = 0; e < NEXP; ++e) rc8 |= rs(16, 2000 + e, slab[e], WB, aq, QKB, y1, K, NROWS, (int64_t) RB);
    const int rcB = bt(16, (const void *const *) slab, NEXP, WB, aq, QKB, y8, K, NROWS);
    printf("rc single=%d batch=%d\n", rc8, rcB);

    for (int e = 0; e < NEXP; ++e) {
        rs(16, 3000 + e, slab[e], WB, aq, QKB, y1, K, NROWS, (int64_t) RB);
        memcpy(ref + (size_t) e * NROWS, y1, (size_t) NROWS * sizeof(float));
    }
    double mx = 0.0, mag = 0.0;
    for (int i = 0; i < NEXP * NROWS; ++i) {
        double d = fabs((double) y8[i] - (double) ref[i]);
        if (d > mx) mx = d;
        if (fabs((double) ref[i]) > mag) mag = fabs((double) ref[i]);
    }
    printf("batch vs per-expert: ref0=%.6g max_abs=%.3e max|ref|=%.3e rel=%.3e\n",
           (double) ref[0], mx, mag, mag > 0 ? mx / mag : 0.0);

    const int REP = 20;
    double t0 = now();
    for (int r = 0; r < REP; ++r)
        for (int e = 0; e < NEXP; ++e) rs(16, 5000 + e + r * 16, slab[e], WB, aq, QKB, y1, K, NROWS, (int64_t) RB);
    double t1 = now();
    for (int r = 0; r < REP; ++r) bt(16, (const void *const *) slab, NEXP, WB, aq, QKB, y8, K, NROWS);
    double t2 = now();
    const double per8 = (t1 - t0) / REP, perB = (t2 - t1) / REP;
    printf("8 separate launches %.2f ms | 1 batched %.2f ms | %.2fx per layer\n",
           per8 * 1e3, perB * 1e3, perB > 0 ? per8 / perB : 0.0);
    printf("extrapolated 24 layers/token: %.1f ms vs %.1f ms\n", per8 * 24e3, perB * 24e3);
    return 0;
}
