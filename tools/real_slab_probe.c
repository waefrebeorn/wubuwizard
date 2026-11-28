/* The decisive probe: same kernel, same geometry, but the bytes come from the REAL GGUF through a FILE-BACKED mapping
 * instead of anonymous memory. Five synthetic probes priced residency, faults, streams and locks and none reproduced the
 * engine's 1.25 us/row. This one changes exactly one variable - the page source - and nothing else.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <windows.h>
#include "wubu_iq2q8.h"
#include "wubu_iq_c11.h"
#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include <assert.h>
#include "ggml-common.h"

void wubu_iq_rows(int ggml_type, const void *w, size_t row_bytes, int n,
                  const void *const *act, int nt, float *const *out, int r0, int r1);
double in2_hires_now(void){ static LARGE_INTEGER f; LARGE_INTEGER c; if(!f.QuadPart) QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c); return (double)c.QuadPart/(double)f.QuadPart; }
double in2_ty_sec[256]; long in2_ty_calls[256]; int in2_ty_par = 0; int in2_ty_depth = 0;

#define ROW_BYTES 594          /* 2304 elements of IQ2_XXS = 9 blocks x 66 B */

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "C:/Users/eman5/models/inkling-2-mini/Inkling-2-Mini-UD-IQ2_XXS.gguf";
    const long long off = argc > 2 ? atoll(argv[2]) : 2003254464LL;
    const int rows = argc > 3 ? atoi(argv[3]) : 200000;

    HANDLE f = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);
    if (f == INVALID_HANDLE_VALUE) { printf("open failed %lu\n", GetLastError()); return 2; }
    HANDLE m = CreateFileMappingA(f, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!m) { printf("map failed %lu\n", GetLastError()); return 2; }
    const unsigned char *base = (const unsigned char *) MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    if (!base) { printf("view failed %lu\n", GetLastError()); return 2; }
    const unsigned char *w = base + off;

    float *x = (float *) malloc(sizeof(float) * 2304);
    in2_q8k_blk *aq = (in2_q8k_blk *) malloc(sizeof(in2_q8k_blk) * 9);
    float *o = (float *) malloc(sizeof(float) * 64);
    for (int i = 0; i < 2304; ++i) x[i] = ((i * 37) % 199 - 99) / 99.0f;
    in2_q8k_quantise(x, 9, aq);

    const void *ap[1] = { aq };
    /* pass 1 = first touch of these file pages; pass 2 = re-read of whatever survived */
    for (int pass = 0; pass < 2; ++pass) {
        double t0 = in2_hires_now();
        size_t done = 0;
        for (int r = 0; r < rows; ++r) {
            float *op[1] = { o };
            wubu_iq_rows(16, w + (size_t) r * ROW_BYTES, ROW_BYTES, 2304, ap, 1, op, 0, 1);
            done += 2304;
        }
        double dt = in2_hires_now() - t0;
        printf("REAL SLAB pass %d: rows=%d  %.2f MB  %.3f ns/elem  %.2f GB/s  %.3f us/row\n",
               pass + 1, rows, (double) rows * ROW_BYTES / 1048576.0,
               dt * 1e9 / (double) done, (double) done * 4.0 / dt / 1e9,
               dt * 1e6 / (double) rows);
    }
    (void) ap;
    return 0;
}
