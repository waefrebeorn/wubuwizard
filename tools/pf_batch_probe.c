/* pf_batch_probe.c - LP-01's gate number, measured before any engine surgery.
 *
 * The question: for the exact weight type this model uses (IQ2_XXS = 16), how much faster is one batched
 * matmul over N tokens than N single-token gemv calls? The engine currently issues the latter, at nine sites,
 * for every projection in every layer, for every token of a prefill. Its 0.11 s/token prefill is therefore an
 * N-gemv-per-matrix number, not an attention number.
 *
 * Everything measured is real: real slab size, real quant type, real block layout from the model's own dims.
 * The values are synthetic because the cost does not depend on them - which is also why this probe cannot say
 * anything about accuracy. It answers throughput only.
 *
 * Usage: pf_batch_probe [n_tok] [n_in] [n_out] [type]
 *        defaults: 128 2304 2304 16
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

void quantized_matmul(const float *x, const void *W, int weight_type,
                      int64_t n_rows, int64_t n_cols, int64_t col_stride_bytes, float *y);
void quantized_matmul_batched(const float *x, const void *W, int weight_type,
                              int64_t n_rows, int64_t n_cols, int64_t col_stride_bytes,
                              int N, float *y);
/* Block sizes inlined instead of linking gguf_reader.c, which needs POSIX sys/mman.h and does not build on this
 * host outside the project's own build script. Every block is 256 weights. */
static long long row_bytes_of(int type, long long n) {
    long long bpb = 0;
    switch (type) {
        case 16: bpb = 66;  break;   /* IQ2_XXS */
        case 22: bpb = 82;  break;   /* IQ2_S   */
        case 23: bpb = 98;  break;   /* IQ3_S   */
        case 18: bpb = 136; break;   /* IQ4_XS  */
        case 12: bpb = 144; break;   /* Q4_K    */
        case 14: bpb = 210; break;   /* Q6_K    */
        case 8:  bpb = 34;  break;   /* Q8_0    */
        default: return -1;
    }
    if (n % 256) return -1;
    return (n / 256) * bpb;
}

static double now_s(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    const int N     = (argc > 1) ? atoi(argv[1]) : 128;
    const int64_t K = (argc > 2) ? atoll(argv[2]) : 2304;   /* n_in  */
    const int64_t M = (argc > 3) ? atoll(argv[3]) : 2304;   /* n_out */
    const int ty    = (argc > 4) ? atoi(argv[4]) : 16;      /* IQ2_XXS */
    if (N <= 0 || K <= 0 || M <= 0) { fprintf(stderr, "bad dims\n"); return 2; }

    const long long row_bytes = row_bytes_of(ty, K);
    if (row_bytes <= 0) { fprintf(stderr, "row_bytes_of(%d,%lld) failed\n", ty, (long long) K); return 3; }
    const size_t wbytes = (size_t) row_bytes * (size_t) M;

    unsigned char *w = (unsigned char *) malloc(wbytes);
    float *x = (float *) malloc((size_t) N * (size_t) K * sizeof(float));
    float *ya = (float *) malloc((size_t) N * (size_t) M * sizeof(float));
    float *yb = (float *) malloc((size_t) N * (size_t) M * sizeof(float));
    if (!w || !x || !ya || !yb) { fprintf(stderr, "oom\n"); return 4; }

    /* Deterministic, non-degenerate bytes: a fixed LCG so runs are comparable. Values do not affect cost. */
    uint32_t s = 12345u;
    for (size_t i = 0; i < wbytes; ++i) { s = s * 1664525u + 1013904223u; w[i] = (unsigned char) (s >> 24); }
    /* Random scale bytes decode to inf/NaN (fp16 bit patterns), which makes the reference magnitude infinite and the
     * whole comparison degenerate - the exact failure this project already retracted a speedup over. Force every block's
     * fp16 scale to 1.0h (0x00 0x3C) so the outputs are finite and comparable. Block layout: IQ2_XXS 66 B / 256 weights. */
    if (ty == 16) {
        for (size_t b = 0; b + 66 <= wbytes; b += 66) { w[b] = 0x00; w[b + 1] = 0x3C; }
    }
    for (size_t i = 0; i < (size_t) N * (size_t) K; ++i) {
        s = s * 1664525u + 1013904223u;
        x[i] = ((float) (int32_t) (s >> 8) / 8388608.0f) - 1.0f;
    }

    printf("probe: N=%d K=%lld M=%lld type=%d  slab=%.2f MB  row=%lld B\n",
           N, (long long) K, (long long) M, ty, (double) wbytes / 1048576.0, row_bytes);

    /* Warm both paths once so the first-touch page faults are not charged to whichever runs first. */
    for (int t = 0; t < 1; ++t)
        quantized_matmul(x, w, ty, K, M, 0, ya);
    quantized_matmul_batched(x, w, ty, K, M, 0, (N < 8 ? N : 8), yb);

    double t0 = now_s();
    for (int t = 0; t < N; ++t)
        quantized_matmul(x + (size_t) t * K, w, ty, K, M, 0, ya + (size_t) t * M);
    double t_gemv = now_s() - t0;

    t0 = now_s();
    quantized_matmul_batched(x, w, ty, K, M, 0, N, yb);
    double t_gemm = now_s() - t0;

    double maxd = 0.0, maxref = 0.0;
    for (size_t i = 0; i < (size_t) N * M; ++i) {
        const double d = (double) ya[i] - (double) yb[i];
        if (d < 0) { if (-d > maxd) maxd = -d; } else if (d > maxd) maxd = d;
        const double r = (double) yb[i] < 0 ? -(double) yb[i] : (double) yb[i];
        if (r > maxref) maxref = r;
    }

    const double mac = (double) N * (double) K * (double) M;
    printf("  %6d single-token gemv   %8.2f ms   %7.2f GMAC/s\n", N, t_gemv * 1e3, mac / t_gemv / 1e9);
    printf("  1 batched matmul        %8.2f ms   %7.2f GMAC/s\n", t_gemm * 1e3, mac / t_gemm / 1e9);
    printf("  SPEEDUP %.2fx   max|diff|=%.3e  max|ref|=%.3e  rel=%.3e\n",
           t_gemv / t_gemm, maxd, maxref, maxref > 0 ? maxd / maxref : 0.0);
    if (maxref == 0.0)
        printf("  WARNING: reference magnitude is zero - this comparison proves nothing (see the retraction rule).\n");

    free(w); free(x); free(ya); free(yb);
    return 0;
}

/* The GPU dims hook is not used by this probe; stub it so the real quantised kernels can be linked. */
void wubu_dims_sync_gpu(void) { }
