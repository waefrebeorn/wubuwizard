/*
 * wubu_gpu.c -- host side of the CUDA DLL bridge.
 *
 * The DLL is built by nvcc/MSVC (src/wubu_cuda_bridge.cu, src/wubu_cuda_quant.cu)
 * and loaded at runtime, because MinGW's ld cannot link MSVC-ABI objects. Every
 * entry point here is a function pointer resolved once with GetProcAddress.
 *
 * Threading: the engine's decode path is single-threaded around these calls, so
 * no locking is used. Device pointers returned by wubu_gpu_upload() are owned by
 * the caller and freed with wubu_gpu_free().
 */
#include "wubu_gpu.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#endif

typedef int (*fn_avail)(void);
typedef int (*fn_name)(char *, int);
typedef int (*fn_meminfo)(unsigned long long *, unsigned long long *);
typedef int (*fn_fit)(unsigned long long, unsigned long long, unsigned long long, unsigned long long);
typedef int (*fn_tables)(const void *, const void *);
typedef void *(*fn_alloc)(unsigned long long);
typedef int (*fn_upload)(void *, const void *, unsigned long long);
typedef int (*fn_free)(void *);
typedef int (*fn_download)(void *, const void *, unsigned long long);
typedef int (*fn_q8k)(const float *, void *, int);
typedef int (*fn_mv_dev)(const void *, int, const void *, float *, int, int, long long);
typedef int (*fn_mv2)(const void *, const void *, float *, int, int, long long);

static struct {
    int ready;
    int tried;
#ifdef _WIN32
    HMODULE h;
#endif
    fn_meminfo meminfo;
    fn_fit fit;
    fn_alloc alloc;
    fn_upload upload;
    fn_free free_dev;
    fn_download download;
    fn_q8k q8k;
    fn_mv_dev mv;
    fn_mv2 mv2;   /* warp-per-column IQ2_S kernel (v3) */
    fn_mv2 mv3;   /* warp-per-column IQ3_XXS kernel */
    fn_mv2 mv4;   /* warp-per-column IQ4_XS kernel */
    fn_mv2 mv5;   /* warp-per-column IQ2_XS kernel */
} G;

int wubu_gpu_init(void)
{
    if (G.tried) return G.ready;
    G.tried = 1;
#ifdef _WIN32
    const char *path = getenv("WUBU_CUDA_DLL");
    if (!path) path = "wubu_cuda.dll";
    G.h = LoadLibraryA(path);
    if (!G.h) return 0;
    fn_avail avail = (fn_avail)(void *)GetProcAddress(G.h, "wubu_cuda_available");
    fn_name name = (fn_name)(void *)GetProcAddress(G.h, "wubu_cuda_device_name");
    G.meminfo = (fn_meminfo)(void *)GetProcAddress(G.h, "wubu_cuda_meminfo");
    G.fit = (fn_fit)(void *)GetProcAddress(G.h, "wubu_cuda_fit");
    fn_tables tables = (fn_tables)(void *)GetProcAddress(G.h, "wubu_cuda_upload_tables");
    G.alloc = (fn_alloc)(void *)GetProcAddress(G.h, "wubu_cuda_alloc");
    G.upload = (fn_upload)(void *)GetProcAddress(G.h, "wubu_cuda_upload");
    G.free_dev = (fn_free)(void *)GetProcAddress(G.h, "wubu_cuda_free_dev");
    G.download = (fn_download)(void *)GetProcAddress(G.h, "wubu_cuda_download");
    G.q8k = (fn_q8k)(void *)GetProcAddress(G.h, "wubu_cuda_quantize_q8k_dev");
    G.mv = (fn_mv_dev)(void *)GetProcAddress(G.h, "wubu_cuda_matvec_dev");
    G.mv2 = (fn_mv2)(void *)GetProcAddress(G.h, "wubu_cuda_matvec_iq2s_v3");
    G.mv3 = (fn_mv2)(void *)GetProcAddress(G.h, "wubu_cuda_matvec_iq3xxs");
    G.mv4 = (fn_mv2)(void *)GetProcAddress(G.h, "wubu_cuda_matvec_iq4xs");
    G.mv5 = (fn_mv2)(void *)GetProcAddress(G.h, "wubu_cuda_matvec_iq2xs");
    if (!avail || !G.alloc || !G.upload || !G.free_dev || !G.download || !G.q8k || !G.mv) return 0;
    if (!avail()) return 0;
    if (tables) {
        extern const void *wubu_vecdot_iq2s_grid(void);
        extern const void *wubu_vecdot_iq3s_grid(void);
        if (tables(wubu_vecdot_iq2s_grid(), wubu_vecdot_iq3s_grid()) != 0) return 0;
    }
    /* IQ3_XXS grid + sign tables, needed by the device kernel for type 18. */
    {
        fn_mv2 up3 = (fn_mv2)(void *)GetProcAddress(G.h, "wubu_cuda_upload_tables_iq3xxs");
        extern const void *wubu_vecdot_iq3_xxs_grid(void);
        extern const void *wubu_vecdot_iq3_xxs_signs(void);
        if (up3 && up3(wubu_vecdot_iq3_xxs_grid(), wubu_vecdot_iq3_xxs_signs(), 0, 0, 0, 0) == 0) {
            /* uploaded; the kernel is usable */
        } else {
            G.mv3 = NULL;
        }
    }
    /* IQ4_XS code table for the type-23 device kernel. */
    {
        fn_mv2 up4 = (fn_mv2)(void *)GetProcAddress(G.h, "wubu_cuda_upload_table_iq4nl");
        extern const void *wubu_vecdot_iq4_xs_table(void);
        if (!(up4 && up4(wubu_vecdot_iq4_xs_table(), 0, 0, 0, 0, 0) == 0)) G.mv4 = NULL;
    }
    /* IQ2_XS 512-entry grid for the type-17 device kernel. */
    {
        fn_mv2 up5 = (fn_mv2)(void *)GetProcAddress(G.h, "wubu_cuda_upload_tables_iq2xs");
        extern const void *wubu_vecdot_iq2_xs_grid(void);
        if (!(up5 && up5(wubu_vecdot_iq2_xs_grid(), 0, 0, 0, 0, 0) == 0)) G.mv5 = NULL;
    }
    char dev[192] = {0};
    if (name) name(dev, (int)sizeof(dev));
    fprintf(stderr, "[GPU] %s\n", dev[0] ? dev : "CUDA device ready");
    G.ready = 1;
#endif
    return G.ready;
}

int wubu_gpu_available(void)
{
    return wubu_gpu_init();
}

void wubu_gpu_stats(unsigned long long *total, unsigned long long *freeb)
{
    if (total) *total = 0;
    if (freeb) *freeb = 0;
    if (!wubu_gpu_init() || !G.meminfo) return;
    G.meminfo(total, freeb);
}

int wubu_gpu_fits(unsigned long long bytes, unsigned long long reserve)
{
    if (!wubu_gpu_init() || !G.fit) return 0;
    return G.fit(bytes, 0, 0, reserve);
}

void *wubu_gpu_upload(const void *host, size_t bytes)
{
    if (!wubu_gpu_init() || !host || bytes == 0) return NULL;
    void *d = G.alloc((unsigned long long)bytes);
    if (!d) return NULL;
    if (G.upload(d, host, (unsigned long long)bytes) != 0) {
        G.free_dev(d);
        return NULL;
    }
    return d;
}

void wubu_gpu_free(void *d)
{
    if (d && G.free_dev) G.free_dev(d);
}

/* y = W[n][K] . x[K] entirely on the device. W and the scratch are cached by the
 * caller; only x (K floats) and y (n floats) cross PCIe per call. */
/* Per-type accounting, printed at exit when WUBU_GPU_STATS=1. */
static struct { double ms; long calls; long bytes; } S[32];
static int S_hooked;

static void wubu_gpu_stats_dump(void)
{
    if (!getenv("WUBU_GPU_STATS")) return;
    fprintf(stderr, "[GPU stats] type  calls   ms_total  MiB(touched)\n");
    for (int t = 0; t < 32; ++t) {
        if (!S[t].calls) continue;
        fprintf(stderr, "[GPU stats] %4d %7ld %9.1f %10.1f\n",
                t, S[t].calls, S[t].ms, (double)S[t].bytes / 1048576.0);
    }
}

static double gpu_now_ms(void)
{
#ifdef _WIN32
    LARGE_INTEGER f, c;
    static LARGE_INTEGER freq;
    static int init;
    if (!init) { QueryPerformanceFrequency(&freq); init = 1; }
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)freq.QuadPart;
#else
    return 0.0;
#endif
}

/* Fused FFN for decode: gate, up, gelu*up and down without leaving the device. One
 * upload of `hidden` floats and one download of `hidden` floats per layer instead of
 * three round trips, and the tanh GELU moves off the CPU. Returns 0 on success so the
 * caller can fall back to the per-tensor CPU path. */
int wubu_gpu_ffn(const float *x, const void *d_wg, const void *d_wu, const void *d_wd,
                 int ggml_type, int hidden, int ff,
                 long long sg, long long su, long long sd, float *out)
{
    if (!wubu_gpu_init() || !x || !d_wg || !d_wu || !d_wd || !out) return -1;
    typedef int (*fn_ffn_t)(const void *, const void *, const void *, int,
                            const float *, float *, float *, float *,
                            int, int, long long, long long, long long);
    static fn_ffn_t ffn_fn;
    static int looked;
    if (!looked) {
        looked = 1;
        ffn_fn = (fn_ffn_t)(void *)GetProcAddress(G.h, "wubu_cuda_ffn_dev");
    }
    if (!ffn_fn) return -2;

    static float *d_x, *d_gate, *d_up, *d_out;
    static int cap_h, cap_f;
    if (cap_h < hidden) {
        if (d_x) G.free_dev(d_x);
        if (d_out) G.free_dev(d_out);
        d_x = (float *)G.alloc((unsigned long long)((size_t)hidden * sizeof(float)));
        d_out = (float *)G.alloc((unsigned long long)((size_t)hidden * sizeof(float)));
        if (!d_x || !d_out) { cap_h = 0; return -3; }
        cap_h = hidden;
    }
    if (cap_f < ff) {
        if (d_gate) G.free_dev(d_gate);
        if (d_up) G.free_dev(d_up);
        d_gate = (float *)G.alloc((unsigned long long)((size_t)ff * sizeof(float)));
        d_up = (float *)G.alloc((unsigned long long)((size_t)ff * sizeof(float)));
        if (!d_gate || !d_up) { cap_f = 0; return -3; }
        cap_f = ff;
    }
    if (G.upload(d_x, x, (unsigned long long)((size_t)hidden * sizeof(float))) != 0) return -4;
    if (ffn_fn(d_wg, d_wu, d_wd, ggml_type, d_x, d_gate, d_up, d_out,
               hidden, ff, sg, su, sd) != 0) return -5;
    if (G.download(out, d_out, (unsigned long long)((size_t)hidden * sizeof(float))) != 0) return -6;
    return 0;
}

/* The activations for q, k and v in a layer are the same buffer, so uploading and
 * quantising it three times per layer is pure overhead. prepare() stages an input;
 * matvec_cached() then reuses it. A mismatch in K or a missing prepare falls back to
 * staging again, so a stale cache cannot silently produce wrong numbers. */
static const float *g_staged_ptr;
static int g_staged_K;

int wubu_gpu_stage_input(const float *x, int K);

/* Scratch buffers shared by wubu_gpu_matvec and the staged-input variant, so q/k/v in a
 * layer can reuse one activation upload and one Q8_K quantisation. */
static float *d_x, *d_y;
static void *d_q8;
static int cap_x, cap_y, cap_q8;
static int g_staged_K;


static int gpu_ensure_buffers(int K, int n_cols)
{
    if (cap_x < K) {
        if (d_x) G.free_dev(d_x);
        if (d_q8) G.free_dev(d_q8);
        d_x = (float *)G.alloc((unsigned long long)((size_t)K * sizeof(float)));
        d_q8 = G.alloc((unsigned long long)(((size_t)(K / 256) + 1) * 292));
        if (!d_x || !d_q8) { cap_x = 0; g_staged_K = 0; return -2; }
        cap_x = K;
        cap_q8 = K;
    }
    if (cap_y < n_cols) {
        if (d_y) G.free_dev(d_y);
        d_y = (float *)G.alloc((unsigned long long)((size_t)n_cols * sizeof(float)));
        if (!d_y) { cap_y = 0; return -3; }
        cap_y = n_cols;
    }
    (void)cap_q8;
    return 0;
}

/* Run the type-specific kernel on the already-staged Q8_K input and download. */
static int gpu_launch_and_download(const void *d_w, int ggml_type,
                                   int K, int n_cols, long long col_stride, float *y)
{
    if (ggml_type == 22 && G.mv2) {
        if (G.mv2(d_w, d_q8, d_y, K, n_cols, col_stride) != 0) return -6;
    } else if (ggml_type == 18 && G.mv3) {
        if (G.mv3(d_w, d_q8, d_y, K, n_cols, col_stride) != 0) return -6;
    } else if (ggml_type == 23 && G.mv4) {
        if (G.mv4(d_w, d_q8, d_y, K, n_cols, col_stride) != 0) return -6;
    } else if (ggml_type == 17 && G.mv5) {
        if (G.mv5(d_w, d_q8, d_y, K, n_cols, col_stride) != 0) return -6;
    } else if (G.mv(d_w, ggml_type, d_q8, d_y, K, n_cols, col_stride) != 0) {
        return -6;
    }
    if (G.download(y, d_y, (unsigned long long)((size_t)n_cols * sizeof(float))) != 0) return -7;
    return 0;
}

/* Stage an activation vector: upload + Q8_K quantise, to be reused by
 * wubu_gpu_matvec_cached for every weight matrix that reads the same input. The caller
 * owns the "x is still this buffer" invariant - the engine only uses it for q/k/v. */
int wubu_gpu_stage_input(const float *x, int K)
{
    if (!wubu_gpu_init() || !x || K <= 0 || (K % 256) != 0) return -1;
    if (gpu_ensure_buffers(K, 1) != 0) return -2;
    if (G.upload(d_x, x, (unsigned long long)((size_t)K * sizeof(float))) != 0) return -3;
    if (G.q8k(d_x, d_q8, K) != 0) return -4;
    g_staged_K = K;
    return 0;
}

int wubu_gpu_matvec_cached(const void *d_w, int ggml_type, int K, int n_cols,
                           long long col_stride, float *y)
{
    if (!S_hooked) { S_hooked = 1; atexit(wubu_gpu_stats_dump); }
    const double t0 = gpu_now_ms();
    if (!d_w || !y || K <= 0 || n_cols <= 0 || K != g_staged_K) return -1;
    if (gpu_ensure_buffers(K, n_cols) != 0) return -2;
    const int rc = gpu_launch_and_download(d_w, ggml_type, K, n_cols, col_stride, y);
    if (rc != 0) return rc;
    if (ggml_type >= 0 && ggml_type < 32) {
        S[ggml_type].calls++;
        S[ggml_type].ms += gpu_now_ms() - t0;
        S[ggml_type].bytes += (long)n_cols * (long)col_stride;
    }
    return 0;
}

/* Fused q/k/v: one upload, one quantisation, three kernels, one download of q|k|v. */
int wubu_gpu_qkv(const void *d_wq, const void *d_wk, const void *d_wv, int ggml_type,
                 const float *x, int K, int qd, int kvd,
                 long long sq, long long sk, long long sv, float *q, float *k, float *v)
{
    if (!wubu_gpu_init() || !x || !d_wq || !d_wk || !d_wv || !q || !k || !v) return -1;
    typedef int (*fn_qkv_t)(const void *, const void *, const void *, int, const float *,
                            float *, float *, float *, int, int, int,
                            long long, long long, long long);
    static fn_qkv_t qkv_fn;
    static int looked;
    if (!looked) { looked = 1; qkv_fn = (fn_qkv_t)(void *)GetProcAddress(G.h, "wubu_cuda_qkv_dev"); }
    if (!qkv_fn) return -2;

    static float *dqkv, *dx;
    static int cap_qkv, cap_x;
    const int total = qd + 2 * kvd;
    if (cap_qkv < total) {
        if (dqkv) G.free_dev(dqkv);
        dqkv = (float *)G.alloc((unsigned long long)((size_t)total * sizeof(float)));
        if (!dqkv) { cap_qkv = 0; return -3; }
        cap_qkv = total;
    }
    if (cap_x < K) {
        if (dx) G.free_dev(dx);
        dx = (float *)G.alloc((unsigned long long)((size_t)K * sizeof(float)));
        if (!dx) { cap_x = 0; return -3; }
        cap_x = K;
    }
    if (G.upload(dx, x, (unsigned long long)((size_t)K * sizeof(float))) != 0) return -4;
    if (qkv_fn(d_wq, d_wk, d_wv, ggml_type, dx, dqkv, dqkv + qd, dqkv + qd + kvd,
               K, qd, kvd, sq, sk, sv) != 0) return -5;
    if (G.download(q, dqkv, (unsigned long long)((size_t)total * sizeof(float))) != 0) return -6;
    /* q|k|v arrived contiguously; split into the caller's three buffers. */
    memcpy(k, q + qd, (size_t)kvd * sizeof(float));
    memcpy(v, q + qd + kvd, (size_t)kvd * sizeof(float));
    return 0;
}

int wubu_gpu_matvec(const void *d_w, int ggml_type, const float *x,
                    int K, int n_cols, long long col_stride, float *y)
{
    if (!S_hooked) { S_hooked = 1; atexit(wubu_gpu_stats_dump); }
    const double t0 = gpu_now_ms();
    if (!wubu_gpu_init() || !d_w || !x || !y || K <= 0 || n_cols <= 0) return -1;

    const int brc = gpu_ensure_buffers(K, n_cols);
    if (brc != 0) return brc;

    if (G.upload(d_x, x, (unsigned long long)((size_t)K * sizeof(float))) != 0) return -4;
    if (G.q8k(d_x, d_q8, K) != 0) return -5;
    g_staged_K = K;
    const int lrc = gpu_launch_and_download(d_w, ggml_type, K, n_cols, col_stride, y);
    if (lrc != 0) return lrc;
    if (ggml_type >= 0 && ggml_type < 32) {
        S[ggml_type].calls++;
        S[ggml_type].ms += gpu_now_ms() - t0;
        S[ggml_type].bytes += (long)n_cols * (long)col_stride;
    }
    return 0;
}
