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
    fn_mv2 mv2;   /* lane-per-grid-entry IQ2_S kernel */
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
    G.mv2 = (fn_mv2)(void *)GetProcAddress(G.h, "wubu_cuda_matvec_iq2s_v2");
    if (!avail || !G.alloc || !G.upload || !G.free_dev || !G.download || !G.q8k || !G.mv) return 0;
    if (!avail()) return 0;
    if (tables) {
        extern const void *wubu_vecdot_iq2s_grid(void);
        extern const void *wubu_vecdot_iq3s_grid(void);
        if (tables(wubu_vecdot_iq2s_grid(), wubu_vecdot_iq3s_grid()) != 0) return 0;
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
int wubu_gpu_matvec(const void *d_w, int ggml_type, const float *x,
                    int K, int n_cols, long long col_stride, float *y)
{
    if (!wubu_gpu_init() || !d_w || !x || !y || K <= 0 || n_cols <= 0) return -1;
    static float *d_x;
    static void *d_q8;
    static float *d_y;
    static int cap_x, cap_y, cap_q8;

    if (cap_x < K) {
        if (d_x) G.free_dev(d_x);
        if (d_q8) G.free_dev(d_q8);
        d_x = (float *)G.alloc((unsigned long long)((size_t)K * sizeof(float)));
        d_q8 = G.alloc((unsigned long long)(((size_t)(K / 256) + 1) * 292));
        if (!d_x || !d_q8) { cap_x = 0; return -2; }
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

    if (G.upload(d_x, x, (unsigned long long)((size_t)K * sizeof(float))) != 0) return -4;
    if (G.q8k(d_x, d_q8, K) != 0) return -5;
    /* IQ2_S has the dp4a lane-per-grid-entry kernel: 0.35 ms for a 15360x3840
     * matvec device-resident, vs 0.9 ms for the generic warp-per-column one. */
    if (ggml_type == 22 && G.mv2) {
        if (G.mv2(d_w, d_q8, d_y, K, n_cols, col_stride) != 0) return -6;
    } else if (G.mv(d_w, ggml_type, d_q8, d_y, K, n_cols, col_stride) != 0) {
        return -6;
    }
    if (G.download(y, d_y, (unsigned long long)((size_t)n_cols * sizeof(float))) != 0) return -7;
    return 0;
}
