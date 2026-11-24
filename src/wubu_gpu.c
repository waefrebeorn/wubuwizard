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
typedef int (*fn_mv_mrow)(const void *, const float *, int, int, long long, int, float *);

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
    fn_mv_mrow mrow;  /* M-row IQ2_S prefill kernel (weight decoded once per MT rows) */
    fn_mv_mrow mrow3; /* M-row IQ3_S prefill kernel */
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
    /* Wait policy first: cudaSetDeviceFlags only works before the context exists, and this
     * is the first CUDA call in the process. G4_WAIT=spin|yield|block selects it; spin is
     * the default because ~200 syncs per token each pay the wake-up when the thread sleeps. */
    {
        typedef int (*fn_wait_t)(int);
        fn_wait_t w = (fn_wait_t)(void *)GetProcAddress(G.h, "wubu_cuda_set_wait_policy");
        if (w) {
            const char *e = getenv("G4_WAIT");
            int mode = 0;
            if (e && e[0] == 'y') mode = 1;
            else if (e && e[0] == 'b') mode = 2;
            else if (e && (e[0] == 'o' || e[0] == 'n')) mode = -1;   /* leave the default */
            const int rc = (mode < 0) ? 0 : w(mode);
            printf("[GPU] wait policy %s (rc=%d)\n",
                   mode == 0 ? "spin" : (mode == 1 ? "yield" : "block"), rc);
        }
    }
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
    G.mrow = (fn_mv_mrow)(void *)GetProcAddress(G.h, "wubu_cuda_matvec_iq2s_mrow");
    G.mrow3 = (fn_mv_mrow)(void *)GetProcAddress(G.h, "wubu_cuda_matvec_iq3s_mrow");
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
    /* IQ2_XS 512-entry grid for the type-17 device kernel. Without this upload the kernel
     * returns 0 for every column, which reads as an attention error of exactly the output
     * scale rather than as a missing table. */
    {
        fn_mv2 up5 = (fn_mv2)(void *)GetProcAddress(G.h, "wubu_cuda_upload_tables_iq2xs");
        extern const void *wubu_vecdot_iq2_xs_grid(void);
        const int up5_rc = (up5 && wubu_vecdot_iq2_xs_grid())
                             ? up5(wubu_vecdot_iq2_xs_grid(), 0, 0, 0, 0, 0) : -99;
        if (getenv("G4_DEBUG_TABLES"))
            fprintf(stderr, "[TABLES] iq2xs up5=%p grid=%p rc=%d\n",
                    (void *)up5, wubu_vecdot_iq2_xs_grid(), up5_rc);
        if (up5_rc != 0) G.mv5 = NULL;
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
int wubu_gpu_ffn3(const float *x, const void *d_wg, const void *d_wu, const void *d_wd,
                  int tg, int tu, int td, int hidden, int ff,
                  long long sg, long long su, long long sd, float *out)
{
    typedef int (*fn_f3_t)(const void *, const void *, const void *, int, int, int,
                           const float *, float *, float *, float *, int, int,
                           long long, long long, long long);
    static fn_f3_t dev3;
    static int looked;
    if (!wubu_gpu_init() || !x || !out) return -1;
    if (!looked) { looked = 1; dev3 = (fn_f3_t)(void *)GetProcAddress(G.h, "wubu_cuda_ffn_dev3"); }
    if (!dev3) return -2;

    static float *dx, *dg, *du, *dout;
    static int cap_x, cap_f, cap_o;
    if (cap_x < hidden) {
        if (dx) G.free_dev(dx);
        dx = (float *)G.alloc((unsigned long long)((size_t)hidden * sizeof(float)));
        if (!dx) { cap_x = 0; return -3; }
        cap_x = hidden;
    }
    if (cap_f < ff) {
        if (dg) { G.free_dev(dg); G.free_dev(du); }
        dg = (float *)G.alloc((unsigned long long)((size_t)ff * sizeof(float)));
        du = (float *)G.alloc((unsigned long long)((size_t)ff * sizeof(float)));
        if (!dg || !du) { cap_f = 0; return -3; }
        cap_f = ff;
    }
    if (cap_o < hidden) {
        if (dout) G.free_dev(dout);
        dout = (float *)G.alloc((unsigned long long)((size_t)hidden * sizeof(float)));
        if (!dout) { cap_o = 0; return -3; }
        cap_o = hidden;
    }
    if (G.upload(dx, x, (unsigned long long)((size_t)hidden * sizeof(float))) != 0) return -4;
    if (dev3(d_wg, d_wu, d_wd, tg, tu, td, dx, dg, du, dout, hidden, ff, sg, su, sd) != 0) return -5;
    if (G.download(out, dout, (unsigned long long)((size_t)hidden * sizeof(float))) != 0) return -6;
    return 0;
}

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

/* staged q and attention row, exposed for the G4_QDUMP / G4_ATTN_DUMP comparisons */
static float g_qdump[8192];
static float g_attn_host[8192];

/* Read a few bytes back from a device buffer: used to check that a tensor was uploaded at
 * all, which "returns zeros" and "never uploaded" look identical without. */
int wubu_gpu_peek(const void *dev, void *host, long long n) {
    if (!dev || !host || n <= 0) return -1;
    if (!wubu_gpu_init()) return -2;
    return G.download(host, dev, (unsigned long long)n) == 0 ? 0 : -3;
}

int wubu_gpu_get_attn_dump(float *out, int n) {
    if (!out || n <= 0 || n > 8192) return -1;
    memcpy(out, g_attn_host, (size_t)n * sizeof(float));
    return 0;
}

int wubu_gpu_get_qdump(float *out, int n) {
    if (!out || n <= 0 || n > 8192) return -1;
    memcpy(out, g_qdump, (size_t)n * sizeof(float));
    return 0;
}

/* ---- device attention half-layer -----------------------------------------------
 * One submission: q,k,v matvecs -> per-head RMS norms -> RoPE -> KV append -> attention
 * -> o-projection. Owns a per-layer device KV cache of MAX_KV_POS positions; the caller
 * primes it from the CPU cache and keeps its own cache current with the k/v that come
 * back, so falling back past capacity stays correct. */
#define MAX_LAYERS_DEV 128
#define MAX_KV_POS 512

static void *g_kv_k[MAX_LAYERS_DEV];
static void *g_kv_v[MAX_LAYERS_DEV];
static void *g_qn_dev[MAX_LAYERS_DEV];
static void *g_kn_dev[MAX_LAYERS_DEV];
static const float *g_qn_src[MAX_LAYERS_DEV];
static const float *g_kn_src[MAX_LAYERS_DEV];
static void *g_freqs_dev;
static const float *g_freqs_src;

typedef int (*fn_half_t)(const void *, const void *, const void *, const void *,
                         int, int, int, int, const float *,
                         float *, float *, float *, float *, float *, float *, float *,
                         const float *, const float *,
                         int, int, int, int, int, int, int, int, int,
                         long long, long long, long long, long long,
                         float, int, float, const float *);
static fn_half_t g_half_fn;
static int g_half_looked;

typedef int (*fn_stage_t)(const void *, const void *, const void *,
                          int, int, int, const float *,
                          float *, float *, float *,
                          float *, float *, float *,
                          const float *, const float *,
                          int, int, int, int, int, int, int, int, int,
                          long long, long long, long long,
                          float, int, float, const float *);
static fn_stage_t g_stage_fn;
static int g_stage_looked;

/* Run the attention stage and return the attention row itself (qd floats), for the
 * G4_ATTN_ADIFF instrument. Appends to the device KV like the composite does. */
int wubu_gpu_attn_stage_only(int layer_id, const void *d_wq, const void *d_wk, const void *d_wv,
                             int tq, int tk, int tv, const float *x,
                             int hidden, int qd, int kvd, int hd, int n_heads, int n_kv_heads,
                             int pos, int kv_start, int kv_len,
                             long long sq, long long sk, long long sv,
                             const float *qn, const float *kn, const float *freqs,
                             int n_rot, float rope_base, float *attn_out)
{
    if (!wubu_gpu_init() || layer_id < 0 || layer_id >= MAX_LAYERS_DEV) return -1;
    if (!x || !attn_out || pos >= MAX_KV_POS || kv_len <= 0) return -1;
    if (!g_stage_looked) {
        g_stage_looked = 1;
        g_stage_fn = (fn_stage_t)(void *)GetProcAddress(G.h, "wubu_cuda_attn_stage");
    }
    if (!g_stage_fn) return -2;
    if (!g_kv_k[layer_id]) {
        g_kv_k[layer_id] = G.alloc((unsigned long long)((size_t)MAX_KV_POS * kvd * sizeof(float)));
        g_kv_v[layer_id] = G.alloc((unsigned long long)((size_t)MAX_KV_POS * kvd * sizeof(float)));
        if (!g_kv_k[layer_id] || !g_kv_v[layer_id]) return -3;
    }
    static float *scratch, *dx;
    static int cap_scratch, cap_x;
    const int total = qd + kvd + kvd + qd;
    if (cap_scratch < total) {
        if (scratch) G.free_dev(scratch);
        scratch = (float *)G.alloc((unsigned long long)((size_t)total * sizeof(float)));
        if (!scratch) { cap_scratch = 0; return -3; }
        cap_scratch = total;
    }
    if (cap_x < hidden) {
        if (dx) G.free_dev(dx);
        dx = (float *)G.alloc((unsigned long long)((size_t)hidden * sizeof(float)));
        if (!dx) { cap_x = 0; return -3; }
        cap_x = hidden;
    }
    float *d_q = scratch, *d_k = d_q + qd, *d_v = d_k + kvd, *d_attn = d_v + kvd;
    if (g_qn_src[layer_id] != qn || !g_qn_dev[layer_id]) {
        if (g_qn_dev[layer_id]) G.free_dev(g_qn_dev[layer_id]);
        g_qn_dev[layer_id] = G.alloc((unsigned long long)((size_t)hd * sizeof(float)));
        if (!g_qn_dev[layer_id]) return -4;
        if (G.upload(g_qn_dev[layer_id], qn, (unsigned long long)((size_t)hd * sizeof(float))) != 0) return -4;
        g_qn_src[layer_id] = qn;
    }
    if (g_kn_src[layer_id] != kn || !g_kn_dev[layer_id]) {
        if (g_kn_dev[layer_id]) G.free_dev(g_kn_dev[layer_id]);
        g_kn_dev[layer_id] = G.alloc((unsigned long long)((size_t)hd * sizeof(float)));
        if (!g_kn_dev[layer_id]) return -4;
        if (G.upload(g_kn_dev[layer_id], kn, (unsigned long long)((size_t)hd * sizeof(float))) != 0) return -4;
        g_kn_src[layer_id] = kn;
    }
    if (freqs && (g_freqs_src != freqs || !g_freqs_dev)) {
        if (g_freqs_dev) G.free_dev(g_freqs_dev);
        g_freqs_dev = G.alloc((unsigned long long)((size_t)(n_rot / 2) * sizeof(float)));
        if (!g_freqs_dev) return -4;
        if (G.upload(g_freqs_dev, freqs, (unsigned long long)((size_t)(n_rot / 2) * sizeof(float))) != 0) return -4;
        g_freqs_src = freqs;
    }
    if (G.upload(dx, x, (unsigned long long)((size_t)hidden * sizeof(float))) != 0) return -5;
    const double theta_scale = pow((double)rope_base, -2.0 / (double)n_rot);
    if (g_stage_fn(d_wq, d_wk, d_wv, tq, tk, tv, dx, d_q, d_k, d_v,
                   (float *)g_kv_k[layer_id], (float *)g_kv_v[layer_id], d_attn,
                   (const float *)g_qn_dev[layer_id], (const float *)g_kn_dev[layer_id],
                   hidden, qd, kvd, hd, n_heads, n_kv_heads, pos, kv_start, kv_len,
                   sq, sk, sv, (float)pos, n_rot, (float)theta_scale,
                   (const float *)g_freqs_dev) != 0) return -6;
    if (G.download(attn_out, d_attn, (unsigned long long)((size_t)qd * sizeof(float))) != 0) return -7;
    return 0;
}

int wubu_gpu_attn_half_capacity(void) { return MAX_KV_POS; }

int wubu_gpu_attn_half_prime(int layer_id, const float *cpu_k, const float *cpu_v,
                             int n, int kvd)
{
    if (layer_id < 0 || layer_id >= MAX_LAYERS_DEV || !cpu_k || !cpu_v || kvd <= 0) return -1;
    if (n <= 0 || n > MAX_KV_POS) return -1;
    if (!g_kv_k[layer_id]) {
        g_kv_k[layer_id] = G.alloc((unsigned long long)((size_t)MAX_KV_POS * kvd * sizeof(float)));
        g_kv_v[layer_id] = G.alloc((unsigned long long)((size_t)MAX_KV_POS * kvd * sizeof(float)));
        if (!g_kv_k[layer_id] || !g_kv_v[layer_id]) return -2;
    }
    if (G.upload(g_kv_k[layer_id], cpu_k, (unsigned long long)((size_t)n * kvd * sizeof(float))) != 0) return -3;
    if (G.upload(g_kv_v[layer_id], cpu_v, (unsigned long long)((size_t)n * kvd * sizeof(float))) != 0) return -3;
    return 0;
}

int wubu_gpu_attn_half(int layer_id, const void *d_wq, const void *d_wk, const void *d_wv,
                       const void *d_wo, int tq, int tk, int tv, int to, const float *x,
                       int hidden, int qd, int kvd, int hd, int n_heads, int n_kv_heads,
                       int pos, int kv_start, int kv_len,
                       long long sq, long long sk, long long sv, long long so,
                       const float *qn, const float *kn, const float *freqs,
                       int n_rot, float rope_base, float *out, float *k_out, float *v_out);

/* Attention only: the same composite with the o-projection skipped (d_wo NULL), returning
 * the attention rows so a gate can compare that stage on its own. */
int wubu_gpu_attn_only(int layer_id, const void *d_wq, const void *d_wk, const void *d_wv,
                       int tq, int tk, int tv, const float *x,
                       int hidden, int qd, int kvd, int hd, int n_heads, int n_kv_heads,
                       int pos, int kv_start, int kv_len,
                       long long sq, long long sk, long long sv,
                       const float *qn, const float *kn, const float *freqs,
                       int n_rot, float rope_base, float *attn_out, float *k_out, float *v_out)
{
    return wubu_gpu_attn_half(layer_id, d_wq, d_wk, d_wv, NULL, tq, tk, tv, 0, x,
                              hidden, qd, kvd, hd, n_heads, n_kv_heads, pos, kv_start, kv_len,
                              sq, sk, sv, 0, qn, kn, freqs, n_rot, rope_base,
                              attn_out, k_out, v_out);
}

int wubu_gpu_attn_half(int layer_id, const void *d_wq, const void *d_wk, const void *d_wv,
                       const void *d_wo, int tq, int tk, int tv, int to, const float *x,
                       int hidden, int qd, int kvd, int hd, int n_heads, int n_kv_heads,
                       int pos, int kv_start, int kv_len,
                       long long sq, long long sk, long long sv, long long so,
                       const float *qn, const float *kn, const float *freqs,
                       int n_rot, float rope_base, float *out, float *k_out, float *v_out)
{
    if (!wubu_gpu_init() || layer_id < 0 || layer_id >= MAX_LAYERS_DEV) return -1;
    if (pos >= MAX_KV_POS || kv_len <= 0) return -1;
    if (!g_half_looked) {
        g_half_looked = 1;
        g_half_fn = (fn_half_t)(void *)GetProcAddress(G.h, "wubu_cuda_attn_half_dev");
    }
    if (!g_half_fn) return -2;
    if (!g_kv_k[layer_id]) {
        g_kv_k[layer_id] = G.alloc((unsigned long long)((size_t)MAX_KV_POS * kvd * sizeof(float)));
        g_kv_v[layer_id] = G.alloc((unsigned long long)((size_t)MAX_KV_POS * kvd * sizeof(float)));
        if (!g_kv_k[layer_id] || !g_kv_v[layer_id]) return -3;
    }
    static float *scratch, *dx;
    static int cap_scratch, cap_x;
    const int total = qd + kvd + kvd + qd + hidden;
    if (cap_scratch < total) {
        if (scratch) G.free_dev(scratch);
        scratch = (float *)G.alloc((unsigned long long)((size_t)total * sizeof(float)));
        if (!scratch) { cap_scratch = 0; return -3; }
        cap_scratch = total;
    }
    if (cap_x < hidden) {
        if (dx) G.free_dev(dx);
        dx = (float *)G.alloc((unsigned long long)((size_t)hidden * sizeof(float)));
        if (!dx) { cap_x = 0; return -3; }
        cap_x = hidden;
    }
    float *d_q = scratch, *d_k = d_q + qd, *d_v = d_k + kvd, *d_attn = d_v + kvd;
    float *d_out = d_attn + qd;

    if (g_qn_src[layer_id] != qn || !g_qn_dev[layer_id]) {
        if (g_qn_dev[layer_id]) G.free_dev(g_qn_dev[layer_id]);
        g_qn_dev[layer_id] = G.alloc((unsigned long long)((size_t)hd * sizeof(float)));
        if (!g_qn_dev[layer_id]) return -4;
        if (G.upload(g_qn_dev[layer_id], qn, (unsigned long long)((size_t)hd * sizeof(float))) != 0) return -4;
        g_qn_src[layer_id] = qn;
    }
    if (g_kn_src[layer_id] != kn || !g_kn_dev[layer_id]) {
        if (g_kn_dev[layer_id]) G.free_dev(g_kn_dev[layer_id]);
        g_kn_dev[layer_id] = G.alloc((unsigned long long)((size_t)hd * sizeof(float)));
        if (!g_kn_dev[layer_id]) return -4;
        if (G.upload(g_kn_dev[layer_id], kn, (unsigned long long)((size_t)hd * sizeof(float))) != 0) return -4;
        g_kn_src[layer_id] = kn;
    }
    if (freqs && (g_freqs_src != freqs || !g_freqs_dev)) {
        if (g_freqs_dev) G.free_dev(g_freqs_dev);
        g_freqs_dev = G.alloc((unsigned long long)((size_t)(n_rot / 2) * sizeof(float)));
        if (!g_freqs_dev) return -4;
        if (G.upload(g_freqs_dev, freqs, (unsigned long long)((size_t)(n_rot / 2) * sizeof(float))) != 0) return -4;
        g_freqs_src = freqs;
    }

    if (G.upload(dx, x, (unsigned long long)((size_t)hidden * sizeof(float))) != 0) return -5;
    const double theta_scale = pow((double)rope_base, -2.0 / (double)n_rot);
    if (g_half_fn(d_wq, d_wk, d_wv, d_wo, tq, tk, tv, to, dx, d_q, d_k, d_v,
                  (float *)g_kv_k[layer_id], (float *)g_kv_v[layer_id], d_attn, d_out,
                  (const float *)g_qn_dev[layer_id], (const float *)g_kn_dev[layer_id],
                  hidden, qd, kvd, hd, n_heads, n_kv_heads, pos, kv_start, kv_len,
                  sq, sk, sv, so, (float)pos, n_rot, (float)theta_scale,
                  (const float *)g_freqs_dev) != 0) return -6;
    if (getenv("G4_QDUMP") && qd <= 8192)
        G.download(g_qdump, d_q, (unsigned long long)((size_t)qd * sizeof(float)));
    /* G4_ATTN_DUMP: keep the device attention row so the o-projection that follows it can be
     * compared against the CPU's, which is the only stage left between the row and the
     * residual that the fused path could be getting wrong. */
    if (getenv("G4_ATTN_DUMP") && qd <= 8192)
        G.download(g_attn_host, d_attn, (unsigned long long)((size_t)qd * sizeof(float)));
    if (G.download(out, d_out, (unsigned long long)((size_t)hidden * sizeof(float))) != 0) return -7;
    if (k_out && G.download(k_out, d_k, (unsigned long long)((size_t)kvd * sizeof(float))) != 0) return -8;
    if (v_out && G.download(v_out, d_v, (unsigned long long)((size_t)kvd * sizeof(float))) != 0) return -8;
    return 0;
}

/* Fused q/k/v with a type per projection. On the 12B model attn_q is IQ2_S and attn_k is
 * IQ3_S, so the single-type fused path only covered part of the layers and the rest fell
 * back to three separate round trips (2.5 device calls per layer, measured). One
 * quantisation of the shared input, three kernels, one submission. */
typedef int (*fn_qkv_mrow_t)(const void *, int, const void *, int, const void *, int,
                            const float *, int, int, int, int, int,
                            long long, long long, long long, float *, float *, float *);

/* Fused Q/K/V for a prompt: one staging of the normalised rows, one Q8_K quantisation, three
 * M-row kernels. The N == 1 path (wubu_gpu_qkv3) does the same for a single row. */
int wubu_gpu_qkv_mrow(const void *d_wq, int tq, const void *d_wk, int tk, const void *d_wv, int tv,
                      const float *x, int K, int M, int nq, int nk, int nv,
                      long long sq, long long sk, long long sv, float *q, float *k, float *v)
{
    if (!wubu_gpu_init()) return -1;
    static fn_qkv_mrow_t f; static int looked;
    if (!looked) { looked = 1; f = (fn_qkv_mrow_t)(void *)GetProcAddress(G.h, "wubu_cuda_qkv_mrow"); }
    if (!f) return -1;
    return f(d_wq, tq, d_wk, tk, d_wv, tv, x, K, M, nq, nk, nv, sq, sk, sv, q, k, v);
}

int wubu_gpu_qkv3(const void *d_wq, const void *d_wk, const void *d_wv,
                  int tq, int tk, int tv, const float *x,
                  int K, int qd, int kvd, long long sq, long long sk, long long sv,
                  float *q, float *k, float *v)
{
    typedef int (*fn_qkv3_t)(const void *, const void *, const void *, int, int, int,
                             const float *, float *, float *, float *, int, int, int,
                             long long, long long, long long);
    static fn_qkv3_t dev3;
    static int looked;
    if (!wubu_gpu_init() || !x || !q || !k || !v) return -1;
    if (!looked) { looked = 1; dev3 = (fn_qkv3_t)(void *)GetProcAddress(G.h, "wubu_cuda_qkv_dev3"); }
    if (!dev3) return -2;

    static float *scratch;
    static int cap;
    const int total = qd + kvd + kvd;
    if (cap < total) {
        if (scratch) G.free_dev(scratch);
        scratch = (float *)G.alloc((unsigned long long)((size_t)total * sizeof(float)));
        if (!scratch) { cap = 0; return -3; }
        cap = total;
    }
    float *d_q = scratch, *d_k = scratch + qd, *d_v = scratch + qd + kvd;
    static float *dx;
    static int cap_x;
    if (cap_x < K) {
        if (dx) G.free_dev(dx);
        dx = (float *)G.alloc((unsigned long long)((size_t)K * sizeof(float)));
        if (!dx) { cap_x = 0; return -3; }
        cap_x = K;
    }
    if (G.upload(dx, x, (unsigned long long)((size_t)K * sizeof(float))) != 0) return -4;
    if (dev3(d_wq, d_wk, d_wv, tq, tk, tv, dx, d_q, d_k, d_v, K, qd, kvd, sq, sk, sv) != 0) return -5;
    if (G.download(q, d_q, (unsigned long long)((size_t)qd * sizeof(float))) != 0) return -6;
    if (G.download(k, d_k, (unsigned long long)((size_t)kvd * sizeof(float))) != 0) return -6;
    if (G.download(v, d_v, (unsigned long long)((size_t)kvd * sizeof(float))) != 0) return -6;
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

/* Per-stage wall time inside the matvec path, so the difference between the isolated
 * bench (0.110 ms for the o-projection shape) and what the engine charges can be split
 * into upload / quantise / launch+download / sync instead of guessed at. */
double wubu_gpu_t_upload, wubu_gpu_t_q8, wubu_gpu_t_call;
long wubu_gpu_n_matvec;
/* Per-section attribution, so the section timers in the model can be checked against the
 * calls they claim to be timing instead of taken on trust. */
int wubu_gpu_tag;
static double g_tag_ms[8];
static long g_tag_n[8];
static const char *const g_tag_name[8] = {
    "none", "qkv", "o-proj", "ffn", "head", "other", "other2", "other3"
};

void wubu_gpu_stage_report(void)
{
    if (wubu_gpu_n_matvec <= 0) return;
    printf("[GPU] %ld matvecs: upload %.1f ms, q8k %.1f ms, launch+download %.1f ms\n",
           wubu_gpu_n_matvec, wubu_gpu_t_upload, wubu_gpu_t_q8, wubu_gpu_t_call);
    for (int t = 0; t < 8; ++t) {
        if (g_tag_n[t] <= 0) continue;
        printf("[GPU]   %-7s %5ld calls  %8.1f ms total  %7.3f ms/call\n",
               g_tag_name[t], g_tag_n[t], g_tag_ms[t], g_tag_ms[t] / (double)g_tag_n[t]);
    }
}

/* M-row IQ2_S prefill. Returns non-zero when the DLL has no such entry, so the caller can
 * fall back to whatever it did before. */
int wubu_gpu_matvec_iq2s_mrow(const void *d_w, const float *x, int K, int n_cols,
                              long long col_stride, int M, float *y)
{
    if (!wubu_gpu_init() || !G.mrow) return -1;
    return G.mrow(d_w, x, K, n_cols, col_stride, M, y);
}

typedef int (*fn_attn_causal)(const float *, const float *, const float *, float *, const int *,
                             int, int, int, int, int, int, int, int);

/* Batched causal attention on the device. Returns 0 on success, non-zero to fall back to CPU. */
int wubu_gpu_attn_causal(const float *q, const float *k, const float *v, float *out,
                         const int *pos, int N, int n_heads, int kv_heads, int hd,
                         int qd, int kvd, int kv_size, int window)
{
    if (!wubu_gpu_init()) return -1;
    fn_attn_causal f = (fn_attn_causal)(void *)GetProcAddress(G.h, "wubu_cuda_attn_causal");
    if (!f) return -1;
    return f(q, k, v, out, pos, N, n_heads, kv_heads, hd, qd, kvd, kv_size, window);
}

typedef void (*fn_pool_stats)(void);

/* Print the device workspace pool: every scratch buffer the kernels declared, by role. */
void wubu_gpu_pool_stats(void)
{
    if (!wubu_gpu_init()) return;
    fn_pool_stats f = (fn_pool_stats)(void *)GetProcAddress(G.h, "wubu_cuda_pool_stats");
    if (f) f();
}

int wubu_gpu_matvec_iq3s_mrow(const void *d_w, const float *x, int K, int n_cols,
                              long long col_stride, int M, float *y)
{
    if (!wubu_gpu_init() || !G.mrow3) return -1;
    return G.mrow3(d_w, x, K, n_cols, col_stride, M, y);
}

int wubu_gpu_matvec(const void *d_w, int ggml_type, const float *x,
                    int K, int n_cols, long long col_stride, float *y)
{
    if (!S_hooked) { S_hooked = 1; atexit(wubu_gpu_stats_dump); }
    const double t0 = gpu_now_ms();
    if (!wubu_gpu_init() || !d_w || !x || !y || K <= 0 || n_cols <= 0) return -1;

    const int brc = gpu_ensure_buffers(K, n_cols);
    if (brc != 0) return brc;

    /* Timing is off unless G4_PROF is set: two clock reads per matvec is 400 per token and
     * that showed up as 11.4 -> 11.2 tok/s. */
    static int prof_on = -1;
    if (prof_on < 0) { const char *e = getenv("G4_PROF"); prof_on = (e && *e == '1'); }
    const double _s0 = prof_on ? gpu_now_ms() : 0.0;
    if (G.upload(d_x, x, (unsigned long long)((size_t)K * sizeof(float))) != 0) return -4;
    double _s1 = prof_on ? gpu_now_ms() : 0.0; if (prof_on) wubu_gpu_t_upload += _s1 - _s0;
    if (G.q8k(d_x, d_q8, K) != 0) return -5;
    double _s2 = prof_on ? gpu_now_ms() : 0.0; if (prof_on) wubu_gpu_t_q8 += _s2 - _s1;
    g_staged_K = K;
    const int lrc = gpu_launch_and_download(d_w, ggml_type, K, n_cols, col_stride, y);
    if (lrc != 0) return lrc;
    if (prof_on) {
        const double _e = gpu_now_ms();
        wubu_gpu_t_call += _e - _s2;
        ++wubu_gpu_n_matvec;
        {
            const int tg = (wubu_gpu_tag >= 0 && wubu_gpu_tag < 8) ? wubu_gpu_tag : 0;
            g_tag_ms[tg] += _e - _s0;
            ++g_tag_n[tg];
        }
    }
    if (ggml_type >= 0 && ggml_type < 32) {
        S[ggml_type].calls++;
        S[ggml_type].ms += gpu_now_ms() - t0;
        S[ggml_type].bytes += (long)n_cols * (long)col_stride;
    }
    return 0;
}
