/*
 * wubu_kernel.c — Hardware-agnostic kernel dispatch, CPU baseline.
 *
 * WASTE reference (https://github.com/sqliteai/waste):
 *   Adopted the kernel dispatch table pattern from waste_kernels[]
 *   but implemented it fully self-contained in C11. Device
 *   backends register function pointers at startup.
 *
 * The CPU baseline is the portable reference implementation of
 * every kernel type — correct within FP32 precision, used as
 * fallback when no device backend is registered or when it
 * reports unsupported.
 */
#include "wubu_kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdarg.h>

/* ------------------------------------------------------------------ */
/* Global state                                                     */
/* ------------------------------------------------------------------ */
static wubu_kernel_backend_t *g_backends = NULL;
static wubu_backend_id_t g_forced_backend = WUBU_BACKEND_AUTO;

/* CPU baseline function pointers */
static wubu_gemm_fn      g_cpu_gemm     = NULL;
static wubu_gemv_fn      g_cpu_gemv     = NULL;
static wubu_attn_fn      g_cpu_attn     = NULL;
static wubu_rope_fn      g_cpu_rope     = NULL;
static wubu_softmax_fn   g_cpu_softmax  = NULL;
#include "wubu_activations.h"
static wubu_rmsnorm_fn   g_cpu_rmsnorm  = NULL;
static wubu_quantize_fn  g_cpu_quantize = NULL;
static wubu_dequantize_fn g_cpu_dequantize = NULL;

/* Flag: true if the cpu_b (static) entry is in the list */
static int g_cpu_b_registered = 0;

/* CPU feature detection result — populated at init */
static wubu_cpu_features_t g_cpu_feat = {0};

/* ------------------------------------------------------------------ */
/* CPU baseline implementations                            */
/* ------------------------------------------------------------------ */

static void cpu_gemm(const float *A, const float *B, float *C,
                           int M, int K, int N, float beta) {
    float alpha = 1.0f;
    for (int i = 0; i < M; i++) {
        const float *ar = A + (size_t)i * K;
        float *cr = C + (size_t)i * N;
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += ar[k] * B[(size_t)k * N + j];
            if (beta == 0.0f) cr[j] = alpha * sum;
            else cr[j] = alpha * sum + beta * cr[j];
        }
    }
}

static void cpu_gemv(const float *A, const float *x, float *y,
                           int M, int K) {
    for (int i = 0; i < M; i++) {
        const float *ar = A + (size_t)i * K;
        float s = 0.0f;
        for (int k = 0; k < K; k++) s += ar[k] * x[k];
        y[i] = s;
    }
}

static void cpu_softmax(float *logits, int M, int N) {
    wubu_softmax_rows(logits, M, N);
}

static void cpu_rmsnorm(float *x, const float *gamma,
                                    const float *beta, int M, int d,
                                    float eps) {
    for (int i = 0; i < M; i++) {
        float *row = x + (size_t)i * d;
        float sum_sq = 0.0f;
        for (int j = 0; j < d; j++) sum_sq += row[j] * row[j];
        float rsqrt = 1.0f / sqrtf(sum_sq / (float)d + eps);
        for (int j = 0; j < d; j++)
            row[j] = row[j] * rsqrt * gamma[j] + beta[j];
    }
}

static void cpu_quantize(const float *fp32, int8_t *q, float *scales,
                                  int M, int K, int bits) {
    float qmax = (float)((1 << bits) - 1);
    (void)bits;
    for (int i = 0; i < M; i++) {
        const float *row = fp32 + (size_t)i * K;
        float amax = 0.0f;
        for (int k = 0; k < K; k++) {
            float a = fabsf(row[k]);
            if (a > amax) amax = a;
        }
        float scale = (amax > 1e-8f) ? amax / qmax : 1e-8f;
        scales[i] = scale;
        float inv = 1.0f / scale;
        for (int k = 0; k < K; k++) {
            int v = (int)roundf(row[k] * inv);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            q[(size_t)i * K + k] = (int8_t)v;
        }
    }
}

static void cpu_dequantize(const int8_t *q, const float *scales,
                                       const float *zeros, float *fp32,
                                       int M, int K, int bits) {
    (void)bits;
    for (int i = 0; i < M; i++) {
        const int8_t *qr = q + (size_t)i * K;
        float *row = fp32 + (size_t)i * K;
        float s = scales[i];
        float z = (zeros ? zeros[i] : 0.0f);
        for (int k = 0; k < K; k++) row[k] = s * ((float)qr[k] - z);
    }
}

/* CPU baseline: causal (masked) attention.
 * Q/K/V are [M, n_heads, d] (M = tokens, N = n_heads * d, d = head_dim).
 * out same layout.  Uses -1e30f sentinel (NOT -INFINITY under -ffast-math). */
static void cpu_attention(const float *Q, const float *K, const float *V,
                          float *out, int M, int N, int d, int n_heads, float scale) {
    for (int tok = 0; tok < M; tok++) {
        for (int h = 0; h < n_heads; h++) {
            float *scores = (float *)malloc((tok + 1) * sizeof(float));
            if (!scores) continue;
            float maxv = -1e30f;
            const float *q = Q + (size_t)(tok * n_heads + h) * d;
            for (int j = 0; j <= tok; j++) {
                const float *k = K + (size_t)(j * n_heads + h) * d;
                float dot = 0.0f;
                for (int i = 0; i < d; i++) dot += q[i] * k[i];
                float s = dot * scale;
                scores[j] = s;
                if (s > maxv) maxv = s;
            }
            float sum = 0.0f;
            for (int j = 0; j <= tok; j++) {
                scores[j] = expf(scores[j] - maxv);
                sum += scores[j];
            }
            if (sum > 0.0f) {
                for (int j = 0; j <= tok; j++) scores[j] /= sum;
            } else {
                float u = 1.0f / (float)(tok + 1);
                for (int j = 0; j <= tok; j++) scores[j] = u;
            }
            float *o = out + (size_t)(tok * n_heads + h) * d;
            for (int i = 0; i < d; i++) o[i] = 0.0f;
            for (int j = 0; j <= tok; j++) {
                float w = scores[j];
                const float *v = V + (size_t)(j * n_heads + h) * d;
                for (int i = 0; i < d; i++) o[i] += w * v[i];
            }
            free(scores);
        }
    }
}

/* CPU baseline: RoPE (rotary positional embedding).
 * Standard interleaved layout: rotate pairs within head_dim/2. */
static void cpu_rope(float *q, float *k, int d, int seq_len,
                     float theta, int offset) {
    int half = d / 2;
    if (half > 64 || half <= 0) return;
    float inv_freq[64];
    for (int i = 0; i < half; i++)
        inv_freq[i] = 1.0f / powf(theta, (float)(2 * i) / (float)d);
    for (int pos = 0; pos < seq_len; pos++) {
        int gp = pos + offset;
        float *qx = q + (size_t)pos * d;
        float *kx = k + (size_t)pos * d;
        for (int i = 0; i < half; i++) {
            float f = (float)gp * inv_freq[i];
            float c = cosf(f), s = sinf(f);
            float q0 = qx[i], q1 = qx[half + i];
            qx[i]       = q0 * c - q1 * s;
            qx[half+i]  = q0 * s + q1 * c;
            float k0 = kx[i], k1 = kx[half + i];
            kx[i]       = k0 * c - k1 * s;
            kx[half+i]  = k0 * s + k1 * c;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Backend management                                     */
int wubu_kernel_init(void) {
    g_cpu_gemm      = cpu_gemm;
    g_cpu_gemv      = cpu_gemv;
    g_cpu_attn      = cpu_attention;
    g_cpu_rope      = cpu_rope;
    g_cpu_softmax   = cpu_softmax;
    g_cpu_rmsnorm   = cpu_rmsnorm;
    g_cpu_quantize  = cpu_quantize;
    g_cpu_dequantize = cpu_dequantize;
    g_forced_backend = WUBU_BACKEND_AUTO;
    g_cpu_b_registered = 0;
    g_backends = NULL;

    /* Auto-detect CPU features for Roofline optimization */
    wubu_cpu_detect(&g_cpu_feat);

    /* Register device backends (CUDA, Vulkan, Metal, etc.) if compiled in */
    wubu_kernel_register_backends();

    return 0;
}

void wubu_kernel_shutdown(void) {
    /* Remove all registered backends; cpu_b is never malloc'd */
    while (g_backends) {
        wubu_kernel_backend_t *next = g_backends->next;
        free(g_backends);  /* cpu_b is NEVER in this list */
        g_backends = next;
    }
    g_backends = NULL;
    g_cpu_b_registered = 0;
    g_cpu_gemm = NULL;
    g_cpu_gemv = NULL;
    g_cpu_softmax = NULL;
    g_cpu_rmsnorm = NULL;
    g_cpu_quantize = NULL;
    g_cpu_dequantize = NULL;
}

int wubu_kernel_register(wubu_backend_id_t id, const char *name,
                                    wubu_kernel_backend_t *backend) {
    if (!backend || !backend->gemm) return -1;
    if (!name) name = wubu_backend_name(id);

    /* Check for duplicate id */
    for (wubu_kernel_backend_t *cur = g_backends; cur; cur = cur->next) {
        if (cur->id == id) return -1;
    }

    wubu_kernel_backend_t *b = (wubu_kernel_backend_t *)
        malloc(sizeof(wubu_kernel_backend_t));
    if (!b) return -2;
    memcpy(b, backend, sizeof(wubu_kernel_backend_t));
    b->name = name;
    b->next = g_backends;
    g_backends = b;
    return 0;
}

int wubu_kernel_unregister(wubu_backend_id_t id) {
    wubu_kernel_backend_t **pp = &g_backends;
    while (*pp) {
        if ((*pp)->id == id) {
            wubu_kernel_backend_t *victim = *pp;
            *pp = victim->next;
            free(victim);  /* only malloc'd entries are in the list */
            return 0;
        }
        pp = &(*pp)->next;
    }
    return -1;
}

int wubu_kernel_force_backend(wubu_backend_id_t id) {
    g_forced_backend = id;
    return 0;
}

const char *wubu_backend_name(wubu_backend_id_t id) {
    switch (id) {
        case WUBU_BACKEND_AUTO:     return "auto";
        case WUBU_BACKEND_SCALAR:   return "cpu-scalar";
        case WUBU_BACKEND_CPU_SIMD: return "cpu-simd";
        case WUBU_BACKEND_CUDA:     return "cuda";
        case WUBU_BACKEND_METAL:    return "metal";
        case WUBU_BACKEND_VULKAN:   return "vulkan";
        case WUBU_BACKEND_ROCM:     return "rocm";
        case WUBU_BACKEND_BLAS:     return "blas";
        default:                    return "unknown";
    }
}

const char *wubu_kernel_active_backend(wubu_kernel_type_t type) {
    (void)type;
    if (g_forced_backend != WUBU_BACKEND_AUTO)
        return wubu_backend_name(g_forced_backend);

    /* Find first registered device backend that supports this type */
    for (wubu_kernel_backend_t *b = g_backends; b; b = b->next) {
        if (b->id == WUBU_BACKEND_SCALAR) continue;
        if (b->supports) {
            int ok = 0;
            switch (type) {
            case WUBU_KERN_GEMM:      ok = (b->gemm != NULL); break;
            case WUBU_KERN_GEMV:      ok = (b->gemv != NULL); break;
            case WUBU_KERN_ATTN:      ok = (b->attn != NULL); break;
            case WUBU_KERN_ROPE:      ok = (b->rope != NULL); break;
            case WUBU_KERN_SOFTMAX:   ok = (b->softmax != NULL); break;
            case WUBU_KERN_LAYER_NORM:ok = (b->rmsnorm != NULL); break;
            case WUBU_KERN_QUANT:     ok = (b->quantize != NULL); break;
            case WUBU_KERN_DEQUANT:   ok = (b->dequantize != NULL); break;
            default: ok = 0; break;
            }
            if (ok) return b->name;
        }
    }
    return "cpu-scalar";
}

/* ------------------------------------------------------------------ */
/* wubu_kernel_run — variadic dispatch                          */
/* ------------------------------------------------------------------ */
int wubu_kernel_run(wubu_kernel_type_t type, ...) {
    va_list args;
    va_start(args, type);
    int rc = 0;

    /* Resolve backend */
    wubu_kernel_backend_t *best = NULL;
    if (g_forced_backend != WUBU_BACKEND_AUTO) {
        for (wubu_kernel_backend_t *b = g_backends; b; b = b->next) {
            if (b->id == g_forced_backend) { best = b; break; }
        }
        if (!best) { rc = -1; goto done; }
    } else {
        /* Prefer registered device backends, fall to CPU scalar */
        for (wubu_kernel_backend_t *b = g_backends; b; b = b->next) {
            if (b->id == WUBU_BACKEND_SCALAR) continue;
            if (!b->supports) continue;
            int ok = 0;
            switch (type) {
            case WUBU_KERN_GEMM:      ok = (b->gemm != NULL); break;
            case WUBU_KERN_GEMV:      ok = (b->gemv != NULL); break;
            case WUBU_KERN_ATTN:      ok = (b->attn != NULL); break;
            case WUBU_KERN_ROPE:      ok = (b->rope != NULL); break;
            case WUBU_KERN_SOFTMAX:   ok = (b->softmax != NULL); break;
            case WUBU_KERN_LAYER_NORM:ok = (b->rmsnorm != NULL); break;
            case WUBU_KERN_QUANT:     ok = (b->quantize != NULL); break;
            case WUBU_KERN_DEQUANT:   ok = (b->dequantize != NULL); break;
            default: ok = 0; break;
            }
            if (ok) { best = b; break; }
        }
        if (!best) {
            /* Fall through to scalar baseline */
            for (wubu_kernel_backend_t *b = g_backends; b; b = b->next) {
                if (b->id == WUBU_BACKEND_SCALAR) { best = b; break; }
            }
        }
    }

    switch (type) {
    case WUBU_KERN_GEMM: {
        wubu_gemm_fn fn = (best && best->gemm) ? best->gemm : g_cpu_gemm;
        if (!fn) { rc = -3; goto done; }
        const float *A = va_arg(args, const float *);
        const float *B = va_arg(args, const float *);
        float *C = va_arg(args, float *);
        int M = va_arg(args, int);
        int K = va_arg(args, int);
        int N = va_arg(args, int);
        double beta = va_arg(args, double);
        fn(A, B, C, M, K, N, (float)beta);
        break;
    }
    case WUBU_KERN_GEMV: {
        wubu_gemv_fn fn = (best && best->gemv) ? best->gemv : g_cpu_gemv;
        if (!fn) { rc = -3; goto done; }
        const float *A = va_arg(args, const float *);
        const float *x = va_arg(args, const float *);
        float *y = va_arg(args, float *);
        int M = va_arg(args, int);
        int K = va_arg(args, int);
        fn(A, x, y, M, K);
        break;
    }
    case WUBU_KERN_SOFTMAX: {
        wubu_softmax_fn fn = (best && best->softmax) ? best->softmax : g_cpu_softmax;
        if (!fn) { rc = -3; goto done; }
        float *logits = va_arg(args, float *);
        int M = va_arg(args, int);
        int N = va_arg(args, int);
        fn(logits, M, N);
        break;
    }
    case WUBU_KERN_LAYER_NORM: {
        wubu_rmsnorm_fn fn = (best && best->rmsnorm) ? best->rmsnorm : g_cpu_rmsnorm;
        if (!fn) { rc = -3; goto done; }
        float *x = va_arg(args, float *);
        const float *gamma = va_arg(args, const float *);
        const float *beta = va_arg(args, const float *);
        int M = va_arg(args, int);
        int d = va_arg(args, int);
        double eps = va_arg(args, double);
        fn(x, gamma, beta, M, d, (float)eps);
        break;
    }
    case WUBU_KERN_QUANT: {
        wubu_quantize_fn fn = (best && best->quantize) ? best->quantize : g_cpu_quantize;
        if (!fn) { rc = -3; goto done; }
        const float *fp32 = va_arg(args, const float *);
        int8_t *q = va_arg(args, int8_t *);
        float *scales = va_arg(args, float *);
        int M = va_arg(args, int);
        int K = va_arg(args, int);
        int bits = va_arg(args, int);
        fn(fp32, q, scales, M, K, bits);
        break;
    }
    case WUBU_KERN_DEQUANT: {
        wubu_dequantize_fn fn = (best && best->dequantize) ? best->dequantize : g_cpu_dequantize;
        if (!fn) { rc = -3; goto done; }
        const int8_t *q = va_arg(args, const int8_t *);
        const float *scales = va_arg(args, const float *);
        const float *zeros = va_arg(args, const float *);
        float *fp32 = va_arg(args, float *);
        int M = va_arg(args, int);
        int K = va_arg(args, int);
        int bits = va_arg(args, int);
        fn(q, scales, zeros, fp32, M, K, bits);
        break;
    }
    case WUBU_KERN_ATTN: {
        wubu_attn_fn fn = (best && best->attn) ? best->attn : g_cpu_attn;
        if (!fn) { rc = -3; goto done; }
        const float *Q = va_arg(args, const float *);
        const float *K = va_arg(args, const float *);
        const float *V = va_arg(args, const float *);
        float *out = va_arg(args, float *);
        int M = va_arg(args, int);
        int N = va_arg(args, int);
        int d = va_arg(args, int);
        int n_heads = va_arg(args, int);
        double scale = va_arg(args, double);
        fn(Q, K, V, out, M, N, d, n_heads, (float)scale);
        break;
    }
    case WUBU_KERN_ROPE: {
        wubu_rope_fn fn = (best && best->rope) ? best->rope : g_cpu_rope;
        if (!fn) { rc = -3; goto done; }
        float *q = va_arg(args, float *);
        float *k = va_arg(args, float *);
        int d = va_arg(args, int);
        int seq_len = va_arg(args, int);
        double theta = va_arg(args, double);
        int offset = va_arg(args, int);
        fn(q, k, d, seq_len, (float)theta, offset);
        break;
    }
    default:
        rc = -1;
        break;
    }

done:
    va_end(args);
    return rc;
}
/* CPU baseline public wrappers */
void wubu_kernel_gemm_scalar(const float *A, const float *B, float *C,
                                    int M, int K, int N, float beta) {
    cpu_gemm(A, B, C, M, K, N, beta);
}
void wubu_kernel_gemv_scalar(const float *A, const float *x, float *y,
                                    int M, int K) {
    cpu_gemv(A, x, y, M, K);
}
void wubu_kernel_softmax_scalar(float *logits, int M, int N) {
    cpu_softmax(logits, M, N);
}
void wubu_kernel_rmsnorm_scalar(float *x, const float *gamma,
                                       const float *beta, int M, int d,
                                       float eps) {
    cpu_rmsnorm(x, gamma, beta, M, d, eps);
}
void wubu_kernel_quantize_scalar(const float *fp32, int8_t *q,
                                        float *scales, int M, int K, int bits) {
    cpu_quantize(fp32, q, scales, M, K, bits);
}
void wubu_kernel_dequantize_scalar(const int8_t *q, const float *scales,
                                          const float *zeros, float *fp32,
                                          int M, int K, int bits) {
    cpu_dequantize(q, scales, zeros, fp32, M, K, bits);
}
void wubu_kernel_attention_scalar(const float *Q, const float *K, const float *V,
                                           float *out, int M, int N, int d, int n_heads,
                                           float scale) {
    cpu_attention(Q, K, V, out, M, N, d, n_heads, scale);
}
void wubu_kernel_rope_scalar(float *q, float *k, int d, int seq_len,
                                    float theta, int offset) {
    cpu_rope(q, k, d, seq_len, theta, offset);
}

/* ================================================================== */
/* CPU Feature Detection — Roofline-aware backend selection          */
/* ================================================================== */

#if defined(__x86_64__) || defined(_M_X64)
#  include <cpuid.h>
#endif
#include <unistd.h>  /* sysconf */

static int detect_avx2(void) {
#if defined(__x86_64__) || defined(_M_X64)
    unsigned eax, ebx, ecx, edx;
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    return (ebx >> 5) & 1;  /* AVX2 bit in EBX of leaf 7 subleaf 0 */
#else
    return 0;
#endif
}

static int detect_avx512(void) {
#if defined(__x86_64__) || defined(_M_X64)
    unsigned eax, ebx, ecx, edx;
    __cpuid(7, eax, ebx, ecx, edx);
    return (ebx >> 16) & 1;  /* AVX512F bit in EBX */
#else
    return 0;
#endif
}

static int detect_fma(void) {
#if defined(__x86_64__) || defined(_M_X64)
    unsigned eax, ebx, ecx, edx;
    __cpuid(1, eax, ebx, ecx, edx);
    return (ecx >> 12) & 1;  /* FMA bit in ECX */
#else
    return 0;
#endif
}

static int detect_n_cores(void) {
#if defined(_WIN32)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwNumberOfProcessors;
#elif defined(__APPLE__)
    int n = 1;
    size_t sz = sizeof(n);
    sysctlbyname("hw.logicalcpu", &n, &sz, NULL, 0);
    return n;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
#endif
}

static int detect_l1d_kb(void) {
#if defined(__linux__)
    FILE *f = fopen("/sys/devices/system/cpu/cpu0/cache/index0/size", "r");
    if (!f) return 32;  /* conservative default */
    char buf[64];
    if (!fgets(buf, sizeof(buf), f)) { fclose(f); return 32; }
    fclose(f);
    int val = 0;
    for (char *p = buf; *p && *p != 'K'; p++) {
        if (*p >= '0' && *p <= '9') val = val * 10 + (*p - '0');
    }
    return val > 0 ? val : 32;
#else
    return 32;
#endif
}

static int detect_l2_kb(void) {
#if defined(__linux__)
    FILE *f = fopen("/sys/devices/system/cpu/cpu0/cache/index1/size", "r");
    if (!f) return 256;
    char buf[64];
    if (!fgets(buf, sizeof(buf), f)) { fclose(f); return 256; }
    fclose(f);
    int val = 0;
    for (char *p = buf; *p && *p != 'K'; p++) {
        if (*p >= '0' && *p <= '9') val = val * 10 + (*p - '0');
    }
    return val > 0 ? val : 256;
#else
    return 256;
#endif
}

static int detect_l3_kb(void) {
#if defined(__linux__)
    FILE *f = fopen("/sys/devices/system/cpu/cpu0/cache/index2/size", "r");
    if (!f) return 0;  /* 0 means unknown/unified */
    char buf[64];
    if (!fgets(buf, sizeof(buf), f)) { fclose(f); return 0; }
    fclose(f);
    int val = 0;
    for (char *p = buf; *p && *p != 'K'; p++) {
        if (*p >= '0' && *p <= '9') val = val * 10 + (*p - '0');
    }
    return val;
#else
    return 0;
#endif
}

static float estimate_mem_bw_gbs(void) {
#if defined(__linux__)
    /* Estimate from /proc/meminfo total RAM and typical DDR5 bandwidth.
     * For WSL2 substrate: 6P cores, ~13GB RAM, DDR5-4800.
     * Theoretical peak per channel: 38.4 GB/s.
     * Single-channel WSL2: ~25 GB/s practical. */
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 25.0f;
    char buf[256];
    long total_kb = 0;
    while (fgets(buf, sizeof(buf), f)) {
        if (strncmp(buf, "MemTotal:", 9) == 0) {
            sscanf(buf + 9, "%ld", &total_kb);
            break;
        }
    }
    fclose(f);
    /* Rough model: dual-channel DDR5-4800 = 76.8 GB/s theoretical,
     * ~50 GB/s practical on WSL2. Scale with RAM to detect if
     * we're on a bigger machine. */
    if (total_kb > 32L * 1024 * 1024) return 60.0f;  /* >32GB = big machine */
    return 50.0f;  /* WSL2 ~13GB, DDR5 */
#else
    return 25.0f;
#endif
}

const wubu_cpu_features_t wubu_cpu_features = {0};

int wubu_cpu_detect(wubu_cpu_features_t *cpu) {
    if (!cpu) return -1;
    cpu->has_avx2    = detect_avx2();
    cpu->has_avx512  = detect_avx512();
    cpu->has_fma     = detect_fma();
    cpu->l1d_kb      = detect_l1d_kb();
    cpu->l2_kb       = detect_l2_kb();
    cpu->l3_kb       = detect_l3_kb();
    cpu->n_cores     = detect_n_cores();
    cpu->mem_bw_gbs  = estimate_mem_bw_gbs();
    return 0;
}

/* Auto-select the best kernel backend based on detected hardware.
 * Called at init if no device backend has been explicitly registered. */
wubu_backend_id_t wubu_kernel_auto_select(wubu_kernel_type_t type) {
    (void)type;
    wubu_cpu_features_t cpu;
    wubu_cpu_detect(&cpu);

    /* WSL2-specific tuning: AVX2 is always present on x86_64 WSL2 */
    if (cpu.has_avx2 && cpu.has_fma) {
        /* Use AVX2-FMA backend for compute-bound kernels */
        return WUBU_BACKEND_CPU_SIMD;
    }
    return WUBU_BACKEND_SCALAR;
}
