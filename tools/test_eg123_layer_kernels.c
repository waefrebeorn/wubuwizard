/* test_eg123_layer_kernels.c -- EG-123: device RMS norm / residual / RoPE vs the CPU.
 *
 * These are the building blocks for a device-resident layer, so each one is compared
 * element-by-element against the CPU routine it replaces (g4_rms_norm, the residual add,
 * g4_rope_apply). A device layer is only worth having if moving the arithmetic onto the
 * card does not change it.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

typedef int (*fn_rms_t)(const void *, const void *, float *, int, int, float, int);
typedef int (*fn_add_t)(const void *, const void *, float *, int);
typedef int (*fn_rope_t)(const void *, const void *, float *, float *, float, int, int, int,
                         int, float, const void *, int, int, int);
typedef int (*fn_attn_t)(const void *, const void *, const void *, float *, int, int, int,
                         int, int, int, int);

static uint64_t rng = 0x5EED1234ULL;
static float rnd(void)
{
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return ((float)(int32_t)(rng >> 32) / 2147483648.0f);
}

/* Straight transcript of g4_rms_norm() from src/wubu_gemma4_model.c. */
static void cpu_rms_norm(const float *x, const float *w, int n, float eps, float *out, int has_w)
{
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    const float inv = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) out[i] = has_w ? x[i] * inv * w[i] : x[i] * inv;
}

/* Transcript of the Q loop in g4_rope_apply(). */
static void cpu_rope(float *q, int d_head, int q_stride, int n_rot, float pos,
                     float theta_scale, const float *freqs)
{
    const int half = n_rot / 2;
    for (int h = 0; h < q_stride; h += d_head) {
        float *qv = q + h;
        for (int j = 0; j < half; j++) {
            const float ff = freqs ? freqs[j] : 1.0f;
            if (ff > 1e20f) continue;
            const float theta = pos * powf(theta_scale, (float)j) / ff;
            const float ct = cosf(theta), st = sinf(theta);
            const float q0 = qv[j], q1 = qv[j + half];
            qv[j] = q0 * ct - q1 * st;
            qv[j + half] = q0 * st + q1 * ct;
        }
    }
}

int main(int argc, char **argv)
{
    HMODULE h = LoadLibraryA(argc > 1 ? argv[1] : "wubu_cuda.dll");
    if (!h) { printf("test_eg123: SKIP (no dll)\n"); return 0; }
    fn_rms_t rms = (fn_rms_t)(void *)GetProcAddress(h, "wubu_cuda_rms_norm_host");
    fn_add_t add = (fn_add_t)(void *)GetProcAddress(h, "wubu_cuda_residual_add_host");
    fn_rope_t rope = (fn_rope_t)(void *)GetProcAddress(h, "wubu_cuda_rope_host");
    CHECK(rms && add && rope);
    if (!rms || !add || !rope) { FreeLibrary(h); return 1; }

    /* ---- RMS norm: hidden-shaped batch, with and without a weight ---- */
    {
        enum { ROWS = 6, N = 3840 };
        static float x[ROWS][N], w[N], ref[ROWS][N], got[ROWS][N];
        for (int i = 0; i < N; ++i) { w[i] = 1.0f + 0.1f * rnd(); x[0][i] = rnd(); }
        for (int r = 1; r < ROWS; ++r) for (int i = 0; i < N; ++i) x[r][i] = rnd();
        for (int r = 0; r < ROWS; ++r) cpu_rms_norm(x[r], w, N, 1e-6f, ref[r], 1);
        CHECK(rms(x, w, &got[0][0], ROWS, N, 1e-6f, 1) == 0);
        double worst = 0.0;
        for (int r = 0; r < ROWS; ++r)
            for (int i = 0; i < N; ++i) {
                const double d = fabs((double)got[r][i] - (double)ref[r][i]);
                if (d > worst) worst = d;
            }
        printf("EG-123 rms_norm weighted: rows=%d worst_abs=%.3e\n", ROWS, worst);
        CHECK(worst < 1e-4);

        for (int r = 0; r < ROWS; ++r) cpu_rms_norm(x[r], NULL, N, 1e-6f, ref[r], 0);
        CHECK(rms(x, NULL, &got[0][0], ROWS, N, 1e-6f, 0) == 0);
        worst = 0.0;
        for (int r = 0; r < ROWS; ++r)
            for (int i = 0; i < N; ++i) {
                const double d = fabs((double)got[r][i] - (double)ref[r][i]);
                if (d > worst) worst = d;
            }
        printf("EG-123 rms_norm unweighted: rows=%d worst_abs=%.3e\n", ROWS, worst);
        CHECK(worst < 1e-4);
    }

    /* ---- residual add ---- */
    {
        enum { N2 = 3840 };
        static float a[N2], b[N2], out[N2];
        for (int i = 0; i < N2; ++i) { a[i] = rnd(); b[i] = rnd(); }
        CHECK(add(a, b, out, N2) == 0);
        double worst = 0.0;
        for (int i = 0; i < N2; ++i) {
            const double d = fabs((double)out[i] - ((double)a[i] + (double)b[i]));
            if (d > worst) worst = d;
        }
        printf("EG-123 residual_add: worst_abs=%.3e\n", worst);
        CHECK(worst < 1e-6);
    }

    /* ---- RoPE on a 16-head Q row with the model's rope parameters ---- */
    {
        enum { HEADS = 16, HD = 256, QD = HEADS * HD, NROT = 256 };
        static float q[QD], qr[QD], qg[QD], freqs[NROT / 2];
        const float base = 1000000.0f;
        const float theta_scale = powf(base, -2.0f / (float)NROT);
        for (int i = 0; i < QD; ++i) q[i] = rnd();
        for (int i = 0; i < NROT / 2; ++i) freqs[i] = 1.0f + 0.01f * rnd();
        memcpy(qr, q, sizeof(q));
        cpu_rope(qr, HD, QD, NROT, 37.0f, theta_scale, freqs);
        CHECK(rope(q, NULL, qg, NULL, 37.0f, HD, QD, 0, NROT, theta_scale,
                   freqs, HEADS, 0, 1) == 0);
        double worst = 0.0;
        for (int i = 0; i < QD; ++i) {
            const double d = fabs((double)qg[i] - (double)qr[i]);
            if (d > worst) worst = d;
        }
        printf("EG-123 rope: heads=%d head_dim=%d worst_abs=%.3e\n", HEADS, HD, worst);
        CHECK(worst < 1e-4);
    }

    /* ---- decode attention vs the sliding-attention CPU routine ---- */
    {
        fn_attn_t attn = (fn_attn_t)(void *)GetProcAddress(h, "wubu_cuda_attn_host");
        CHECK(attn != NULL);
        if (attn) {
            enum { HEADS = 8, KVH = 4, HD = 64, KVD = KVH * HD, KCAP = 32, KVLEN = 20 };
            static float q[HEADS * HD], kc[KCAP * KVD], vc[KCAP * KVD];
            static float ref[HEADS * HD], got[HEADS * HD];
            for (int i = 0; i < HEADS * HD; ++i) q[i] = rnd();
            for (int i = 0; i < KCAP * KVD; ++i) { kc[i] = rnd(); vc[i] = rnd(); }

            /* CPU transcript of g4_sliding_attn_cpu for one position, kv_start = 0. */
            for (int hq = 0; hq < HEADS; ++hq) {
                const int kv_h = hq / (HEADS / KVH);
                const float *qvec = q + hq * HD;
                float s[KVLEN], mx = -3.4e38f;
                for (int kp = 0; kp < KVLEN; ++kp) {
                    const float *kv = kc + (size_t)kp * KVD + (size_t)kv_h * HD;
                    float sc = 0.0f;
                    for (int d = 0; d < HD; ++d) sc += qvec[d] * kv[d];
                    s[kp] = sc;
                    if (sc > mx) mx = sc;
                }
                float sum = 0.0f;
                for (int kp = 0; kp < KVLEN; ++kp) { s[kp] = expf(s[kp] - mx); sum += s[kp]; }
                const float inv = 1.0f / (sum + 1e-10f);
                for (int d = 0; d < HD; ++d) {
                    float acc = 0.0f;
                    for (int kp = 0; kp < KVLEN; ++kp)
                        acc += s[kp] * inv * vc[(size_t)kp * KVD + (size_t)kv_h * HD + d];
                    ref[hq * HD + d] = acc;
                }
            }
            CHECK(attn(q, kc, vc, got, HEADS, KVH, HD, 0, KVLEN, KVD, KCAP) == 0);
            double worst = 0.0;
            for (int i = 0; i < HEADS * HD; ++i) {
                const double dd = fabs((double)got[i] - (double)ref[i]);
                if (dd > worst) worst = dd;
            }
            printf("EG-123 attn: heads=%d kv_heads=%d kv_len=%d worst_abs=%.3e\n",
                   HEADS, KVH, KVLEN, worst);
            CHECK(worst < 1e-4);
        }
    }

    FreeLibrary(h);
    if (failures) { fprintf(stderr, "test_eg123_layer_kernels: FAIL (%d)\n", failures); return 1; }
    printf("test_eg123_layer_kernels: PASS\n");
    return 0;
}
