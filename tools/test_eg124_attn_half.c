/* test_eg124_attn_half.c -- EG-124: the composite attention half-layer vs a CPU sequence.
 *
 * Everything else about the composite is already gated piece by piece (EG-123: RMS norm,
 * residual, RoPE, attention, KV round trip; EG-111/120/121: the quantised dots). This gate
 * checks the ASSEMBLY: same synthetic IQ2_S weights, same activation, same KV contents,
 * one submission on the device against the step-by-step CPU sequence. A single wrong
 * column stops the work.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>

#include "gguf_reader.h"
#include "wubu_vecdot_iq.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

typedef int (*fn_half_t)(const void *, const void *, const void *, const void *, int,
                         const void *, const void *, const void *, const void *,
                         const float *, const float *, int, int,
                         const void *, const void *,
                         float *, float *, float *, float *, float *,
                         int, int, int, int, int,
                         int, int, int, int, float);

static uint64_t rng = 0xA77E5EEDULL;
static float rnd(void)
{
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return ((float)(int32_t)(rng >> 32) / 2147483648.0f);
}

enum {
    HIDDEN = 256, HD = 64, NHEADS = 4, QD = NHEADS * HD,
    NKV = 2, KVD = NKV * HD, NROT = 64, KCAP = 16, POS = 4, KVLEN = POS + 1
};

static uint8_t wq[QD * 82], wk[KVD * 82], wv[KVD * 82], wo[HIDDEN * 82];
static float x[HIDDEN], qn[HD], kn[HD], freqs[NROT / 2];
static float kc[KCAP * KVD], vc[KCAP * KVD];

static void fill_block(uint8_t *blk)
{
    /* Advance the xorshift per byte. The first version read (rng >> 32) without advancing,
     * so every block was identical and the o-projection produced the same value for every
     * column - which looked like a matvec bug for a while. */
    for (int i = 0; i < 82; ++i) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        blk[i] = (uint8_t)(rng >> 32);
    }
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    const uint16_t d16 = (uint16_t)(0x2C00u + (rng & 0x3FFu));   /* fp16 ~0.03..0.07 */
    memcpy(blk, &d16, 2);
}

/* One column's dot against a Q8_K row, using the gated CPU port. */
static float cpu_col(const uint8_t *w, int col, const block_q8_K *q8)
{
    float s = 0.0f;
    if (wubu_vec_dot_iq2_s_q8_k_scalar(HIDDEN, &s, w + (size_t)col * 82, q8) != 0) s = NAN;
    return s;
}

static void cpu_rms(const float *in, const float *w, int n, float *out, int has_w)
{
    float ss = 0.0f;
    for (int i = 0; i < n; ++i) ss += in[i] * in[i];
    const float inv = 1.0f / sqrtf(ss / (float)n + 1e-6f);
    for (int i = 0; i < n; ++i) out[i] = has_w ? in[i] * inv * w[i] : in[i] * inv;
}

static void cpu_rope(float *q, float *k, float pos, float theta_scale)
{
    const int half = NROT / 2;
    for (int h = 0; h < NHEADS; ++h) {
        float *qv = q + h * HD;
        for (int j = 0; j < half; ++j) {
            const float ff = freqs[j];
            if (ff > 1e20f) continue;
            const float th = pos * powf(theta_scale, (float)j) / ff;
            const float ct = cosf(th), st = sinf(th);
            const float q0 = qv[j], q1 = qv[j + half];
            qv[j] = q0 * ct - q1 * st;
            qv[j + half] = q0 * st + q1 * ct;
        }
    }
    for (int h = 0; h < NKV; ++h) {
        float *kv = k + h * HD;
        for (int j = 0; j < half; ++j) {
            const float ff = freqs[j];
            if (ff > 1e20f) continue;
            const float th = pos * powf(theta_scale, (float)j) / ff;
            const float ct = cosf(th), st = sinf(th);
            const float k0 = kv[j], k1 = kv[j + half];
            kv[j] = k0 * ct - k1 * st;
            kv[j + half] = k0 * st + k1 * ct;
        }
    }
}

int main(int argc, char **argv)
{
    HMODULE h = LoadLibraryA(argc > 1 ? argv[1] : "wubu_cuda.dll");
    if (!h) { printf("test_eg124: SKIP (no dll)\n"); return 0; }
    fn_half_t half = (fn_half_t)(void *)GetProcAddress(h, "wubu_cuda_attn_half_host");
    CHECK(half != NULL);
    if (!half) { FreeLibrary(h); return 1; }
    /* The IQ2_S grid must be on the device or every dot returns exactly 0 and the whole
     * comparison reads as a kernel that "ran fine". */
    {
        typedef int (*fn_tab_t)(const void *, const void *);
        fn_tab_t tab = (fn_tab_t)(void *)GetProcAddress(h, "wubu_cuda_upload_tables");
        extern const void *wubu_vecdot_iq2s_grid(void);
        extern const void *wubu_vecdot_iq3s_grid(void);
        CHECK(tab != NULL);
        if (tab) CHECK(tab(wubu_vecdot_iq2s_grid(), wubu_vecdot_iq3s_grid()) == 0);
    }

    for (int c = 0; c < QD; ++c) fill_block(wq + c * 82);
    for (int c = 0; c < KVD; ++c) { fill_block(wk + c * 82); fill_block(wv + c * 82); }
    for (int c = 0; c < HIDDEN; ++c) fill_block(wo + c * 82);
    for (int i = 0; i < HIDDEN; ++i) x[i] = rnd();
    for (int i = 0; i < HD; ++i) { qn[i] = 1.0f + 0.2f * rnd(); kn[i] = 1.0f + 0.2f * rnd(); }
    for (int i = 0; i < NROT / 2; ++i) freqs[i] = 1.0f + 0.01f * rnd();
    for (int i = 0; i < POS * KVD; ++i) { kc[i] = rnd(); vc[i] = rnd(); }

    const float theta_scale = powf(4096.0f, -2.0f / (float)NROT);

    static float dq_[QD], dk_[KVD], dv_[KVD], dattn_[QD], dout_[HIDDEN];
    CHECK(half(wq, wk, wv, wo, 22, x, qn, kn, freqs, kc, vc, KVD, KCAP, NULL, NULL,
               dq_, dk_, dv_, dattn_, dout_, HIDDEN, QD, HD, NHEADS, NKV,
               POS, 0, KVLEN, NROT, theta_scale) == 0);

    /* ---- CPU sequence, same inputs ---- */
    static block_q8_K q8;
    quantize_row_q8_K(x, &q8, HIDDEN);

    float cq[QD], ck[KVD], cv[KVD], cattn[QD], cout[HIDDEN];
    for (int c = 0; c < QD; ++c) cq[c] = cpu_col(wq, c, &q8);
    for (int c = 0; c < KVD; ++c) { ck[c] = cpu_col(wk, c, &q8); cv[c] = cpu_col(wv, c, &q8); }

    float tq[HD];
    for (int hd_i = 0; hd_i < NHEADS; ++hd_i) cpu_rms(cq + hd_i * HD, qn, HD, cq + hd_i * HD, 1);
    for (int hd_i = 0; hd_i < NKV; ++hd_i) {
        cpu_rms(ck + hd_i * HD, kn, HD, ck + hd_i * HD, 1);
        cpu_rms(cv + hd_i * HD, NULL, HD, cv + hd_i * HD, 0);
    }
    (void)tq;
    cpu_rope(cq, ck, (float)POS, theta_scale);

    double wq_ = 0.0, wk_ = 0.0, wv_ = 0.0;
    for (int i = 0; i < QD; ++i) { const double d = fabs((double)dq_[i] - (double)cq[i]); if (d > wq_) wq_ = d; }
    for (int i = 0; i < KVD; ++i) {
        double d = fabs((double)dk_[i] - (double)ck[i]); if (d > wk_) wk_ = d;
        d = fabs((double)dv_[i] - (double)cv[i]);        if (d > wv_) wv_ = d;
    }

    /* store the current token, then attend over rows [0, KVLEN) */
    memcpy(kc + (size_t)POS * KVD, ck, sizeof(ck));
    memcpy(vc + (size_t)POS * KVD, cv, sizeof(cv));
    static float sc[KVLEN];
    for (int hq = 0; hq < NHEADS; ++hq) {
        const int kv_h = hq / (NHEADS / NKV);
        float mx = -3.4e38f;
        for (int kp = 0; kp < KVLEN; ++kp) {
            const float *kvp = kc + (size_t)kp * KVD + (size_t)kv_h * HD;
            float s = 0.0f;
            for (int d = 0; d < HD; ++d) s += cq[hq * HD + d] * kvp[d];
            sc[kp] = s;
            if (s > mx) mx = s;
        }
        float sum = 0.0f;
        for (int kp = 0; kp < KVLEN; ++kp) { sc[kp] = expf(sc[kp] - mx); sum += sc[kp]; }
        const float inv = 1.0f / (sum + 1e-10f);
        for (int d = 0; d < HD; ++d) {
            float acc = 0.0f;
            for (int kp = 0; kp < KVLEN; ++kp)
                acc += sc[kp] * inv * vc[(size_t)kp * KVD + (size_t)kv_h * HD + d];
            cattn[hq * HD + d] = acc;
        }
    }
    double wa = 0.0;
    for (int i = 0; i < QD; ++i) { const double d = fabs((double)dattn_[i] - (double)cattn[i]); if (d > wa) wa = d; }

    /* Feed the DEVICE attention into the CPU o-projection reference: the composite
     * quantises the attention it just computed, so comparing against a CPU attention that
     * differs by 1e-7 would measure the rounding of the second quantisation, not the
     * o-projection stage. */
    static block_q8_K q8o;
    quantize_row_q8_K(dattn_, &q8o, QD);
    for (int c = 0; c < HIDDEN; ++c) {
        float s = 0.0f;
        wubu_vec_dot_iq2_s_q8_k_scalar(QD, &s, wo + (size_t)c * 82, &q8o);
        cout[c] = s;
    }
    double wo_ = 0.0;
    for (int i = 0; i < HIDDEN; ++i) { const double d = fabs((double)dout_[i] - (double)cout[i]); if (d > wo_) wo_ = d; }

    printf("EG-124 attn_half: q=%.3e k=%.3e v=%.3e attn=%.3e out=%.3e\n", wq_, wk_, wv_, wa, wo_);
    printf("EG-124 q[0..5] dev:");
    for (int i = 0; i < 6; ++i) printf(" %.4f", dq_[i]);
    printf("\nEG-124 q[0..5] cpu:");
    for (int i = 0; i < 6; ++i) printf(" %.4f", cq[i]);
    printf("\nEG-124 k[0..5] dev:");
    for (int i = 0; i < 6; ++i) printf(" %.4f", dk_[i]);
    printf("\nEG-124 k[0..5] cpu:");
    for (int i = 0; i < 6; ++i) printf(" %.4f", ck[i]);
    printf("\nout[0..5] dev:");
    for (int i = 0; i < 6; ++i) printf(" %.4f", dout_[i]);
    printf("\nout[0..5] cpu:");
    for (int i = 0; i < 6; ++i) printf(" %.4f", cout[i]);
    printf("\nattn[0..3] dev:");
    for (int i = 0; i < 4; ++i) printf(" %.5f", dattn_[i]);
    printf("\n");
    CHECK(wq_ < 1e-4);
    CHECK(wk_ < 1e-4);
    CHECK(wv_ < 1e-4);
    CHECK(wa < 1e-3);
    CHECK(wo_ < 1e-3);

    FreeLibrary(h);
    if (failures) { fprintf(stderr, "test_eg124_attn_half: FAIL (%d)\n", failures); return 1; }
    printf("test_eg124_attn_half: PASS\n");
    return 0;
}
