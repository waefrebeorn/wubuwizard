/* wubu_sd_unet.c -- LDM UNet forward for the WuBu stable-diffusion engine.
 * SD 1.5 topology (Anything-V5 / DreamShaper8):
 *   in 4ch -> 320 -> 640 -> 1280 -> 1280, cross-attn vs CLIP 77x768.
 *   ResBlock: GroupNorm(32) + SiLU + 3x3, time-emb Linear, skip 1x1.
 *   SpatialTransformer: LN + self-attn + cross-attn + GEGLU FFN.
 *   Downsample 3x3 s2, Upsample nearest2x + 3x3.
 * Weights stay quantized in the GGUF blob; per-tensor F32 lazy cache
 * (same pattern as wubu_sd_clip). C11, self-contained.
 */
#include "wubu_sd_unet.h"
#include "wubu_sd_ops.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif
#include <math.h>
#include <omp.h>

#define U_CH 320          /* model_channels */
#define U_TIME 1280       /* time_embed_dim = 320*4 */
#define U_CTX 768         /* CLIP context dim */
#define U_CTX_LEN 77
#define U_HEADS 8

/* DEBUG HOOK (cfg_diff): capture attn2 outputs for cond vs uncond. */
int g_attn2_capture = 0;
char g_attn2_prefix[128] = {0};
float g_attn2_out[3 * 1280 * 4096];
int g_attn2_slot = 0;
int g_attn2_slot2 = -1;
/* stage capture: 0=off, 1=middle-block input, 2=after middle transformer,
 * 3=final pre-conv activations (out.0 gn out), 4=out.2 conv out (eps),
 * 5=output block N result (g_stage_block), 6=after output block transformer */
#include <time.h>
/* perf counters: 0=conv 1=linear 2=gn 3=silu 4=attn 5=ffn 6=lookup 7=other */
double g_t[8];
int g_timing = 0;
static double t_now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9; }

int g_stage_capture = 0;
int g_stage_block = 0;
int g_stage_count = 0;
float g_stage_out[5][1280 * 64 * 64];

/* per-block channels (SD1.5): block -> in_channels */
/* input blocks (SD1.5 GGUF ground truth):
 *   0: conv 4->320; 1: res320+attn; 2: res320+attn; 3: DOWN;
 *   4: res640+attn; 5: res640+attn; 6: DOWN; 7: res1280+attn; 8: res1280+attn;
 *   9: DOWN; 10: res1280; 11: res1280. */
static const int IN_TYPE[12] = { /* 0=conv,1=res+attn,2=res,3=down */
    0, 1, 1, 3, 1, 1, 3, 1, 1, 3, 2, 2 };
/* channel after each input block (the skip buffer's channels) */
static const int IN_OUT_CH[12] = { 320, 320, 320, 320, 640, 640, 640, 1280, 1280, 1280, 1280, 1280 };

/* output blocks: prev_ch (cur channels), skip_ch (from IN_OUT_CH[skip]),
 * out_ch, attn, up (subblock index of the upsample conv: 1 for block 2,
 * 2 for blocks 5/8 — GGUF naming varies!), skip-from-input-block index. */
typedef struct { int prev_ch; int skip_ch; int out_ch; int attn; int up; int skip; } out_cfg_t;
static const out_cfg_t OUT_CFG[12] = {
    {1280, 1280, 1280, 0, 0, 11}, {1280, 1280, 1280, 0, 0, 10}, {1280, 1280, 1280, 0, 1, 9},
    {1280, 1280, 1280, 1, 0, 8},  {1280, 1280, 1280, 1, 0, 7},  {1280, 640, 1280, 1, 2, 6},
    {1280, 640, 640, 1, 0, 5},    {640,  640,  640, 1, 0, 4},   {640,  320, 640, 1, 2, 3},
    {640,  320, 320, 1, 0, 2},    {320,  320,  320, 1, 0, 1},   {320,  320, 320, 1, 0, 0},
};

struct wubu_sd_unet {
    gguf_ctx *gguf;
    /* no F32 weight cache — weights stay in the mmap'd blob */
    /* DeepCache (SD_DEEPCACHE=1): cached 64x64 feature from the last
     * full pass (output of output_blocks.10, 320ch). Two slots: CFG
     * cond/uncond passes have different deep features. On cached steps
     * the UNet runs conv_in + input_blocks.1 + output_blocks.11 +
     * conv_out only (paper: 2.3x, 0.05 CLIP drop). */
    float *dc_feat[2];   /* [pass] 320*64*64 cached deep feature */
    int dc_valid;
    int dc_interval;     /* full pass every N steps (default 5) */
    int dc_step;         /* current step counter (set per step) */
    int dc_pass;         /* 0=cond, 1=uncond (set per call) */
    int *dc_sched_full;  /* per-step full-pass map (1=full), or NULL */
    int dc_sched_n;      /* number of steps in the map */
    int lat_h, lat_w;    /* latent spatial dims (default 64x64) */
};

/* raw (quantized, never-dequantized) weight descriptor */
typedef struct {
    const void *ptr;
    int type;
    int64_t K, N;      /* dims[0] (contraction), dims[1] (columns) */
    int64_t n_elems;
    int found;
} unet_raw_t;

/* ---------------- quantized weight access (no F32 cache) ---------------- */
static unet_raw_t unet_get_raw(wubu_sd_unet_t *u, const char *name) {
    unet_raw_t r = {0};
    gguf_tensor_info *ti = gguf_find_tensor(u->gguf, name);
    if (!ti) { fprintf(stderr, "UNET: missing %s\n", name); return r; }
    r.ptr = (const uint8_t *)u->gguf->data_blob + ti->data_offset;
    r.type = ti->ggml_type;
    r.n_elems = 1;
    for (int d = 0; d < ti->n_dims; d++) r.n_elems *= ti->dims[d];
    r.K = ti->n_dims > 0 ? ti->dims[0] : 1;
    r.N = ti->n_dims > 1 ? ti->dims[1] : 1;
    r.found = 1;
    return r;
}

/* linear helper: out[M][N] = x[M][K] @ W^T[K][N] + b[N], W raw blob.
 * W is column-major [K][N] (ggml dims[0]=K innermost). */
static void unet_linear(const void *W, int wtype, const float *b, int N, int K,
                        const float *x, int M, float *out) {
    double t0 = g_timing ? t_now() : 0;
    wubu_sd_linear_qb(x, W, wtype, b, M, K, N, out);
    if (g_timing) {
        double dt = t_now() - t0;
        g_t[1] += dt;
        if (dt > 0.01)
            fprintf(stderr, "LIN %dx%d->%d: %.1fms\n", M, K, N, dt * 1e3);
    }
}

/* tiny F32 read for 1D params (biases, layernorm) — a few KB each,
 * never the big matmul weights. No cache (weights stay in the blob). */
static float *unet_get_f32(wubu_sd_unet_t *u, const char *name) {
    gguf_tensor_info *ti = gguf_find_tensor(u->gguf, name);
    if (!ti) { fprintf(stderr, "UNET: missing %s\n", name); return NULL; }
    int64_t ne = 1;
    for (int d = 0; d < ti->n_dims; d++) ne *= ti->dims[d];
    float *f = (float *)malloc((size_t)ne * sizeof(float));
    if (!f) return NULL;
    if (gguf_read_tensor_f32(u->gguf, ti, f, ne) != (int)ne) { free(f); return NULL; }
    return f;
}

/* quantized conv2d helper: fetch raw weight, run conv2d_q.
 * us>0 = fused nearest-upsample factor (see wubu_sd_conv2d_q). */
static int unet_conv_q(wubu_sd_unet_t *u, const char *wname, const char *bname,
                       const float *x, int C_in, int H, int W,
                       int C_out, int KH, int KW, int stride,
                       int pad_h, int pad_w, float *y, int us) {
    unet_raw_t w = unet_get_raw(u, wname);
    if (!w.found) return -1;
    float *b = bname ? unet_get_f32(u, bname) : NULL;
    double t0 = g_timing ? t_now() : 0;
    wubu_sd_conv2d_q(x, 1, C_in, H, W, w.ptr, w.type, b, C_out, KH, KW,
                     stride, pad_h, pad_w, y, NULL, NULL, us);
    if (g_timing) {
        double dt = t_now() - t0;
        g_t[0] += dt;
        if (dt > 0.005)
            fprintf(stderr, "CONV %s %dx%d %d->%d s%d p%d: %.1fms\n",
                    wname, H, W, C_in, C_out, stride, pad_h, dt * 1e3);
    }
    free(b);
    return 0;
}

#if defined(__ARM_NEON) && defined(__aarch64__)
/* NEON exp2: 2^x = 2^i * 2^f, i = round(x), f in [-0.5,0.5].
 * 2^i via exponent insertion (exact); 2^f = e^(f*ln2) via degree-8
 * Horner (error ~9e-9 = sub-ULP vs F32 near 1.0). NOTE: the FEXPA
 * hardware instruction is SVE-only (ARMv8.2+) — A72 has no SVE, so
 * this poly is the fast path here (~10 NEON ops per 4 elements vs a
 * libm exp2f call per element). llama.cpp PR 7154 pattern. */
static inline float32x4_t exp2q_f32(float32x4_t x) {
    float32x4_t i = vrndnq_f32(x);                    /* nearest int */
    float32x4_t f = vsubq_f32(x, i);                  /* [-0.5,0.5] */
    int32x4_t ie = vcvtq_s32_f32(i);
    ie = vshlq_n_s32(vaddq_s32(ie, vdupq_n_s32(127)), 23);
    float32x4_t pow2i = vreinterpretq_f32_s32(ie);    /* exact 2^i */
    float32x4_t u = vmulq_f32(f, vdupq_n_f32(0.6931471805599453f)); /* f*ln2 */
    float32x4_t p = vdupq_n_f32(2.4801587301587302e-05f); /* 1/8! */
    p = vfmaq_f32(vdupq_n_f32(1.984126984126984e-04f), p, u); /* 1/7! */
    p = vfmaq_f32(vdupq_n_f32(1.3888888888888888e-03f), p, u); /* 1/6! */
    p = vfmaq_f32(vdupq_n_f32(8.3333333333333332e-03f), p, u); /* 1/5! */
    p = vfmaq_f32(vdupq_n_f32(4.1666666666666666e-02f), p, u); /* 1/4! */
    p = vfmaq_f32(vdupq_n_f32(1.6666666666666666e-01f), p, u); /* 1/3! */
    p = vfmaq_f32(vdupq_n_f32(0.5f), p, u);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, u);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, u);           /* e^u */
    return vmulq_f32(pow2i, p);
}
#endif /* __ARM_NEON && __aarch64__ */

/* silu in place — exp2(x*log2e). On AArch64 the NEON path uses the
 * polynomial exp2q_f32 above (~10 vector ops / 4 elems vs a libm
 * exp2f call per element). llama.cpp PR 7154: 2x on SiLU/SoftMax.
 * Fallback: scalar exp2f (x86, non-NEON ARM). */
static void unet_silu(float *x, int n) {
    double t0 = g_timing ? t_now() : 0;
    const float log2e = 1.4426950408889634f;
#if defined(__ARM_NEON) && defined(__aarch64__)
    const float32x4_t le = vdupq_n_f32(log2e), one = vdupq_n_f32(1.0f);
    int i = 0;
    #pragma omp parallel for lastprivate(i)
    for (i = 0; i <= n - 8; i += 8) {
        float32x4_t s0 = vld1q_f32(x + i), s1 = vld1q_f32(x + i + 4);
        float32x4_t e0 = exp2q_f32(vmulq_f32(vnegq_f32(s0), le));
        float32x4_t e1 = exp2q_f32(vmulq_f32(vnegq_f32(s1), le));
        vst1q_f32(x + i,     vdivq_f32(s0, vaddq_f32(one, e0)));
        vst1q_f32(x + i + 4, vdivq_f32(s1, vaddq_f32(one, e1)));
    }
    #pragma omp parallel for
    for (int j = i; j < n; j++) {
        float s = x[j];
        x[j] = s / (1.0f + exp2f(-s * log2e));
    }
#else
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        float s = x[i];
        x[i] = s / (1.0f + exp2f(-s * log2e));
    }
#endif
    if (g_timing) g_t[3] += t_now() - t0;
}

/* groupnorm in place, G=32 */
static void unet_gn(float *x, int C, int HW, const float *g, const float *b) {
    double t0 = g_timing ? t_now() : 0;
    wubu_sd_groupnorm(x, 1, C, 1, HW, 32, 1e-6f, g, b, x);
    if (g_timing) g_t[2] += t_now() - t0;
}

/* timestep embedding: sinusoidal 320 -> Linear -> silu -> Linear -> 1280 */
static void unet_time_embed(wubu_sd_unet_t *u, int t, float *out /*[1280]*/) {
    float emb[U_CH];
    /* sinusoidal embedding (half = 160), matches diffusers get_timestep_embedding
     * with flip_sin_to_cos=TRUE (SD1.5 default): COS in the first half,
     * SIN in the second. [sin,cos] order would feed the MLP wrong inputs. */
    const int half = U_CH / 2;
    for (int i = 0; i < half; i++) {
        float w = expf(-logf(10000.0f) * (float)i / (float)half);
        emb[i] = cosf((float)t * w);
        emb[i + half] = sinf((float)t * w);
    }
    unet_raw_t l1 = unet_get_raw(u, "model.diffusion_model.time_embed.0.weight");
    float *l1b = unet_get_f32(u, "model.diffusion_model.time_embed.0.bias");
    unet_raw_t l2 = unet_get_raw(u, "model.diffusion_model.time_embed.2.weight");
    float *l2b = unet_get_f32(u, "model.diffusion_model.time_embed.2.bias");
    if (!l1.found || !l2.found) { fprintf(stderr, "time_embed missing\n"); return; }
    float mid[U_TIME];
    unet_linear(l1.ptr, l1.type, l1b, U_TIME, U_CH, emb, 1, mid);
    unet_silu(mid, U_TIME);
    unet_linear(l2.ptr, l2.type, l2b, U_TIME, U_TIME, mid, 1, out);
    free(l1b); free(l2b);
}

/* ResBlock: x [C][H][W] + temb[1280] -> out [Cout][H][W].
 * names: model.diffusion_model.<prefix>.0.*  (prefix = input_blocks.N / etc.) */
static int unet_resblock(wubu_sd_unet_t *u, const char *prefix,
                         float *x, int C, int Cout, int H, int W,
                         const float *temb, float *out) {
    char nm[256];
    float *scratch = (float *)malloc((size_t)Cout * H * W * sizeof(float));
    if (!scratch) return -1;
    int HW = H * W;

    /* CRITICAL: x is the caller's `cur` buffer (skips[i-1]) — MUST NOT be
     * modified in place. The skip connection adds the ORIGINAL x, and the
     * caller still needs it for an output block. Copy to a norm buffer. */
    float *xn = (float *)malloc((size_t)C * H * W * sizeof(float));
    if (!xn) { free(scratch); return -1; }
    memcpy(xn, x, (size_t)C * H * W * sizeof(float));

    /* in_layers: groupnorm(32,C) -> silu -> conv3x3 C->Cout */
    snprintf(nm, sizeof(nm), "%s.in_layers.0.weight", prefix);
    float *gn_w = unet_get_f32(u, nm);
    snprintf(nm, sizeof(nm), "%s.in_layers.0.bias", prefix);
    float *gn_b = unet_get_f32(u, nm);
    snprintf(nm, sizeof(nm), "%s.in_layers.2.weight", prefix);
    char conv_nm[256]; snprintf(conv_nm, sizeof(conv_nm), "%s", nm);
    snprintf(nm, sizeof(nm), "%s.in_layers.2.bias", prefix);
    char conv_bn[256]; snprintf(conv_bn, sizeof(conv_bn), "%s", nm);
    if (!gn_w) { free(scratch); free(xn); return -1; }

    unet_gn(xn, C, HW, gn_w, gn_b);
    unet_silu(xn, C * HW);
    if (unet_conv_q(u, conv_nm, conv_bn, xn, C, H, W, Cout, 3, 3, 1, 1, 1,
                    scratch, 0) != 0) { free(scratch); free(xn); return -1; }
    if (g_timing) { int nn=0; for (int z=0;z<Cout*H*W;z++) if (isnan(scratch[z])) {nn++;break;} if (nn) fprintf(stderr, "NAN after conv1 %s\n", prefix); }
    free(xn);

    /* emb_layers: nn.Sequential(SiLU, Linear(1280 -> Cout)) — GGUF stores
     * emb_layers.1.weight = the Linear; SiLU is the non-parametric .0 and
     * comes FIRST. y = Linear(silu(temb)). ADDED channel-wise to the
     * activations (h = h + emb_out — the original LDM ResBlock form; the
     * multiplicative (1+y) form is SD3/DiT-style and WRONG here). */
    snprintf(nm, sizeof(nm), "%s.emb_layers.1.weight", prefix);
    unet_raw_t emb_w = unet_get_raw(u, nm);
    snprintf(nm, sizeof(nm), "%s.emb_layers.1.bias", prefix);
    float *emb_b = unet_get_f32(u, nm);
    if (!emb_w.found) { free(scratch); return -1; }
    float emb_in[U_TIME];   /* silu(temb) — temb is shared, don't clobber */
    for (int i = 0; i < U_TIME; i++) emb_in[i] = temb[i] / (1.0f + expf(-temb[i]));
    float emb_out[2048];
    unet_linear(emb_w.ptr, emb_w.type, emb_b, Cout, U_TIME, emb_in, 1, emb_out);
    #pragma omp parallel for
    for (int c = 0; c < Cout; c++)
        for (int p = 0; p < HW; p++)
            scratch[(size_t)c * HW + p] += emb_out[c];

    /* out_layers: groupnorm(32,Cout) -> silu -> conv3x3 Cout->Cout */
    snprintf(nm, sizeof(nm), "%s.out_layers.0.weight", prefix);
    float *gn2_w = unet_get_f32(u, nm);
    snprintf(nm, sizeof(nm), "%s.out_layers.0.bias", prefix);
    float *gn2_b = unet_get_f32(u, nm);
    snprintf(nm, sizeof(nm), "%s.out_layers.3.weight", prefix);
    char conv2_nm[256]; snprintf(conv2_nm, sizeof(conv2_nm), "%s", nm);
    snprintf(nm, sizeof(nm), "%s.out_layers.3.bias", prefix);
    char conv2_bn[256]; snprintf(conv2_bn, sizeof(conv2_bn), "%s", nm);
    if (!gn2_w) { free(scratch); return -1; }
    unet_gn(scratch, Cout, HW, gn2_w, gn2_b);
    unet_silu(scratch, Cout * HW);
    if (unet_conv_q(u, conv2_nm, conv2_bn, scratch, Cout, H, W, Cout, 3, 3, 1, 1, 1,
                    out, 0) != 0) { free(scratch); return -1; }

    /* skip_connection: 1x1 conv C->Cout (only present when C != Cout) */
    if (C != Cout) {
        snprintf(nm, sizeof(nm), "%s.skip_connection.weight", prefix);
        char skip_nm[256]; snprintf(skip_nm, sizeof(skip_nm), "%s", nm);
        snprintf(nm, sizeof(nm), "%s.skip_connection.bias", prefix);
        char skip_bn[256]; snprintf(skip_bn, sizeof(skip_bn), "%s", nm);
        float *sx = (float *)malloc((size_t)Cout * HW * sizeof(float));
        if (!sx) { free(scratch); return -1; }
        if (unet_conv_q(u, skip_nm, skip_bn, x, C, H, W, Cout, 1, 1, 1, 0, 0,
                        sx, 0) != 0) { free(sx); free(scratch); return -1; }
        #pragma omp parallel for
        for (int i = 0; i < Cout * HW; i++) out[i] += sx[i];
        free(sx);
    } else {
        #pragma omp parallel for
        for (int i = 0; i < Cout * HW; i++) out[i] += x[i];
    }
    free(scratch);
    free(gn_w); free(gn_b); free(gn2_w); free(gn2_b);
    return 0;
}

/* spatial attention over the H*W spatial positions (SD transformer:
 * input [C][H][W] -> treat each spatial position as a token).
 * attn1 = self (n = H*W), attn2 = cross (k/v from ctx 77x768).
 * heads = 8, head_dim = C/8. */
/* fast exp for x <= 0 (softmax after max-subtraction): exp(x) = 2^(x*log2e),
 * degree-5 exp2 poly on [-0.5,0.5] + exponent-field scaling. ~3ns vs ~15ns
 * for expf; relative error ~1e-6 (irrelevant for softmax weights). */
static inline float fast_expf(float x) {
    float t = x * 1.4426950408889634f;
    float n = rintf(t);
    float f = t - n;
    float p = 1.0f + f * (0.6931471805599453f + f * (0.2402265069591007f +
              f * (0.0555041086648216f + f * (0.00961812910762848f +
              f * 0.001333355814642844f))));
    union { float f; uint32_t u; } uu;
    /* Guard: when n < -126, 2^n underflows to 0 in IEEE-754 (8-bit exponent
     * with bias 127).  Without this, (uint32_t)(int)n + 127u wraps to a huge
     * unsigned value and << 23 produces a completely wrong (huge/negative)
     * bit pattern instead of a denormal/0.  This broke softmax at higher C. */
    if (n < -126.0f) return 0.0f;
    uu.u = ((uint32_t)(int)n + 127u) << 23;
    return p * uu.f;
}

static int unet_attn(wubu_sd_unet_t *u, const char *prefix,
                     float *x, int C, int H, int W,
                     const float *ctx /*[77][768]*/, int is_cross) {
    double t0 = g_timing ? t_now() : 0;
    char nm[256];
    const int n = H * W;
    const int hd = C / U_HEADS;
    const int dim = C;
    float *q = (float *)malloc((size_t)n * dim * sizeof(float));
    float *k = (float *)malloc((size_t)n * dim * sizeof(float));
    float *v = (float *)malloc((size_t)n * dim * sizeof(float));
    if (!q || !k || !v) return -1;
    /* projections (all Linear C->C) */
    snprintf(nm, sizeof(nm), "%s.to_q.weight", prefix);
    unet_raw_t qw = unet_get_raw(u, nm);
    snprintf(nm, sizeof(nm), "%s.to_q.bias", prefix);
    float *qb = unet_get_f32(u, nm);
    snprintf(nm, sizeof(nm), "%s.to_k.weight", prefix);
    unet_raw_t kw = unet_get_raw(u, nm);
    snprintf(nm, sizeof(nm), "%s.to_k.bias", prefix);
    float *kb = unet_get_f32(u, nm);
    snprintf(nm, sizeof(nm), "%s.to_v.weight", prefix);
    unet_raw_t vw = unet_get_raw(u, nm);
    snprintf(nm, sizeof(nm), "%s.to_v.bias", prefix);
    float *vb = unet_get_f32(u, nm);
    if (!qw.found || !kw.found || !vw.found) return -1;
    /* NOTE: SD1.5 middle_block attention has NO biases — qb/kb/vb may be
     * NULL and unet_linear handles NULL bias (adds 0). */
    unet_linear(qw.ptr, qw.type, qb, dim, dim, x, n, q);
    if (is_cross) {
        /* k/v from ctx [77][768] -> [77][dim] — replace the (unused)
         * n-length k/v buffers; they are freed once at the end. */
        float *kctx = (float *)malloc((size_t)U_CTX_LEN * dim * sizeof(float));
        float *vctx = (float *)malloc((size_t)U_CTX_LEN * dim * sizeof(float));
        unet_linear(kw.ptr, kw.type, kb, dim, U_CTX, ctx, U_CTX_LEN, kctx);
        unet_linear(vw.ptr, vw.type, vb, dim, U_CTX, ctx, U_CTX_LEN, vctx);
        free(k); free(v);
        k = kctx; v = vctx;
    } else {
        unet_linear(kw.ptr, kw.type, kb, dim, dim, x, n, k);
        unet_linear(vw.ptr, vw.type, vb, dim, dim, x, n, v);
    }
    /* attention: out[i] = sum_j softmax(q_i . k_j / sqrt(hd)) v_j.
     * k/v are transposed to head-major contiguous [U_HEADS][Tk][hd] so the
     * j-loop inner products walk sequential memory (the [j][dim] layout
     * strides 3KB per j — cache-thrash). */
    const int Tk = is_cross ? U_CTX_LEN : n;
    float *kt = (float *)malloc((size_t)U_HEADS * Tk * hd * sizeof(float));
    float *vt = (float *)malloc((size_t)U_HEADS * Tk * hd * sizeof(float));
    #pragma omp parallel for collapse(2)
    for (int h = 0; h < U_HEADS; h++)
        for (int j = 0; j < Tk; j++) {
            const float *kj = k + (size_t)j * dim + (size_t)h * hd;
            const float *vj = v + (size_t)j * dim + (size_t)h * hd;
            float *kth = kt + ((size_t)h * Tk + j) * hd;
            float *vth = vt + ((size_t)h * Tk + j) * hd;
            memcpy(kth, kj, (size_t)hd * sizeof(float));
            memcpy(vth, vj, (size_t)hd * sizeof(float));
        }
    /* NOTE: k/v are NOT freed here — they are freed once at the end
     * (cross path: k/v are kctx/vctx; self path: the originals). */
    /* att row is only used within one (i,h) iteration — a private stack
     * array (NOT a full n*U_HEADS*Tk matrix: 536MB at 64x64 self-attn). */
    float *out = (float *)calloc((size_t)n * dim, sizeof(float));
    const float scale = 1.0f / sqrtf((float)hd);
#if defined(__AVX2__) && defined(__FMA__)
    const int hd8 = hd / 8;   /* hd is always a multiple of 8 here */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; i++)
        for (int h = 0; h < U_HEADS; h++) {
            float att_h[4096]; /* private per iteration: Tk <= 4096 */
            const float *qi = q + (size_t)i * dim + (size_t)h * hd;
            const float *kth = kt + (size_t)h * Tk * hd;
            const float *vth = vt + (size_t)h * Tk * hd;
            float mx = -INFINITY;
            for (int j = 0; j < Tk; j++) {
                const float *kj = kth + (size_t)j * hd;
                __m256 acc = _mm256_setzero_ps();
                for (int d8 = 0; d8 < hd8; d8++)
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(qi + 8 * d8),
                                          _mm256_loadu_ps(kj + 8 * d8), acc);
                float t[8];
                _mm256_storeu_ps(t, acc);
                float s = (t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7]) * scale;
                att_h[j] = s;
                if (s > mx) mx = s;
            }
            float sum = 0;
            for (int j = 0; j < Tk; j++) {
                float e = fast_expf(att_h[j] - mx);
                att_h[j] = e;
                sum += e;
            }
            const float inv_sum = 1.0f / sum;
            float *oi = out + (size_t)i * dim + (size_t)h * hd;
            __m256 oacc[20];
            for (int d8 = 0; d8 < hd8; d8++) oacc[d8] = _mm256_setzero_ps();
            for (int j = 0; j < Tk; j++) {
                __m256 wv = _mm256_set1_ps(att_h[j] * inv_sum);
                const float *vj = vth + (size_t)j * hd;
                for (int d8 = 0; d8 < hd8; d8++)
                    oacc[d8] = _mm256_fmadd_ps(wv, _mm256_loadu_ps(vj + 8 * d8), oacc[d8]);
            }
            for (int d8 = 0; d8 < hd8; d8++)
                _mm256_storeu_ps(oi + 8 * d8, oacc[d8]);
        }
#elif defined(__ARM_NEON)
    /* NEON attention, i-BATCHED x4 for QK^T: the QK^T dot reads kth[j]
     * once per 4 i's (kth 655KB/head was re-read once per i → 21GB/call
     * of L2 traffic at ~32GB/s shared); batching cuts that 4x. The AV
     * pass stays per-i (oacc[40] register window, no spills — a 4x-i AV
     * batch would need 160 live accumulators). collapse(2) over (i4,h)
     * keeps all threads on the same kth/vth stream (shared L2 hits).
     * fast_expf softmax = x86 path, bit-identical (verified 2026-08-10). */
    const int hd4 = hd / 4;
    const int ni = (n / 4) * 4;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i4 = 0; i4 < ni; i4 += 4)
        for (int h = 0; h < U_HEADS; h++) {
            float att_h[4][4096]; /* private: 4 i's x Tk (Tk <= 4096) */
            const float *qi[4];
            for (int a = 0; a < 4; a++)
                qi[a] = q + (size_t)(i4 + a) * dim + (size_t)h * hd;
            const float *kth = kt + (size_t)h * Tk * hd;
            const float *vth = vt + (size_t)h * Tk * hd;
            float mx[4] = { -INFINITY, -INFINITY, -INFINITY, -INFINITY };
            for (int j = 0; j < Tk; j++) {
                const float *kj = kth + (size_t)j * hd;
                float32x4_t acc[4] = { vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                                       vdupq_n_f32(0.0f), vdupq_n_f32(0.0f) };
                for (int d4 = 0; d4 < hd4; d4++) {
                    float32x4_t kjv = vld1q_f32(kj + 4 * d4);
                    for (int a = 0; a < 4; a++)
                        acc[a] = vfmaq_f32(acc[a], vld1q_f32(qi[a] + 4 * d4), kjv);
                }
                for (int a = 0; a < 4; a++) {
                    float s = vaddvq_f32(acc[a]) * scale;
                    att_h[a][j] = s;
                    if (s > mx[a]) mx[a] = s;
                }
            }
            float sum[4] = { 0, 0, 0, 0 };
            for (int j = 0; j < Tk; j++)
                for (int a = 0; a < 4; a++) {
                    float e = fast_expf(att_h[a][j] - mx[a]);
                    att_h[a][j] = e;
                    sum[a] += e;
                }
            float inv_sum[4];
            for (int a = 0; a < 4; a++) inv_sum[a] = 1.0f / sum[a];
            for (int a = 0; a < 4; a++) {   /* AV per-i (register window) */
                float *oi = out + (size_t)(i4 + a) * dim + (size_t)h * hd;
                float32x4_t oacc[40];  /* hd/4 <= 40 (hd=160 middle-block) */
                for (int d4 = 0; d4 < hd4; d4++) oacc[d4] = vdupq_n_f32(0.0f);
                const float inv = inv_sum[a];
                for (int j = 0; j < Tk; j++) {
                    const float *vj = vth + (size_t)j * hd;
                    float w = att_h[a][j] * inv;
                    for (int d4 = 0; d4 < hd4; d4++)
                        oacc[d4] = vfmaq_n_f32(oacc[d4], vld1q_f32(vj + 4 * d4), w);
                }
                for (int d4 = 0; d4 < hd4; d4++)
                    vst1q_f32(oi + 4 * d4, oacc[d4]);
            }
        }
    for (int i = ni; i < n; i++)   /* odd last i rows (n % 4) */
        for (int h = 0; h < U_HEADS; h++) {
            float att_h[4096];
            const float *qi = q + (size_t)i * dim + (size_t)h * hd;
            const float *kth = kt + (size_t)h * Tk * hd;
            const float *vth = vt + (size_t)h * Tk * hd;
            float mx = -INFINITY;
            for (int j = 0; j < Tk; j++) {
                const float *kj = kth + (size_t)j * hd;
                float32x4_t acc = vdupq_n_f32(0.0f);
                for (int d4 = 0; d4 < hd4; d4++)
                    acc = vfmaq_f32(acc, vld1q_f32(qi + 4 * d4),
                                         vld1q_f32(kj + 4 * d4));
                float s = vaddvq_f32(acc) * scale;
                att_h[j] = s;
                if (s > mx) mx = s;
            }
            float sum = 0;
            for (int j = 0; j < Tk; j++) {
                float e = fast_expf(att_h[j] - mx);
                att_h[j] = e;
                sum += e;
            }
            const float inv_sum = 1.0f / sum;
            float *oi = out + (size_t)i * dim + (size_t)h * hd;
            float32x4_t oacc[40];
            for (int d4 = 0; d4 < hd4; d4++) oacc[d4] = vdupq_n_f32(0.0f);
            for (int j = 0; j < Tk; j++) {
                float32x4_t wv = vdupq_n_f32(att_h[j] * inv_sum);
                const float *vj = vth + (size_t)j * hd;
                for (int d4 = 0; d4 < hd4; d4++)
                    oacc[d4] = vfmaq_f32(oacc[d4], wv, vld1q_f32(vj + 4 * d4));
            }
            for (int d4 = 0; d4 < hd4; d4++)
                vst1q_f32(oi + 4 * d4, oacc[d4]);
        }
#else
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; i++)
        for (int h = 0; h < U_HEADS; h++) {
            float att_h[4096]; /* private per iteration: Tk <= 4096 */
            const float *qi = q + (size_t)i * dim + (size_t)h * hd;
            const float *kth = kt + (size_t)h * Tk * hd;
            const float *vth = vt + (size_t)h * Tk * hd;
            float mx = -INFINITY;
            for (int j = 0; j < Tk; j++) {
                const float *kj = kth + (size_t)j * hd;
                float s = 0;
                for (int d = 0; d < hd; d++) s += qi[d] * kj[d];
                s *= scale;
                att_h[j] = s;
                if (s > mx) mx = s;
            }
            float sum = 0;
            for (int j = 0; j < Tk; j++) {
                att_h[j] = expf(att_h[j] - mx);
                sum += att_h[j];
            }
            float *oi = out + (size_t)i * dim + (size_t)h * hd;
            for (int j = 0; j < Tk; j++) {
                float w_ = att_h[j] / sum;
                const float *vj = vth + (size_t)j * hd;
                for (int d = 0; d < hd; d++) oi[d] += w_ * vj[d];
            }
        }
#endif
    free(kt); free(vt);
    /* out_proj */
    snprintf(nm, sizeof(nm), "%s.to_out.0.weight", prefix);
    unet_raw_t ow = unet_get_raw(u, nm);
    snprintf(nm, sizeof(nm), "%s.to_out.0.bias", prefix);
    float *ob = unet_get_f32(u, nm);
    if (!ow.found) return -1;
    unet_linear(ow.ptr, ow.type, ob, dim, dim, out, n, x);
    free(q); free(k); free(v); free(out);
    free(qb); free(kb); free(vb); free(ob);
    if (g_timing) g_t[4] += t_now() - t0;
    return 0;
}

/* fast tanh via fast_expf: tanh(z) = (1-e)/(1+e), e = exp(-2|z|). */
static inline float fast_tanhf(float z) {
    float az = fabsf(z);
    float e = fast_expf(-2.0f * az);
    float t = (1.0f - e) / (1.0f + e);
    return copysignf(t, z);
}

/* GEGLU FFN: Linear C->8C (half = 4C for gelu) -> Linear 4C->C */
static int unet_ffn(wubu_sd_unet_t *u, const char *prefix, float *x, int C, int n) {
    double t0 = g_timing ? t_now() : 0;
    char nm[256];
    const int hidden = C * 8;
    snprintf(nm, sizeof(nm), "%s.net.0.proj.weight", prefix);
    unet_raw_t p_w = unet_get_raw(u, nm);
    snprintf(nm, sizeof(nm), "%s.net.0.proj.bias", prefix);
    float *p_b = unet_get_f32(u, nm);
    snprintf(nm, sizeof(nm), "%s.net.2.weight", prefix);
    unet_raw_t f_w = unet_get_raw(u, nm);
    snprintf(nm, sizeof(nm), "%s.net.2.bias", prefix);
    float *f_b = unet_get_f32(u, nm);
    if (!p_w.found || !f_w.found) return -1;
    float *h = (float *)malloc((size_t)n * hidden * sizeof(float));
    if (!h) return -1;
    unet_linear(p_w.ptr, p_w.type, p_b, hidden, C, x, n, h);
    /* GEGLU: split hidden into a|b, h = a * gelu(b). MUST write to a
     * separate buffer: in-place compression races (row i reads up to
     * (i+1)*8C while row i+1 writes from (i+1)*4C). */
    float *g = (float *)malloc((size_t)n * (hidden / 2) * sizeof(float));
    if (!g) { free(h); return -1; }
    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        for (int j = 0; j < hidden / 2; j++) {
            float a = h[(size_t)i * hidden + j];
            float b = h[(size_t)i * hidden + j + hidden / 2];
            float ge = 0.5f * b * (1.0f + fast_tanhf(0.7978845608028654f * (b + 0.044715f * b * b * b)));
            g[(size_t)i * (hidden / 2) + j] = a * ge;
        }
    free(h);
    unet_linear(f_w.ptr, f_w.type, f_b, C, hidden / 2, g, n, x);
    free(g);
    if (g_timing) g_t[5] += t_now() - t0;
    return 0;
}

/* spatial transformer: norm(group) -> proj_in 1x1 -> attn1 self -> attn2
 * cross -> ffn -> proj_out 1x1 -> residual. prefix = ...1.transformer_blocks.0 */
static int unet_transformer(wubu_sd_unet_t *u, const char *prefix,
                            float *x, int C, int H, int W, const float *ctx) {
    char nm[256];
    int HW = H * W;
    /* outer norm: GroupNorm(32, C) — SD1.5 SpatialTransformer norm is
     * GROUPnorm (unlike CLIP). The inner norm1/2/3 are LayerNorm. */
    snprintf(nm, sizeof(nm), "%s.norm.weight", prefix);
    float *ln_w = unet_get_f32(u, nm);
    snprintf(nm, sizeof(nm), "%s.norm.bias", prefix);
    float *ln_b = unet_get_f32(u, nm);
    /* proj_in 1x1 */
    snprintf(nm, sizeof(nm), "%s.proj_in.weight", prefix);
    char pi_nm[256]; snprintf(pi_nm, sizeof(pi_nm), "%s", nm);
    snprintf(nm, sizeof(nm), "%s.proj_in.bias", prefix);
    char pi_bn[256]; snprintf(pi_bn, sizeof(pi_bn), "%s", nm);
    if (!ln_w) return -1;
    /* groupnorm over C (per spatial position) */
    float *ln = (float *)malloc((size_t)C * HW * sizeof(float));
    if (!ln) return -1;
    /* CRITICAL: the transformer's outer residual is out = x + proj_out(...).
     * proj_in OVERWRITES x below, so save the original input FIRST. */
    float *xorig = (float *)malloc((size_t)C * HW * sizeof(float));
    if (!xorig) { free(ln); return -1; }
    memcpy(xorig, x, (size_t)C * HW * sizeof(float));
    wubu_sd_groupnorm(x, 1, C, H, W, 32, 1e-6f, ln_w, ln_b, ln);
    if (unet_conv_q(u, pi_nm, pi_bn, ln, C, H, W, C, 1, 1, 1, 0, 0, x, 0) != 0) { free(ln); free(xorig); return -1; }
    /* LAYOUT FIX: conv output is [C][H*W] (NCHW), but the LayerNorm/attn/FFN
     * below use [H*W][C] (NHWC) indexing: x[p*C + c]. Transpose so the
     * transformer blocks see spatial-first data. (Channel-first conv ->
     * spatial-first transformer.) */
    {
        float *xt = (float *)malloc((size_t)C * HW * sizeof(float));
        if (!xt) { free(ln); free(xorig); return -1; }
        for (int p = 0; p < HW; p++)
            for (int c = 0; c < C; c++)
                xt[(size_t)p * C + c] = x[(size_t)c * HW + p];
        memcpy(x, xt, (size_t)C * HW * sizeof(float));
        free(xt);
    }
    if (g_timing) { int nn=0; for (int z=0;z<C*HW;z++) if (isnan(x[z])) {nn++;break;} if (nn) fprintf(stderr, "NAN after proj_in %s\n", prefix); }
    free(ln);
    /* transformer_blocks.0 */
    char tb[300];
    snprintf(tb, sizeof(tb), "%s.transformer_blocks.0", prefix);
    /* norm1 -> attn1 (self) -> residual */
    snprintf(nm, sizeof(nm), "%s.norm1.weight", tb);
    float *n1w = unet_get_f32(u, nm);
    snprintf(nm, sizeof(nm), "%s.norm1.bias", tb);
    float *n1b = unet_get_f32(u, nm);
    float *ln2 = (float *)malloc((size_t)C * HW * sizeof(float));
    #pragma omp parallel for
    for (int p = 0; p < HW; p++) {
        const float *xr = x + (size_t)p * C;
        float mean = 0, var = 0;
        for (int c = 0; c < C; c++) { mean += xr[c]; var += xr[c] * xr[c]; }
        mean /= C; var = var / C - mean * mean;
        float inv = 1.0f / sqrtf(var + 1e-5f);
        float *lr = ln2 + (size_t)p * C;
        for (int c = 0; c < C; c++) lr[c] = (xr[c] - mean) * inv * n1w[c] + (n1b ? n1b[c] : 0);
    }
    snprintf(nm, sizeof(nm), "%s.attn1", tb);
    if (unet_attn(u, nm, ln2, C, H, W, ctx, 0) == 0)
        #pragma omp parallel for
        for (int i = 0; i < C * HW; i++) x[i] += ln2[i];
    free(ln2);
    /* norm2 -> attn2 (cross) -> residual */
    snprintf(nm, sizeof(nm), "%s.norm2.weight", tb);
    float *n2w = unet_get_f32(u, nm);
    snprintf(nm, sizeof(nm), "%s.norm2.bias", tb);
    float *n2b = unet_get_f32(u, nm);
    float *ln3 = (float *)malloc((size_t)C * HW * sizeof(float));
    #pragma omp parallel for
    for (int p = 0; p < HW; p++) {
        const float *xr = x + (size_t)p * C;
        float mean = 0, var = 0;
        for (int c = 0; c < C; c++) { mean += xr[c]; var += xr[c] * xr[c]; }
        mean /= C; var = var / C - mean * mean;
        float inv = 1.0f / sqrtf(var + 1e-5f);
        float *lr = ln3 + (size_t)p * C;
        for (int c = 0; c < C; c++) lr[c] = (xr[c] - mean) * inv * n2w[c] + (n2b ? n2b[c] : 0);
    }
    snprintf(nm, sizeof(nm), "%s.attn2", tb);
    if (unet_attn(u, nm, ln3, C, H, W, ctx, 1) == 0) {
        #pragma omp parallel for
        for (int i = 0; i < C * HW; i++) x[i] += ln3[i];
        /* DEBUG HOOK (cfg_diff): capture the attn2 OUTPUT for the first
         * transformer block to compare cond vs uncond contributions. */
        if (g_attn2_capture && g_attn2_slot >= 0 &&
            (g_attn2_prefix[0] == 0 || strstr(tb, g_attn2_prefix))) {
            memcpy(g_attn2_out + (size_t)g_attn2_slot * C * HW, ln3,
                   (size_t)C * HW * sizeof(float));
            g_attn2_slot = -1;
            if (g_attn2_slot2 >= 0) { /* also capture x post-attn2-residual */
                memcpy(g_attn2_out + (size_t)g_attn2_slot2 * C * HW, x,
                       (size_t)C * HW * sizeof(float));
                g_attn2_slot2 = -1;
            }
        }
    }
    free(ln3);
    /* norm3 -> ffn -> residual */
    snprintf(nm, sizeof(nm), "%s.norm3.weight", tb);
    float *n3w = unet_get_f32(u, nm);
    snprintf(nm, sizeof(nm), "%s.norm3.bias", tb);
    float *n3b = unet_get_f32(u, nm);
    float *ln4 = (float *)malloc((size_t)C * HW * sizeof(float));
    #pragma omp parallel for
    for (int p = 0; p < HW; p++) {
        const float *xr = x + (size_t)p * C;
        float mean = 0, var = 0;
        for (int c = 0; c < C; c++) { mean += xr[c]; var += xr[c] * xr[c]; }
        mean /= C; var = var / C - mean * mean;
        float inv = 1.0f / sqrtf(var + 1e-5f);
        float *lr = ln4 + (size_t)p * C;
        for (int c = 0; c < C; c++) lr[c] = (xr[c] - mean) * inv * n3w[c] + (n3b ? n3b[c] : 0);
    }
    snprintf(nm, sizeof(nm), "%s.ff", tb);
    if (unet_ffn(u, nm, ln4, C, HW) == 0)
        #pragma omp parallel for
        for (int i = 0; i < C * HW; i++) x[i] += ln4[i];
    free(ln4);
    /* proj_out 1x1 -> residual: out = xorig + proj_out(blocks_out) */
    /* LAYOUT FIX (inverse): x is [HW][C] from transformer blocks, but
     * proj_out conv expects NCHW [C][H][W]. Transpose back first. */
    {
        float *xt = (float *)malloc((size_t)C * HW * sizeof(float));
        if (!xt) { free(xorig); return -1; }
        for (int p = 0; p < HW; p++)
            for (int c = 0; c < C; c++)
                xt[(size_t)c * HW + p] = x[(size_t)p * C + c];
        memcpy(x, xt, (size_t)C * HW * sizeof(float));
        free(xt);
    }
    snprintf(nm, sizeof(nm), "%s.proj_out.weight", prefix);
    char po_nm[256]; snprintf(po_nm, sizeof(po_nm), "%s", nm);
    snprintf(nm, sizeof(nm), "%s.proj_out.bias", prefix);
    char po_bn[256]; snprintf(po_bn, sizeof(po_bn), "%s", nm);
    float *po = (float *)malloc((size_t)C * HW * sizeof(float));
    memcpy(po, x, (size_t)C * HW * sizeof(float));
    if (unet_conv_q(u, po_nm, po_bn, po, C, H, W, C, 1, 1, 1, 0, 0, x, 0) != 0) { free(po); free(xorig); return -1; }
    #pragma omp parallel for
    for (int i = 0; i < C * HW; i++) x[i] += xorig[i];
    free(po); free(xorig);
    free(n1w); free(n1b); free(n2w); free(n2b); free(n3w); free(n3b);
    free(ln_w); free(ln_b);
    return 0;
}

/* downsample: conv 3x3 s2 (prefix.0.op) */
static int unet_downsample(wubu_sd_unet_t *u, const char *prefix,
                           float *x, int C, int H, int W, float *out) {
    char nm[256];
    snprintf(nm, sizeof(nm), "%s.op.weight", prefix);
    char op_nm[256]; snprintf(op_nm, sizeof(op_nm), "%s", nm);
    snprintf(nm, sizeof(nm), "%s.op.bias", prefix);
    char op_bn[256]; snprintf(op_bn, sizeof(op_bn), "%s", nm);
    return unet_conv_q(u, op_nm, op_bn, x, C, H, W, C, 3, 3, 2, 1, 1, out, 0);
}

/* upsample: nearest2x + conv 3x3 at prefix.<up_sub>.conv — the subblock
 * index varies by block (1 for out.2, 2 for out.5/out.8). Fused into
 * the conv's im2col (us=2) — no 4x buffer materialized (same win as
 * the VAE upsample). */
static int unet_upsample(wubu_sd_unet_t *u, const char *prefix, int up_sub,
                         float *x, int C, int H, int W, float *out) {
    char nm[256];
    snprintf(nm, sizeof(nm), "%s.%d.conv.weight", prefix, up_sub);
    char up_nm[256]; snprintf(up_nm, sizeof(up_nm), "%s", nm);
    snprintf(nm, sizeof(nm), "%s.%d.conv.bias", prefix, up_sub);
    char up_bn[256]; snprintf(up_bn, sizeof(up_bn), "%s", nm);
    return unet_conv_q(u, up_nm, up_bn, x, C, H, W, C, 3, 3, 1, 1, 1, out,
                       /* us = */ 2);
}

wubu_sd_unet_t *wubu_sd_unet_load(void *ctx) {
    wubu_sd_unet_t *u = (wubu_sd_unet_t *)calloc(1, sizeof(wubu_sd_unet_t));
    if (!u) return NULL;
    u->gguf = (gguf_ctx *)ctx;
    u->lat_h = 64; u->lat_w = 64;  /* default; override via set_resolution */
    /* mmap the data blob — quantized path reads weights straight from it */
    {
        gguf_ctx *g = (gguf_ctx *)ctx;
        if (!g->data_blob) gguf_buffer_data(g);
    }
    /* DeepCache: SD_DEEPCACHE=1 enables (interval SD_DEEPCACHE_INTERVAL,
     * default 5 — the paper's sweet spot). */
    if (getenv("SD_DEEPCACHE")) {
        u->dc_interval = getenv("SD_DEEPCACHE_INTERVAL")
                       ? atoi(getenv("SD_DEEPCACHE_INTERVAL")) : 5;
        if (u->dc_interval < 2) u->dc_interval = 5;
        u->dc_feat[0] = (float *)malloc((size_t)320 * 64 * 64 * sizeof(float));
        u->dc_feat[1] = (float *)malloc((size_t)320 * 64 * 64 * sizeof(float));
        if (!u->dc_feat[0] || !u->dc_feat[1]) {
            free(u->dc_feat[0]); free(u->dc_feat[1]);
            u->dc_feat[0] = NULL; u->dc_feat[1] = NULL; u->dc_interval = 0;
        }
        fprintf(stderr, "[unet] DeepCache enabled (interval=%d)\n", u->dc_interval);
    }
    return u;
}

void wubu_sd_unet_free(wubu_sd_unet_t *u) {
    if (!u) return;
    free(u->dc_feat[0]);
    free(u->dc_feat[1]);
    free(u->dc_sched_full);
    free(u);
}

void wubu_sd_unet_clear_cache(wubu_sd_unet_t *u) {
    /* no cache to clear — weights stay in the mmap'd blob */
    (void)u;
}

int wubu_sd_unet_forward(wubu_sd_unet_t *u,
                         const float *latent, int t,
                         const float *ctx,
                         float *out) {
    /* DeepCache: on cached steps run only conv_in + input_blocks.1 +
     * output_blocks.11 + conv_out, reusing the deep feature cached by
     * the last full pass (output of output_blocks.10, 320ch @ 64x64)
     * and the cached skip[1]. Training-free (paper arXiv 2312.00858):
     * 2.3x with ~0.05 CLIP-score drop. Full pass every dc_interval
     * steps. NOTE: changes the output vs the full UNet (that's the
     * quality/speed tradeoff). Cadence is per STEP (the caller sets
     * dc_step once per step via wubu_sd_unet_set_step — both CFG
     * calls of one step share the same decision). */
    /* DeepCache cached-pass decision: cached iff (a) a schedule array is
     * installed and this step is NOT a full step, or (b) no schedule and
     * step % interval != 0 (uniform fallback). */
    int dc_cached = 0;
    if (u->dc_interval > 0 && u->dc_valid) {
        if (u->dc_sched_n > 0)
            dc_cached = !u->dc_sched_full[u->dc_step];
        else
            dc_cached = (u->dc_step % u->dc_interval) != 0;
    }
    if (dc_cached) {
        /* ---- cached (cheap) pass ---- */
        const int LH = u->lat_h, LW = u->lat_w;
        const size_t AREA = (size_t)LH * LW;
        float *x = (float *)malloc((size_t)320 * AREA * sizeof(float));
        if (unet_conv_q(u, "model.diffusion_model.input_blocks.0.0.weight",
                        "model.diffusion_model.input_blocks.0.0.bias",
                        latent, 4, LH, LW, 320, 3, 3, 1, 1, 1, x, 0) != 0) { free(x); return -1; }
        /* reference cached pass: also run input_blocks.1 (res+attn); block
         * 11's skip = in1 (the truncated down path's last sample) */
        float *rb = (float *)malloc((size_t)320 * AREA * sizeof(float));
        float temb[U_TIME];
        unet_time_embed(u, t, temb);
        if (unet_resblock(u, "model.diffusion_model.input_blocks.1.0",
                          x, 320, 320, LH, LW, temb, rb) != 0) { free(x); free(rb); return -1; }
        if (unet_transformer(u, "model.diffusion_model.input_blocks.1.1",
                             rb, 320, LH, LW, ctx) != 0) { free(x); free(rb); return -1; }
        /* output_blocks.11: concat(cached deep feat + in1) -> res -> attn.
         * Full pass feeds block 11 with cur (block 10's post-attn output)
         * + skips[0] (conv_in) — the reference cached pass instead uses
         * in1 (input_blocks.1 output) as block 11's skip (paper's cheap
         * approximation). */
        float *cat = (float *)malloc((size_t)640 * AREA * sizeof(float));
        #pragma omp parallel for
        for (size_t p = 0; p < AREA; p++) {
            for (int c = 0; c < 320; c++)
                cat[(size_t)c * AREA + p] = u->dc_feat[u->dc_pass][(size_t)c * AREA + p];
            for (int c = 0; c < 320; c++)
                cat[(size_t)(320 + c) * AREA + p] = rb[(size_t)c * AREA + p];
        }
        free(x);
        if (getenv("SD_DC_DEBUG")) {
            double sx = 0; for (size_t z = 0; z < 320*AREA; z++) sx += rb[z]*rb[z];
            fprintf(stderr, "[dc] CACHED rb(in1) rms=%.4f\n", sqrt(sx/(320.0*AREA)));
            double st = 0; for (int z = 0; z < U_TIME; z++) st += temb[z]*temb[z];
            fprintf(stderr, "[dc] CACHED temb rms=%.4f | cat[0..3]=%.6f %.6f %.6f %.6f cat[320*AREA..]=%.6f %.6f %.6f %.6f\n",
                sqrt(st/U_TIME),
                cat[0], cat[1], cat[2], cat[3],
                cat[(size_t)320*AREA], cat[(size_t)320*AREA+1],
                cat[(size_t)320*AREA+2], cat[(size_t)320*AREA+3]);
        }
        float *o11 = (float *)malloc((size_t)320 * AREA * sizeof(float));
        if (unet_resblock(u, "model.diffusion_model.output_blocks.11.0",
                          cat, 640, 320, LH, LW, temb, o11) != 0) { free(cat); free(rb); free(o11); return -1; }
        free(cat);
        if (getenv("SD_DC_DEBUG")) {
            double s = 0; for (size_t z = 0; z < 320*AREA; z++) s += o11[z]*o11[z];
            fprintf(stderr, "[dc] CACHED o11 pre-attn rms=%.4f (full=3.0749)\n", sqrt(s/(320.0*AREA)));
        }
        if (unet_transformer(u, "model.diffusion_model.output_blocks.11.1",
                             o11, 320, LH, LW, ctx) != 0) { free(rb); free(o11); return -1; }
        free(rb);
        if (unet_transformer(u, "model.diffusion_model.output_blocks.11.1",
                             o11, 320, LH, LW, ctx) != 0) { free(o11); return -1; }
        /* out: gn -> silu -> conv */
        float *gn_w = unet_get_f32(u, "model.diffusion_model.out.0.weight");
        float *gn_b = unet_get_f32(u, "model.diffusion_model.out.0.bias");
        if (!gn_w) { free(o11); return -1; }
        unet_gn(o11, 320, (int)AREA, gn_w, gn_b);
        unet_silu(o11, (int)(320 * AREA));
        int rc = unet_conv_q(u, "model.diffusion_model.out.2.weight",
                             "model.diffusion_model.out.2.bias",
                             o11, 320, LH, LW, 4, 3, 3, 1, 1, 1, out, 0);
        free(o11);
        return rc;
    }
    /* timestep embedding */
    float temb[U_TIME];
    unet_time_embed(u, t, temb);
    if (getenv("SD_DC_DEBUG")) {
        double st = 0; for (int z = 0; z < U_TIME; z++) st += temb[z]*temb[z];
        fprintf(stderr, "[dc] FULL temb rms=%.4f (cached=0.2902)\n", sqrt(st/U_TIME));
    }

    /* conv_in: 4 -> 320 */
    float *x = (float *)malloc((size_t)320 * u->lat_h * u->lat_w * sizeof(float));
    if (unet_conv_q(u, "model.diffusion_model.input_blocks.0.0.weight",
                    "model.diffusion_model.input_blocks.0.0.bias",
                    latent, 4, u->lat_h, u->lat_w, 320, 3, 3, 1, 1, 1, x, 0) != 0) return -1;

    /* input blocks. Ownership: skips[i] OWNS its buffer; `cur` borrows
     * skips[i] until the next block replaces it. */
    float *skips[12];
    memset(skips, 0, sizeof(skips));
    int H = u->lat_h, W = u->lat_w;
    float *cur = x;   /* after conv_in, cur = x (320ch @ latent) */
    for (int i = 0; i < 12; i++) {
        char prefix[128];
        int type = IN_TYPE[i];
        if (type == 3) {  /* downsample only */
            snprintf(prefix, sizeof(prefix), "model.diffusion_model.input_blocks.%d.0", i);
            int C = IN_OUT_CH[i - 1];
            float *ds = (float *)malloc((size_t)C * (H / 2) * (W / 2) * sizeof(float));
            if (unet_downsample(u, prefix, cur, C, H, W, ds) != 0) return -1;
            /* cur (skips[i-1]'s buffer) is STILL NEEDED by an output block
             * (every skip feeds exactly one out block) — never free here. */
            cur = ds;
            H /= 2; W /= 2;
        } else if (type == 1 || type == 2) {  /* resblock [+ transformer] */
            snprintf(prefix, sizeof(prefix), "model.diffusion_model.input_blocks.%d.0", i);
            int C = IN_OUT_CH[i];
            int Cin = (i == 1) ? 320 : IN_OUT_CH[i - 1];  /* res in-ch = prev out-ch */
            float *rb = (float *)malloc((size_t)C * H * W * sizeof(float));
            if (unet_resblock(u, prefix, cur, Cin, C, H, W, temb, rb) != 0) return -1;
            /* cur is still skips[i-1]'s buffer — do NOT free it here */
            cur = rb;
            if (type == 1) {
                snprintf(prefix, sizeof(prefix), "model.diffusion_model.input_blocks.%d.1", i);
                if (unet_transformer(u, prefix, cur, C, H, W, ctx) != 0) return -1;
            }
        }
        skips[i] = cur;
        /* NaN checkpoint (debug) */
        if (g_timing) {
            int cc = (type == 3) ? IN_OUT_CH[i - 1] : IN_OUT_CH[i];
            int nn = 0;
            for (int z = 0; z < cc * H * W; z++) if (isnan(cur[z])) { nn++; break; }
            if (nn) fprintf(stderr, "NAN at input_blocks.%d (C=%d H=%d W=%d)\n", i, cc, H, W);
        }
        /* capture input_blocks.1 output (320ch @ 64x64) for sd.cpp parity */
        if (g_stage_capture & 1 && i == 1)
            memcpy(g_stage_out[0], cur, (size_t)320 * 64 * 64 * sizeof(float));
        /* cur is now OWNED by skips[i] */
    }

    /* middle block: res -> attn -> res. NOTE: cur is skips[11] — do NOT
     * free it here (output block 0 concats it). */
    {
        float *m = (float *)malloc((size_t)1280 * H * W * sizeof(float));
        float *m2 = (float *)malloc((size_t)1280 * H * W * sizeof(float));
        if (g_stage_capture & 2) memcpy(g_stage_out[1], cur, (size_t)1280 * H * W * sizeof(float));
        if (unet_resblock(u, "model.diffusion_model.middle_block.0", cur, 1280, 1280, H, W, temb, m) != 0) return -1;
        /* middle attention: uses transformer-style attn with ctx (cross) */
        if (unet_transformer(u, "model.diffusion_model.middle_block.1", m, 1280, H, W, ctx) != 0) return -1;
        if (g_stage_capture & 4) memcpy(g_stage_out[2], m, (size_t)1280 * H * W * sizeof(float));
        if (unet_resblock(u, "model.diffusion_model.middle_block.2", m, 1280, 1280, H, W, temb, m2) != 0) return -1;
        free(m);
        cur = m2;
    }

    /* output blocks: concat prev + skip, res, [attn], [up] */
    for (int i = 0; i < 12; i++) {
        const out_cfg_t *cfg = &OUT_CFG[i];
        int Cin = cfg->prev_ch + cfg->skip_ch;  /* concat dims */
        int C = cfg->out_ch;
        /* concat cur (prev_ch) + skips[cfg->skip] (skip_ch) -> [Cin] */
        float *cat = (float *)malloc((size_t)Cin * H * W * sizeof(float));
        #pragma omp parallel for
        for (int p = 0; p < H * W; p++) {
            for (int c = 0; c < cfg->prev_ch; c++)
                cat[(size_t)c * (H * W) + p] = cur[(size_t)c * (H * W) + p];
            for (int c = 0; c < cfg->skip_ch; c++)
                cat[(size_t)(cfg->prev_ch + c) * (H * W) + p] = skips[cfg->skip][(size_t)c * (H * W) + p];
        }
        free(cur);
        char prefix[128];
        snprintf(prefix, sizeof(prefix), "model.diffusion_model.output_blocks.%d.0", i);
        float *rb = (float *)malloc((size_t)C * H * W * sizeof(float));
        if (unet_resblock(u, prefix, cat, Cin, C, H, W, temb, rb) != 0) return -1;
        free(cat);
        cur = rb;
        if (g_stage_capture == 5 && g_stage_block == i) {
            memcpy(g_stage_out[0], cur, (size_t)C * H * W * sizeof(float));
            g_stage_count = C * H * W;
        }
        /* debug: compare full-pass block 11 pre-attn with cached */
        if (getenv("SD_DC_DEBUG") && i == 11) {
            double s = 0; for (int z = 0; z < C*H*W; z++) s += cur[z]*cur[z];
            fprintf(stderr, "[dc] FULL o11 pre-attn rms=%.4f\n", sqrt(s/(double)(C*H*W)));
            double s2 = 0; for (int z = 0; z < cfg->skip_ch*H*W; z++) s2 += skips[cfg->skip][z]*skips[cfg->skip][z];
            fprintf(stderr, "[dc] FULL o11 skip[%d] rms=%.4f\n", cfg->skip, sqrt(s2/(double)(cfg->skip_ch*H*W)));
            double s3 = 0; for (int z = 0; z < 320*4096; z++) s3 += u->dc_feat[u->dc_pass][z]*u->dc_feat[u->dc_pass][z];
            fprintf(stderr, "[dc] FULL o11 dc_feat rms=%.4f | cur(pre-res)[0..3]=%.6f %.6f %.6f %.6f\n", sqrt(s3/(320.0*4096)),
                cur[0], cur[1], cur[2], cur[3]);
        }
        /* dc_feat sanity: after block 10's attn, dc_feat must equal cur */
        if (getenv("SD_DC_DEBUG") && i == 10) {
            double s = 0; for (int z = 0; z < 320*4096; z++) s += u->dc_feat[u->dc_pass][z]*u->dc_feat[u->dc_pass][z];
            fprintf(stderr, "[dc] dc_feat[%d] rms=%.4f cur rms=", u->dc_pass, sqrt(s/(320.0*4096)));
            double s2 = 0; for (int z = 0; z < C*H*W; z++) s2 += cur[z]*cur[z];
            fprintf(stderr, "%.4f | cur[0..3]=%.6f %.6f %.6f %.6f dc_feat[0..3]=%.6f %.6f %.6f %.6f\n",
                sqrt(s2/(double)(C*H*W)),
                cur[0], cur[1], cur[2], cur[3],
                u->dc_feat[u->dc_pass][0], u->dc_feat[u->dc_pass][1],
                u->dc_feat[u->dc_pass][2], u->dc_feat[u->dc_pass][3]);
        }
        /* DeepCache: capture the deep feature after output_blocks.10
         * (320ch @ 64x64) INCLUDING its attention — the cached pass
         * resumes from here. Indexed by CFG pass (cond/uncond). */
        if (u->dc_interval > 0 && i == 10 && u->dc_feat[u->dc_pass] &&
            C == 320 && H == u->lat_h && W == u->lat_w) {
            memcpy(u->dc_feat[u->dc_pass], cur, (size_t)320 * H * W * sizeof(float));
            if (getenv("SD_DC_DEBUG")) {
                double s = 0; for (int z = 0; z < 320*H*W; z++) s += cur[z]*cur[z];
                fprintf(stderr, "[dc] captured feat@i10 pass=%d rms=%.4f\n",
                        u->dc_pass, sqrt(s/(320.0*H*W)));
            }
        }
        if (cfg->attn) {
            snprintf(prefix, sizeof(prefix), "model.diffusion_model.output_blocks.%d.1", i);
            if (unet_transformer(u, prefix, cur, C, H, W, ctx) != 0) return -1;
            if (g_stage_capture == 6 && g_stage_block == i)
                memcpy(g_stage_out[0], cur, (size_t)C * H * W * sizeof(float));
            /* DeepCache: block 10's feature is captured AFTER attn too —
             * the cached pass feeds block 11 with block 10's full output. */
            if (u->dc_interval > 0 && i == 10 && u->dc_feat[u->dc_pass] &&
                C == 320 && H == u->lat_h && W == u->lat_w) {
                memcpy(u->dc_feat[u->dc_pass], cur, (size_t)320 * H * W * sizeof(float));
                if (getenv("SD_DC_DEBUG")) {
                    double s = 0; for (int z = 0; z < 320*H*W; z++) s += cur[z]*cur[z];
                    fprintf(stderr, "[dc] captured feat@i10 POST-attn pass=%d rms=%.4f\n",
                            u->dc_pass, sqrt(s/(320.0*H*W)));
                }
            }
        }
        if (cfg->up) {
            /* upsample conv subblock varies (1 for block 2, 2 for 5/8) */
            snprintf(prefix, sizeof(prefix), "model.diffusion_model.output_blocks.%d", i);
            float *up = (float *)malloc((size_t)C * (2 * H) * (2 * W) * sizeof(float));
            if (unet_upsample(u, prefix, cfg->up, cur, C, H, W, up) != 0) return -1;
            free(cur); cur = up;
            H *= 2; W *= 2;
        }
    }

    /* out: groupnorm(32, 320) -> silu -> conv 320->4 */
    float *gn_w = unet_get_f32(u, "model.diffusion_model.out.0.weight");
    float *gn_b = unet_get_f32(u, "model.diffusion_model.out.0.bias");
    if (!gn_w) return -1;
    if (g_stage_capture & 8) memcpy(g_stage_out[3], cur, (size_t)320 * H * W * sizeof(float));
    unet_gn(cur, 320, H * W, gn_w, gn_b);
    unet_silu(cur, 320 * H * W);
    if (unet_conv_q(u, "model.diffusion_model.out.2.weight",
                    "model.diffusion_model.out.2.bias",
                    cur, 320, H, W, 4, 3, 3, 1, 1, 1, out, 0) != 0) return -1;
    if (g_stage_capture & 16) memcpy(g_stage_out[4], out, (size_t)4 * H * W * sizeof(float));
    free(cur);
    for (int i = 0; i < 12; i++) free(skips[i]);
    /* DeepCache: full pass completed — cache is valid */
    if (u->dc_interval > 0) u->dc_valid = 1;
    return 0;
}

/* DeepCache cadence control: caller sets the STEP index (0-based) once
 * per denoising step, before the forward calls. Both CFG calls of one
 * step share the same decision. */
void wubu_sd_unet_set_step(wubu_sd_unet_t *u, int step) {
    if (u) u->dc_step = step;
}

/* DeepCache CFG pass selector: 0 = cond (text-conditional), 1 = uncond.
 * Set before each forward call; the cached deep feature is per-pass. */
void wubu_sd_unet_set_pass(wubu_sd_unet_t *u, int pass) {
    if (u) u->dc_pass = (pass != 0) ? 1 : 0;
}

/* DeepCache schedule: install a per-step full-pass map (1 = run the full
 * UNet, 0 = cached pass). Mirrors the paper's non-uniform quad-center
 * schedule (full passes concentrated at early high-noise steps). The
 * caller frees the array after installing (we copy it). */
void wubu_sd_unet_set_dc_schedule(wubu_sd_unet_t *u, const int *full_map, int n) {
    if (!u || !full_map || n <= 0) return;
    free(u->dc_sched_full);
    u->dc_sched_full = (int *)malloc((size_t)n * sizeof(int));
    if (!u->dc_sched_full) { u->dc_sched_n = 0; return; }
    memcpy(u->dc_sched_full, full_map, (size_t)n * sizeof(int));
    u->dc_sched_n = n;
}

/* Set the latent spatial resolution (HxW, both multiples of 8). The UNet
 * conv/attn kernels are spatial-agnostic (they take H/W as args), so a
 * smaller latent — e.g. 30x52 for the 240x416 display instead of the
 * default 64x64 — cuts UNet FLOPs ~proportionally to the pixel area.
 * DeepCache features are reallocated to the new area. Returns 0 ok. */
int wubu_sd_unet_set_resolution(wubu_sd_unet_t *u, int h, int w)
{
    if (!u || h <= 0 || w <= 0) return -1;
    if (h % 8 != 0 || w % 8 != 0) return -1;  /* mid-block downsamples by 8 */
    u->lat_h = h;
    u->lat_w = w;
    if (u->dc_interval > 0) {
        size_t area = (size_t)h * (size_t)w;
        free(u->dc_feat[0]); free(u->dc_feat[1]);
        u->dc_feat[0] = (float *)malloc((size_t)320 * area * sizeof(float));
        u->dc_feat[1] = (float *)malloc((size_t)320 * area * sizeof(float));
        if (!u->dc_feat[0] || !u->dc_feat[1]) {
            free(u->dc_feat[0]); free(u->dc_feat[1]);
            u->dc_feat[0] = NULL; u->dc_feat[1] = NULL; u->dc_interval = 0;
        }
    }
    return 0;
}
