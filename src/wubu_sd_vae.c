/* wubu_sd_vae.c -- LDM AutoencoderKL DECODER forward (SD 1.5 family).
 *
 * Topology (verified against the Anything-V5 GGUF, dims = [kh,kw,in,out]):
 *   conv_in: 4 -> 512 (3x3)
 *   mid:     resnet(512) -> attn(512, 1x1 convs, GroupNorm32) -> resnet(512)
 *   up.3:    3x resnet 512->512, upsample
 *   up.2:    3x resnet 512->512, upsample
 *   up.1:    resnet 512->256 + 2x resnet 256->256, upsample
 *   up.0:    resnet 256->128 + 2x resnet 128->128   (NO upsample)
 *   norm_out: GroupNorm32(128), conv_out: 128 -> 3 (3x3)
 * Latent 64x64 -> image 512x512 (8x). C11, lazy F32 weight cache.
 */
#include "wubu_sd_vae.h"
#include "wubu_sd_ops.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#define VAE_PREFIX "first_stage_model.decoder"

struct wubu_sd_vae {
    gguf_ctx *gguf;
    /* no F32 weight cache — weights stay in the mmap'd blob */
};

/* raw (quantized) weight descriptor */
typedef struct {
    const void *ptr;
    int type;
    int64_t K, N;
    int64_t n_elems;
    int found;
} vae_raw_t;

static vae_raw_t vae_get_raw(wubu_sd_vae_t *v, const char *name) {
    vae_raw_t r = {0};
    gguf_tensor_info *ti = gguf_find_tensor(v->gguf, name);
    if (!ti) { fprintf(stderr, "VAE: missing %s\n", name); return r; }
    r.ptr = (const uint8_t *)v->gguf->data_blob + ti->data_offset;
    r.type = ti->ggml_type;
    r.n_elems = 1;
    for (int d = 0; d < ti->n_dims; d++) r.n_elems *= ti->dims[d];
    r.K = ti->n_dims > 0 ? ti->dims[0] : 1;
    r.N = ti->n_dims > 1 ? ti->dims[1] : 1;
    r.found = 1;
    return r;
}

/* tiny F32 read for 1D params (biases) — a few KB, no cache */
static float *vae_get_f32(wubu_sd_vae_t *v, const char *name) {
    gguf_tensor_info *ti = gguf_find_tensor(v->gguf, name);
    if (!ti) { fprintf(stderr, "VAE: missing %s\n", name); return NULL; }
    int64_t ne = 1;
    for (int d = 0; d < ti->n_dims; d++) ne *= ti->dims[d];
    float *f = (float *)malloc((size_t)ne * sizeof(float));
    if (!f) return NULL;
    if (gguf_read_tensor_f32(v->gguf, ti, f, ne) != (int)ne) { free(f); return NULL; }
    return f;
}

/* quantized conv2d helper: fetch raw weight, run conv2d_q. */
static int vae_conv_q(wubu_sd_vae_t *v, const char *wname, const char *bname,
                      const float *x, int C_in, int H, int W,
                      int C_out, int KH, int KW, int stride,
                      int pad_h, int pad_w, float *y) {
    double t0 = 0;
    if (getenv("SD_VAE_TIMING")) t0 = (double)clock() / CLOCKS_PER_SEC;
    vae_raw_t w = vae_get_raw(v, wname);
    if (!w.found) return -1;
    float *b = bname ? vae_get_f32(v, bname) : NULL;
    wubu_sd_conv2d_q(x, 1, C_in, H, W, w.ptr, w.type, b, C_out, KH, KW,
                     stride, pad_h, pad_w, y, NULL, NULL);
    free(b);
    if (getenv("SD_VAE_TIMING"))
        fprintf(stderr, "  [vae-conv] %s %dx%d C%d->%d k%dx%d s%d: %.3fs\n",
                wname, W, H, C_in, C_out, KH, KW, stride,
                (double)clock() / CLOCKS_PER_SEC - t0);
    return 0;
}

/* groupnorm with G=32 groups (VAE standard), in place */
static void vae_gn(float *x, int C, int HW, const float *g, const float *b) {
    wubu_sd_groupnorm(x, 1, C, 1, HW, 32, 1e-6f, g, b, x);
}

/* VAE resnet block: gn32 -> silu -> conv1 (3x3) -> gn32 -> silu ->
 * conv2 (3x3); nin_shortcut 1x1 when in != out; residual add.
 * prefix = "first_stage_model.decoder.up.N.block.M" or "...mid.block_N" */
static int vae_resnet(wubu_sd_vae_t *v, const char *prefix,
                      float *x, int C, int Cout, int H, int W, float *out) {
    char nm[256];
    int HW_ = H * W;
    /* conv2d is NOT in-place safe (writes channel co clobber input channels
     * still needed by co+1) — always distinct in/out buffers. h holds the
     * INPUT after groupnorm (C channels), h2 the conv output (Cout).
     * conv2 reads h2 and must write to a THIRD buffer (h is C-sized and
     * may be smaller than Cout). */
    float *h  = (float *)malloc((size_t)C * HW_ * sizeof(float));
    float *h2 = (float *)malloc((size_t)Cout * HW_ * sizeof(float));
    float *h3 = (float *)malloc((size_t)Cout * HW_ * sizeof(float));
    float *sx = (float *)malloc((size_t)Cout * HW_ * sizeof(float));
    if (!h || !h2 || !h3 || !sx) return -1;
    /* conv1 */
    snprintf(nm, sizeof(nm), "%s.norm1.weight", prefix);
    float *n1w = vae_get_f32(v, nm);
    snprintf(nm, sizeof(nm), "%s.norm1.bias", prefix);
    float *n1b = vae_get_f32(v, nm);
    snprintf(nm, sizeof(nm), "%s.conv1.weight", prefix);
    char c1_nm[256]; snprintf(c1_nm, sizeof(c1_nm), "%s", nm);
    snprintf(nm, sizeof(nm), "%s.conv1.bias", prefix);
    char c1_bn[256]; snprintf(c1_bn, sizeof(c1_bn), "%s", nm);
    if (!n1w) return -1;
    memcpy(h, x, (size_t)C * HW_ * sizeof(float));
    vae_gn(h, C, HW_, n1w, n1b);
    wubu_sd_silu(h, C * HW_);
    if (vae_conv_q(v, c1_nm, c1_bn, h, C, H, W, Cout, 3, 3, 1, 1, 1, h2) != 0) return -1;
    /* conv2 */
    snprintf(nm, sizeof(nm), "%s.norm2.weight", prefix);
    float *n2w = vae_get_f32(v, nm);
    snprintf(nm, sizeof(nm), "%s.norm2.bias", prefix);
    float *n2b = vae_get_f32(v, nm);
    snprintf(nm, sizeof(nm), "%s.conv2.weight", prefix);
    char c2_nm[256]; snprintf(c2_nm, sizeof(c2_nm), "%s", nm);
    snprintf(nm, sizeof(nm), "%s.conv2.bias", prefix);
    char c2_bn[256]; snprintf(c2_bn, sizeof(c2_bn), "%s", nm);
    if (!n2w) return -1;
    vae_gn(h2, Cout, HW_, n2w, n2b);
    wubu_sd_silu(h2, Cout * HW_);
    if (vae_conv_q(v, c2_nm, c2_bn, h2, Cout, H, W, Cout, 3, 3, 1, 1, 1, h3) != 0) return -1;
    /* shortcut */
    if (C != Cout) {
        snprintf(nm, sizeof(nm), "%s.nin_shortcut.weight", prefix);
        char sw_nm[256]; snprintf(sw_nm, sizeof(sw_nm), "%s", nm);
        snprintf(nm, sizeof(nm), "%s.nin_shortcut.bias", prefix);
        char sw_bn[256]; snprintf(sw_bn, sizeof(sw_bn), "%s", nm);
        if (vae_conv_q(v, sw_nm, sw_bn, x, C, H, W, Cout, 1, 1, 1, 0, 0, sx) != 0) return -1;
    } else {
        memcpy(sx, x, (size_t)Cout * HW_ * sizeof(float));
    }
    #pragma omp parallel for
    for (int i = 0; i < Cout * HW_; i++) h3[i] += sx[i];
    memcpy(out, h3, (size_t)Cout * HW_ * sizeof(float));
    free(h); free(h2); free(h3); free(sx);
    free(n1w); free(n1b); free(n2w); free(n2b);
    return 0;
}

/* single-head attention over spatial positions with 1x1 conv proj.
 * prefix = "...mid.attn_1" (or up.N.attn if present). */
static int vae_attn(wubu_sd_vae_t *v, const char *prefix,
                    float *x, int C, int H, int W) {
    char nm[256];
    int n = H * W;
    float *h = (float *)malloc((size_t)C * n * sizeof(float));
    float *q = (float *)malloc((size_t)C * n * sizeof(float));
    float *k = (float *)malloc((size_t)C * n * sizeof(float));
    float *v_ = (float *)malloc((size_t)C * n * sizeof(float));
    float *out = (float *)calloc((size_t)C * n, sizeof(float));
    if (!h || !q || !k || !v_ || !out) return -1;
    /* norm */
    snprintf(nm, sizeof(nm), "%s.norm.weight", prefix);
    float *nw = vae_get_f32(v, nm);
    snprintf(nm, sizeof(nm), "%s.norm.bias", prefix);
    float *nb = vae_get_f32(v, nm);
    if (!nw) return -1;
    memcpy(h, x, (size_t)C * n * sizeof(float));
    vae_gn(h, C, n, nw, nb);
    /* q,k,v 1x1 convs */
    snprintf(nm, sizeof(nm), "%s.q.weight", prefix);
    char q_nm[256]; snprintf(q_nm, sizeof(q_nm), "%s", nm);
    snprintf(nm, sizeof(nm), "%s.q.bias", prefix);
    char q_bn[256]; snprintf(q_bn, sizeof(q_bn), "%s", nm);
    snprintf(nm, sizeof(nm), "%s.k.weight", prefix);
    char k_nm[256]; snprintf(k_nm, sizeof(k_nm), "%s", nm);
    snprintf(nm, sizeof(nm), "%s.k.bias", prefix);
    char k_bn[256]; snprintf(k_bn, sizeof(k_bn), "%s", nm);
    snprintf(nm, sizeof(nm), "%s.v.weight", prefix);
    char v_nm[256]; snprintf(v_nm, sizeof(v_nm), "%s", nm);
    snprintf(nm, sizeof(nm), "%s.v.bias", prefix);
    char v_bn[256]; snprintf(v_bn, sizeof(v_bn), "%s", nm);
    if (vae_conv_q(v, q_nm, q_bn, h, C, H, W, C, 1, 1, 1, 0, 0, q) != 0) return -1;
    if (vae_conv_q(v, k_nm, k_bn, h, C, H, W, C, 1, 1, 1, 0, 0, k) != 0) return -1;
    if (vae_conv_q(v, v_nm, v_bn, h, C, H, W, C, 1, 1, 1, 0, 0, v_) != 0) return -1;
    /* softmax attention, single head, scale 1/sqrt(C) */
    const float scale = 1.0f / sqrtf((float)C);
    float *att = (float *)malloc((size_t)n * n * sizeof(float));
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        const float *qi = q + (size_t)i * C;
        float mx = -INFINITY;
        for (int j = 0; j < n; j++) {
            const float *kj = k + (size_t)j * C;
            float s = 0;
            for (int d = 0; d < C; d++) s += qi[d] * kj[d];
            s *= scale;
            att[(size_t)i * n + j] = s;
            if (s > mx) mx = s;
        }
        float sum = 0;
        for (int j = 0; j < n; j++) {
            att[(size_t)i * n + j] = expf(att[(size_t)i * n + j] - mx);
            sum += att[(size_t)i * n + j];
        }
        float *oi = out + (size_t)i * C;
        for (int j = 0; j < n; j++) {
            float w_ = att[(size_t)i * n + j] / sum;
            const float *vj = v_ + (size_t)j * C;
            for (int d = 0; d < C; d++) oi[d] += w_ * vj[d];
        }
    }
    free(att);
    /* proj_out 1x1, then residual */
    snprintf(nm, sizeof(nm), "%s.proj_out.weight", prefix);
    char po_nm[256]; snprintf(po_nm, sizeof(po_nm), "%s", nm);
    snprintf(nm, sizeof(nm), "%s.proj_out.bias", prefix);
    char po_bn[256]; snprintf(po_bn, sizeof(po_bn), "%s", nm);
    if (vae_conv_q(v, po_nm, po_bn, out, C, H, W, C, 1, 1, 1, 0, 0, h) != 0) return -1;
    #pragma omp parallel for
    for (int i = 0; i < C * n; i++) x[i] += h[i];
    free(h); free(q); free(k); free(v_); free(out);
    free(nw); free(nb);
    return 0;
}

/* upsample: nearest 2x + 3x3 conv. prefix = "...up.N.upsample" */
static int vae_upsample(wubu_sd_vae_t *v, const char *prefix,
                        float *x, int C, int H, int W, float *out) {
    char nm[256];
    snprintf(nm, sizeof(nm), "%s.conv.weight", prefix);
    char up_nm[256]; snprintf(up_nm, sizeof(up_nm), "%s", nm);
    snprintf(nm, sizeof(nm), "%s.conv.bias", prefix);
    char up_bn[256]; snprintf(up_bn, sizeof(up_bn), "%s", nm);
    float *up = (float *)malloc((size_t)C * (2 * H) * (2 * W) * sizeof(float));
    if (!up) return -1;
    wubu_sd_upsample2x(x, 1, C, H, W, up);
    int rc = vae_conv_q(v, up_nm, up_bn, up, C, 2 * H, 2 * W, C, 3, 3, 1, 1, 1, out);
    free(up);
    return rc;
}

wubu_sd_vae_t *wubu_sd_vae_load(void *ctx) {
    wubu_sd_vae_t *v = (wubu_sd_vae_t *)calloc(1, sizeof(wubu_sd_vae_t));
    if (!v) return NULL;
    v->gguf = (gguf_ctx *)ctx;
    /* mmap the data blob — quantized path reads weights straight from it */
    {
        gguf_ctx *g = (gguf_ctx *)ctx;
        if (!g->data_blob) gguf_buffer_data(g);
    }
    return v;
}

void wubu_sd_vae_free(wubu_sd_vae_t *v) {
    if (!v) return;
    free(v);
}

void wubu_sd_vae_clear_cache(wubu_sd_vae_t *v) {
    /* no cache to clear — weights stay in the mmap'd blob */
    (void)v;
}

int wubu_sd_vae_decode(wubu_sd_vae_t *v, const float *latent, int H, int W,
                       float *out) {
    /* conv_in 4 -> 512 */
    char nm[256];
    float *cur = (float *)malloc((size_t)512 * H * W * sizeof(float));
    if (!cur) return -1;
    if (vae_conv_q(v, VAE_PREFIX ".conv_in.weight", VAE_PREFIX ".conv_in.bias",
                   latent, 4, H, W, 512, 3, 3, 1, 1, 1, cur) != 0) { free(cur); return -1; }

    /* mid: resnet -> attn -> resnet (512) */
    float *m1 = (float *)malloc((size_t)512 * H * W * sizeof(float));
    if (vae_resnet(v, VAE_PREFIX ".mid.block_1", cur, 512, 512, H, W, m1) != 0) return -1;
    free(cur); cur = m1;
    if (vae_attn(v, VAE_PREFIX ".mid.attn_1", cur, 512, H, W) != 0) return -1;
    float *m2 = (float *)malloc((size_t)512 * H * W * sizeof(float));
    if (vae_resnet(v, VAE_PREFIX ".mid.block_2", cur, 512, 512, H, W, m2) != 0) return -1;
    free(cur); cur = m2;

    /* up blocks: execution order 3,2,1,0 (sd.cpp: i = num_res-1 .. 0).
     * Tables indexed by block number (UP_CH[3] = up.3's channels). */
    static const int UP_CH[4] = { 128, 256, 512, 512 }; /* up.0..up.3 out */
    static const int UP_IN[4]  = { 256, 512, 512, 512 }; /* up.0..up.3 first resnet in */
    double t0 = omp_get_wtime();
    for (int i = 3; i >= 0; i--) {
        char prefix[128];
        int C = UP_CH[i];
        int Cin = UP_IN[i];
        for (int j = 0; j < 3; j++) {
            snprintf(prefix, sizeof(prefix), VAE_PREFIX ".up.%d.block.%d", i, j);
            int rbC = (j == 0) ? Cin : C;
            float *rb = (float *)malloc((size_t)C * H * W * sizeof(float));
            if (vae_resnet(v, prefix, cur, rbC, C, H, W, rb) != 0) return -1;
            free(cur); cur = rb;
        }
        if (i != 0) {
            snprintf(prefix, sizeof(prefix), VAE_PREFIX ".up.%d.upsample", i);
            float *up = (float *)malloc((size_t)C * (2 * H) * (2 * W) * sizeof(float));
            if (vae_upsample(v, prefix, cur, C, H, W, up) != 0) return -1;
            free(cur); cur = up;
            H *= 2; W *= 2;
        }
        fprintf(stderr, "[vae] up.%d done (H=%d W=%d): %.1fs\n", i, H, W,
                omp_get_wtime() - t0);
    }

    /* norm_out + conv_out 128 -> 3 */
    float *nw = vae_get_f32(v, VAE_PREFIX ".norm_out.weight");
    float *nb = vae_get_f32(v, VAE_PREFIX ".norm_out.bias");
    if (!nw) return -1;
    vae_gn(cur, 128, H * W, nw, nb);
    wubu_sd_silu(cur, 128 * H * W);
    if (vae_conv_q(v, VAE_PREFIX ".conv_out.weight", VAE_PREFIX ".conv_out.bias",
                   cur, 128, H, W, 3, 3, 3, 1, 1, 1, out) != 0) { free(cur); return -1; }
    free(cur);
    return 0;
}
