/**
 * wubu_siglip.c — SigLIP vision encoder (SmolVLM2-2.2B mmproj)
 *
 * The SmolVLM2 mmproj carries a SigLIP-SO400M vision tower:
 *   - patch embed: conv 14x14, stride 14, 3 -> 1152 channels
 *   - position embd: learned [729, 1152] (27x27 grid for 378x378 input;
 *     the GGUF says image_size 384 but position_embd.dims[1] = 729 =
 *     27*27 -> the encoder consumes 27x27 patches)
 *   - 27 transformer blocks: LN -> MHSA (16 heads, hd 72, separate
 *     q/k/v with biases) -> residual; LN -> MLP (1152->4304 GELU ->
 *     1152) -> residual
 *   - post_ln LayerNorm
 *   - projector: 3x3 spatial merge (scale_factor 3) -> fc
 *     [9*1152=10368, 2048] (no bias) -> 81 tokens x 2048
 *
 * Written from the GGUF schema directly (the V1 llama.cpp does not
 * build SmolVLM2; the recovered wubu_vision_encoder.c is the Qwen3.6
 * 3D-ViT and does NOT match these tensors).
 */
#include "wubu_siglip.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ================================================================
// LayerNorm (mean/var in double, like the model's reference impl)
// ================================================================
static void siglip_ln(const float *x, int n, int d,
                      const float *weight, const float *bias, float eps,
                      float *out) {
    for (int s = 0; s < n; s++) {
        const float *inp = x + (size_t)s * d;
        float *oup = out + (size_t)s * d;
        double mean = 0.0;
        for (int i = 0; i < d; i++) mean += inp[i];
        mean /= d;
        double var = 0.0;
        for (int i = 0; i < d; i++) { double dd = inp[i] - mean; var += dd * dd; }
        var /= d;
        float inv_std = 1.0f / sqrtf((float)var + eps);
        for (int i = 0; i < d; i++)
            oup[i] = (inp[i] - (float)mean) * inv_std * weight[i] + (bias ? bias[i] : 0.0f);
    }
}

// ================================================================
// GELU (tanh approximation — llama.cpp ggml_gelu)
// ================================================================
static inline float siglip_gelu(float x) {
    /* Match llama.cpp ggml_gelu_f32 exactly:
     *   0.5f*x*(1.0f + tanhf(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)))
     * NOT the factored x^3 form — the (1 + c*x^2) form lets the compiler
     * emit an FMA for the inner expression under -ffp-contract=fast,
     * matching llama.cpp's bit-exact rounding. */
    return 0.5f * x * (1.0f + tanhf(0.79788456080286535587989211986876f * x * (1.0f + 0.044715f * x * x)));
}

// ================================================================
// Loader
// ================================================================
bool wubu_siglip_init(wubu_siglip_t *enc, const char *path) {
    memset(enc, 0, sizeof(*enc));
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) { fprintf(stderr, "SigLIP: failed to open %s\n", path); return false; }
    gguf_buffer_data(ctx);

    #define LOAD(name, ptr) do { \
        gguf_tensor_info *t__ = gguf_find_tensor(ctx, name); \
        if (t__) { \
            int64_t ne__ = 1; for (int d__ = 0; d__ < t__->n_dims; d__++) ne__ *= t__->dims[d__]; \
            ptr = (float *)malloc((size_t)ne__ * sizeof(float)); \
            gguf_read_tensor_f32(ctx, t__, ptr, ne__); \
        } else { fprintf(stderr, "SigLIP: missing %s\n", name); goto fail; } \
    } while (0)

    LOAD("v.patch_embd.weight", enc->patch_w);   // [14,14,3] -> 588
    LOAD("v.patch_embd.bias",   enc->patch_b);   // [1152]
    LOAD("v.position_embd.weight", enc->pos_w);  // [1152, 729]
    LOAD("v.post_ln.weight",    enc->post_ln_w); // [1152]
    LOAD("v.post_ln.bias",      enc->post_ln_b); // [1152]

    enc->blocks = (wubu_siglip_block_t *)calloc(SIGLIP_N_BLOCKS, sizeof(wubu_siglip_block_t));
    if (!enc->blocks) goto fail;
    for (int l = 0; l < SIGLIP_N_BLOCKS; l++) {
        char nm[128];
        wubu_siglip_block_t *b = &enc->blocks[l];
        snprintf(nm, sizeof(nm), "v.blk.%d.ln1.weight", l); LOAD(nm, b->ln1_w);
        snprintf(nm, sizeof(nm), "v.blk.%d.ln1.bias",   l); LOAD(nm, b->ln1_b);
        snprintf(nm, sizeof(nm), "v.blk.%d.attn_q.weight", l); LOAD(nm, b->q_w);
        snprintf(nm, sizeof(nm), "v.blk.%d.attn_q.bias",   l); LOAD(nm, b->q_b);
        snprintf(nm, sizeof(nm), "v.blk.%d.attn_k.weight", l); LOAD(nm, b->k_w);
        snprintf(nm, sizeof(nm), "v.blk.%d.attn_k.bias",   l); LOAD(nm, b->k_b);
        snprintf(nm, sizeof(nm), "v.blk.%d.attn_v.weight", l); LOAD(nm, b->v_w);
        snprintf(nm, sizeof(nm), "v.blk.%d.attn_v.bias",   l); LOAD(nm, b->v_b);
        snprintf(nm, sizeof(nm), "v.blk.%d.attn_out.weight", l); LOAD(nm, b->o_w);
        snprintf(nm, sizeof(nm), "v.blk.%d.attn_out.bias",   l); LOAD(nm, b->o_b);
        snprintf(nm, sizeof(nm), "v.blk.%d.ln2.weight", l); LOAD(nm, b->ln2_w);
        snprintf(nm, sizeof(nm), "v.blk.%d.ln2.bias",   l); LOAD(nm, b->ln2_b);
        /* llama.cpp mtmd clip.cpp swaps ffn up/down for SmolVLM2
         * (is_ffn_swapped: ff_down_w->ne[0] == n_embd). The GGUF file
         * stores ffn_up=(4304,1152) and ffn_down=(1152,4304), i.e. the
         * names are reversed relative to the computation. After the
         * swap, dims are (in, out): up maps 1152->4304, down 4304->1152. */
        snprintf(nm, sizeof(nm), "v.blk.%d.ffn_up.weight", l);   LOAD(nm, b->ffn_down_w);  // file "up" is the real DOWN
        snprintf(nm, sizeof(nm), "v.blk.%d.ffn_up.bias",   l);   LOAD(nm, b->ffn_down_b);
        snprintf(nm, sizeof(nm), "v.blk.%d.ffn_down.weight", l); LOAD(nm, b->ffn_up_w);    // file "down" is the real UP
        snprintf(nm, sizeof(nm), "v.blk.%d.ffn_down.bias",   l); LOAD(nm, b->ffn_up_b);
    }
    LOAD("mm.model.fc.weight", enc->proj_w);  // [10368, 2048] = (in, out)
    #undef LOAD
    gguf_close(ctx);
    return true;

fail:
    gguf_close(ctx);
    wubu_siglip_free(enc);
    return false;
}

void wubu_siglip_free(wubu_siglip_t *enc) {
    if (!enc) return;
    free(enc->patch_w); free(enc->patch_b); free(enc->pos_w);
    free(enc->post_ln_w); free(enc->post_ln_b); free(enc->proj_w);
    if (enc->blocks) {
        for (int l = 0; l < SIGLIP_N_BLOCKS; l++) {
            wubu_siglip_block_t *b = &enc->blocks[l];
            free(b->ln1_w); free(b->ln1_b); free(b->q_w); free(b->q_b);
            free(b->k_w); free(b->k_b); free(b->v_w); free(b->v_b);
            free(b->o_w); free(b->o_b); free(b->ln2_w); free(b->ln2_b);
            free(b->ffn_up_w); free(b->ffn_up_b); free(b->ffn_down_w); free(b->ffn_down_b);
        }
        free(enc->blocks);
        enc->blocks = NULL;
    }
    memset(enc, 0, sizeof(*enc));
}

// ================================================================
// Forward
// input:  [B, C, H, W] float, values in [0,1] (caller applies mean/std)
// output: [B, 81, 2048] projected embeddings (3x3 merge, fc 10368->2048)
// ================================================================
void wubu_siglip_forward(wubu_siglip_t *enc, const float *input, int B, int C, int H, int W,
                         float *output) {
    const int PH = 14, PW = 14;           // patch size
    const int gh = H / PH, gw = W / PW;   // grid (27x27 for 378x378)
    const int NP = gh * gw;               // patches (729)
    if (NP != SIGLIP_N_POS) {
        fprintf(stderr, "SigLIP: image %dx%d -> grid %dx%d (%d patches); expected %d\n",
                H, W, gh, gw, NP, SIGLIP_N_POS);
        return;
    }
    const int NTOK = NP / 9;              // 81 merged tokens

    for (int b = 0; b < B; b++) {
        const float *img = input + (size_t)b * C * H * W;
        float *tok = (float *)malloc((size_t)NP * SIGLIP_HIDDEN * sizeof(float));
        float *buf = (float *)malloc((size_t)NP * SIGLIP_HIDDEN * sizeof(float));
        float *q   = (float *)malloc((size_t)NP * SIGLIP_HIDDEN * sizeof(float));
        float *k   = (float *)malloc((size_t)NP * SIGLIP_HIDDEN * sizeof(float));
        float *v   = (float *)malloc((size_t)NP * SIGLIP_HIDDEN * sizeof(float));
        float *o   = (float *)malloc((size_t)NP * SIGLIP_HIDDEN * sizeof(float));
        float *ffn_in = (float *)malloc((size_t)NP * SIGLIP_INTERMEDIATE * sizeof(float));
        if (!tok || !buf || !q || !k || !v || !o || !ffn_in) {
            fprintf(stderr, "SigLIP: alloc failed\n");
            free(tok); free(buf); free(q); free(k); free(v); free(o); free(ffn_in);
            return;
        }

        // ---- Patch embed (conv 14x14 stride 14, 3ch -> 1152) ----
        // patch_w layout: GGUF dims (14,14,3,1152) with dims[0] contiguous:
        //   mem index = ky + 14*kx + 196*ci + 588*c
        // out[gy,gx,c] = sum_{ky,kx,ci} img[...] * patch_w[ky,kx,ci,c] + patch_b[c]
        for (int gy = 0; gy < gh; gy++) {
            for (int gx = 0; gx < gw; gx++) {
                float *t = tok + ((size_t)gy * gw + gx) * SIGLIP_HIDDEN;
                for (int c = 0; c < SIGLIP_HIDDEN; c++) {
                    float sum = enc->patch_b[c];
                    const float *pw_c = enc->patch_w + (size_t)c * (14 * 14 * 3);
                    for (int ci = 0; ci < C; ci++) {
                        const float *prow = img + ci * H * W + (size_t)gy * PH * W + gx * PW;
                        for (int ky = 0; ky < PH; ky++)
                            for (int kx = 0; kx < PW; kx++) {
                                int widx = ky + 14 * kx + 196 * ci;
                                sum += prow[(size_t)ky * W + kx] * pw_c[widx];
                            }
                    }
                    t[c] = sum;
                }
            }
        }
        // ---- Add position embeddings ----
        for (int p = 0; p < NP; p++) {
            float *t = tok + (size_t)p * SIGLIP_HIDDEN;
            const float *pw = enc->pos_w + (size_t)p * SIGLIP_HIDDEN;  // [1152, 729] col p
            for (int c = 0; c < SIGLIP_HIDDEN; c++) t[c] += pw[c];
        }

        // ---- 27 blocks ----
        for (int l = 0; l < SIGLIP_N_BLOCKS; l++) {
            wubu_siglip_block_t *bl = &enc->blocks[l];
            const float eps = 1e-5f;
            // LN1 -> QKV
            siglip_ln(tok, NP, SIGLIP_HIDDEN, bl->ln1_w, bl->ln1_b, eps, buf);
            for (int h = 0; h < SIGLIP_HEADS; h++) {
                const int hd = SIGLIP_HEAD_DIM;
                for (int p = 0; p < NP; p++) {
                    /* LN output is the FULL 1152-dim token vector; head h
                     * selects the OUTPUT rows (h*hd..h*hd+hd) of Wq/Wk/Wv.
                     * (The old buf + h*hd offset read past the buffer at
                     * the last head — heap overflow, non-deterministic.) */
                    const float *x = buf + (size_t)p * SIGLIP_HIDDEN;
                    float *qq = q + ((size_t)p * SIGLIP_HEADS + h) * hd;
                    float *kk = k + ((size_t)p * SIGLIP_HEADS + h) * hd;
                    float *vv = v + ((size_t)p * SIGLIP_HEADS + h) * hd;
                    for (int i = 0; i < hd; i++) {
                        /* GGUF 2D weight (in, out) column-major:
                         * W[out][in] at in + n_in*out, so with oi=out,
                         * j=in: offset = oi*n_in + j = oi*HIDDEN + j. */
                        float sq = bl->q_b[h * hd + i], sk = bl->k_b[h * hd + i], sv = bl->v_b[h * hd + i];
                        const int oi = h * hd + i;
                        for (int j = 0; j < SIGLIP_HIDDEN; j++) {
                            sq += x[j] * bl->q_w[(size_t)oi * SIGLIP_HIDDEN + j];
                            sk += x[j] * bl->k_w[(size_t)oi * SIGLIP_HIDDEN + j];
                            sv += x[j] * bl->v_w[(size_t)oi * SIGLIP_HIDDEN + j];
                        }
                        qq[i] = sq; kk[i] = sk; vv[i] = sv;
                    }
                }
            }
            // Attention (causal-free full attention over patches) with scale 1/sqrt(hd)
            const float scale = 1.0f / sqrtf((float)SIGLIP_HEAD_DIM);
            for (int h = 0; h < SIGLIP_HEADS; h++) {
                for (int p = 0; p < NP; p++) {
                    const float *qq = q + ((size_t)p * SIGLIP_HEADS + h) * SIGLIP_HEAD_DIM;
                    float *oo = o + ((size_t)p * SIGLIP_HEADS + h) * SIGLIP_HEAD_DIM;
                    memset(oo, 0, SIGLIP_HEAD_DIM * sizeof(float));
                    // softmax over all patches
                    float maxs = -1e30f;
                    for (int pp = 0; pp < NP; pp++) {
                        const float *kk = k + ((size_t)pp * SIGLIP_HEADS + h) * SIGLIP_HEAD_DIM;
                        float s = 0.0f;
                        for (int i = 0; i < SIGLIP_HEAD_DIM; i++) s += qq[i] * kk[i];
                        s *= scale;
                        if (s > maxs) maxs = s;
                    }
                    float sum = 0.0f;
                    float att[2048];  // NP max 729
                    for (int pp = 0; pp < NP; pp++) {
                        const float *kk = k + ((size_t)pp * SIGLIP_HEADS + h) * SIGLIP_HEAD_DIM;
                        float s = 0.0f;
                        for (int i = 0; i < SIGLIP_HEAD_DIM; i++) s += qq[i] * kk[i];
                        float a = expf(s * scale - maxs);
                        att[pp] = a;
                        sum += a;
                    }
                    for (int pp = 0; pp < NP; pp++) {
                        const float *vv = v + ((size_t)pp * SIGLIP_HEADS + h) * SIGLIP_HEAD_DIM;
                        float a = att[pp] / sum;
                        for (int i = 0; i < SIGLIP_HEAD_DIM; i++) oo[i] += a * vv[i];
                    }
                }
            }
            // attn out projection + residual
            for (int p = 0; p < NP; p++) {
                float *t = tok + (size_t)p * SIGLIP_HIDDEN;
                const float *oo = o + (size_t)p * SIGLIP_HIDDEN;
                for (int i = 0; i < SIGLIP_HIDDEN; i++) {
                    float s = bl->o_b[i];
                    for (int j = 0; j < SIGLIP_HIDDEN; j++)
                        s += oo[j] * bl->o_w[(size_t)i * SIGLIP_HIDDEN + j];
                    buf[p * SIGLIP_HIDDEN + i] = t[i] + s;  // residual
                }
            }
            // LN2 -> MLP (GELU) -> residual
            siglip_ln(buf, NP, SIGLIP_HIDDEN, bl->ln2_w, bl->ln2_b, eps, tok);
            for (int p = 0; p < NP; p++) {
                const float *x = tok + (size_t)p * SIGLIP_HIDDEN;
                float *fi = ffn_in + (size_t)p * SIGLIP_INTERMEDIATE;
                for (int i = 0; i < SIGLIP_INTERMEDIATE; i++) {
                    float s = bl->ffn_up_b[i];
                    for (int j = 0; j < SIGLIP_HIDDEN; j++)
                        s += x[j] * bl->ffn_up_w[(size_t)i * SIGLIP_HIDDEN + j];
                    fi[i] = siglip_gelu(s);
                }
                float *t = buf + (size_t)p * SIGLIP_HIDDEN;  // saved pre-MLP residual
                for (int i = 0; i < SIGLIP_HIDDEN; i++) {
                    float s = bl->ffn_down_b[i];
                    for (int j = 0; j < SIGLIP_INTERMEDIATE; j++)
                        s += fi[j] * bl->ffn_down_w[(size_t)i * SIGLIP_INTERMEDIATE + j];
                    tok[(size_t)p * SIGLIP_HIDDEN + i] = t[i] + s;
                }
            }
        }

        // ---- post_ln ----
        siglip_ln(tok, NP, SIGLIP_HIDDEN, enc->post_ln_w, enc->post_ln_b, 1e-5f, buf);

        // ---- Projector: 3x3 merge + fc [10368, 2048] ----
        // merge grid: 27x27 -> 9x9 groups of 3x3; token order row-major
        const int mg = 3;  // merge size
        const int og = gh / mg;  // 9
        float *merged = (float *)malloc((size_t)NTOK * 9 * SIGLIP_HIDDEN * sizeof(float));
        for (int oy = 0; oy < og; oy++) {
            for (int ox = 0; ox < og; ox++) {
                float *m = merged + ((size_t)(oy * og + ox)) * 9 * SIGLIP_HIDDEN;
                int mpos = 0;
                for (int dy = 0; dy < mg; dy++)
                    for (int dx = 0; dx < mg; dx++) {
                        int p = (oy * mg + dy) * gw + (ox * mg + dx);
                        memcpy(m + (size_t)mpos * SIGLIP_HIDDEN,
                               buf + (size_t)p * SIGLIP_HIDDEN,
                               SIGLIP_HIDDEN * sizeof(float));
                        mpos++;
                    }
            }
        }
        for (int t = 0; t < NTOK; t++) {
            const float *m = merged + (size_t)t * 9 * SIGLIP_HIDDEN;
            float *out = output + ((size_t)b * NTOK + t) * SIGLIP_PROJ;
            for (int i = 0; i < SIGLIP_PROJ; i++) {
                float s = 0.0f;
                /* W[out][in] at out*n_in + in, n_in = 9*1152 = 10368 */
                for (int j = 0; j < 9 * SIGLIP_HIDDEN; j++)
                    s += m[j] * enc->proj_w[(size_t)i * (9 * SIGLIP_HIDDEN) + j];
                out[i] = s;
            }
        }

        free(merged);
        free(tok); free(buf); free(q); free(k); free(v); free(o); free(ffn_in);
    }
}
