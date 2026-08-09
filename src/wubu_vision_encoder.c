/**
 * wubu_vision_encoder.c — 3D ViT vision encoder port (Qwen3.6-35B-A3B)
 *
 * 27-layer Vision Transformer with 3D patch embedding (temporal_patch=2),
 * spatial_merge_size=2, 16-head GQA attention, GELU activation.
 * Weights loaded from mmproj GGUF file.
 *
 * RECOVERED (2026-08-09): a marathon JB-theme batch (47f6ad2) overwrote
 * this encoder inside wubu_vision.c. Restored from git (58f0a07) into its
 * own module so the encoder AND the JB theme coexist. See
 * include/wubu_vision_encoder.h for the API.
 */
#include "wubu_vision_encoder.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ================================================================
// LayerNorm
// ================================================================
void vision_layer_norm(const float *x, int n, int d,
                       const float *weight, const float *bias, float eps,
                       float *out) {
    for (int s = 0; s < n; s++) {
        const float *inp = x + s * d;
        double mean = 0.0, var = 0.0;
        for (int i = 0; i < d; i++) mean += inp[i];
        mean /= d;
        for (int i = 0; i < d; i++) var += (inp[i] - mean) * (inp[i] - mean);
        var = var / d;
        float inv_std = 1.0f / (sqrtf((float)var) + eps);
        for (int i = 0; i < d; i++)
            out[s * d + i] = (inp[i] - (float)mean) * inv_std * weight[i] + bias[i];
    }
}

// ================================================================
// Load vision encoder weights from GGUF
// ================================================================
bool vision_encoder_init(vision_encoder_t *enc, const char *path) {
    memset(enc, 0, sizeof(*enc));
    
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) { fprintf(stderr, "Vision: failed to open %s\n", path); return false; }
    gguf_buffer_data(ctx);
    
    // Load top-level tensors
    #define LOAD_TENSOR(name, ptr) do { \
        gguf_tensor_info *t__ = gguf_find_tensor(ctx, name); \
        if (t__) { \
            int64_t ne__ = 1; for (int d__ = 0; d__ < t__->n_dims; d__++) ne__ *= t__->dims[d__]; \
            ptr = (float *)malloc(ne__ * sizeof(float)); \
            gguf_read_tensor_f32(ctx, t__, ptr, ne__); \
        } else fprintf(stderr, "Vision: missing %s\n", name); \
    } while(0)
    
    LOAD_TENSOR("v.patch_embd.weight", enc->patch_embd_weight);
    LOAD_TENSOR("v.patch_embd.weight.1", enc->patch_embd_weight2);
    LOAD_TENSOR("v.patch_embd.bias", enc->patch_embd_bias);
    LOAD_TENSOR("v.position_embd.weight", enc->pos_embd_weight);
    LOAD_TENSOR("v.post_ln.weight", enc->post_ln_weight);
    LOAD_TENSOR("v.post_ln.bias", enc->post_ln_bias);
    LOAD_TENSOR("mm.0.weight", enc->mm0_weight);
    LOAD_TENSOR("mm.0.bias", enc->mm0_bias);
    LOAD_TENSOR("mm.2.weight", enc->mm2_weight);
    LOAD_TENSOR("mm.2.bias", enc->mm2_bias);
    
    // Load per-layer tensors
    for (int l = 0; l < V_N_LAYERS; l++) {
        char name[256];
        vision_layer_weights_t *layer = &enc->layers[l];
        
        snprintf(name, sizeof(name), "v.blk.%d.ln1.weight", l);
        LOAD_TENSOR(name, layer->ln1_weight);
        snprintf(name, sizeof(name), "v.blk.%d.ln1.bias", l);
        LOAD_TENSOR(name, layer->ln1_bias);
        
        snprintf(name, sizeof(name), "v.blk.%d.attn_qkv.weight", l);
        LOAD_TENSOR(name, layer->attn_qkv_weight);
        snprintf(name, sizeof(name), "v.blk.%d.attn_qkv.bias", l);
        LOAD_TENSOR(name, layer->attn_qkv_bias);
        
        snprintf(name, sizeof(name), "v.blk.%d.attn_out.weight", l);
        LOAD_TENSOR(name, layer->attn_out_weight);
        snprintf(name, sizeof(name), "v.blk.%d.attn_out.bias", l);
        LOAD_TENSOR(name, layer->attn_out_bias);
        
        snprintf(name, sizeof(name), "v.blk.%d.ln2.weight", l);
        LOAD_TENSOR(name, layer->ln2_weight);
        snprintf(name, sizeof(name), "v.blk.%d.ln2.bias", l);
        LOAD_TENSOR(name, layer->ln2_bias);
        
        snprintf(name, sizeof(name), "v.blk.%d.ffn_up.weight", l);
        LOAD_TENSOR(name, layer->ffn_up_weight);
        snprintf(name, sizeof(name), "v.blk.%d.ffn_up.bias", l);
        LOAD_TENSOR(name, layer->ffn_up_bias);
        
        snprintf(name, sizeof(name), "v.blk.%d.ffn_down.weight", l);
        LOAD_TENSOR(name, layer->ffn_down_weight);
        snprintf(name, sizeof(name), "v.blk.%d.ffn_down.bias", l);
        LOAD_TENSOR(name, layer->ffn_down_bias);
        
        layer->loaded = (layer->attn_qkv_weight != NULL);
        if (!layer->loaded)
            fprintf(stderr, "Vision: layer %d incomplete\n", l);
    }
    
    gguf_close(ctx);
    enc->loaded = (enc->patch_embd_weight != NULL);
    if (enc->loaded) printf("Vision encoder: loaded %d layers\n", V_N_LAYERS);
    fflush(stdout);
    return enc->loaded;
}

void vision_encoder_free(vision_encoder_t *enc) {
    if (!enc) return;
    free(enc->patch_embd_weight);
    free(enc->patch_embd_weight2);
    free(enc->patch_embd_bias);
    free(enc->pos_embd_weight);
    free(enc->post_ln_weight);
    free(enc->post_ln_bias);
    free(enc->mm0_weight);
    free(enc->mm0_bias);
    free(enc->mm2_weight);
    free(enc->mm2_bias);
    for (int l = 0; l < V_N_LAYERS; l++) {
        vision_layer_weights_t *layer = &enc->layers[l];
        free(layer->ln1_weight); free(layer->ln1_bias);
        free(layer->attn_qkv_weight); free(layer->attn_qkv_bias);
        free(layer->attn_out_weight); free(layer->attn_out_bias);
        free(layer->ln2_weight); free(layer->ln2_bias);
        free(layer->ffn_up_weight); free(layer->ffn_up_bias);
        free(layer->ffn_down_weight); free(layer->ffn_down_bias);
    }
    memset(enc, 0, sizeof(*enc));
}

// ================================================================
// Single ViT layer forward
// ================================================================
static void vision_layer_forward(
    const vision_layer_weights_t *w,
    const float *x, int n,     // x: [n, V_HIDDEN]
    float *output)             // output: [n, V_HIDDEN]
{
    // LayerNorm 1
    float *normed = (float *)malloc(n * V_HIDDEN * sizeof(float));
    vision_layer_norm(x, n, V_HIDDEN, w->ln1_weight, w->ln1_bias, 1e-6f, normed);
    
    // QKV projection: [n, 1152] @ [1152, 3456] -> [n, 3456]
    float *qkv = (float *)malloc(n * 3456 * sizeof(float));
    #pragma omp parallel for collapse(2)
    for (int s = 0; s < n; s++)
        for (int j = 0; j < 3456; j++) {
            double sum = w->attn_qkv_bias[j];
            for (int k = 0; k < V_HIDDEN; k++)
                sum += (double)normed[s * V_HIDDEN + k] * (double)w->attn_qkv_weight[k * 3456 + j];
            qkv[s * 3456 + j] = (float)sum;
        }
    
    // Multi-head attention: split Q [n,1152], K [n,1152], V [n,1152]
    // 16 heads, head_dim=72
    float *q = qkv;  // [n, 1152]
    float *k = qkv + n * V_HIDDEN;  // [n, 1152]
    float *v = qkv + n * V_HIDDEN * 2;  // [n, 1152]
    
    float *attn_out = (float *)calloc(n * V_HIDDEN, sizeof(float));
    
    // Per-head attention: out[h][s] = sum_t softmax(s@t/√d) * v[t]
    float scale = 1.0f / sqrtf((float)V_HEAD_DIM);
    for (int h = 0; h < V_N_HEADS; h++) {
        for (int s = 0; s < n; s++) {
            const float *q_s = q + s * V_HIDDEN + h * V_HEAD_DIM;
            
            // Compute attention scores for all t
            float *scores = (float *)malloc(n * sizeof(float));
            float max_s = -1e30f;
            for (int t = 0; t < n; t++) {
                const float *k_t = k + t * V_HIDDEN + h * V_HEAD_DIM;
                double sum = 0.0;
                for (int d = 0; d < V_HEAD_DIM; d++)
                    sum += (double)q_s[d] * (double)k_t[d];
                scores[t] = (float)(sum * scale);
                if (scores[t] > max_s) max_s = scores[t];
            }
            
            // Softmax
            double sum_exp = 0.0;
            for (int t = 0; t < n; t++)
                sum_exp += expf(scores[t] - max_s);
            float inv_sum = 1.0f / ((float)sum_exp + 1e-30f);
            
            // Weighted sum of V
            float *out_s = attn_out + s * V_HIDDEN + h * V_HEAD_DIM;
            memset(out_s, 0, V_HEAD_DIM * sizeof(float));
            for (int t = 0; t < n; t++) {
                float wgt = expf(scores[t] - max_s) * inv_sum;
                const float *v_t = v + t * V_HIDDEN + h * V_HEAD_DIM;
                for (int d = 0; d < V_HEAD_DIM; d++)
                    out_s[d] += wgt * v_t[d];
            }
            free(scores);
        }
    }
    
    // Attention output projection
    float *attn_proj = (float *)malloc(n * V_HIDDEN * sizeof(float));
    #pragma omp parallel for collapse(2)
    for (int s = 0; s < n; s++)
        for (int j = 0; j < V_HIDDEN; j++) {
            double sum = w->attn_out_bias[j];
            for (int k = 0; k < V_HIDDEN; k++)
                sum += (double)attn_out[s * V_HIDDEN + k] * (double)w->attn_out_weight[k * V_HIDDEN + j];
            attn_proj[s * V_HIDDEN + j] = (float)sum;
        }
    
    // Residual: x = x + attn_proj
    float *residual = (float *)malloc(n * V_HIDDEN * sizeof(float));
    for (int s = 0; s < n * V_HIDDEN; s++)
        residual[s] = x[s] + attn_proj[s];
    
    // LayerNorm 2
    float *normed2 = (float *)malloc(n * V_HIDDEN * sizeof(float));
    vision_layer_norm(residual, n, V_HIDDEN, w->ln2_weight, w->ln2_bias, 1e-6f, normed2);
    
    // FFN up projection + GELU
    float *ffn_up = (float *)malloc(n * V_INTERMEDIATE * sizeof(float));
    #pragma omp parallel for collapse(2)
    for (int s = 0; s < n; s++)
        for (int j = 0; j < V_INTERMEDIATE; j++) {
            double sum = w->ffn_up_bias[j];
            for (int k = 0; k < V_HIDDEN; k++)
                sum += (double)normed2[s * V_HIDDEN + k] * (double)w->ffn_up_weight[k * V_INTERMEDIATE + j];
            ffn_up[s * V_INTERMEDIATE + j] = gelu_tanh((float)sum);
        }
    
    // FFN down projection
    float *ffn_down = (float *)malloc(n * V_HIDDEN * sizeof(float));
    #pragma omp parallel for collapse(2)
    for (int s = 0; s < n; s++)
        for (int j = 0; j < V_HIDDEN; j++) {
            double sum = w->ffn_down_bias[j];
            for (int k = 0; k < V_INTERMEDIATE; k++)
                sum += (double)ffn_up[s * V_INTERMEDIATE + k] * (double)w->ffn_down_weight[k * V_HIDDEN + j];
            ffn_down[s * V_HIDDEN + j] = (float)sum;
        }
    
    // Residual: output = residual + ffn_down
    for (int s = 0; s < n * V_HIDDEN; s++)
        output[s] = residual[s] + ffn_down[s];
    
    free(normed); free(qkv); free(attn_out); free(attn_proj);
    free(residual); free(normed2); free(ffn_up); free(ffn_down);
}

// ================================================================
// Full vision encoder forward
// ================================================================
void vision_encoder_forward(const vision_encoder_t *enc,
                            const float *pixels, int B, int C, int H, int W,
                            float *output) {
    if (!enc->loaded) return;
    
    int patch_h = H / V_PATCH_SIZE;
    int patch_w = W / V_PATCH_SIZE;
    int merged_h = patch_h / 2;  // spatial_merge_size=2
    int merged_w = patch_w / 2;
    int n_merged = merged_h * merged_w * V_TEMP_PATCH;  // *2 for temporal
    
    if (n_merged > V_MAX_POS) n_merged = V_MAX_POS;
    
    // Allocate hidden states (pre-merge: n_patches_total = patch_h * patch_w * V_TEMP_PATCH)
    int n_patches_total = patch_h * patch_w * V_TEMP_PATCH;
    if (n_patches_total > V_MAX_POS) n_patches_total = V_MAX_POS; // cap to position embedding limit
    float *hidden = (float *)malloc(n_patches_total * V_HIDDEN * sizeof(float));
    
    // === Patch embedding (3D convolution) ===
    // Two temporal kernels: one for each temporal_patch
    for (int b = 0; b < B; b++) {
        for (int tp = 0; tp < V_TEMP_PATCH; tp++) {
            const float *kernel = (tp == 0) ? enc->patch_embd_weight : enc->patch_embd_weight2;
            
            for (int ph = 0; ph < patch_h; ph++) {
                for (int pw = 0; pw < patch_w; pw++) {
                    int idx = (b * V_TEMP_PATCH + tp) * (patch_h * patch_w) + ph * patch_w + pw;
                    if (idx >= n_patches_total) break;
                    
                    float *out = hidden + idx * V_HIDDEN;
                    memcpy(out, enc->patch_embd_bias, V_HIDDEN * sizeof(float));
                    
                    // 3D convolution: [16,16,3,1152] kernel over [16×16×3] patch
                    for (int c = 0; c < C; c++) {
                        for (int ky = 0; ky < V_PATCH_SIZE; ky++) {
                            for (int kx = 0; kx < V_PATCH_SIZE; kx++) {
                                float pixel = pixels[(b * C + c) * (H * W) + (ph * V_PATCH_SIZE + ky) * W + (pw * V_PATCH_SIZE + kx)];
                                for (int f = 0; f < V_HIDDEN; f++) {
                                    out[f] += pixel * kernel[(ky * V_PATCH_SIZE + kx) * (C * V_HIDDEN) + c * V_HIDDEN + f];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // === Add position embeddings (before ViT, on un-merged patches) ===
    for (int i = 0; i < n_patches_total && i < V_MAX_POS; i++)
        for (int f = 0; f < V_HIDDEN; f++)
            hidden[i * V_HIDDEN + f] += enc->pos_embd_weight[f * V_MAX_POS + i];
    
    // === 27 ViT layers ===
    float *vit_inp = hidden;
    float *vit_out = (float *)malloc(n_patches_total * V_HIDDEN * sizeof(float));
    
    for (int l = 0; l < V_N_LAYERS; l++) {
        if (!enc->layers[l].loaded) {
            memcpy(vit_out, vit_inp, n_patches_total * V_HIDDEN * sizeof(float));
        } else {
            vision_layer_forward(&enc->layers[l], vit_inp, n_patches_total, vit_out);
        }
        // Swap
        if (l < V_N_LAYERS - 1) {
            memcpy(vit_inp, vit_out, n_patches_total * V_HIDDEN * sizeof(float));
        }
    }
    
    // === Post layer norm ===
    vision_layer_norm(vit_out, n_patches_total, V_HIDDEN, enc->post_ln_weight, enc->post_ln_bias, 1e-6f, vit_out);
    
    // === Spatial merge: 2×2 concatenate (not average!) ===
    // Each group of 4 spatial neighbors → concatenated along feature dim → 4608
    #define V_MERGE_DIM (V_HIDDEN * 4)  // 4608
    float *merged = (float *)malloc(n_merged * V_MERGE_DIM * sizeof(float));
    memset(merged, 0, n_merged * V_MERGE_DIM * sizeof(float));
    
    for (int tp = 0; tp < V_TEMP_PATCH; tp++) {
        int temporal_base = tp * (patch_h * patch_w);  // contiguous: tp=0:0..255, tp=1:256..511
        for (int mh = 0; mh < merged_h; mh++) {
            for (int mw = 0; mw < merged_w; mw++) {
                int dst_idx = tp * (merged_h * merged_w) + mh * merged_w + mw;
                float *dst_row = merged + dst_idx * V_MERGE_DIM;
                for (int dy = 0; dy < 2; dy++) {
                    for (int dx = 0; dx < 2; dx++) {
                        int src_flat = temporal_base + (mh * 2 + dy) * patch_w + (mw * 2 + dx);
                        const float *src_row = vit_out + src_flat * V_HIDDEN;
                        int chan_offset = (dy * 2 + dx) * V_HIDDEN;
                        for (int f = 0; f < V_HIDDEN; f++)
                            dst_row[chan_offset + f] = src_row[f];
                    }
                }
            }
        }
    }
    
    // === MMProj: mm0 → GELU → mm2 (per merged token) ===
    if (enc->mm0_weight) {
        #define V_PROJ_DIM V_MERGE_DIM  // 4608
        for (int i = 0; i < n_merged; i++) {
            const float *row = merged + i * V_PROJ_DIM;
            float mm0_buf[4608];
            // mm.0: [4608] @ [4608,4608] + bias → GELU
            for (int j = 0; j < 4608; j++) {
                double sum = enc->mm0_bias[j];
                for (int k = 0; k < 4608; k++)
                    sum += (double)row[k] * (double)enc->mm0_weight[k * 4608 + j];
                mm0_buf[j] = gelu_tanh((float)sum);
            }
            // mm.2: [4608] @ [4608,2048] + bias → output
            float *out_row = output + i * V_OUT_HIDDEN;
            for (int j = 0; j < V_OUT_HIDDEN; j++) {
                double sum = enc->mm2_bias[j];
                for (int k = 0; k < 4608; k++)
                    sum += (double)mm0_buf[k] * (double)enc->mm2_weight[k * V_OUT_HIDDEN + j];
                out_row[j] = (float)sum;
            }
        }
    } else {
        // No merger — just flatten (shouldn't happen for Qwen3VL)
        for (int i = 0; i < n_merged; i++)
            memcpy(output + i * V_OUT_HIDDEN, merged + i * V_MERGE_DIM,
                   (V_MERGE_DIM < V_OUT_HIDDEN ? V_MERGE_DIM : V_OUT_HIDDEN) * sizeof(float));
    }
    
    free(hidden);
    free(vit_out);
    free(merged);
}
