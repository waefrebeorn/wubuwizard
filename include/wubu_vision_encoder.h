#ifndef WUBU_VISION_ENCODER_H
#define WUBU_VISION_ENCODER_H

/**
 * wubu_vision_encoder.h -- the multimodal vision encoder (ViT + mm projector).
 *
 * The Qwen3.6-35B-A3B mmproj vision tower: 27-layer ViT with 3D patch
 * embedding (temporal_patch=2), spatial_merge_size=2, 16-head attention,
 * GELU. Loads from an mmproj GGUF and emits text-dimension vision tokens
 * for the language model.
 *
 * Kept SEPARATE from wubu_vision.h (the JB token-compression theme) -- a
 * 2026-08-02 marathon batch overwrote this encoder in wubu_vision.c; it
 * was recovered from git history (58f0a07/47f6ad2~1) into its own module
 * so both the encoder AND the JB utilities coexist.
 */

#include "gguf_reader.h"
#include <stdbool.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// Helper: GELU activation (tanh approximation)
static inline float gelu_tanh(float x) {
    // GELU with tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float c = (float)(0.7978845608028654 * (x + 0.044715 * x * x * x));
    return 0.5f * x * (1.0f + tanhf(c));
}

// Vision encoder dimensions (from GGUF)
#define V_HIDDEN       1152   // vision token dimension
#define V_INTERMEDIATE 4304   // ViT MLP hidden
#define V_N_HEADS      16     // attention heads
#define V_HEAD_DIM     72     // head_dim = 1152/16
#define V_N_LAYERS     27     // ViT depth
#define V_PATCH_SIZE   16     // spatial patch
#define V_TEMP_PATCH   2      // temporal patch (frames)
#define V_MAX_POS      2304   // max position embeddings
#define V_OUT_HIDDEN   2048   // projection to match text

// Single vision layer weights
typedef struct {
    // Pre-attention LayerNorm
    float *ln1_weight;    // [V_HIDDEN]
    float *ln1_bias;      // [V_HIDDEN]

    // Attention QKV projection (fused)
    float *attn_qkv_weight; // [V_HIDDEN, V_HIDDEN * 3] = [1152, 3456]
    float *attn_qkv_bias;   // [3456]

    // Attention output projection
    float *attn_out_weight; // [V_HIDDEN, V_HIDDEN] = [1152, 1152]
    float *attn_out_bias;   // [1152]

    // Post-attention LayerNorm
    float *ln2_weight;    // [V_HIDDEN]
    float *ln2_bias;      // [V_HIDDEN]

    // FFN up projection
    float *ffn_up_weight; // [V_HIDDEN, V_INTERMEDIATE] = [1152, 4304]
    float *ffn_up_bias;   // [4304]

    // FFN down projection
    float *ffn_down_weight; // [V_INTERMEDIATE, V_HIDDEN] = [4304, 1152]
    float *ffn_down_bias;   // [1152]

    bool loaded;
} vision_layer_weights_t;

// Full vision encoder
typedef struct {
    // Patch embedding (2 kernels for temporal_patch_size=2)
    float *patch_embd_weight;   // [16, 16, 3, 1152] conv kernel
    float *patch_embd_weight2;  // [16, 16, 3, 1152] temporal kernel 2
    float *patch_embd_bias;     // [1152]

    // Position embeddings
    float *pos_embd_weight;     // [1152, V_MAX_POS] = [1152, 2304]

    // Post-norm
    float *post_ln_weight;      // [V_HIDDEN]
    float *post_ln_bias;        // [V_HIDDEN]

    // MM projector (2-layer MLP with GELU)
    float *mm0_weight;          // [4608, 4608]
    float *mm0_bias;            // [4608]
    float *mm2_weight;          // [4608, V_OUT_HIDDEN] = [4608, 2048]
    float *mm2_bias;            // [V_OUT_HIDDEN]

    // ViT transformer layers
    vision_layer_weights_t layers[V_N_LAYERS];

    bool loaded;
} vision_encoder_t;

// Load vision encoder weights from an mmproj GGUF
bool vision_encoder_init(vision_encoder_t *enc, const char *path);

// Free all encoder weights
void vision_encoder_free(vision_encoder_t *enc);

// Forward pass.
// input:  [B, C, H, W] float, normalized [0,1]
// output: [B, n_patches, V_OUT_HIDDEN] where n_patches = (H/16)*(W/16)/(2*2) * temporal_patches
void vision_encoder_forward(const vision_encoder_t *enc,
                            const float *input, int B, int C, int H, int W,
                            float *output);

#ifdef __cplusplus
}
#endif

#endif // WUBU_VISION_ENCODER_H
