#ifndef WUBU_VISION_MOONDREAM_H
#define WUBU_VISION_MOONDREAM_H

/**
 * Moondream3 Vision Encoder — SigLIP-style ViT C Port
 *
 * Architecture (from config.py VisionConfig):
 *   - Depth:      27 layers
 *   - Hidden:     1152 (enc_dim)
 *   - Intermediate: 4304 (enc_ff_dim)
 *   - Heads:      16 (head_dim=72)
 *   - Patch size: 14×14
 *   - Crop size:  378×378
 *   - Grid:       27×27 = 729 patches
 *   - Proj out:   2048 (proj_out_dim)
 *   - Proj inner: 8192 (proj_inner_dim)
 *   - Activation: GELU (tanh approx)
 *   - Weights:    f32, loaded from moondream3_vision_weights.bin
 *
 * Forward: patch_embed → 27× ViT block → post_ln → proj_mlp → exp_map → Poincaré
 */

#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── Constants ──────────────────────────────────────────────────────────
#define VM_ENC_DIM       1152
#define VM_ENC_FF_DIM    4304
#define VM_ENC_N_LAYERS  27
#define VM_ENC_N_HEADS   16
#define VM_HEAD_DIM      72   // 1152 / 16
#define VM_PATCH_SIZE    14
#define VM_CROP_SIZE     378
#define VM_GRID_SIZE     27   // 378 / 14
#define VM_N_PATCHES     729  // 27 * 27
#define VM_PATCH_DIM     588  // 14 * 14 * 3
#define VM_PROJ_INNER    8192
#define VM_PROJ_OUT      2048

// ── Weight structure (mirrors build_vision_model layout) ───────────────
typedef struct {
    // Patch embedding: Linear(588 → 1152)
    float *patch_emb_weight;  // [1152, 588]
    float *patch_emb_bias;    // [1152]

    // Position embedding: learned
    float *pos_emb;           // [1, 729, 1152]

    // Post-layer norm
    float *post_ln_weight;    // [1152]
    float *post_ln_bias;      // [1152]

    // Projection MLP (global + reconstructed → output tokens)
    float *proj_mlp_fc1_weight; // [8192, 2304]
    float *proj_mlp_fc1_bias;   // [8192]
    float *proj_mlp_fc2_weight; // [2048, 8192]
    float *proj_mlp_fc2_bias;   // [2048]

    // Transformer blocks
    struct {
        float *ln1_weight;        // [1152]
        float *ln1_bias;          // [1152]
        float *attn_qkv_weight;   // [3456, 1152]
        float *attn_qkv_bias;     // [3456]
        float *attn_proj_weight;  // [1152, 1152]
        float *attn_proj_bias;    // [1152]
        float *ln2_weight;        // [1152]
        float *ln2_bias;          // [1152]
        float *mlp_fc1_weight;    // [4304, 1152]
        float *mlp_fc1_bias;      // [4304]
        float *mlp_fc2_weight;    // [1152, 4304]
        float *mlp_fc2_bias;      // [1152]
    } blocks[VM_ENC_N_LAYERS];
} vm_weights_t;

// ── Runtime state ──────────────────────────────────────────────────────
typedef struct {
    vm_weights_t w;
    bool loaded;
    // Scratch buffers for intermediate computations
    float *scratch;  // allocated at init, sized for max temp tensor
} vm_state_t;

// ── API ────────────────────────────────────────────────────────────────

/**
 * Load vision encoder weights from binary dump.
 * @param state      Uninitialized state (caller allocates)
 * @param bin_path   Path to moondream3_vision_weights.bin
 * @param index_path Path to moondream3_vision_index.json
 * @return true on success
 */
bool vm_init(vm_state_t *state, const char *bin_path, const char *index_path);

/**
 * Free all allocated memory.
 */
void vm_free(vm_state_t *state);

/**
 * Run vision encoder forward pass.
 * @param state   Initialized state
 * @param pixels  Input image [3, H, W] float32, RGB, normalized [-1, 1]
 * @param H       Image height (must == VM_CROP_SIZE)
 * @param W       Image width  (must == VM_CROP_SIZE)
 * @param output  Output buffer [VM_N_PATCHES, VM_PROJ_OUT] flattened row-major
 */
void vm_forward(vm_state_t *state, const float *pixels, int H, int W, float *output);

// ── Utility ────────────────────────────────────────────────────────────

/**
 * Create patches from image (equivalent to vision.create_patches).
 * @param pixels    [3, H, W] float32, normalized [-1, 1]
 * @param H, W      Image dimensions (H == W == VM_CROP_SIZE)
 * @param patches   Output [VM_N_PATCHES, VM_PATCH_DIM] flattened row-major
 */
void vm_create_patches(const float *pixels, int H, int W, float *patches);

/**
 * Compute GELU with tanh approximation (F.gelu(approximate="tanh")).
 */
static inline float vm_gelu(float x) {
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x3)));
}

/**
 * LayerNorm (equivalent to torch F.layer_norm).
 * @param x         Input [N] flattened
 * @param weight    [N] scale
 * @param bias      [N] shift
 * @param n         Size of normalized dimension
 * @param eps       Small constant (default 1e-5)
 */
void vm_layer_norm(float *x, const float *weight, const float *bias, int n, float eps);

/**
 * Full multi-token SDPA over all tokens: output = softmax(QK^T/√d) V
 * q, k, v: [n_tokens, D] where D = n_heads * head_dim
 * output: [n_tokens, D]
 * attn_scratch: [n_tokens * max_tokens] temp for softmax scores
 * max_tokens: stride for attn_scratch (allows scratch reuse between calls)
 * n_tokens: actual number of tokens in this call
 * n_heads: number of attention heads
 * head_dim: dimension per head (D / n_heads)
 */
void vm_attention(const float *q, const float *k, const float *v,
                  float *output, float *attn_scratch,
                  int max_tokens, int n_tokens, int n_heads, float head_dim);

/**
 * Linear layer: y = x @ W^T + bias
 * @param x       [in_features] input
 * @param w       [out_features, in_features] weight matrix (row-major)
 * @param bias    [out_features] bias
 * @param out     [out_features] output
 * @param in_dim  input feature count
 * @param out_dim output feature count
 */
void vm_linear(const float *x, const float *w, const float *bias,
               float *out, int in_dim, int out_dim);

/**
 * Exponential map to Poincaré ball (from mobius operations).
 * Projects Euclidean vector to hyperbolic space.
 * @param vec   [n] input Euclidean vector
 * @param out   [n] output tangent vector in Poincaré ball
 * @param n     dimension
 * @param R     ball radius (default 0.956)
 */
void vm_exp_map(const float *vec, float *out, int n, float R);

#ifdef __cplusplus
}
#endif

#endif // WUBU_VISION_MOONDREAM_H
