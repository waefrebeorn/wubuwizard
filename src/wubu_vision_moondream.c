/**
 * wubu_vision_moondream.c — Moondream3 SigLIP-style ViT forward pass in C
 *
 * Loads dumped f32 weights from moondream3_vision_weights.bin and runs:
 *   patch_embed → 27× ViT block → post_ln → proj_mlp → exp_map → Poincaré
 *
 * Each ViT block:
 *   x = x + attention(layer_norm(x))
 *   x = x + mlp(layer_norm(x))
 *
 * MLP: fc1 → GELU(tanh) → fc2
 * Attention: QKV fused linear → SDPA → proj
 *
 * Weight loading: index.json maps tensor names → offset/size in .bin
 * Weight layout mirrors build_vision_model() in vision.py
 */

#include "wubu_vision_moondream.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <json-c/json.h>

// ── Tensor name → struct field mapping ─────────────────────────────────

// Helper: map tensor name to (field_ptr, expected_bytes) for reading.
// Returns 1 if matched, 0 if unknown.
static int map_tensor(const char *name, vm_state_t *state,
                      float **field, size_t *expected_bytes) {
    vm_weights_t *w = &state->w;
    int n = -1;

    // Parse: model.vision.blocks.N.{sub}.{param}
    // Or:     model.vision.{other}.{param}
    if (sscanf(name, "model.vision.blocks.%d.", &n) == 1) {
        if (n < 0 || n >= VM_ENC_N_LAYERS) return 0;
        // Find the rest after "model.vision.blocks.N." where N is variable length
        const char *rest = strstr(name, "blocks.");
        if (!rest) return 0;
        rest = strchr(rest + 7, '.'); // skip "blocks.N"
        if (!rest) return 0;
        rest++; // skip the dot

        if (!strcmp(rest, "attn.proj.bias"))       { *field = w->blocks[n].attn_proj_bias;       *expected_bytes = VM_ENC_DIM * sizeof(float); }
        else if (!strcmp(rest, "attn.proj.weight"))  { *field = w->blocks[n].attn_proj_weight;     *expected_bytes = (size_t)VM_ENC_DIM * VM_ENC_DIM * sizeof(float); }
        else if (!strcmp(rest, "attn.qkv.bias"))     { *field = w->blocks[n].attn_qkv_bias;       *expected_bytes = (size_t)3 * VM_ENC_DIM * sizeof(float); }
        else if (!strcmp(rest, "attn.qkv.weight"))   { *field = w->blocks[n].attn_qkv_weight;     *expected_bytes = (size_t)3 * VM_ENC_DIM * VM_ENC_DIM * sizeof(float); }
        else if (!strcmp(rest, "ln1.bias"))           { *field = w->blocks[n].ln1_bias;           *expected_bytes = VM_ENC_DIM * sizeof(float); }
        else if (!strcmp(rest, "ln1.weight"))         { *field = w->blocks[n].ln1_weight;         *expected_bytes = VM_ENC_DIM * sizeof(float); }
        else if (!strcmp(rest, "ln2.bias"))           { *field = w->blocks[n].ln2_bias;           *expected_bytes = VM_ENC_DIM * sizeof(float); }
        else if (!strcmp(rest, "ln2.weight"))         { *field = w->blocks[n].ln2_weight;         *expected_bytes = VM_ENC_DIM * sizeof(float); }
        else if (!strcmp(rest, "mlp.fc1.bias"))       { *field = w->blocks[n].mlp_fc1_bias;       *expected_bytes = VM_ENC_FF_DIM * sizeof(float); }
        else if (!strcmp(rest, "mlp.fc1.weight"))     { *field = w->blocks[n].mlp_fc1_weight;     *expected_bytes = (size_t)VM_ENC_FF_DIM * VM_ENC_DIM * sizeof(float); }
        else if (!strcmp(rest, "mlp.fc2.bias"))       { *field = w->blocks[n].mlp_fc2_bias;       *expected_bytes = VM_ENC_DIM * sizeof(float); }
        else if (!strcmp(rest, "mlp.fc2.weight"))     { *field = w->blocks[n].mlp_fc2_weight;     *expected_bytes = (size_t)VM_ENC_DIM * VM_ENC_FF_DIM * sizeof(float); }
        else return 0;
        return 1;
    }

    // Non-block tensors
    if (!strcmp(name, "model.vision.patch_emb.bias"))     { *field = w->patch_emb_bias;       *expected_bytes = VM_ENC_DIM * sizeof(float); }
    else if (!strcmp(name, "model.vision.patch_emb.weight")) { *field = w->patch_emb_weight;  *expected_bytes = (size_t)VM_ENC_DIM * VM_PATCH_DIM * sizeof(float); }
    else if (!strcmp(name, "model.vision.pos_emb"))          { *field = w->pos_emb;            *expected_bytes = (size_t)VM_N_PATCHES * VM_ENC_DIM * sizeof(float); }
    else if (!strcmp(name, "model.vision.post_ln.bias"))     { *field = w->post_ln_bias;       *expected_bytes = VM_ENC_DIM * sizeof(float); }
    else if (!strcmp(name, "model.vision.post_ln.weight"))   { *field = w->post_ln_weight;     *expected_bytes = VM_ENC_DIM * sizeof(float); }
    else if (!strcmp(name, "model.vision.proj_mlp.fc1.bias"))   { *field = w->proj_mlp_fc1_bias;   *expected_bytes = VM_PROJ_INNER * sizeof(float); }
    else if (!strcmp(name, "model.vision.proj_mlp.fc1.weight")) { *field = w->proj_mlp_fc1_weight; *expected_bytes = (size_t)VM_PROJ_INNER * 2 * VM_ENC_DIM * sizeof(float); }
    else if (!strcmp(name, "model.vision.proj_mlp.fc2.bias"))   { *field = w->proj_mlp_fc2_bias;   *expected_bytes = VM_PROJ_OUT * sizeof(float); }
    else if (!strcmp(name, "model.vision.proj_mlp.fc2.weight")) { *field = w->proj_mlp_fc2_weight; *expected_bytes = (size_t)VM_PROJ_OUT * VM_PROJ_INNER * sizeof(float); }
    else return 0;
    return 1;
}

// ── Weight Loading ─────────────────────────────────────────────────────

static void *load_tensor(const char *name, struct json_object *index,
                          FILE *bin, vm_state_t *state) {
    struct json_object *entry, *joff, *jsz;
    if (!json_object_object_get_ex(index, name, &entry)) {
        fprintf(stderr, "[vm] WARNING: tensor '%s' not found in index\n", name);
        return NULL;
    }
    json_object_object_get_ex(entry, "offset", &joff);
    json_object_object_get_ex(entry, "size_bytes", &jsz);
    long offset = (long)json_object_get_int64(joff);
    size_t size = (size_t)json_object_get_int64(jsz);

    float *field = NULL;
    size_t expected = 0;
    if (!map_tensor(name, state, &field, &expected)) {
        fprintf(stderr, "[vm] WARNING: no struct mapping for '%s'\n", name);
        return NULL;
    }

    if (size != expected) {
        fprintf(stderr, "[vm] WARNING: '%s' size mismatch: bin=%zu expected=%zu\n",
                name, size, expected);
        // Still proceed — the .bin is ground truth
    }

    if (fseek(bin, offset, SEEK_SET) != 0) {
        fprintf(stderr, "[vm] ERROR: seek failed for '%s' at offset %ld\n", name, offset);
        return NULL;
    }
    if (fread(field, 1, size, bin) != size) {
        fprintf(stderr, "[vm] ERROR: read failed for '%s' (%zu bytes)\n", name, size);
        return NULL;
    }
    return field;
}

// ── Init / Free ─────────────────────────────────────────────────────────

bool vm_init(vm_state_t *state, const char *bin_path, const char *index_path) {
    memset(state, 0, sizeof(*state));

    // 1. Allocate all weight pointers
    vm_weights_t *w = &state->w;
    w->patch_emb_weight   = (float *)calloc((size_t)VM_ENC_DIM * VM_PATCH_DIM, sizeof(float));
    w->patch_emb_bias     = (float *)calloc(VM_ENC_DIM, sizeof(float));
    w->pos_emb            = (float *)calloc((size_t)VM_N_PATCHES * VM_ENC_DIM, sizeof(float));
    w->post_ln_weight     = (float *)calloc(VM_ENC_DIM, sizeof(float));
    w->post_ln_bias       = (float *)calloc(VM_ENC_DIM, sizeof(float));
    w->proj_mlp_fc1_weight = (float *)calloc((size_t)VM_PROJ_INNER * 2 * VM_ENC_DIM, sizeof(float));
    w->proj_mlp_fc1_bias   = (float *)calloc(VM_PROJ_INNER, sizeof(float));
    w->proj_mlp_fc2_weight = (float *)calloc((size_t)VM_PROJ_OUT * VM_PROJ_INNER, sizeof(float));
    w->proj_mlp_fc2_bias   = (float *)calloc(VM_PROJ_OUT, sizeof(float));

    for (int i = 0; i < VM_ENC_N_LAYERS; i++) {
        float **ptrs[] = {
            &w->blocks[i].ln1_weight, &w->blocks[i].ln1_bias,
            &w->blocks[i].attn_qkv_weight, &w->blocks[i].attn_qkv_bias,
            &w->blocks[i].attn_proj_weight, &w->blocks[i].attn_proj_bias,
            &w->blocks[i].ln2_weight, &w->blocks[i].ln2_bias,
            &w->blocks[i].mlp_fc1_weight, &w->blocks[i].mlp_fc1_bias,
            &w->blocks[i].mlp_fc2_weight, &w->blocks[i].mlp_fc2_bias,
        };
        size_t sizes[] = {
            VM_ENC_DIM, VM_ENC_DIM,
            (size_t)3 * VM_ENC_DIM * VM_ENC_DIM, (size_t)3 * VM_ENC_DIM,
            (size_t)VM_ENC_DIM * VM_ENC_DIM, VM_ENC_DIM,
            VM_ENC_DIM, VM_ENC_DIM,
            (size_t)VM_ENC_FF_DIM * VM_ENC_DIM, VM_ENC_FF_DIM,
            (size_t)VM_ENC_DIM * VM_ENC_FF_DIM, VM_ENC_DIM,
        };
        for (int j = 0; j < 12; j++)
            *ptrs[j] = (float *)calloc(sizes[j], sizeof(float));
    }

    // 2. Allocate large scratch (max intermediate = patches + qkv + attn_output + mlp_hidden + proj)
    //    Max usage: proj_hidden[N * 8192] + proj_input[N * 2304] = ~7.5M floats
    size_t max_scratch = (size_t)VM_N_PATCHES * (VM_PROJ_INNER + 2 * VM_ENC_DIM);
    state->scratch = (float *)calloc(max_scratch, sizeof(float));

    // 3. Load JSON index
    struct json_object *index = json_object_from_file(index_path);
    if (!index) {
        fprintf(stderr, "[vm] ERROR: failed to parse %s\n", index_path);
        goto fail;
    }

    // 4. Open binary weight file
    FILE *bin = fopen(bin_path, "rb");
    if (!bin) {
        fprintf(stderr, "[vm] ERROR: cannot open %s\n", bin_path);
        json_object_put(index);
        goto fail;
    }

    // 5. Load each tensor name from the JSON index
    //    Iterate over all keys in the JSON object
    json_object_object_foreach(index, key, val) {
        (void)val;  // unused — we use key to look up
        // Skip metadata fields that aren't tensor names
        if (strcmp(key, "n_elems") == 0 || strcmp(key, "offset") == 0 ||
            strcmp(key, "size_bytes") == 0 || strcmp(key, "shape") == 0)
            continue;
        if (!load_tensor(key, index, bin, state)) {
            fprintf(stderr, "[vm] WARNING: failed to load '%s'\n", key);
        }
    }

    fclose(bin);
    json_object_put(index);

    state->loaded = true;
    fprintf(stderr, "[vm] init OK: %s + %s\n", bin_path, index_path);
    return true;

fail:
    vm_free(state);
    return false;
}

void vm_free(vm_state_t *state) {
    if (!state) return;
    vm_weights_t *w = &state->w;

    // Non-block weights
    free(w->patch_emb_weight);   w->patch_emb_weight = NULL;
    free(w->patch_emb_bias);     w->patch_emb_bias = NULL;
    free(w->pos_emb);            w->pos_emb = NULL;
    free(w->post_ln_weight);     w->post_ln_weight = NULL;
    free(w->post_ln_bias);       w->post_ln_bias = NULL;
    free(w->proj_mlp_fc1_weight); w->proj_mlp_fc1_weight = NULL;
    free(w->proj_mlp_fc1_bias);   w->proj_mlp_fc1_bias = NULL;
    free(w->proj_mlp_fc2_weight); w->proj_mlp_fc2_weight = NULL;
    free(w->proj_mlp_fc2_bias);   w->proj_mlp_fc2_bias = NULL;

    // Block weights
    for (int i = 0; i < VM_ENC_N_LAYERS; i++) {
        free(w->blocks[i].ln1_weight);      w->blocks[i].ln1_weight = NULL;
        free(w->blocks[i].ln1_bias);        w->blocks[i].ln1_bias = NULL;
        free(w->blocks[i].attn_qkv_weight); w->blocks[i].attn_qkv_weight = NULL;
        free(w->blocks[i].attn_qkv_bias);   w->blocks[i].attn_qkv_bias = NULL;
        free(w->blocks[i].attn_proj_weight); w->blocks[i].attn_proj_weight = NULL;
        free(w->blocks[i].attn_proj_bias);  w->blocks[i].attn_proj_bias = NULL;
        free(w->blocks[i].ln2_weight);      w->blocks[i].ln2_weight = NULL;
        free(w->blocks[i].ln2_bias);        w->blocks[i].ln2_bias = NULL;
        free(w->blocks[i].mlp_fc1_weight);  w->blocks[i].mlp_fc1_weight = NULL;
        free(w->blocks[i].mlp_fc1_bias);    w->blocks[i].mlp_fc1_bias = NULL;
        free(w->blocks[i].mlp_fc2_weight);  w->blocks[i].mlp_fc2_weight = NULL;
        free(w->blocks[i].mlp_fc2_bias);    w->blocks[i].mlp_fc2_bias = NULL;
    }

    free(state->scratch);
    state->scratch = NULL;
    state->loaded = false;
}

// ── Core Ops ───────────────────────────────────────────────────────────

void vm_create_patches(const float *pixels, int H, int W, float *patches) {
    // Equivalent to vision.create_patches()
    // Input:  pixels[3][H][W], float32, normalized [-1, 1]
    // Output: patches[729][588]
    //
    // Algorithm:
    //   for each 14×14 patch in the 27×27 grid:
    //     extract [3, 14, 14] → flatten to [588]
    //     store in patches[py * 27 + px]

    const int ps = VM_PATCH_SIZE;
    const int gs = VM_GRID_SIZE;

    for (int py = 0; py < gs; py++) {
        for (int px = 0; px < gs; px++) {
            int patch_idx = py * gs + px;
            float *out = &patches[patch_idx * VM_PATCH_DIM];
            int off = 0;
            for (int c = 0; c < 3; c++) {
                for (int j = 0; j < ps; j++) {
                    for (int i = 0; i < ps; i++) {
                        int y = py * ps + j;
                        int x = px * ps + i;
                        out[off++] = pixels[c * H * W + y * W + x];
                    }
                }
            }
        }
    }
}

void vm_layer_norm(float *x, const float *weight, const float *bias,
                   int n, float eps) {
    // Standard LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
    float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    for (int i = 0; i < n; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= n;
    float inv_std = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < n; i++) {
        x[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

void vm_attention(const float *q, const float *k, const float *v,
                  float *output, float *attn_scratch,
                  int N, int n_tokens, int n_heads, float head_dim) {
    // Full scaled dot-product attention: output = softmax(QK^T/√d) V
    // q, k, v: [n_tokens, D] where D = n_heads * head_dim
    // output: [n_tokens, D]
    // attn_scratch: [n_tokens * n_tokens] temp for softmax scores
    // N: max tokens (for scratch stride), n_tokens: actual tokens this call
    //
    // For each head h:
    //   attn[t][s] = sum_i q[t][h*hd+i] * k[s][h*hd+i] / sqrt(hd)
    //   attn[t][:] = softmax(attn[t][:])
    //   output[t][h*hd+i] = sum_s attn[t][s] * v[s][h*hd+i]

    float scale = 1.0f / sqrtf(head_dim);

    for (int h = 0; h < n_heads; h++) {
        int base = (int)(h * head_dim);
        int D = (int)(n_heads * head_dim);

        // Compute attention scores: Q @ K^T per head
        for (int t = 0; t < n_tokens; t++) {
            for (int s = 0; s < n_tokens; s++) {
                float score = 0.0f;
                for (int i = 0; i < (int)head_dim; i++) {
                    score += q[t * D + base + i] * k[s * D + base + i];
                }
                score *= scale;
                attn_scratch[t * N + s] = score;
            }
        }

        // Softmax over last dimension (over s)
        for (int t = 0; t < n_tokens; t++) {
            float max_val = attn_scratch[t * N];
            for (int s = 1; s < n_tokens; s++) {
                if (attn_scratch[t * N + s] > max_val)
                    max_val = attn_scratch[t * N + s];
            }
            float sum = 0.0f;
            for (int s = 0; s < n_tokens; s++) {
                attn_scratch[t * N + s] = expf(attn_scratch[t * N + s] - max_val);
                sum += attn_scratch[t * N + s];
            }
            float inv_sum = 1.0f / (sum + 1e-30f);
            for (int s = 0; s < n_tokens; s++) {
                attn_scratch[t * N + s] *= inv_sum;
            }
        }

        // Weighted sum: output[t][h] = sum_s attn[t][s] * v[s][h]
        for (int t = 0; t < n_tokens; t++) {
            for (int i = 0; i < (int)head_dim; i++) {
                float wsum = 0.0f;
                for (int s = 0; s < n_tokens; s++) {
                    wsum += attn_scratch[t * N + s] * v[s * D + base + i];
                }
                output[t * D + base + i] = wsum;
            }
        }
    }
}

void vm_linear(const float *x, const float *w, const float *bias,
               float *out, int in_dim, int out_dim) {
    // y = x @ W^T + bias
    // x: [in_dim], w: [out_dim, in_dim] row-major
    for (int i = 0; i < out_dim; i++) {
        float sum = bias ? bias[i] : 0.0f;
        for (int j = 0; j < in_dim; j++) {
            sum += x[j] * w[i * in_dim + j];
        }
        out[i] = sum;
    }
}

void vm_exp_map(const float *vec, float *out, int n, float R) {
    // Exponential map: project Euclidean vector to Poincaré ball
    // exp_0(v) = tanh(||v||/R) * v / (R * ||v||)
    // For ||v|| = 0, returns 0

    float norm = 0.0f;
    for (int i = 0; i < n; i++) norm += vec[i] * vec[i];
    norm = sqrtf(norm);

    if (norm < 1e-8f) {
        for (int i = 0; i < n; i++) out[i] = 0.0f;
        return;
    }

    float factor = tanhf(norm / R) / (R * norm);
    for (int i = 0; i < n; i++) out[i] = vec[i] * factor;
}

// ── Forward Pass ───────────────────────────────────────────────────────

void vm_forward(vm_state_t *state, const float *pixels, int H, int W,
                float *output) {
    if (!state->loaded) {
        fprintf(stderr, "[vm] ERROR: not initialized\n");
        return;
    }

    vm_weights_t *w = &state->w;
    float *scratch = state->scratch;  // large enough for max intermediate

    // Dimension pointers for readability
    int N = VM_N_PATCHES;    // 729
    int D = VM_ENC_DIM;      // 1152
    int F = VM_ENC_FF_DIM;   // 4304
    int Pd = VM_PATCH_DIM;   // 588

    // 1. Create patches: [N, Pd]
    //    patches_ptr starts at scratch[0]
    float *patches = scratch;
    vm_create_patches(pixels, H, W, patches);

    // 2. Patch embedding: Linear(Pd → D) → [N, D]
    //    x starts after patches
    float *x = patches + N * Pd;
    // x = patches @ W.T + bias  for each patch
    for (int i = 0; i < N; i++) {
        vm_linear(&patches[i * Pd], w->patch_emb_weight, w->patch_emb_bias,
                  &x[i * D], Pd, D);
    }

    // 3. Add position embeddings
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            x[i * D + j] += w->pos_emb[i * D + j];
        }
    }

    // 4. Transformer blocks
    float *residual = scratch + (N * Pd + N * D) / sizeof(float);  // temp
    for (int l = 0; l < VM_ENC_N_LAYERS; l++) {
        // Block: x = x + attention(layer_norm(x))
        //        x = x + mlp(layer_norm(x))

        // --- Attention sub-block ---
        memcpy(residual, x, N * D * sizeof(float));

        // LayerNorm
        for (int i = 0; i < N; i++) {
            vm_layer_norm(&x[i * D], w->blocks[l].ln1_weight,
                          w->blocks[l].ln1_bias, D, 1e-5f);
        }

        // QKV projection: [N, D] → [N, 3456] → split into Q, K, V each [N, 1152]
        float *qkv = scratch;  // reuse
        for (int i = 0; i < N; i++) {
            vm_linear(&x[i * D], w->blocks[l].attn_qkv_weight,
                      w->blocks[l].attn_qkv_bias, &qkv[i * 3 * D], D, 3 * D);
        }

        // Split QKV and do attention
        float *q = scratch;
        float *k = scratch + N * D;
        float *v = scratch + 2 * N * D;
        // In-place reshape: qkv [N, 3D] → q [N, D], k [N, D], v [N, D]
        // Already laid out this way since we write to qkv linearly

        // Attention per head — full SDPA over all N tokens
        float *attn_out = scratch + 3 * N * D;
        float *attn_scratch = scratch + 4 * N * D;  // [N, N] attention score matrix
        float head_dim = (float)D / (float)VM_ENC_N_HEADS;
        vm_attention(q, k, v, attn_out, attn_scratch, N, N, VM_ENC_N_HEADS, head_dim);

        // Output projection
        float *proj_out = scratch;  // reuse qkv space
        for (int i = 0; i < N; i++) {
            vm_linear(&attn_out[i * D], w->blocks[l].attn_proj_weight,
                      w->blocks[l].attn_proj_bias, &proj_out[i * D], D, D);
        }

        // Residual connection
        for (int i = 0; i < N * D; i++)
            x[i] = residual[i] + proj_out[i];

        // --- MLP sub-block ---
        memcpy(residual, x, N * D * sizeof(float));

        // LayerNorm
        for (int i = 0; i < N; i++) {
            vm_layer_norm(&x[i * D], w->blocks[l].ln2_weight,
                          w->blocks[l].ln2_bias, D, 1e-5f);
        }

        // MLP: fc1 [D → F] → GELU → fc2 [F → D]
        float *mlp_hidden = scratch;  // [N, F]
        for (int i = 0; i < N; i++) {
            vm_linear(&x[i * D], w->blocks[l].mlp_fc1_weight,
                      w->blocks[l].mlp_fc1_bias, &mlp_hidden[i * F], D, F);
            // GELU in-place
            for (int j = 0; j < F; j++)
                mlp_hidden[i * F + j] = vm_gelu(mlp_hidden[i * F + j]);
        }

        float *mlp_out = scratch + N * F;  // [N, D]
        for (int i = 0; i < N; i++) {
            vm_linear(&mlp_hidden[i * F], w->blocks[l].mlp_fc2_weight,
                      w->blocks[l].mlp_fc2_bias, &mlp_out[i * D], F, D);
        }

        // Residual connection
        for (int i = 0; i < N * D; i++)
            x[i] = residual[i] + mlp_out[i];
    }

    // 5. Post-layer norm
    for (int i = 0; i < N; i++) {
        vm_layer_norm(&x[i * D], w->post_ln_weight, w->post_ln_bias, D, 1e-5f);
    }

    // 6. Projection MLP: [N, 1152] → concat? → [N, 2304] → fc1 → GELU → fc2 → [N, 2048]
    //    NOTE: The full projection concatenates global_features with "reconstructed"
    //    (from a decoder). For standalone vision encoder, we just project global_features.
    //    The proj_mlp.fc1 expects [2304] input = 2 × 1152 (global + reconstructed).
    //    Since we don't have reconstructed features in standalone mode,
    //    we pad with zeros or use a simpler projection.
    //
    //    For now: feed [global, zeros] through proj_mlp.
    int Pout = VM_PROJ_OUT;
    float *proj_input = scratch;  // [N, 2304]
    for (int i = 0; i < N; i++) {
        memcpy(&proj_input[i * 2 * D], &x[i * D], D * sizeof(float));
        memset(&proj_input[i * 2 * D + D], 0, D * sizeof(float));  // zeros for reconstructed
    }

    float *proj_hidden = scratch + N * 2 * D;  // [N, 8192]
    for (int i = 0; i < N; i++) {
        vm_linear(&proj_input[i * 2 * D], w->proj_mlp_fc1_weight,
                  w->proj_mlp_fc1_bias, &proj_hidden[i * VM_PROJ_INNER],
                  2 * D, VM_PROJ_INNER);
        for (int j = 0; j < VM_PROJ_INNER; j++)
            proj_hidden[i * VM_PROJ_INNER + j] = vm_gelu(proj_hidden[i * VM_PROJ_INNER + j]);
    }

    // Final projection → output
    for (int i = 0; i < N; i++) {
        vm_linear(&proj_hidden[i * VM_PROJ_INNER], w->proj_mlp_fc2_weight,
                  w->proj_mlp_fc2_bias, &output[i * Pout],
                  VM_PROJ_INNER, Pout);
    }

    // 7. Exponential map → Poincaré ball
    float R = 0.956f;  // Default curvature radius
    float *poincare = scratch;  // temp
    for (int i = 0; i < N; i++) {
        vm_exp_map(&output[i * Pout], &poincare[i * Pout], Pout, R);
    }
    memcpy(output, poincare, N * Pout * sizeof(float));
}
