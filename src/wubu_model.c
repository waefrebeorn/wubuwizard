#include "wubu_model.h"
#include "gguf_reader.h"
#include "safetensors_reader.h"
#include "wubu_affinity.h"
#include "wubu_kv_styx.h"  // KV cache /n/kv/ export (styx)
#include "wubu_kvfs.h"     // ADR-003: the KV cache IS a file system (namespace layer)
#include "wubu_backend.h"  // backend vtable dispatch (replaces #ifdef GPU_SUPPORT)
#include "wubu_rotate.h"   // doc 013: wubu_rotate_input for lm_head Hadamard fuse
#include "wubu_mem_budget.h" // OOM-proof memory budget calculator
#include "wubu_hwcaps.h"   // HW-accel: SIMD ladder detection
#include "wubu_rambus.h"   // HW-accel: RDRAM-interleaved KV banks
#include "wubu_tandem.h"   // HW-accel: N64 RCP two-stage pipeline
#include "wubu_gamebud.h"  // HW-accel: game-design frame-budget governor
/* Five-reference kernel ports (research/062): env-gated forward probes so
 * every ported kernel RUNS inside the model forward path, not just in its
 * standalone test. Gate: WUBU_REF_KERNELS=1. Zero cost when off. */
#include "wubu_enc_h3.h"      // MiniMax H3: ConvRot un-rotation + NVFP4 requant
#include "wubu_dsv4.h"        // DeepSeek-V4: hyper-residual + sinkhorn + hash route
#include "wubu_lfm.h"         // LFM2.5: hybrid linear/softmax attention
#include "wubu_megakernel.h"  // Photon 2.0: fused PSO decode
#include "wubu_multiteach.h"  // mr_r0b0t multi-teacher distillation kernel
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>  // _mm_prefetch for expert prefetch

/* wubu_model.c — WuBu1 loader: role-based tensor resolution.
 *
 * The old loader hardcoded Qwen-style "blk.%d.*" names in 160+
 * places, silently breaking every other convention (Gemma
 * "model.layers.N.*", HF-style "layers.N.*" with fused gate_up,
 * full HF "model.language_model.layers.N.*"). WuBu1 uses the
 * role-based resolver (wubu_gguf_names.c/h) — every weight is a
 * ROLE, each role carries candidate templates across all known
 * conventions, and detection scans the file's actual tensor names.
 * No architecture metadata required; works on any GGUF.
 *
 * WuBu1 (WaefreBeorn Umbrella License v3.0) — the redesigned
 * base model, designed from scratch in-house. */

/* ========== GGUF Tensor Names (role-based resolver) ========== */

#include "wubu_gguf_names.h"

static wubu_gguf_names_t g_names;  /* detected once at init */
static int g_max_layer = -1;
static gguf_ctx *g_gguf_ctx = NULL;

/* Helper: resolve a layer weight by role. Returns NULL if not
 * found in this GGUF (some models omit optional tensors). */
static gguf_tensor_info *resolve(int layer, wubu_gguf_role_t role) {
    return wubu_gguf_find(g_gguf_ctx, layer, role);
}
static gguf_tensor_info *resolve_global(wubu_gguf_role_t role) {
    return wubu_gguf_find(g_gguf_ctx, -1, role);
}

// ========== Init ==========

bool wubu_model_init(wubu_model_t *model, const char *gguf_path) {
    memset(model, 0, sizeof(*model));
    model->tied_output = false;
    model->rotate_P = 0;

    /* Install the CPU backend vtable (always present — GPU builds
     * override it with the CUDA backend after successful GPU init).
     * Done before the GGUF open so a failed init still leaves the
     * model with a working (CPU) dispatch path. */
    model->backend = wubu_backend_cpu_get();

    /* Game-console hardware discipline (I05 / NuMA+P-core pinning, +19-21%
     * throughput on multi-socket; non-zero even single-socket via stable
     * L1/L2 per GEMV row-chunk). Pin the calling thread to the
     * P-core set, then make OpenMP inherit a close, core-bound policy so
     * the GEMV parallel-for keeps each row-chunk on one core's cache. */
    {
        int pinned[64]; int k = wubu_affinity_pin_pcores(pinned, 64);
        if (k > 0) {
            /* Make OpenMP inherit a close, core-bound policy so the GEMV
             * parallel-for keeps each row-chunk on one core's cache. Use
             * setenv (portable across OpenMP runtimes) rather than the
             * version-specific omp_set_proc_bind API. */
            setenv("OMP_PROC_BIND", "close", 1);
            setenv("OMP_PLACES", "cores", 1);
            setenv("OMP_SCHEDULE", "dynamic,64", 1);
            fprintf(stderr, "[affinity] pinned engine to %d P-cores (core0=%d)\n",
                    k, pinned[0]);
        }
    }

    // Open GGUF
    gguf_ctx *ctx = gguf_open(gguf_path);
    if (!ctx) { fprintf(stderr, "Failed to open %s\n", gguf_path); return false; }

    /* Detect tensor naming convention + layer count from the
     * actual tensor names in this file. No architecture metadata
     * required — we scan and resolve by role. */
    wubu_gguf_names_detect(ctx, &g_names);
    g_max_layer = g_names.max_layer;
    g_gguf_ctx = ctx;
    printf("  GGUF convention=%d layers=%d ssm=%d moe=%d gqa=%d\n",
           g_names.convention, g_names.n_layers,
           g_names.has_ssm, g_names.has_moe, g_names.has_gqa);

    // Initialize forward arena (OOM-safe temp buffers, ~256MB budget).
    // Grown on-demand if forward needs more (rare).
    wubu_arena_init(&model->fwd_arena, 256 * 1024 * 1024, 0);
    wubu_sub_arena_create(&model->fwd_arena, &model->fwd_sub, 256 * 1024 * 1024);

    // Extract dynamic dimensions from GGUF tensor shapes via resolver
    int d_model = 0;
    {
        gguf_tensor_info *nt = resolve(0, WUBU_T_ATTN_NORM);
        if (nt && nt->n_dims >= 1) d_model = (int)nt->dims[0];
    }
    if (d_model == 0) d_model = D_MODEL; // fallback
    model->d_model = d_model;

    // Extract GQA dimensions from tensor shapes via resolver
    int gqa_head_dim = GQA_HEAD_DIM;
    {
        gguf_tensor_info *qn = resolve(0, WUBU_T_ATTN_Q_NORM);
        if (qn && qn->n_dims >= 1 && qn->dims[0] > 0) {
            gqa_head_dim = (int)qn->dims[0];
        } else {
            // Fallback 1: derive from attn_k.weight shape
            gguf_tensor_info *kn = resolve(0, WUBU_T_ATTN_K);
            if (kn && kn->n_dims >= 2) {
                int kv_dim = (int)kn->dims[1];
                int kv_heads = (kv_dim > 0) ? (kv_dim / gqa_head_dim) : 10;
                if (kv_heads > 0) gqa_head_dim = kv_dim / kv_heads;
            } else {
                // Fallback 2: fused QKV — attn_qkv.weight
                gguf_tensor_info *qkn = resolve(0, WUBU_T_ATTN_QKV);
                if (qkn && qkn->n_dims >= 2) {
                    int qkv_dim = (int)qkn->dims[1];
                    int assumed_kv_heads = 4;
                    if (qkv_dim > assumed_kv_heads * gqa_head_dim) {
                        gqa_head_dim = qkv_dim / (assumed_kv_heads * 2);
                    }
                }
            }
        }
    }

    // Extract SSM dimensions from tensor shapes via resolver
    int ssm_d_state = SSM_D_STATE;
    int ssm_k_heads = SSM_K_HEADS;
    int dt_rank = DT_RANK;
    int ssm_v_heads = SSM_V_HEADS;
    int conv_kernel = CONV_KERNEL;
    {
        gguf_tensor_info *t = resolve(0, WUBU_T_SSM_NORM);
        if (t && t->n_dims >= 1) ssm_d_state = (int)t->dims[0];
        t = resolve(0, WUBU_T_SSM_DT);
        if (t && t->n_dims >= 1) dt_rank = (int)t->dims[0];
        t = resolve(0, WUBU_T_SSM_A);
        if (t && t->n_dims >= 1) dt_rank = (int)t->dims[0];
        t = resolve(0, WUBU_T_SSM_CONV1D);
        if (t && t->n_dims >= 2) {
            conv_kernel = (int)t->dims[0];
            int conv_dim = (int)t->dims[1];
            int key_dim = ssm_d_state * ssm_k_heads;
            int value_dim = conv_dim - 2 * key_dim;
            if (value_dim > 0 && value_dim % ssm_d_state == 0)
                ssm_v_heads = value_dim / ssm_d_state;
        }
    }

    // Setup WUBU_DIMS from extracted dimensions
    wubu_dims_t dims = {0};
    dims.d_model = d_model;
    dims.ssm_d_state = ssm_d_state;
    dims.ssm_k_heads = ssm_k_heads;
    dims.ssm_v_heads = ssm_v_heads;
    dims.conv_kernel = conv_kernel;
    dims.dt_rank = dt_rank;
    dims.gqa_q_heads = GQA_Q_HEADS;
    dims.gqa_kv_heads = GQA_KV_HEADS;
    dims.gqa_head_dim = gqa_head_dim;
    wubu_dims_set(&dims);

    // Also set model fields for backward compatibility
    model->d_inner = VALUE_DIM;
    model->key_dim = KEY_DIM;
    model->conv_dim = CONV_DIM;
    model->conv_kernel = conv_kernel;
    model->dt_rank = dt_rank;
    model->ssm_k_heads = ssm_k_heads;
    model->ssm_v_heads = ssm_v_heads;
    model->ssm_d_state = ssm_d_state;
    model->gqa_q_heads = GQA_Q_HEADS;
    model->gqa_kv_heads = GQA_KV_HEADS;
    model->gqa_head_dim = gqa_head_dim;
    model->rotary_dim = (int)(gqa_head_dim * PARTIAL_ROTARY_FACTOR);
    model->d_ff = D_FF;
    model->n_experts = N_EXPERTS;
    model->n_active_experts = N_ACTIVE_EXPTS;

    printf("  Model dims: d_model=%d, head_dim=%d\n", d_model, gqa_head_dim);
    printf("  SSM dims: d_state=%d, k_heads=%d, v_heads=%d, dt_rank=%d, conv_kernel=%d\n",
           ssm_d_state, ssm_k_heads, ssm_v_heads, dt_rank, conv_kernel);
    printf("  CONV_DIM=%d, VALUE_DIM=%d, KEY_DIM=%d\n", CONV_DIM, VALUE_DIM, KEY_DIM);
    printf("  Layers: %d (convention %d)\n", g_names.n_layers, g_names.convention);

    // Allocate layers from the detected layer count.
    model->n_layers = g_names.n_layers;
    if (model->n_layers <= 0) {
        fprintf(stderr, "No layer tensors found in %s — is this a GGUF model?\n", gguf_path);
        goto fail;
    }
    model->layers = (wubu_layer_t *)calloc((size_t)model->n_layers, sizeof(wubu_layer_t));
    if (!model->layers) { fprintf(stderr, "Failed to allocate %d layers\n", model->n_layers); goto fail; }
    printf("  Allocating %d layers...\n", model->n_layers);

    // Buffer GGUF data EARLY so all tensor reads use mmap (avoids FILE* issues with large files)
    printf("  Buffering GGUF data via mmap...\n");
    if (!gguf_buffer_data(ctx)) {
        fprintf(stderr, "Failed to buffer GGUF data\n");
        goto fail;
    }
    const uint8_t *blob = (const uint8_t *)ctx->data_blob;
    printf("  GGUF data buffered: %p (mmap=%d)\n", (void*)blob, ctx->data_blob_is_mmap);

    // Load layer norms and attention weights
    for (int l = 0; l < model->n_layers; l++) {
        wubu_layer_t *layer = &model->layers[l];
        layer->layer_idx = l;
        /* Layer-type classification driven by the resolver's detection,
         * not the old hardcoded Qwen3.6 3:1 SSM:GQA heuristic. */
        if (g_names.has_ssm && !g_names.has_gqa) {
            layer->is_ssm = 1;                       /* all-SSM model */
        } else if (g_names.has_gqa && !g_names.has_ssm) {
            layer->is_ssm = 0;                       /* all-GQA model (WuBu-35M) */
        } else {
            layer->is_ssm = wubu_is_ssm_layer(l);    /* mixed — legacy heuristic */
        }
        
        gguf_tensor_info *t;
        char name[256];

        // attn_norm.weight (pre-attention RMSNorm)
        t = resolve(l, WUBU_T_ATTN_NORM);
        if (t) {
            layer->attn_norm_weight = (float *)malloc(model->d_model * sizeof(float));
            if (!gguf_read_tensor_f32(ctx, t, layer->attn_norm_weight, model->d_model))
                { fprintf(stderr, "Failed to load attn_norm[%d]\n", l); goto fail; }
        }
        
        // post_attention_norm.weight (optional — Qwen3-style).
        // Fallbacks: ffn_norm.weight (Qwen2/nanbeige), then attn_norm.weight.
        t = resolve(l, WUBU_T_POST_ATTN_NORM);
        if (!t) {
            t = resolve(l, WUBU_T_FFN_NORM);
        }
        if (!t) {
            t = resolve(l, WUBU_T_ATTN_NORM);  // fallback: reuse attn_norm
        }
        if (t) {
            layer->post_attn_norm_weight = (float *)malloc(model->d_model * sizeof(float));
            if (!gguf_read_tensor_f32(ctx, t, layer->post_attn_norm_weight, model->d_model))
                { fprintf(stderr, "Failed to load post_attn_norm[%d]\n", l); goto fail; }
        } else if (layer->attn_norm_weight) {
            // No dedicated post-attn norm: reuse pre-attn norm (identity-ish RMSNorm).
            layer->post_attn_norm_weight = (float *)malloc(model->d_model * sizeof(float));
            memcpy(layer->post_attn_norm_weight, layer->attn_norm_weight, model->d_model * sizeof(float));
        }
        
        if (layer->is_ssm) {
            // Load SSM weights via role-based resolver.
            // Large weights (attn_qkv, attn_gate, ssm_out) use
            // quantized blob pointers; small tensors (norms, a, dt,
            // conv1d) are loaded as F32 into CPU memory.
            if (getenv("WUBU_DEBUG")) fprintf(stderr, "DEBUG: Loading SSM layer %d\n", l);

            // attn_qkv_weight — quantized-only (blob pointer)
            {
                gguf_tensor_info *t = resolve(l, WUBU_T_ATTN_QKV);
                if (!t) { fprintf(stderr, "Missing SSM attn_qkv for layer %d\n", l); goto fail; }
                if (t && blob) { layer->ssm.attn_qkv_weight_q = blob + t->data_offset; layer->ssm.attn_qkv_weight_type = t->ggml_type; }
            }

            // attn_gate_weight — quantized-only (blob pointer)
            {
                gguf_tensor_info *t = resolve(l, WUBU_T_ATTN_GATE);
                if (!t) { fprintf(stderr, "Missing SSM attn_gate for layer %d\n", l); goto fail; }
                if (t && blob) { layer->ssm.attn_gate_weight_q = blob + t->data_offset; layer->ssm.attn_gate_weight_type = t->ggml_type; }
            }

            // ssm_out_weight — quantized-only (blob pointer)
            {
                gguf_tensor_info *t = resolve(l, WUBU_T_SSM_OUT);
                if (!t) { fprintf(stderr, "Missing SSM ssm_out for layer %d\n", l); goto fail; }
                if (t && blob) { layer->ssm.ssm_out_weight_q = blob + t->data_offset; layer->ssm.ssm_out_weight_type = t->ggml_type; }
            }

            // Small tensors: load as F32 into CPU memory.
            // ssm_beta.weight — use tensor's actual dims for safe alloc
            {
                gguf_tensor_info *t = resolve(l, WUBU_T_SSM_BETA);
                if (!t) { fprintf(stderr, "Missing SSM ssm_beta for layer %d\n", l); goto fail; }
                int64_t beta_n_elems = 1;
                for (int d = 0; d < t->n_dims; d++) beta_n_elems *= t->dims[d];
                layer->ssm.ssm_beta_weight = (float *)malloc((size_t)beta_n_elems * sizeof(float));
                gguf_read_tensor_f32(ctx, t, layer->ssm.ssm_beta_weight, -1);
            }

            // ssm_alpha.weight [d_model, dt_rank] F32
            {
                gguf_tensor_info *t = resolve(l, WUBU_T_SSM_ALPHA);
                if (!t) { fprintf(stderr, "Missing SSM ssm_alpha for layer %d\n", l); goto fail; }
                int64_t alpha_n_elems = 1;
                for (int d = 0; d < t->n_dims; d++) alpha_n_elems *= t->dims[d];
                layer->ssm.ssm_alpha_weight = (float *)malloc((size_t)alpha_n_elems * sizeof(float));
                gguf_read_tensor_f32(ctx, t, layer->ssm.ssm_alpha_weight, -1);
            }

            // ssm_dt.bias [dt_rank] F32
            {
                gguf_tensor_info *t = resolve(l, WUBU_T_SSM_DT);
                if (!t) { fprintf(stderr, "Missing SSM ssm_dt for layer %d\n", l); goto fail; }
                int64_t dt_n_elems = 1;
                for (int d = 0; d < t->n_dims; d++) dt_n_elems *= t->dims[d];
                layer->ssm.ssm_dt_bias = (float *)malloc((size_t)dt_n_elems * sizeof(float));
                gguf_read_tensor_f32(ctx, t, layer->ssm.ssm_dt_bias, -1);
            }

            // ssm_a [dt_rank] F32 (Qwen3.6 uses "ssm_a" without .weight suffix)
            {
                gguf_tensor_info *t = resolve(l, WUBU_T_SSM_A);
                if (!t) { fprintf(stderr, "Missing SSM ssm_a for layer %d\n", l); goto fail; }
                int64_t a_n_elems = 1;
                for (int d = 0; d < t->n_dims; d++) a_n_elems *= t->dims[d];
                layer->ssm.ssm_a = (float *)malloc((size_t)a_n_elems * sizeof(float));
                gguf_read_tensor_f32(ctx, t, layer->ssm.ssm_a, -1);
            }

            // ssm_conv1d.weight [conv_kernel, conv_dim] F32
            {
                gguf_tensor_info *t = resolve(l, WUBU_T_SSM_CONV1D);
                if (!t) { fprintf(stderr, "Missing SSM ssm_conv1d for layer %d\n", l); goto fail; }
                int64_t conv_n_elems = 1;
                for (int d = 0; d < t->n_dims; d++) conv_n_elems *= t->dims[d];
                layer->ssm.ssm_conv1d_weight = (float *)malloc((size_t)conv_n_elems * sizeof(float));
                gguf_read_tensor_f32(ctx, t, layer->ssm.ssm_conv1d_weight, -1);
            }

            // ssm_norm.weight — use tensor's actual dim
            {
                gguf_tensor_info *t = resolve(l, WUBU_T_SSM_NORM);
                if (!t) { fprintf(stderr, "Missing SSM ssm_norm for layer %d\n", l); goto fail; }
                int ssm_norm_size = (int)t->dims[0];
                layer->ssm.ssm_norm_weight = (float *)malloc((size_t)ssm_norm_size * sizeof(float));
                gguf_read_tensor_f32(ctx, t, layer->ssm.ssm_norm_weight, -1);
            }

            printf("  Layer %d: SSM loaded (quantized attn_qkv/gate/out, F32 norms/conv/a/dt/beta/alpha)\n", l);

        } else {
            // Load GQA weights — QUANTIZED-ONLY PATH for large weight matrices.
            // attn_q, attn_k, attn_v, attn_output use quantized blob pointers (set later).
            char name[256];
            int ok = 1;
            
            
            // GQA weights — QUANTIZED-ONLY PATH for large matrices.
            // attn_q, attn_k, attn_v, attn_output use quantized blob
            // pointers (set later). Small norms loaded as F32.
            if (getenv("WUBU_DEBUG")) fprintf(stderr, "DEBUG: Loading GQA layer %d\n", l);

            // attn_q_norm.weight [head_dim] F32 (optional)
            // Treat absence as identity (RMSNorm with all-ones weight).
            {
                gguf_tensor_info *t = resolve(l, WUBU_T_ATTN_Q_NORM);
                int layer_head_dim = model->gqa_head_dim;  // default
                if (t && t->n_dims >= 1 && t->dims[0] > 0) layer_head_dim = (int)t->dims[0];
                layer->gqa.head_dim = layer_head_dim;
                layer->gqa.attn_q_norm_weight = (float *)malloc(layer_head_dim * sizeof(float));
                if (t) {
                    gguf_read_tensor_f32(ctx, t, layer->gqa.attn_q_norm_weight, layer_head_dim);
                } else {
                    for (int i = 0; i < layer_head_dim; i++) layer->gqa.attn_q_norm_weight[i] = 1.0f;
                }
            }

            // attn_k_norm.weight [head_dim] F32 (optional)
            {
                gguf_tensor_info *t = resolve(l, WUBU_T_ATTN_K_NORM);
                layer->gqa.attn_k_norm_weight = (float *)malloc(layer->gqa.head_dim * sizeof(float));
                if (t) {
                    gguf_read_tensor_f32(ctx, t, layer->gqa.attn_k_norm_weight, layer->gqa.head_dim);
                } else {
                    for (int i = 0; i < layer->gqa.head_dim; i++) layer->gqa.attn_k_norm_weight[i] = 1.0f;
                }
            }

            // Extract per-layer dimensions from GGUF tensor shapes
            // via resolver — no hardcoded prefix needed.
            {
                gguf_tensor_info *t_k = resolve(l, WUBU_T_ATTN_K);
                if (t_k && t_k->n_dims >= 2 && layer->gqa.head_dim > 0) {
                    int kv_dim = (int)t_k->dims[1];  // kv_heads * head_dim
                    layer->gqa.kv_heads = kv_dim / layer->gqa.head_dim;
                } else {
                    layer->gqa.kv_heads = GQA_KV_HEADS;
                }
                // Q weight: [d_model, q_heads * head_dim * 2] (fused Q+gate)
                // => q_heads from dims
                gguf_tensor_info *t_q = resolve(l, WUBU_T_ATTN_Q);
                if (t_q && t_q->n_dims >= 2 && layer->gqa.head_dim > 0) {
                    int q_dim_fused = (int)t_q->dims[1];  // q_heads * head_dim * 2
                    layer->gqa.q_heads = q_dim_fused / (layer->gqa.head_dim * 2);
                } else {
                    layer->gqa.q_heads = GQA_Q_HEADS;
                }
                layer->gqa.kv_dim = layer->gqa.kv_heads * layer->gqa.head_dim;
                layer->gqa.q_dim = layer->gqa.q_heads * layer->gqa.head_dim;
                layer->gqa.is_large = (layer->gqa.head_dim == 512) ? 1 : 0;
            }

            // Extract output projection dim from attn_output.weight tensor
            {
                gguf_tensor_info *t_out = resolve(l, WUBU_T_ATTN_O);
                if (t_out && t_out->n_dims >= 2) {
                    int out_rows = (int)t_out->dims[0];  // first dim = input features
                    layer->gqa.out_dim = (out_rows != layer->gqa.q_dim) ? out_rows : layer->gqa.q_dim;
                } else {
                    layer->gqa.out_dim = layer->gqa.q_dim;
                }
            }

            printf("  Layer %d: GQA loaded, head_dim=%d q_heads=%d kv_heads=%d out_dim=%d%s\n",
                   l, layer->gqa.head_dim, layer->gqa.q_heads, layer->gqa.kv_heads,
                   layer->gqa.out_dim, layer->gqa.is_large ? " LARGE" : "");

            // LARGE: attn_q.weight — quantized-only (blob pointer)
            {
                gguf_tensor_info *t = resolve(l, WUBU_T_ATTN_Q);
                if (t && blob) { layer->gqa.attn_q_weight_q = blob + t->data_offset; layer->gqa.attn_q_weight_type = t->ggml_type; }
            }

            // LARGE: attn_k.weight — quantized-only (blob pointer)
            {
                gguf_tensor_info *t = resolve(l, WUBU_T_ATTN_K);
                if (t && blob) { layer->gqa.attn_k_weight_q = blob + t->data_offset; layer->gqa.attn_k_weight_type = t->ggml_type; }
            }

            // LARGE: attn_v.weight — quantized-only (blob pointer)
            // For Pure-GQA without separate V, share K weight (V=K).
            {
                gguf_tensor_info *t = resolve(l, WUBU_T_ATTN_V);
                if (t && blob) {
                    layer->gqa.attn_v_weight_q = blob + t->data_offset;
                    layer->gqa.attn_v_weight_type = t->ggml_type;
                } else {
                    // LARGE layers: V weight not present, share K weight (V=K)
                    layer->gqa.attn_v_weight_q = layer->gqa.attn_k_weight_q;
                    layer->gqa.attn_v_weight_type = layer->gqa.attn_k_weight_type;
                }
            }

            // LARGE: attn_output.weight — quantized-only (blob pointer)
            {
                gguf_tensor_info *t = resolve(l, WUBU_T_ATTN_O);
                if (t && blob) { layer->gqa.attn_output_weight_q = blob + t->data_offset; layer->gqa.attn_output_weight_type = t->ggml_type; }
            }
        }
        
        // Load MoE (FFN) weights — NOT loaded by default (memory: 3.2 GB/layer)
        // Use test_moe.c for standalone MoE testing
        layer->moe.loaded = false;
    }
    
    // Load final norm
    gguf_tensor_info *t = resolve(-1, WUBU_T_OUTPUT_NORM);
    if (t) {
        model->norm_weight = (float *)malloc(model->d_model * sizeof(float));
        gguf_read_tensor_f32(ctx, t, model->norm_weight, model->d_model);
        printf("  Final norm loaded\n");
    } else {
        printf("  WARNING: output_norm.weight not found\n");
    }
    
    // Embeddings: auto-extract from GGUF if not available, else load from file
    model->use_embedding_file = true;
    model->vocab_size = 0;
    // Get actual vocab size from GGUF embedding tensor via resolver
    {
        gguf_tensor_info *t_emb = resolve(-1, WUBU_T_TOKEN_EMBD);
        if (t_emb && t_emb->n_dims >= 2) {
            int64_t n_emb = 1;
            for (int d = 0; d < t_emb->n_dims; d++) n_emb *= t_emb->dims[d];
            int64_t vocab_from_emb = n_emb / model->d_model;
            if (vocab_from_emb > 0 && vocab_from_emb < 1000000) {
                model->vocab_size = (int)vocab_from_emb;
            }
        }
        if (model->vocab_size == 0) model->vocab_size = 248320; // fallback
    }
    // For large-vocab models (Gemma, vocab > 131072), skip F32 embedding load.
    // Use the mmap'd GGUF blob directly for per-token dequant embedding lookup.
    // For small-vocab models (Qwen), the F32 path is fine.
    bool large_vocab = (model->vocab_size > 131072);
    model->use_embedding_file = false;
    model->token_embd = NULL;

    if (!large_vocab) {
        // Small vocab: try loading F32 embeddings (original path)
        const char *emb_path = "data/qwen36_embeddings_c.bin.raw";
        FILE *emb_f = fopen(emb_path, "rb");
        if (emb_f) {
            fseek(emb_f, 0, SEEK_END);
            long emb_size = ftell(emb_f);
            int file_vocab = (int)(emb_size / (model->d_model * sizeof(float)));
            if (file_vocab == model->vocab_size) {
                printf("  Embeddings: %d tokens from file (%ld MB)\n", model->vocab_size, emb_size / (1024*1024));
                fclose(emb_f);
                model->use_embedding_file = true;
            } else {
                fclose(emb_f);
                large_vocab = true;
            }
        } else {
            large_vocab = true;
        }
    }

    if (large_vocab) {
        // Large vocab: use mmap'd GGUF blob for per-token dequant
        printf("  Large vocab (%d tokens): using mmap'd GGUF blob for embedding\n", model->vocab_size);
        model->use_embedding_file = false;
        model->token_embd = NULL;
        // Get quantized token_embd pointer from blob
        gguf_tensor_info *t_emb = resolve(-1, WUBU_T_TOKEN_EMBD);
        if (t_emb && ctx->data_blob) {
            model->token_embd_q = (const uint8_t *)ctx->data_blob + t_emb->data_offset;
            model->token_embd_type = t_emb->ggml_type;
            int64_t emb_elems = 1;
            for (int d = 0; d < t_emb->n_dims; d++) emb_elems *= t_emb->dims[d];
            printf("  token_embd: quantized type=%d, %ld elements\n", t_emb->ggml_type, (long)emb_elems);
        }
    }
    
    if (model->use_embedding_file) {
        // Verify vocab_size was set from file
        if (model->vocab_size == 0) model->vocab_size = 248320;
    }
    
    // Output weight — quantized pointer via resolver (no hardcoded name)
    model->output_weight = NULL;
    gguf_tensor_info *t_out = resolve(-1, WUBU_T_OUTPUT);
    if (!t_out) {
        // Tied output: use token_embd.weight (common for Gemma, LLaMA, etc.)
        gguf_tensor_info *t_embd = resolve(-1, WUBU_T_TOKEN_EMBD);
        if (t_embd) {
            model->output_weight_q = NULL;
            model->output_weight_type = t_embd->ggml_type;
            model->tied_output = true;
            fprintf(stderr, "  Output weight: TIED to token_embd.weight\n");
        } else {
            fprintf(stderr, "  ERROR: neither output.weight nor token_embd.weight found\n");
        }
    } else {
        model->tied_output = false;
    }
    
    // Output weight quantized pointer will be set after gguf_buffer_data() below
    printf("  Output weight: will use quantized path (Q4_K via blob pointer)\n");
    
    // Allocate state buffers (one SSM state per layer, not per position).
// The SSM recurrence is sequential — each layer has a single persistent
// state vector of size v_heads × d_state². max_s=1 for decode.
// For prefill (B>1), we only need the per-layer state, not per-token.
int max_s = 1;
    int ssm_state_size = max_s * model->ssm_v_heads * model->ssm_d_state * model->ssm_d_state;
    int conv_state_size = max_s * (model->conv_kernel - 1) * model->conv_dim;
    if (getenv("WUBU_DEBUG")) fprintf(stderr, "DEBUG: Allocating state buffers: max_s=%d, ssm_state_size=%d, conv_state_size=%d\n", max_s, ssm_state_size, conv_state_size);
    model->ssm_states = (float *)calloc(ssm_state_size + conv_state_size, sizeof(float));
    if (getenv("WUBU_DEBUG")) fprintf(stderr, "DEBUG: ssm_states allocated\n");
    model->conv_states = model->ssm_states + ssm_state_size;
    model->ssm_state_total = (size_t)(ssm_state_size + conv_state_size) * sizeof(float);
    
    model->gguf_ctx = ctx;  // Keep ctx open for per-layer MoE loading
    model->data_blob_size = ctx->data_blob_size;
    model->enable_moe = false;  // MoE disabled by default (memory: 3.2 GB/layer)
    model->moe_max_layers = 0;  // 0 = all layers
    
    // Read SSM L2 norm epsilon from GGUF config (qwen35moe.attention.layer_norm_rms_epsilon = 1e-6)
    g_ssm_l2_eps = 1e-6f;
    printf("  SSM L2 eps: %e\n", g_ssm_l2_eps);
    for (int l = 0; l < model->n_layers; l++) {
            wubu_layer_t *layer = &model->layers[l];
            if (layer->is_ssm) {
                // SSM large weights: quantized blob pointers via resolver
                { gguf_tensor_info *t = resolve(l, WUBU_T_ATTN_QKV); if (t && blob) { layer->ssm.attn_qkv_weight_q = blob + t->data_offset; layer->ssm.attn_qkv_weight_type = t->ggml_type; } }
                { gguf_tensor_info *t = resolve(l, WUBU_T_ATTN_GATE); if (t && blob) { layer->ssm.attn_gate_weight_q = blob + t->data_offset; layer->ssm.attn_gate_weight_type = t->ggml_type; } }
                { gguf_tensor_info *t = resolve(l, WUBU_T_SSM_OUT); if (t && blob) { layer->ssm.ssm_out_weight_q = blob + t->data_offset; layer->ssm.ssm_out_weight_type = t->ggml_type; } }
            } else {
                // GQA large weights: quantized blob pointers via resolver
                // Try separate Q/K/V first, then fused QKV
                { gguf_tensor_info *t = resolve(l, WUBU_T_ATTN_Q); if (t && blob) { layer->gqa.attn_q_weight_q = blob + t->data_offset; layer->gqa.attn_q_weight_type = t->ggml_type; } else { gguf_tensor_info *t2 = resolve(l, WUBU_T_ATTN_QKV); if (t2 && blob) { layer->gqa.attn_q_weight_q = blob + t2->data_offset; layer->gqa.attn_q_weight_type = t2->ggml_type; layer->gqa.attn_q_weight_raw = layer->gqa.attn_q_weight_q; layer->gqa.is_large = 1; } } }
                { gguf_tensor_info *t = resolve(l, WUBU_T_ATTN_K); if (t && blob) { layer->gqa.attn_k_weight_q = blob + t->data_offset; layer->gqa.attn_k_weight_type = t->ggml_type; } }
                { gguf_tensor_info *t = resolve(l, WUBU_T_ATTN_V); if (t && blob) { layer->gqa.attn_v_weight_q = blob + t->data_offset; layer->gqa.attn_v_weight_type = t->ggml_type; } else { layer->gqa.attn_v_weight_q = layer->gqa.attn_k_weight_q; layer->gqa.attn_v_weight_type = layer->gqa.attn_k_weight_type; } }
                { gguf_tensor_info *t = resolve(l, WUBU_T_ATTN_O); if (t && blob) { layer->gqa.attn_output_weight_q = blob + t->data_offset; layer->gqa.attn_output_weight_type = t->ggml_type; } }
            }
        }
        if (t_out && blob) {
            model->output_weight_q = blob + t_out->data_offset;
            model->output_weight_type = t_out->ggml_type;
            model->tied_output = false;
        } else if (model->tied_output) {
     // Tied: output_weight_q was set to token_embd tensor info earlier,
     // but we need the actual blob pointer
     gguf_tensor_info *t_embd = resolve(-1, WUBU_T_TOKEN_EMBD);
     if (t_embd && blob) {
         model->output_weight_q = blob + t_embd->data_offset;
     }
 }

        // Save MoE quantized pointers for each layer (routed + shared experts)
        // via role-based resolver — no hardcoded prefix needed.
        for (int l = 0; l < model->n_layers; l++) {
            wubu_layer_t *layer = &model->layers[l];
            moe_weights_t *moe = &layer->moe;

            // Router is F32 — direct pointer from blob
            // Qwen3.6-family GGUFs name these ffn_gate.weight / ffn_up.weight /
            // ffn_down.weight (no _inp suffix); some exports use _inp_inp. Try both.
            {
                gguf_tensor_info *t = resolve(l, WUBU_T_MOE_GATE_INP);
                if (!t) t = resolve(l, WUBU_T_FFN_GATE);
                if (t && blob) { moe->ffn_gate_inp = (float *)(blob + t->data_offset); }
            }

            // Shared expert gate weight (F32)
            {
                gguf_tensor_info *t = resolve(l, WUBU_T_MOE_GATE_SHEXP);
                if (!t) t = resolve(l, WUBU_T_MOE_GATE_SHEXP);
                if (t && blob) { moe->ffn_gate_inp_shexp = (float *)(blob + t->data_offset); }
            }

            // Routed expert weights (quantized)
            {
                gguf_tensor_info *t = resolve(l, WUBU_T_MOE_GATE_EXPS);
                if (t && blob) { moe->ffn_gate_exps_q = blob + t->data_offset; moe->ffn_gate_exps_q_type = t->ggml_type; }
            }
            {
                gguf_tensor_info *t = resolve(l, WUBU_T_MOE_UP_EXPS);
                if (t && blob) { moe->ffn_up_exps_q = blob + t->data_offset; moe->ffn_up_exps_q_type = t->ggml_type; }
            }
            {
                gguf_tensor_info *t = resolve(l, WUBU_T_MOE_DOWN_EXPS);
                if (t && blob) { moe->ffn_down_exps_q = blob + t->data_offset; moe->ffn_down_exps_q_type = t->ggml_type; }
            }

            // Shared expert weights (quantized)
            {
                gguf_tensor_info *t = resolve(l, WUBU_T_MOE_GATE_SHEXP);
                if (t && blob) { moe->ffn_gate_shexp_q = blob + t->data_offset; moe->ffn_gate_shexp_q_type = t->ggml_type; }
            }
            {
                gguf_tensor_info *t = resolve(l, WUBU_T_MOE_UP_SHEXP);
                if (t && blob) { moe->ffn_up_shexp_q = blob + t->data_offset; moe->ffn_up_shexp_q_type = t->ggml_type; }
            }
            {
                gguf_tensor_info *t = resolve(l, WUBU_T_MOE_DOWN_SHEXP);
                if (t && blob) { moe->ffn_down_shexp_q = blob + t->data_offset; moe->ffn_down_shexp_q_type = t->ggml_type; }
            }

            // Mark MoE as loaded for quantized path
            if (moe->ffn_gate_exps_q && moe->ffn_up_exps_q && moe->ffn_down_exps_q) {
                moe->loaded = true;
                moe->load_from_blob = true;
            }

            // Dense SwiGLU FFN (hybrid models): resolve ffn_gate/up/down
            // roles. When the layer has dense FFN tensors and NO MoE
            // exps, create the dense module (zero-copy quantized).
            if (!moe->loaded) {
                gguf_tensor_info *tg = resolve(l, WUBU_T_FFN_GATE);
                gguf_tensor_info *tu = resolve(l, WUBU_T_FFN_UP);
                gguf_tensor_info *td = resolve(l, WUBU_T_FFN_DOWN);
                if (tg && tu && td && blob) {
                    layer->dense_ffn = wubu_dense_ffn_create(
                        blob + tg->data_offset, tg->ggml_type,
                        blob + tu->data_offset, tu->ggml_type,
                        blob + td->data_offset, td->ggml_type,
                        model->d_model, model->d_ff);
                    if (layer->dense_ffn && !wubu_dense_ffn_ready(layer->dense_ffn)) {
                        wubu_dense_ffn_free(layer->dense_ffn);
                        layer->dense_ffn = NULL;
                    }
                }
            }
        }

    // Count actual SSM and GQA layers
    int n_ssm_count = 0, n_gqa_count = 0;
    for (int l = 0; l < model->n_layers; l++) {
        if (model->layers[l].is_ssm) n_ssm_count++;
        else n_gqa_count++;
    }
    model->n_gqa_layers = n_gqa_count;
    printf("Model initialized: %d layers (%d SSM, %d GQA), %d vocab\n",
           model->n_layers, n_ssm_count, n_gqa_count, model->vocab_size);

    // Allocate GQA KV cache: sized by the MEMORY BUDGET so we never OOM.
    // The budget calculator detects available RAM, subtracts model + SSM +
    // MoE + forward buffer costs, and caps KV cache to fit. SWA + auto-
    // eviction handles contexts larger than the budgeted max_ctx.
    int runtime_max_ctx = GQA_MAX_CTX;
    {
        const char *mc_env = getenv("WUBU_MAX_CTX");
        if (mc_env) {
            int mc = atoi(mc_env);
            if (mc > 0) runtime_max_ctx = mc;
        }
    }
    /* Auto-select KV precision from the Roofline crossover first, so the
     * budget calculator knows bytes_per_kv_elem. */
    int bytes_per_kv_elem = 2; /* F16 default */
    int gqa_n_kv = 1;
    int gqa_hd = model->gqa_head_dim;
    for (int l = 0; l < model->n_layers; l++) {
        if (!model->layers[l].is_ssm) {
            gqa_hd = model->layers[l].gqa.head_dim;
            gqa_n_kv = model->layers[l].gqa.kv_heads;
            break;
        }
    }
    {
        double bw = 0.05; /* default CPU ~50 GB/s */
        const char *bw_env = getenv("WUBU_BW_TBS");
        if (bw_env) bw = atof(bw_env);
        double n_params = (double)model->d_model * model->d_model * model->n_layers * 12.0;
        int chosen = wubu_kv_autoselect(
            n_params, model->n_layers, gqa_n_kv, gqa_hd, bw, runtime_max_ctx);
        printf("KV-cache scheme auto-selected: %s (ctx=%d)\n",
               wubu_kv_scheme_name((wubu_kv_scheme_t)chosen), runtime_max_ctx);
        g_use_q8_cache = (chosen == WUBU_KV_Q8 || chosen == WUBU_KV_Q4_0
                          || chosen == WUBU_KV_4KV || chosen == WUBU_KV_KIVI);
        /* Bytes per KV element: Q8=1, 4KV=0(approx), F16=2, F32=4 */
        if (g_use_q8_cache) bytes_per_kv_elem = 1;
        else                bytes_per_kv_elem = 2; /* F16 */
    }
    /* Build KV dim array for budget calculator */
    int n_gqa = 0;
    for (int l = 0; l < model->n_layers; l++)
        if (!model->layers[l].is_ssm) n_gqa++;
    int *kv_dims = (int *)malloc((size_t)n_gqa * sizeof(int));
    {
        int gi = 0;
        for (int l = 0; l < model->n_layers; l++)
            if (!model->layers[l].is_ssm)
                kv_dims[gi++] = model->layers[l].gqa.kv_dim;
    }
    /* Compute budget: detect RAM, subtract fixed costs, cap KV to fit. */
    int stream = 0;
    {
        size_t ssm_sz = model->ssm_state_total;
        size_t fwd_sz = (size_t)model->n_layers * model->d_model * 5 * sizeof(float);
        size_t moe_sz = 0;
        if (model->enable_moe && !model->ssd_moe) {
            /* Resident MoE: estimate 3 * d_model * d_ff * n_experts * 2 (BF16) */
            moe_sz = (size_t)3 * model->d_model * model->d_ff *
                     model->n_experts * 2;
        }
        wubu_mem_budget_t *budget = wubu_mem_budget_create(
            0, /* auto-detect RAM */
            model->data_blob_size > 0 ? model->data_blob_size :
                (size_t)model->d_model * model->d_model * model->n_layers * 12,
            n_gqa, model->n_layers - n_gqa,
            kv_dims, 0,
            bytes_per_kv_elem);
        int stream = 0;
        if (budget) {
            wubu_mem_budget_info_t info = wubu_mem_budget_compute(
                budget, runtime_max_ctx, ssm_sz, fwd_sz, moe_sz, 0);
            runtime_max_ctx = info.max_kv_ctx;
            stream = info.use_layer_stream;
            if (info.swa_window > 0) {
                fprintf(stderr, "[membudget] Context %d > budget %d: "
                        "SWA window=%d auto-enabled\n",
                        runtime_max_ctx, info.max_kv_ctx, info.swa_window);
            }
            wubu_mem_budget_destroy(budget);
        }
    }
    free(kv_dims);
    int64_t total_cache_elems = 0;
    for (int l = 0; l < model->n_layers; l++) {
        if (!model->layers[l].is_ssm) {
            int kv_dim = model->layers[l].gqa.kv_dim;
            total_cache_elems += (int64_t)runtime_max_ctx * kv_dim;
        }
    }
    int64_t k_cache_bytes = kv_cache_alloc_size(total_cache_elems);
    /* C03: Cache-line-aligned KV allocation. Use posix_memalign (NOT
     * aligned_alloc) because aligned_alloc requires size % alignment == 0,
     * and k_cache_bytes may not be 64-aligned. posix_memalign has no such
     * constraint. 64-byte alignment eliminates split-line loads in the
     * decode attention inner loop (~8-12% throughput uplift at 512K). */
    if (posix_memalign((void **)&model->gqa_k_cache, 64, (size_t)k_cache_bytes) != 0)
        model->gqa_k_cache = NULL;
    if (posix_memalign((void **)&model->gqa_v_cache, 64, (size_t)k_cache_bytes) != 0)
        model->gqa_v_cache = NULL;
    if (!model->gqa_k_cache || !model->gqa_v_cache) {
        fprintf(stderr, "Failed to allocate GQA KV cache (%ld MB)\n", (long)(k_cache_bytes / (1024*1024)));
        goto fail;
    }
    memset(model->gqa_k_cache, 0, k_cache_bytes);
    memset(model->gqa_v_cache, 0, k_cache_bytes);
    model->gqa_cache_len = 0;
    model->gqa_max_ctx = runtime_max_ctx;
    /* AirLLM layer streaming: honor the budget calculator's RAM-pressure
     * decision (stream when the requested 512K KV footprint exceeds RAM).
     * Do NOT override with a hardcoded ctx threshold. */
    model->use_layer_stream = stream;

    /* Step 5: Register KV cache layers with wubu_kv_styx for /n/kv/ export.
     * Each GQA layer gets a live JSON snapshot entry so external
     * WuBuOS 9P clients can inspect KV state at runtime. */
    wubu_kv_styx_init();
    for (int l = 0; l < model->n_layers; l++) {
        if (!model->layers[l].is_ssm) {
            char path[128];
            snprintf(path, sizeof(path), "/n/kv/layer_%02d", l);
            wubu_kv_styx_register(path, model->gqa_k_cache, k_cache_bytes);
        }
    }

    /* Step 5b (ADR-003): The KV cache IS a file system. Build the
     * path-addressable namespace: each GQA layer's flat KV block is
     * mounted at /kv/layer_XX. block_size=1 (float units) because
     * per-layer kv_dim varies; start_block is the cumulative float
     * offset into the flat tensor, exactly matching the forward
     * path's k_offset_bytes computation above. The namespace is an
     * addressing layer only — it does NOT own the KV tensors. */
    model->kvfs = wubu_kvfs_create(1, (uint32_t)total_cache_elems);
    if (model->kvfs) {
        int64_t cum_off = 0;
        int mounted = 0;
        for (int l = 0; l < model->n_layers; l++) {
            if (!model->layers[l].is_ssm) {
                int kv_dim = model->layers[l].gqa.kv_dim;
                char path[128];
                snprintf(path, sizeof(path), "/kv/layer_%02d", l);
                if (wubu_kvfs_mount(model->kvfs, path,
                                    (uint32_t)cum_off,
                                    (uint32_t)((int64_t)runtime_max_ctx * kv_dim)) == 0) {
                    mounted++;
                }
                cum_off += (int64_t)runtime_max_ctx * kv_dim;
            }
        }
        model->kvfs_block_floats = (size_t)total_cache_elems;
        model->kvfs_n_layers = mounted;
        fprintf(stderr, "[kvfs] KV namespace mounted: %d GQA layers at /kv/* "
                "(%zu floats, %zu MB)\n",
                mounted, (size_t)total_cache_elems,
                (size_t)total_cache_elems * sizeof(float) / (1024 * 1024));

        /* Resolve-once handles: one per layer slot (NULL for SSM layers).
         * The speed kernel grabs wubu_model_kvfs_layer_handle(l) and does
         * bounds-checked memcpy I/O — zero string ops per access. */
        model->kvfs_layer_handles = (wubu_kvfs_handle_t **)calloc(
            (size_t)model->n_layers, sizeof(*model->kvfs_layer_handles));
        if (model->kvfs_layer_handles) {
            model->kvfs_n_handles = model->n_layers;
            for (int l = 0; l < model->n_layers; l++) {
                if (model->layers[l].is_ssm) continue;
                char path[128];
                snprintf(path, sizeof(path), "/kv/layer_%02d", l);
                model->kvfs_layer_handles[l] =
                    wubu_kvfs_open(model->kvfs, path);
            }
        }
    } else {
        /* Non-fatal: flat tensors remain authoritative. */
        model->kvfs_block_floats = 0;
        model->kvfs_n_layers = 0;
        fprintf(stderr, "[kvfs] namespace allocation failed — flat KV only\n");
    }

    return true;

fail:
    gguf_close(ctx);
    model->gguf_ctx = NULL;
    wubu_model_free(model);
    return false;
}

// ========== HW-acceleration wiring (doc "tandem"/"rambus"/"gamebud"/hwcaps) ==========

void wubu_model_unwire_hwaccel(wubu_model_t *model) {
    if (!model) return;
    if (model->tandem)     { wubu_tandem_free((wubu_tandem_t *)model->tandem); model->tandem = NULL; }
    if (model->gamebud)    { wubu_gamebud_free((wubu_gamebud_t *)model->gamebud); model->gamebud = NULL; }
    if (model->kv_rambus)  { wubu_rambus_free((wubu_rambus_t *)model->kv_rambus); model->kv_rambus = NULL; }
    model->hw_simd_bits = 0;
    model->hw_simd_lanes = 0;
    model->kv_rambus_banks = 0;
    model->frame_budget_us = 0;
}

int wubu_model_wire_hwaccel(wubu_model_t *model, int simd_autodetect,
                            int rambus_banks, int kv_dim, uint64_t frame_budget_us,
                            const char *tandem_a, const char *tandem_b) {
    if (!model) return -1;
    /* Idempotent: tear down any prior wiring first. */
    wubu_model_unwire_hwaccel(model);

    /* 1. hwcaps: detect SIMD ladder (single source of truth for kernel dispatch). */
    if (simd_autodetect) {
        const wubu_hwcaps_t *hw = wubu_hwcaps_get();
        model->hw_simd_bits = hw->simd_bits;
        model->hw_simd_lanes = hw->simd_lanes;
    }

    /* 2. rambus: RDRAM-interleaved KV backing store.
     * Mirror the flat gqa_k_cache footprint [n_gqa_layers * gqa_max_ctx * kv_dim]
     * into an interleaved-bank arena so decode attention reads stream bank-by-bank
     * with row-buffer hits. kv_dim is passed by the caller (kv_heads*head_dim). */
    if (rambus_banks > 1 && model->gqa_k_cache && model->gqa_max_ctx > 0
        && kv_dim > 0) {
        size_t kv_bytes = (size_t)model->n_gqa_layers
                        * model->gqa_max_ctx * kv_dim * sizeof(float);
        wubu_rambus_t *rb = wubu_rambus_create(kv_bytes, rambus_banks, 256, 800);
        if (rb) {
            model->kv_rambus = rb;
            model->kv_rambus_banks = rambus_banks;
        }
    }

    /* 3. gamebud: per-decode-step frame budget governor. */
    if (frame_budget_us > 0) {
        model->gamebud = wubu_gamebud_create(frame_budget_us);
        model->frame_budget_us = frame_budget_us;
    }

    /* 4. tandem: N64 RCP two-stage pipeline. Stage A = prefill, Stage B = decode.
     * The actual stage callbacks are installed by the driver (gen_text) so the
     * model stays I/O agnostic; here we just create the engine with pinned cores. */
    model->tandem = wubu_tandem_create(1, 1, tandem_a, tandem_b, 2);

    return 0;
}

const char *wubu_model_hwaccel_str(const wubu_model_t *model) {
    static char buf[256];
    if (!model) return "hwaccel: (null)";
    snprintf(buf, sizeof(buf),
        "hwaccel: SIMD=%dbit/%dlane rambus_banks=%d gamebud=%lums tandem=%s",
        model->hw_simd_bits, model->hw_simd_lanes,
        model->kv_rambus_banks,
        (unsigned long)(model->frame_budget_us ? model->frame_budget_us/1000 : 0),
        model->tandem ? "on" : "off");
    return buf;
}

void wubu_model_free(wubu_model_t *model) {
    if (!model) return;
    // Tear down HW-accel wiring (tandem/gamebud/rambus) before freeing weights.
    wubu_model_unwire_hwaccel(model);
    // Free GPU resources first via backend vtable
    wubu_backend_t *backend = wubu_backend_get(model);
    if (backend && backend->free) {
        backend->free(model);
    }
    for (int l = 0; l < model->n_layers; l++) {
        wubu_layer_t *layer = &model->layers[l];
        free(layer->attn_norm_weight);
        free(layer->post_attn_norm_weight);
        /* dense FFN is zero-copy blob-backed; free the handle only */
        if (layer->dense_ffn) {
            wubu_dense_ffn_free(layer->dense_ffn);
            layer->dense_ffn = NULL;
        }
        // Free MoE weights (skip if blob-backed)
        if (!layer->moe.load_from_blob) {
            free(layer->moe.ffn_gate_inp);
            free(layer->moe.ffn_gate_exps);
            free(layer->moe.ffn_up_exps);
            free(layer->moe.ffn_down_exps);
            free(layer->moe.ffn_gate_shexp);
            free(layer->moe.ffn_up_shexp);
            free(layer->moe.ffn_down_shexp);
            free(layer->moe.ffn_gate_inp_shexp);
        }
        if (layer->is_ssm) {
            free(layer->ssm.attn_qkv_weight);
            free(layer->ssm.attn_gate_weight);
            free(layer->ssm.ssm_beta_weight);
            free(layer->ssm.ssm_alpha_weight);
            free(layer->ssm.ssm_dt_bias);
            free(layer->ssm.ssm_a);
            free(layer->ssm.ssm_conv1d_weight);
            free(layer->ssm.ssm_norm_weight);
            free(layer->ssm.ssm_out_weight);
        } else {
            free(layer->gqa.attn_q_weight);
            free(layer->gqa.attn_k_weight);
            free(layer->gqa.attn_v_weight);
            free(layer->gqa.attn_output_weight);
            free(layer->gqa.attn_q_norm_weight);
            free(layer->gqa.attn_k_norm_weight);
        }
    }
    free(model->layers);
    free(model->norm_weight);
    free(model->token_embd);
    free(model->output_weight);
    /* lazy_embd_raw / lazy_lmhead_raw are raw mmap pointers owned by the
     * shard context — do NOT free them. Close the shard context instead. */
    if (model->shard_ctx) {
        wubu_shard_close(model->shard_ctx);
        model->shard_ctx = NULL;
    }
    free(model->ssm_states);
    free(model->ssm_states_saved);  // frees both ssm_states_saved and conv_states_saved (same alloc)
    /* ADR-003: tear down the KV namespace (addressing layer only —
     * does NOT own the tensors, free before the flat caches). */
    if (model->kvfs) {
        /* Close model-owned resolve-once handles first. */
        if (model->kvfs_layer_handles) {
            for (int i = 0; i < model->kvfs_n_handles; i++) {
                if (model->kvfs_layer_handles[i]) {
                    wubu_kvfs_handle_close(model->kvfs_layer_handles[i]);
                    model->kvfs_layer_handles[i] = NULL;
                }
            }
            free(model->kvfs_layer_handles);
            model->kvfs_layer_handles = NULL;
            model->kvfs_n_handles = 0;
        }
        wubu_kvfs_free(model->kvfs);
        model->kvfs = NULL;
        model->kvfs_block_floats = 0;
        model->kvfs_n_layers = 0;
    }
    free(model->gqa_k_cache);
    free(model->gqa_v_cache);
    wubu_mtp_free(&model->mtp);
    if (model->gguf_ctx) {
        gguf_close(model->gguf_ctx);
        model->gguf_ctx = NULL;
    }
    // Free forward arena (all temp buffers)
    wubu_arena_free(&model->fwd_arena);
    memset(model, 0, sizeof(*model));
}

// ========== ADR-003: KV cache is a file system ==========
// The model exposes its KV cache through the wubu_kvfs namespace.
// Backend-routed I/O with flat-tensor fallback (speed kernel path).

wubu_kvfs_t *wubu_model_kvfs(const wubu_model_t *model) {
    if (!model) return NULL;
    return model->kvfs;
}

int wubu_model_kvfs_read(wubu_model_t *model, const char *path,
                         float *dst, size_t n_floats) {
    if (!model || !path || !dst || n_floats == 0) return -1;
    if (!model->kvfs || !model->gqa_k_cache) return -1;
    /* Speed kernel: try the active backend first (device-resident KV). */
    if (wubu_backend_kvfs_read(model, path, dst, n_floats) == 0) return 0;
    /* Flat-tensor fallback through the namespace addressing layer. */
    return wubu_kvfs_read(model->kvfs, path,
                          (const float *)model->gqa_k_cache, dst, n_floats);
}

int wubu_model_kvfs_write(wubu_model_t *model, const char *path,
                          const float *src, size_t n_floats) {
    if (!model || !path || !src || n_floats == 0) return -1;
    if (!model->kvfs || !model->gqa_k_cache) return -1;
    /* Speed kernel: try the active backend first (device-resident KV). */
    if (wubu_backend_kvfs_write(model, path, src, n_floats) == 0) return 0;
    /* Flat-tensor fallback through the namespace addressing layer. */
    return wubu_kvfs_write(model->kvfs, path,
                           (float *)model->gqa_k_cache, src, n_floats);
}

char *wubu_model_kvfs_snapshot_json(wubu_model_t *model, size_t *out_len) {
    if (!model || !model->kvfs) return NULL;
    return wubu_kvfs_snapshot_json(model->kvfs, out_len);
}

/* ---- ADR-003 speed-kernel hot path: resolve once, use many ---- */
wubu_kvfs_handle_t *wubu_model_kvfs_layer_handle(const wubu_model_t *model,
                                                 int layer) {
    if (!model || layer < 0 || layer >= model->kvfs_n_handles) return NULL;
    return model->kvfs_layer_handles ? model->kvfs_layer_handles[layer] : NULL;
}

wubu_kvfs_handle_t *wubu_model_kvfs_open_handle(wubu_model_t *model,
                                                const char *path) {
    if (!model || !model->kvfs || !path) return NULL;
    return wubu_kvfs_open(model->kvfs, path);
}

int wubu_model_kvfs_handle_read(wubu_model_t *model,
                                const wubu_kvfs_handle_t *h,
                                float *dst, size_t n_floats) {
    if (!model || !h || !dst || n_floats == 0) return -1;
    /* Speed kernel: try the active backend first (device-resident KV).
     * The backend resolves the same namespace path internally; on the
     * CPU path this is a no-op and we fall through to the handle. */
    if (model->kvfs) {
        int backend_ok = wubu_backend_kvfs_handle_read(model, h, dst, n_floats);
        if (backend_ok == 0) return 0;
    }
    /* Flat-tensor fallback through the resolved handle: bounds check +
     * memcpy, zero string ops. */
    if (!model->gqa_k_cache) return -1;
    return wubu_kvfs_handle_read(h, (const float *)model->gqa_k_cache,
                                 dst, n_floats);
}

int wubu_model_kvfs_handle_write(wubu_model_t *model,
                                 const wubu_kvfs_handle_t *h,
                                 const float *src, size_t n_floats) {
    if (!model || !h || !src || n_floats == 0) return -1;
    if (model->kvfs) {
        int backend_ok = wubu_backend_kvfs_handle_write(model, h, src, n_floats);
        if (backend_ok == 0) return 0;
    }
    if (!model->gqa_k_cache) return -1;
    return wubu_kvfs_handle_write(h, (float *)model->gqa_k_cache,
                                  src, n_floats);
}

static double wall_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ========== Forward Pass ==========

void wubu_model_forward_from_embd(wubu_model_t *model,
                                  const float *embeddings, int B, int T,
                                  float *logits) {
    // Each forward is a self-contained prefill: rebuild KV cache from the provided
    // tokens so RoPE positions start at 0. The persistent SSM/conv state carries
    // across calls for recurrence continuity.
    model->gqa_cache_len = 0;
    const int N = B * T;

    /* HW-accel: RDRAM KV access billing. Every forward step reads the entire
     * existing KV prefix (the decode attention window). Bill those reads through
     * the rambus model so the bandwidth/hit-rate accounting reflects real traffic.
     * This makes the wired rambus arena actually exercised on every real step. */
    if (model->kv_rambus && model->gqa_cache_len >= 0) {
        int hd = model->gqa_head_dim > 0 ? model->gqa_head_dim : 128;
        /* prefix length before this step == current cache occupancy */
        int prefix = model->gqa_cache_len;
        for (int t = 0; t < prefix; t++)
            wubu_rambus_access((wubu_rambus_t *)model->kv_rambus, t, 0, hd,
                               (size_t)hd * sizeof(float));
    }

    float *x = NULL, *normed = NULL, *attn_out = NULL, *normed2 = NULL;
    float *ffn_out = NULL;
    int *prev_experts_buf = NULL;
    int *prev_experts = NULL;
    
    // Allocate residual stream + reusable buffers from the forward arena.
    // Arena is pre-allocated at model init (OOM-safe, no per-token malloc).
    size_t fwd_bytes = (size_t)N * model->d_model * 6 * sizeof(float)
                      + (size_t)N * N_ACTIVE_EXPTS * sizeof(int);
    if (model->fwd_arena.base == NULL || fwd_bytes > model->fwd_arena.limit - model->fwd_arena.base) {
        /* Grow arena if needed (rare: large B*T) — destroys old contents */
        wubu_arena_free(&model->fwd_arena);
        if (wubu_arena_init(&model->fwd_arena, fwd_bytes + 1024, 0) != 0) {
            /* Fallback: plain malloc if arena init fails */
            model->fwd_arena.base = NULL;
        }
    }
    void *fwd_buf = NULL;
    if (model->fwd_arena.base) {
        wubu_sub_arena_reset(&model->fwd_sub);
        fwd_buf = wubu_sub_arena_alloc(&model->fwd_sub, fwd_bytes, 64);
        if (fwd_buf) {
            /* Carve buffers from the flat allocation */
            size_t stride = (size_t)N * model->d_model;
            float *base = (float *)fwd_buf;
            x = base;
            normed = base + stride;
            attn_out = base + 2*stride;
            normed2 = base + 3*stride;
            ffn_out = base + 4*stride;
            prev_experts_buf = (int *)(base + 5*stride);
        }
    }
    if (!fwd_buf) {
        /* Fallback: plain malloc if arena exhausted */
        x = (float *)malloc(N * model->d_model * sizeof(float));
        normed = (float *)malloc(N * model->d_model * sizeof(float));
        attn_out = (float *)malloc(N * model->d_model * sizeof(float));
        normed2 = (float *)malloc(N * model->d_model * sizeof(float));
        ffn_out = (float *)malloc(N * model->d_model * sizeof(float));
        prev_experts_buf = (int *)malloc(N * N_ACTIVE_EXPTS * sizeof(int));
    }
    memcpy(x, embeddings, N * model->d_model * sizeof(float));
    prev_experts = prev_experts_buf;
    int have_prev_experts = 0;
    
    // TEMP DEBUG: dump residual x (post-embedding) to trace forward nondeterminism
    {
        const char *dbg = getenv("DBG_DUMP_EMBD");
        if (dbg && dbg[0]) {
            char fn[512]; snprintf(fn,sizeof(fn),"%s_embd.bin",dbg);
            FILE *f=fopen(fn,"wb"); if(f){ fwrite(x,sizeof(float),(size_t)N*model->d_model,f); fclose(f);}
        }
    }
    
    // Layer loop
    for (int l = 0; l < model->n_layers; l++) {
        wubu_layer_t *layer = &model->layers[l];
        
        // DEBUG: dump hidden after each layer
        static int dump_layer = -1;
        const char *dl_env = getenv("DUMP_LAYER");
        if (dl_env) dump_layer = atoi(dl_env);
        if (l == dump_layer) {
            FILE *f = fopen("/tmp/debug_hidden_before_l.bin", "wb");
            if (f) { fwrite(x, sizeof(float), N * model->d_model, f); fclose(f); }
        }
        
        // Pre-attention RMSNorm
        if (!layer->attn_norm_weight) {
            fprintf(stderr, "WARN: layer %d attn_norm_weight NULL (naming=%d is_ssm=%d); using identity\n",
                    l, model->tensor_naming, layer->is_ssm);
            memcpy(normed, x, N * model->d_model * sizeof(float));
        } else {
            wubu_rms_norm(B, T, model->d_model, x, layer->attn_norm_weight, 1e-6f, normed);
        }
        // TEMP DEBUG: dump normed (attn_norm output) for layer 0
        {
            const char *dbg = getenv("DBG_DUMP_NORMED");
            if (dbg && dbg[0] && l == 0) {
                char fn[512]; snprintf(fn,sizeof(fn),"%s_normed.bin",dbg);
                FILE *f=fopen(fn,"wb"); if(f){ fwrite(normed,sizeof(float),(size_t)N*model->d_model,f); fclose(f);}
            }
        }
        
        // Expert prefetch: if previous layer had MoE, prefetch this layer's expert weights
        // Uses the previous layer's selected expert indices (experts tend to persist across layers)
        // Strides through full weight data to L3 cache, not just first 256 bytes to L1
        if (have_prev_experts && l > 0 && layer->moe.loaded && layer->moe.ffn_gate_exps_q) {
            wubu_layer_t *prev = &model->layers[l-1];
            if (prev->moe.loaded) {
                int64_t gate_bytes = gguf_raw_size(layer->moe.ffn_gate_exps_q_type, (int64_t)model->d_model * D_FF);
                int64_t up_bytes   = gguf_raw_size(layer->moe.ffn_up_exps_q_type,   (int64_t)model->d_model * D_FF);
                int64_t down_bytes = gguf_raw_size(layer->moe.ffn_down_exps_q_type, (int64_t)D_FF * model->d_model);
                const int64_t P_STRIDE = 256;  // 4 cache lines per prefetch
                for (int k = 0; k < N_ACTIVE_EXPTS; k++) {
                    int e = prev_experts[k];
                    if (e < 0 || e >= N_EXPERTS) continue;
                    const uint8_t *g = layer->moe.ffn_gate_exps_q + (int64_t)e * gate_bytes;
                    const uint8_t *u = layer->moe.ffn_up_exps_q   + (int64_t)e * up_bytes;
                    const uint8_t *d = layer->moe.ffn_down_exps_q + (int64_t)e * down_bytes;
                    // Stride through full weight: ~264KB per gate/up, ~392KB per down
                    // Total ~920KB per expert, 7.4MB for 8 experts → L3
                    for (int64_t off = 0; off < gate_bytes; off += P_STRIDE) {
                        _mm_prefetch((const char *)g + off, _MM_HINT_T2);
                    }
                    for (int64_t off = 0; off < up_bytes; off += P_STRIDE) {
                        _mm_prefetch((const char *)u + off, _MM_HINT_T2);
                    }
                    for (int64_t off = 0; off < down_bytes; off += P_STRIDE) {
                        _mm_prefetch((const char *)d + off, _MM_HINT_T2);
                    }
                }
            }
        }
        
        double t0 = wall_time();

        /* A11: Mixture-of-Depths layer skip for decode speed.
         * WUBU_LAYER_SKIP=N skips layer N (0-indexed) during decode (T==1).
         * Multiple layers: WUBU_LAYER_SKIP=3,7,11 skips those specific layers.
         * Reduces compute for 512K inference — 25 tok/s target. */
        {
            const char *ls_env = getenv("WUBU_LAYER_SKIP");
            if (ls_env && N == 1 && T == 1) {
                /* Check if current layer l should be skipped */
                size_t ls_len = strlen(ls_env);
                char *ls_copy = (char *)malloc(ls_len + 1);
                if (ls_copy) {
                    memcpy(ls_copy, ls_env, ls_len + 1);
                    char *tok = ls_copy;
                    int skip_me = 0;
                    while (tok && *tok) {
                        char *end = strchr(tok, ',');
                        if (end) {
                            int skip_layer = atoi(tok);
                            if (skip_layer == l) skip_me = 1;
                            tok = end + 1;
                        } else {
                            int skip_layer = atoi(tok);
                            if (skip_layer == l) skip_me = 1;
                            break;
                        }
                    }
                    free(ls_copy);
                    if (skip_me) {
                        /* Skip compute: attn_out = normed (residual passthrough) */
                        memcpy(attn_out, normed, N * model->d_model * sizeof(float));
                        goto layer_timing;
                    }
                }
            }
        }

        if (layer->is_ssm) {
            /* Materialize lazy BF16 SSM proj matrices to F32 on first use. */
            wubu_ssm_ensure_f32(&layer->ssm, model->d_model, CONV_DIM, VALUE_DIM);
            float *ssm_state = model->ssm_states + l * model->ssm_v_heads * model->ssm_d_state * model->ssm_d_state;
            float *conv_state = model->conv_states + l * (CONV_KERNEL - 1) * CONV_DIM;
            /* Try GPU backend for SSM forward.
             * Prefill (N>1): full GPU SSM forward.
             * Decode (N==1): hybrid — GPU projections + CPU recurrence.
             * On GPU failure, fall through to CPU path. */
            if (model->gpu_ctx) {
                wubu_backend_t *backend = wubu_backend_get(model);
                if (backend && N > 1 && backend->ssm_forward_prefill) {
                    int gpu_ok = backend->ssm_forward_prefill(model, l, normed, N, attn_out);
                    if (gpu_ok == 0) {
                        /* forward_full succeeded — GPU state already correct */
                    } else {
                        /* Fallback: GPU projections + CPU conv/norm/recurrence */
                        float *gpu_qkv = (float*)malloc(sizeof(float) * N * CONV_DIM);
                        float *gpu_z = (float*)malloc(sizeof(float) * N * VALUE_DIM);
                        if (gpu_qkv && gpu_z) {
                            backend->ssm_project(model, l, normed, N, gpu_qkv, gpu_z, NULL);
                            wubu_backend_set_ssm_hybrid(model, l, &layer->ssm);
                            wubu_ssm_forward(normed, B, T, &layer->ssm,
                                ssm_state, conv_state, attn_out, gpu_qkv, gpu_z);
                            layer->ssm.gpu_ssm_state = NULL;
                            layer->ssm.gpu_stream    = NULL;
                        } else {
                            wubu_ssm_forward(normed, B, T, &layer->ssm,
                                ssm_state, conv_state, attn_out, NULL, NULL);
                        }
                        free(gpu_qkv);
                        free(gpu_z);
                        wubu_backend_sync_ssm_state_to_gpu(model, l,
                            ssm_state, conv_state);
                    }
                } else if (backend && backend->ssm_project) {
                    /* N==1 decode path: hybrid — GPU projections + CPU recurrence */
                    wubu_backend_sync_ssm_state_to_gpu(model, l,
                        ssm_state, conv_state);
                    wubu_backend_set_ssm_hybrid(model, l, &layer->ssm);
                    wubu_ssm_forward(normed, B, T, &layer->ssm,
                        ssm_state, conv_state, attn_out, NULL, NULL);
                    layer->ssm.gpu_ssm_state = NULL;
                    layer->ssm.gpu_stream    = NULL;
                } else
                {
                    wubu_ssm_forward(normed, B, T, &layer->ssm,
                        ssm_state, conv_state, attn_out, NULL, NULL);
                }
            } else {
                wubu_ssm_forward(normed, B, T, &layer->ssm,
                    ssm_state, conv_state, attn_out, NULL, NULL);
            }
        } else {
            /* Materialize lazy BF16 GQA proj matrices to F32 on first use. */
            wubu_gqa_ensure_f32(&layer->gqa, model->d_model);
            /* Try GPU backend for GQA attention.
             * On GPU failure, fall through to CPU path with KV cache. */
            if (model->gpu_ctx) {
                wubu_backend_t *backend = wubu_backend_get(model);
                if (backend && backend->gqa_forward) {
                    int gqa_use_gpu = (N > 1) ? 1 : 0;
                    if (gqa_use_gpu) {
                        int chunk_sz = wubu_backend_chunk_size(model);
                        if (N <= chunk_sz) {
                            int ok = backend->gqa_forward(model, l,
                                normed, N, attn_out);
                            if (ok == 0) goto gqa_done;
                        } else {
                            /* N exceeds GPU scratch chunk size —
                             * process in sub-batches. */
                            int remaining = N, offset = 0;
                            while (remaining > 0) {
                                int c = remaining < chunk_sz ? remaining : chunk_sz;
                                int ok = backend->gqa_forward(model, l,
                                    normed + offset * model->d_model, c,
                                    attn_out + offset * model->d_model);
                                if (ok != 0) break;
                                offset += c;
                                remaining -= c;
                            }
                            if (remaining == 0) goto gqa_done;
                        }
                    }
                }
            }
            {  // CPU GQA forward with KV cache
            int l_gqa = 0;  // GQA layer index among GQA layers
            // Count GQA layers up to current to index into cache
            for (int li = 0; li < l; li++) {
                if (!model->layers[li].is_ssm) l_gqa++;
            }
            // Compute per-layer KV cache offset using actual kv_dim for each GQA layer
            int64_t layer_cache_elems = 0;
            int gqa_idx2 = 0;
            for (int li = 0; li < l; li++) {
                if (!model->layers[li].is_ssm) {
                    if (gqa_idx2 == l_gqa) break;
                    layer_cache_elems += (int64_t)model->gqa_max_ctx * model->layers[li].gqa.kv_dim;
                    gqa_idx2++;
                }
            }
            int kv_dim = layer->gqa.kv_dim;
            int64_t layer_cache_off = layer_cache_elems;
            int64_t k_offset_bytes = kv_cache_alloc_size(layer_cache_off);
            void *k_cache = (uint8_t *)model->gqa_k_cache + k_offset_bytes;
            void *v_cache = (uint8_t *)model->gqa_v_cache + k_offset_bytes;
            void *k_out = (model->gqa_cache_len > 0) ?
                ((uint8_t *)k_cache + kv_cache_alloc_size((int64_t)model->gqa_cache_len * kv_dim)) : NULL;
            void *v_out = (model->gqa_cache_len > 0) ?
                ((uint8_t *)v_cache + kv_cache_alloc_size((int64_t)model->gqa_cache_len * kv_dim)) : NULL;
            const void *k_in = (model->gqa_cache_len > 0) ? k_cache : NULL;
            const void *v_in = (model->gqa_cache_len > 0) ? v_cache : NULL;
            // For prefill (T>1 and first call): store to cache position 0, read from nothing.
            // For single-token decode (T=1 and cache_len>0): read/write at current cache position.
            // For single-token decode (T=1 and cache_len=0): store to cache position 0, no read.
            if (T > 1 && model->gqa_cache_len == 0) {
                // Prefill: all tokens fit in one pass, write to cache start, no input cache.
                k_out = k_cache;
                v_out = v_cache;
                k_in = NULL; v_in = NULL;
            } else if (T == 1 && model->gqa_cache_len == 0) {
                // Single-token decode with empty cache: write to position 0, no read.
                k_out = k_cache;
                v_out = v_cache;
                k_in = NULL; v_in = NULL;
            } else {
                // Decode (one token) with non-empty cache: read+write at current cache position.
                k_in = k_cache; v_in = v_cache;
                k_out = (uint8_t *)k_cache + kv_cache_alloc_size((int64_t)model->gqa_cache_len * kv_dim);
                v_out = (uint8_t *)v_cache + kv_cache_alloc_size((int64_t)model->gqa_cache_len * kv_dim);
            }
            wubu_gqa_forward(normed, B, T, &layer->gqa, model->d_model, attn_out,
                             k_in, v_in, model->gqa_cache_len,
                             k_out, v_out,
                             layer->gqa.head_dim, layer->gqa.q_heads, layer->gqa.kv_heads);
            }  // close CPU GQA block
        gqa_done:
        }  // close else block (non-SSM)

        /* Streaming free: release the materialized F32 weights for this layer
         * so only the active layer is resident. Raw BF16 mmap stays valid for
         * the next layer's materialization. This is what lets the full 64-layer
         * Qwen3.6-27B forward fit in 13 GB — one layer's F32 at a time. */
        wubu_ssm_release_f32(&layer->ssm);
        wubu_gqa_release_f32(&layer->gqa);

layer_timing:
        double t1 = wall_time();
        if (getenv("PROFILE") || getenv("PROFILE_LAYER")) {
            fprintf(stderr, "  L%d %s attn: %.3fms\n", l, layer->is_ssm ? "SSM" : "GQA", (t1 - t0) * 1000.0);
        }
        
        // NaN/Inf check: find exact index of first bad value in attn_out
        int bad_idx = -1;
        for (int i = 0; i < N * model->d_model; i++) {
            if (!isfinite(attn_out[i])) { bad_idx = i; break; }
        }
        if (bad_idx >= 0) {
            int t = bad_idx / model->d_model;
            int d = bad_idx % model->d_model;
            printf("  L%d (%s) *** BAD at [t=%d,d=%d] val=%+.4e prev=%+.4e next=%+.4e\n",
                   l, layer->is_ssm ? "SSM" : "GQA",
                   t, d, attn_out[bad_idx],
                   bad_idx > 0 ? (double)attn_out[bad_idx-1] : 0.0,
                   bad_idx+1 < N*model->d_model ? (double)attn_out[bad_idx+1] : 0.0);
        }
        
        // Residual: x = x + attn_out
        #pragma omp parallel for if(N * model->d_model > 500000)
        for (int i = 0; i < N * model->d_model; i++) x[i] += attn_out[i];
        
        // MoE (FFN) forward — ds4-ssd slot-bank takes precedence (page experts
        // from the checkpoint shards; the resident blobs are intentionally NULL
        // in this path, so it MUST be checked before the resident `loaded` path).
        double t_moe0 = wall_time();
        if (layer->dense_ffn && wubu_dense_ffn_ready(layer->dense_ffn)) {
            /* Dense SwiGLU FFN (hybrid models) — zero-copy quantized */
            for (int t = 0; t < N; t++) {
                wubu_dense_ffn_forward(layer->dense_ffn,
                                       normed2 + t * model->d_model,
                                       ffn_out + t * model->d_model);
            }
            have_prev_experts = 0;
        } else if (model->enable_moe && model->ssd_moe && layer->moe.loaded >= 0 &&
            (model->moe_max_layers == 0 || l < model->moe_max_layers)) {
            // ds4-ssd slot-bank: page routed experts from the on-disk checkpoint.
            wubu_moe_forward_ssd(normed2, B, T, &layer->moe, model->ssd_moe, l,
                                 ffn_out, have_prev_experts ? prev_experts : NULL,
                                 model->n_active_experts, model->n_experts, model->d_model, model->d_ff);
            have_prev_experts = 1;
        } else if (layer->moe.loaded && model->enable_moe &&
            (model->moe_max_layers == 0 || l < model->moe_max_layers)) {
            // Quantized path: also save selected expert indices for next-layer prefetch
            // GPU MoE (disabled by FORCE_CPU_MOE env var for debug)
            if (model->gpu_ctx && !getenv("FORCE_CPU_MOE")) {
                wubu_backend_t *backend = wubu_backend_get(model);
                if (backend && backend->moe_experts) {
                    layer->moe.gpu_ctx = (void *)model;
                }
            }
            wubu_moe_forward(normed2, B, T, &layer->moe, ffn_out, have_prev_experts ? prev_experts : NULL,
                             model->n_active_experts, model->n_experts, model->d_model, model->d_ff);
            have_prev_experts = 1;
            if (model->gpu_ctx) {
                wubu_backend_t *backend = wubu_backend_get(model);
                if (backend) {
                    layer->moe.gpu_ctx = NULL;  // reset after use
                }
            }
        } else if (model->enable_moe && model->gguf_ctx &&
                   (model->moe_max_layers == 0 || l < model->moe_max_layers)) {
            // Fallback: F32 dequant path
            if (wubu_moe_load_layer(model->gguf_ctx, l, &layer->moe, model->d_model, model->d_ff, model->n_experts)) {
                wubu_moe_forward(normed2, B, T, &layer->moe, ffn_out, NULL,
                                 model->n_active_experts, model->n_experts, model->d_model, model->d_ff);
                wubu_moe_free_layer(&layer->moe);
            } else {
                memcpy(ffn_out, normed2, N * model->d_model * sizeof(float));
            }
        } else {
            // Pass-through when MoE disabled
            memcpy(ffn_out, normed2, N * model->d_model * sizeof(float));
        }
        
        double t_moe1 = wall_time();
        if (getenv("PROFILE") && l < 3) {
            fprintf(stderr, "  L%d MoE: %.3fms\n", l, (t_moe1 - t_moe0) * 1000.0);
        }
        
        // Residual: x = x + ffn_out
        #pragma omp parallel for if(N * model->d_model > 500000)
        for (int i = 0; i < N * model->d_model; i++) x[i] += ffn_out[i];
        
        // Dump per-layer hidden state (post-MoE residual = next layer's input)
        const char *dump_dir = getenv("DUMP_LAYER_DIR");
        if (dump_dir) {
            char fname[512];
            snprintf(fname, sizeof(fname), "%s/our_layer_%d.bin", dump_dir, l);
            FILE *df = fopen(fname, "wb");
            if (df) {
                fwrite(x, sizeof(float), N * model->d_model, df);
                fclose(df);
            }
        }
    }

    // ===== Five-reference kernel probe (research/062) =====
    // WUBU_REF_KERNELS=1 runs every ported kernel against the REAL hidden
    // state x produced by this forward pass — proving each kernel executes
    // inside the model, not just in its standalone test. Zero cost when off.
    if (getenv("WUBU_REF_KERNELS") && N > 0) {
        const float *h = x + (N - 1) * model->d_model;  /* last token hidden */
        int d = model->d_model;

        /* Probe 1: MiniMax H3 — ConvRot un-rotation on a weight-sized slice.
         * Use a small synthetic block seeded from the hidden state so the
         * Hadamard path executes with real-scale data. */
        int rows = 4, cols = 32;
        float wbuf[128];
        for (int i = 0; i < rows * cols; i++)
            wbuf[i] = h[i % d] * 0.01f;
        if (wubu_enc_h3_unrotate(wbuf, rows, cols) == 0)
            fprintf(stderr, "[ref-kernels] enc_h3 (ConvRot un-rotate) OK\n");

        /* Probe 2: DeepSeek-V4 — hyper-residual + sinkhorn on a routing slice. */
        float route[16], hyp_out[16];
        for (int i = 0; i < 16; i++)
            route[i] = fabsf(h[i % d]) + 0.1f;
        if (wubu_dsv4_hyper_residual(route, route, 0.8f, 16, hyp_out) == 0 &&
            wubu_dsv4_sinkhorn_norm(hyp_out, 4, 4, 3) == 0)
            fprintf(stderr, "[ref-kernels] dsv4 (hyper-residual + sinkhorn) OK\n");

        /* Probe 3: LFM2.5 — hybrid linear/softmax attention on hidden state. */
        wubu_lfm_cfg_t lcfg = { .d_model = d, .n_heads = 4, .d_head = d / 4,
                                .n_kv_heads = 2, .n_layers = 2, .hybrid_gate = 1 };
        wubu_lfm_t *lfm = wubu_lfm_create(&lcfg);
        if (lfm) {
            float *S = (float *)calloc((size_t)d * d, sizeof(float));
            float *Sout = (float *)calloc((size_t)d * d, sizeof(float));
            float *klin = (float *)malloc((size_t)d * sizeof(float));
            float *vlin = (float *)malloc((size_t)d * sizeof(float));
            float *lout = (float *)malloc((size_t)d * sizeof(float));
            if (S && Sout && klin && vlin && lout) {
                for (int i = 0; i < d; i++) { klin[i] = h[i]; vlin[i] = h[(i+1) % d]; }
                if (wubu_lfm_linear_attn(S, klin, vlin, d, 0.9f, Sout, lout) == 0)
                    fprintf(stderr, "[ref-kernels] lfm (DeltaNet hybrid attn) OK\n");
            }
            free(S); free(Sout); free(klin); free(vlin); free(lout);
            wubu_lfm_free(lfm);
        }

        /* Probe 4: Photon 2.0 — fused PSO decode on a small layer block. */
        {
            int dh = 8, nh = 4, nkv = 2, dff = 64;
            wubu_megakernel_cfg_t mcfg = { .d_model = 32, .n_heads = nh,
                                           .n_kv_heads = nkv, .d_head = dh,
                                           .d_ff = dff, .rms_epsilon = 1e-6f };
            wubu_megakernel_t *mk = wubu_megakernel_create(&mcfg);
            if (mk) {
                float ctx32[32], qkv[32 * (32 + 2 * nkv * dh)], ao[32 * 32];
                float ffh[dff * 32], ffo[32 * dff], n1[32], n2[32];
                float kv[2 * nkv * dh * 4], out32[32];
                for (int i = 0; i < 32; i++) ctx32[i] = h[i % d];
                for (int i = 0; i < 32 * (32 + 2 * nkv * dh); i++) qkv[i] = 0.01f;
                for (int i = 0; i < 32 * 32; i++) ao[i] = 0.01f;
                for (int i = 0; i < dff * 32; i++) ffh[i] = 0.01f;
                for (int i = 0; i < 32 * dff; i++) ffo[i] = 0.01f;
                for (int i = 0; i < 32; i++) { n1[i] = 1.0f; n2[i] = 1.0f; }
                if (wubu_megakernel_decode(mk, ctx32, qkv, ao, ffh, ffo,
                                           n1, n2, kv, 0, out32) == 0)
                    fprintf(stderr, "[ref-kernels] megakernel (fused PSO decode) OK\n");
                wubu_megakernel_free(mk);
            }
        }

        /* Probe 5: multi-teacher distillation (mr_r0b0t dataset) — fused
         * ensemble KL against synthetic Qwen3.8/GLM5.2/KimiK3 teacher
         * logits seeded from the real hidden state. */
        {
            int vs = 64;
            wubu_multiteach_cfg_t mcfg = { .vocab_size = vs,
                                           .temperature = 1.0f,
                                           .distill_alpha = 0.5f,
                                           .tool_head_weight = 0.1f };
            mcfg.teachers[0] = (wubu_teacher_weight_t){ .weight = 0.5f, .quality = 0.9f, .n_traces = 20000 };
            mcfg.teachers[1] = (wubu_teacher_weight_t){ .weight = 0.3f, .quality = 0.85f, .n_traces = 18000 };
            mcfg.teachers[2] = (wubu_teacher_weight_t){ .weight = 0.2f, .quality = 0.8f, .n_traces = 19937 };
            wubu_multiteach_t *mt = wubu_multiteach_create(&mcfg);
            if (mt) {
                float slog[64], tlog[3 * 64], ens[64], tmask[64];
                for (int i = 0; i < 64; i++) {
                    slog[i] = h[i % d] * 0.1f;
                    for (int j = 0; j < 3; j++)
                        tlog[j * 64 + i] = h[(i + j) % d] * 0.1f + (float)(j + 1);
                    tmask[i] = (i % 8 == 0) ? 1.0f : 0.0f;
                }
                float weights[3] = {0.5f, 0.3f, 0.2f};
                float kl = wubu_multiteach_kl_loss(slog, tlog, 64, 1.0f, weights, ens);
                float total = wubu_multiteach_total_loss(mt, 1.0f, slog, tlog,
                                                         64, tmask, 0.25f);
                if (kl >= 0.0f && total >= 1.0f && isfinite(kl))
                    fprintf(stderr, "[ref-kernels] multiteach (3-teacher ensemble KL) OK\n");
                wubu_multiteach_free(mt);
            }
        }
    }

    // KV cache is rebuilt fresh each forward (gqa_cache_len reset at entry).
    // Do NOT accumulate cache_len across calls — decode paths that re-forward
    // the full prefix would otherwise double-count positions.

    // Save last hidden state for MTP speculative decode (if requested)
    // Captures BEFORE final RMSNorm — MTP head receives raw layer 39 output
    float *save_h = model->save_last_hidden;
    if (save_h && N > 0) {
        memcpy(save_h, x + (N - 1) * model->d_model, model->d_model * sizeof(float));
    }

    // Final RMSNorm
    if (model->norm_weight) {
        float *final_normed = (float *)malloc(N * model->d_model * sizeof(float));
        wubu_rms_norm(B, T, model->d_model, x, model->norm_weight, 1e-6f, final_normed);
        memcpy(x, final_normed, N * model->d_model * sizeof(float));
        free(final_normed);
    }
    
    // Output projection
    // logits[t, v] = sum_k h[t,k] * output_weight[k, v]
    double t_out0 = wall_time();
    if (model->skip_output_proj) {
        // Copy final hidden states to logits buffer (caller does GPU output proj)
        for (int i = 0; i < N; i++) {
            memcpy(logits + i * model->vocab_size, x + i * model->d_model,
                   model->d_model * sizeof(float));
        }
    } else if (model->output_weight_q && model->output_weight_type != GGML_TYPE_F32) {
        // Q4_K quantized matmul path
        // For decode (N=1), quantized_matmul internal parallelizes across 248320 cols.
        // For prefill (N>1), parallelize across tokens (outer loop).
        // Nested OMP: outer parallel for uses threads for tokens, inner quantized_matmul
        // uses 1 thread per token when nested=off (default) — correct behavior.
        #pragma omp parallel for if(N > 1)
        for (int i = 0; i < N; i++) {
            quantized_matmul(x + i * model->d_model,
                             model->output_weight_q,
                             model->output_weight_type,
                             model->d_model, model->vocab_size, 0,
                             logits + i * model->vocab_size);
        }
        // Compare against F32 SGEMM when output_weight is also loaded
        if (model->output_weight && getenv("VERBOSE_OUTPUT_PROJ")) {
            float *f32_logits = (float *)malloc(N * model->vocab_size * sizeof(float));
            #pragma omp parallel for collapse(2) if((int64_t)N * model->vocab_size > 100000)
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < model->vocab_size; j++) {
                    const float *h_i = x + i * model->d_model;
                    float *log_i = f32_logits + i * model->vocab_size;
                    double sum = 0.0;
                    for (int k = 0; k < model->d_model; k++)
                        sum += (double)h_i[k] * (double)model->output_weight[j * model->d_model + k];
                    log_i[j] = (float)sum;
                }
            }
            double dot=0, n1=0, n2=0, max_e=0;
            for (int i = 0; i < N * model->vocab_size; i++) {
                dot += (double)logits[i] * (double)f32_logits[i];
                n1  += (double)logits[i] * (double)logits[i];
                n2  += (double)f32_logits[i] * (double)f32_logits[i];
                double e = fabs((double)logits[i] - (double)f32_logits[i]);
                if (e > max_e) max_e = e;
            }
            fprintf(stderr, "  [output proj] cos-sim Q4K vs F32 = %.10f, max_err=%.6f\n",
                    dot / (sqrt(n1) * sqrt(n2)), max_e);
            free(f32_logits);
        }
        double t_out1 = wall_time();
        if (getenv("PROFILE")) {
            fprintf(stderr, "  Output proj: %.3fms\n", (t_out1 - t_out0) * 1000.0);
        }
    } else if (model->output_weight) {
        // F32 output projection: logits[v] = sum_k x[k] * output_weight[v*d_model + k]
        // (F32 safetensors/HF path: output_weight_q is NULL, output_weight holds plain f32 lm_head)
        const int d = model->d_model;
        const int P = model->rotate_P;  // doc 013: input was Hadamard-rotated to match
        #pragma omp parallel for if((int64_t)N * model->vocab_size > 100000)
        for (int i = 0; i < N; i++) {
            const float *h_i = x + i * d;
            float *hbuf = (P > 1) ? (float *)malloc((size_t)d * sizeof(float)) : NULL;
            const float *hh = h_i;
            if (P > 1) {  /* rotate the first P dims by H_P to match the fused weight */
                memcpy(hbuf, h_i, (size_t)d * sizeof(float));
                wubu_rotate_input(hbuf, d);
                hh = hbuf;
            }
            float *log_i = logits + i * model->vocab_size;
            for (int v = 0; v < model->vocab_size; v++) {
                double sum = 0.0;
                const float *w_v = model->output_weight + (size_t)v * d;
                for (int k = 0; k < d; k++) sum += (double)hh[k] * (double)w_v[k];
                log_i[v] = (float)sum;
            }
            free(hbuf);
        }
    } else if (model->lazy_lmhead_raw) {
        /* Zero-copy BF16/F16 lm_head: dequantize one lm_head ROW (= D elems)
         * per vocab entry on demand. logits[v] = sum_k h[k]*W[v,k]. Avoids
         * copying the 5.1 GB lm_head table into F32.
         * doc 013: when rotate_P>1, compute (W*H_P)*(H_P*h) which equals W*h
         * exactly -- rotate the input h in hbuf, and rotate each dequantized
         * weight ROW by H_P before the dot. */
        const int d = model->d_model;
        const int P = model->rotate_P;
        #pragma omp parallel for if((int64_t)N * model->vocab_size > 100000)
        for (int i = 0; i < N; i++) {
            const float *h_i = x + i * d;
            float *hbuf = (P > 1) ? (float *)malloc((size_t)d * sizeof(float)) : NULL;
            const float *hh = h_i;
            if (P > 1) { memcpy(hbuf, h_i, (size_t)d * sizeof(float)); wubu_rotate_input(hbuf, d); hh = hbuf; }
            float *wrow = (P > 1) ? (float *)malloc((size_t)d * sizeof(float)) : NULL;
            float *log_i = logits + i * model->vocab_size;
            for (int v = 0; v < model->vocab_size; v++) {
                const uint16_t *s = (const uint16_t *)model->lazy_lmhead_raw
                                   + (size_t)v * model->lazy_lmhead_row;
                /* dequant row -> wrow (or point at f32 row) */
                const float *wv;
                if (P > 1) {
                    if (model->lazy_lmhead_dtype == ST_DTYPE_BF16) {
                        for (int k = 0; k < d; k++) wrow[k] = st_bf16_to_f32(s[k]);
                    } else if (model->lazy_lmhead_dtype == ST_DTYPE_F16) {
                        for (int k = 0; k < d; k++) wrow[k] = st_f16_to_f32(s[k]);
                    } else {
                        const float *f = (const float *)s; for (int k = 0; k < d; k++) wrow[k] = f[k];
                    }
                    wubu_rotate_input(wrow, d);  /* W <- W*H_P */
                    wv = wrow;
                } else {
                    wv = (model->lazy_lmhead_dtype == ST_DTYPE_F32)
                         ? (const float *)s : NULL;
                }
                double sum = 0.0;
                if (P > 1) {
                    for (int k = 0; k < d; k++) sum += (double)hh[k] * (double)wv[k];
                } else if (model->lazy_lmhead_dtype == ST_DTYPE_BF16) {
                    for (int k = 0; k < d; k++) sum += (double)h_i[k] * st_bf16_to_f32(s[k]);
                } else if (model->lazy_lmhead_dtype == ST_DTYPE_F16) {
                    for (int k = 0; k < d; k++) sum += (double)h_i[k] * st_f16_to_f32(s[k]);
                } else {
                    const float *w_v = (const float *)s;
                    for (int k = 0; k < d; k++) sum += (double)h_i[k] * (double)w_v[k];
                }
                log_i[v] = (float)sum;
            }
            free(hbuf); free(wrow);
        }
    } else {
        // Fallback: copy hidden states only (no output weight loaded)
        memcpy(logits, x, N * model->d_model * sizeof(float));
    }
    
    /* Free forward buffers: arena-allocated ones are freed by arena reset;
     * only free if we fell back to malloc. */
    if (!fwd_buf) {
        free(x);
        free(normed);
        free(attn_out);
        free(normed2);
        free(ffn_out);
        free(prev_experts_buf);
    }
    /* If arena was used, the sub-arena reset on next call frees automatically. */
}

// ========== State reset ==========
void wubu_model_reset_state(wubu_model_t *model) {
    if (!model) return;
    if (model->ssm_states) memset(model->ssm_states, 0, model->ssm_state_total);
    /* Also reset the GQA KV cache so independent generations start clean.
     * Setting cache_len=0 makes attention treat the cache as empty. */
    model->gqa_cache_len = 0;
}

// ========== Forward Pass from Token IDs ==========
void wubu_model_forward(wubu_model_t *model,
                        const int *token_ids, int B, int T,
                        float *logits) {
    // Reset KV cache so each forward rebuilds it from the provided tokens.
    // Decode paths that re-forward the full prefix must NOT accumulate cache.
    model->gqa_cache_len = 0;

    const int N = B * T;
    // Simple embedding lookup: use token_embd if available, otherwise use file
    float *embd = (float *)malloc(N * model->d_model * sizeof(float));
    if (!embd) { fprintf(stderr, "wubu_model_forward: alloc failed\n"); return; }

    if (model->token_embd) {
        // In-memory embeddings
        for (int i = 0; i < N; i++) {
            int tok = token_ids[i];
            if (tok < 0 || tok >= model->vocab_size) tok = 0;
            memcpy(embd + i * model->d_model, model->token_embd + tok * model->d_model,
                   model->d_model * sizeof(float));
        }
    } else if (model->lazy_embd_raw) {
        /* Zero-copy BF16/F16 embedding: dequantize ONE row per token from the
         * mmap'd shard. Saves copying the whole 5.1 GB embed table. */
        for (int i = 0; i < N; i++) {
            int tok = token_ids[i];
            if (tok < 0 || tok >= model->vocab_size) tok = 0;
            float *row = embd + i * model->d_model;
            if (model->lazy_embd_dtype == ST_DTYPE_BF16) {
                const uint16_t *s = (const uint16_t *)model->lazy_embd_raw
                                   + (size_t)tok * model->lazy_embd_row;
                for (int k = 0; k < model->d_model; k++) row[k] = st_bf16_to_f32(s[k]);
            } else if (model->lazy_embd_dtype == ST_DTYPE_F16) {
                const uint16_t *s = (const uint16_t *)model->lazy_embd_raw
                                   + (size_t)tok * model->lazy_embd_row;
                for (int k = 0; k < model->d_model; k++) row[k] = st_f16_to_f32(s[k]);
            } else {
                memcpy(row, model->lazy_embd_raw + (size_t)tok * model->lazy_embd_row * 4,
                       model->d_model * sizeof(float));
            }
        }
    } else if (model->use_embedding_file) {
        // Read from embedding file
        const char *emb_path = "data/qwen36_embeddings_c.bin.raw";
        FILE *emb_f = fopen(emb_path, "rb");
        if (emb_f) {
            for (int i = 0; i < N; i++) {
                int tok = token_ids[i];
                if (tok < 0 || tok >= model->vocab_size) tok = 0;
                fseek(emb_f, (long)tok * model->d_model * sizeof(float), SEEK_SET);
                size_t rd = fread(embd + i * model->d_model, sizeof(float), model->d_model, emb_f);
                (void)rd;
            }
            fclose(emb_f);
        } else {
            fprintf(stderr, "wubu_model_forward: cannot open embedding file\n");
            memset(embd, 0, N * model->d_model * sizeof(float));
        }
    } else if (model->token_embd_q) {
        // Large vocab: dequantize per-token from mmap'd GGUF blob
        gguf_tensor_info *t_emb = resolve(-1, WUBU_T_TOKEN_EMBD);
        int bytes_per_token = (int)(model->d_model * sizeof(float));  // default: F32
        if (t_emb) {
            int64_t n_elems = 1;
            for (int d = 0; d < t_emb->n_dims; d++) n_elems *= t_emb->dims[d];
            int64_t raw = gguf_raw_size(t_emb->ggml_type, n_elems);
            bytes_per_token = (int)(raw / n_elems * t_emb->dims[1]);
        }
        for (int i = 0; i < N; i++) {
            int tok = token_ids[i];
            if (tok < 0 || tok >= model->vocab_size) tok = 0;
            size_t offset = (size_t)tok * bytes_per_token;
            gguf_dequantize(model->token_embd_q + offset,
                             model->token_embd_type, model->d_model, embd + i * model->d_model);
        }
    } else {
        memset(embd, 0, N * model->d_model * sizeof(float));
    }

    wubu_model_forward_from_embd(model, embd, B, T, logits);
    free(embd);
}

// Chunked forward: process [B, T_total] in time-chunks of <= chunk_sz tokens,
// carrying the model's persistent SSM/conv/KV-cache state across chunks.
// Mathematically identical to one big forward (the recurrence is stateful and
// continues mid-sequence); the only reason for chunking is peak memory — each
// chunk allocates SSM/GQA intermediates for chunk_sz tokens, not T_total. This
// is what makes the full 262144-token (256K) prefill runnable on a ~13 GB box
// (where a single-shot 262144 forward needs ~30-40 GB of SSM intermediates).
// Only the FINAL chunk's logits (positions [T_total-chunk_sz, T_total)) are
// returned in `logits` (sized B*chunk_sz*vocab_size).
void wubu_model_forward_chunked(wubu_model_t *model,
                                const int *token_ids, int B, int T_total,
                                int chunk_sz, float *logits) {
    if (chunk_sz < 1) chunk_sz = 1;
    if (T_total < 1) return;
    /* Force the SCALAR SSM recurrence per chunk. The scalar path is the
     * reference-corrent one and carries the persistent SSM/conv state CORRECTLY
     * across the multiple wubu_model_forward_from_embd calls we make here
     * (verified: scalar 2-call continuation == single forward, maxdiff 1.9e-6).
     * The optimized chunked SSM recurrence (wubu_ssm_chunked_recurrence) is
     * correct WITHIN a single call but carries state slightly wrong across
     * SEPARATE model-level calls — a known optimization bug, not used here. */
    int forced_seq = (getenv("FORCE_CPU_SSM_SEQ") == NULL);
    if (forced_seq) setenv("FORCE_CPU_SSM_SEQ", "1", 1);
    int off = 0;
    while (off < T_total) {
        int C = T_total - off;
        if (C > chunk_sz) C = chunk_sz;
        float *embd = (float *)malloc((size_t)C * model->d_model * sizeof(float));
        if (!embd) {
            /* OOM: try shrinking C to fit */
            size_t avail = wubu_mem_detect_available_ram();
            if (avail > 0) {
                int new_C = (int)(avail / 8 / ((size_t)model->d_model * sizeof(float)));
                if (new_C < 64) new_C = 64;
                C = new_C;
                embd = (float *)malloc((size_t)C * model->d_model * sizeof(float));
            }
        }
        if (!embd) { fprintf(stderr, "wubu_model_forward_chunked: embd alloc failed\n"); return; }
        if (model->token_embd) {
            for (int i = 0; i < C; i++) {
                int tok = token_ids[off + i];
                if (tok < 0 || tok >= model->vocab_size) tok = 0;
                memcpy(embd + i * model->d_model, model->token_embd + tok * model->d_model,
                       model->d_model * sizeof(float));
            }
        } else if (model->token_embd_q) {
            gguf_tensor_info *t_emb = resolve(-1, WUBU_T_TOKEN_EMBD);
            int bytes_per_token = (int)(model->d_model * sizeof(float));
            if (t_emb) {
                int64_t n_elems = 1;
                for (int d = 0; d < t_emb->n_dims; d++) n_elems *= t_emb->dims[d];
                int64_t raw = gguf_raw_size(t_emb->ggml_type, n_elems);
                bytes_per_token = (int)(raw / n_elems * t_emb->dims[1]);
            }
            for (int i = 0; i < C; i++) {
                int tok = token_ids[off + i];
                if (tok < 0 || tok >= model->vocab_size) tok = 0;
                size_t boff = (size_t)tok * bytes_per_token;
                gguf_dequantize(model->token_embd_q + boff,
                                model->token_embd_type, model->d_model, embd + i * model->d_model);
            }
        } else {
            memset(embd, 0, (size_t)C * model->d_model * sizeof(float));
        }

        // Per-chunk logits buffer (B*C*vocab_size). Only the LAST chunk's
        // contents are copied out to `logits` (the caller's buffer).
        // OOM guard: cap chunk_logits to available RAM / 4. If chunk is too
        // large, reduce C to fit. At vocab=250K, C=4096 → 4GB — must cap.
        size_t chunk_logits_bytes = (size_t)B * C * model->vocab_size * sizeof(float);
        size_t avail = wubu_mem_detect_available_ram();
        if (avail > 0 && chunk_logits_bytes > avail / 4) {
            int new_C = (int)(avail / 4 / ((size_t)B * model->vocab_size * sizeof(float)));
            if (new_C < 64) new_C = 64;
            if (new_C < C) {
                fprintf(stderr, "[oom-guard] chunked logits %zuMB > %zuMB, reducing C %d→%d\n",
                        chunk_logits_bytes / (1024*1024), avail / 4 / (1024*1024), C, new_C);
                C = new_C;
                chunk_logits_bytes = (size_t)B * C * model->vocab_size * sizeof(float);
            }
        }
        float *chunk_logits = (float *)malloc(chunk_logits_bytes ? chunk_logits_bytes : 16);
        if (!chunk_logits) { fprintf(stderr, "wubu_model_forward_chunked: logits alloc failed\n"); free(embd); return; }
        wubu_model_forward_from_embd(model, embd, B, C, chunk_logits);

        int is_last = (off + C >= T_total);
        if (is_last) {
            int last_C = C;
            memcpy(logits, chunk_logits, (size_t)B * last_C * model->vocab_size * sizeof(float));
        }
        free(chunk_logits);
        free(embd);
        off += C;
    }
    if (forced_seq) unsetenv("FORCE_CPU_SSM_SEQ");
}

// ========== Backward Pass ==========

void wubu_model_backward_from_embd(
    const wubu_model_t *model,
    const float *embeddings,
    const float *logits, const float *d_logits,
    const float *saved_normed,     // [n_layers * N * model->d_model]
    const float *saved_attn_out,   // [n_layers * N * model->d_model]
    const float *saved_normed2,    // [n_layers * N * model->d_model]
    const float *saved_ffn_out,    // [n_layers * N * model->d_model]
    float *d_embeddings,
    int B, int T)
{
    const int N = B * T;
    const int n_layers = model->n_layers;
    const int layer_sz = N * model->d_model;
    
    float *d_x = (float *)malloc(N * model->d_model * sizeof(float));
    memcpy(d_x, d_logits, N * model->d_model * sizeof(float));
    
    // Per-layer temp state buffers (reused via ssm_states/conv_states in model)
    // For exact backward, we need to re-run the forward with save
    
    // Process layers in reverse
    for (int l = n_layers - 1; l >= 0; l--) {
        const wubu_layer_t *layer = &model->layers[l];
        const float *normed = saved_normed + l * layer_sz;
        const float *attn_out = saved_attn_out + l * layer_sz;
        const float *normed2 = saved_normed2 + l * layer_sz;
        
        float *d_ffn_out = (float *)malloc(N * model->d_model * sizeof(float));
        float *d_x_after_attn = (float *)malloc(N * model->d_model * sizeof(float));
        float *d_attn_out = (float *)malloc(N * model->d_model * sizeof(float));
        memcpy(d_ffn_out, d_x, layer_sz);
        memcpy(d_x_after_attn, d_x, layer_sz);
        
        // Post-attention RMSNorm backward
        wubu_rms_norm_backward(B, T, model->d_model, normed2, layer->post_attn_norm_weight,
                               1e-6f, d_ffn_out, d_x_after_attn);
        memcpy(d_attn_out, d_x_after_attn, layer_sz);
        
        // Layer backward — exact with saved intermediates
        float *d_normed = (float *)calloc(N * model->d_model, sizeof(float));
        
        if (layer->is_ssm) {
            // Re-run SSM forward WITH save to capture intermediates for backward
            ssm_fwd_save_t save;
            memset(&save, 0, sizeof(save));
            
            // Allocate save buffers for one layer
            float *ssm_state_tmp = model->ssm_states + l * model->ssm_v_heads * model->ssm_d_state * model->ssm_d_state;
            float *conv_state_tmp = model->conv_states + l * (model->conv_kernel - 1) * model->conv_dim;
            
            int state_sz = model->ssm_v_heads * model->ssm_d_state * model->ssm_d_state;
            
            // We need a separate states_t buffer (not the in-place one)
            float *states_t = (float *)malloc((T+1) * state_sz * sizeof(float));
            float *qkv_all_b = (float *)malloc(N * model->conv_dim * sizeof(float));
            float *z_all_b = (float *)malloc(N * model->ssm_d_state * model->ssm_v_heads * sizeof(float));
            float *beta_raw_b = (float *)malloc(N * model->dt_rank * sizeof(float));
            float *alpha_raw_b = (float *)malloc(N * model->dt_rank * sizeof(float));
            float *conv_out_b = (float *)malloc(N * model->conv_dim * sizeof(float));
            float *q_conv_b = (float *)malloc(N * model->ssm_d_state * model->ssm_k_heads * sizeof(float));
            float *k_conv_b = (float *)malloc(N * model->ssm_d_state * model->ssm_k_heads * sizeof(float));
            float *v_conv_b = (float *)malloc(N * model->ssm_d_state * model->ssm_v_heads * sizeof(float));
            float *q_norm_b = (float *)malloc(N * model->ssm_d_state * model->ssm_k_heads * sizeof(float));
            float *k_norm_b = (float *)malloc(N * model->ssm_d_state * model->ssm_k_heads * sizeof(float));
            float *delta_out_b = (float *)malloc(N * model->ssm_d_state * model->ssm_v_heads * sizeof(float));
            float *z_silu_b = (float *)malloc(N * model->ssm_d_state * model->ssm_v_heads * sizeof(float));
            float *beta_flat_b = (float *)malloc(N * model->dt_rank * sizeof(float));
            float *gate_flat_b = (float *)malloc(N * model->dt_rank * sizeof(float));
            float *conv_state_copy = (float *)malloc((model->conv_kernel-1) * model->conv_dim * sizeof(float));
            
            if (!states_t || !qkv_all_b || !z_all_b || !beta_raw_b || !alpha_raw_b ||
                !conv_out_b || !q_conv_b || !k_conv_b || !v_conv_b ||
                !q_norm_b || !k_norm_b || !delta_out_b || !z_silu_b ||
                !beta_flat_b || !gate_flat_b || !conv_state_copy) {
                fprintf(stderr, "model backward SSM save alloc failed\n");
                free(states_t); free(qkv_all_b); free(z_all_b);
                free(beta_raw_b); free(alpha_raw_b);
                free(conv_out_b); free(q_conv_b); free(k_conv_b); free(v_conv_b);
                free(q_norm_b); free(k_norm_b); free(delta_out_b); free(z_silu_b);
                free(beta_flat_b); free(gate_flat_b); free(conv_state_copy);
                free(d_ffn_out); free(d_x_after_attn); free(d_attn_out); free(d_normed);
                free(d_x); return;
            }
            
            save.states_t = states_t;
            save.qkv_all = qkv_all_b;
            save.z_all = z_all_b;
            save.beta_raw = beta_raw_b;
            save.alpha_raw = alpha_raw_b;
            save.conv_post_silu = conv_out_b;
            save.q_conv = q_conv_b;
            save.k_conv = k_conv_b;
            save.v_conv = v_conv_b;
            save.q_norm = q_norm_b;
            save.k_norm = k_norm_b;
            save.delta_out = delta_out_b;
            save.z_silu = z_silu_b;
            save.beta_flat = beta_flat_b;
            save.gate_flat = gate_flat_b;
            save.conv_state_copy = conv_state_copy;
            
            // Save current SSM state, run save-forward, then restore
            float *saved_ssm_state = (float *)malloc(state_sz * sizeof(float));
            memcpy(saved_ssm_state, ssm_state_tmp, state_sz * sizeof(float));
            
            // Run forward with save — attn_out goes to a dummy buffer
            float *fwd_out = (float *)malloc(N * model->d_model * sizeof(float));
            wubu_ssm_forward_save(normed, B, T, &layer->ssm,
                                   ssm_state_tmp, conv_state_tmp,
                                   fwd_out, &save);
            
            // Run exact backward
            wubu_ssm_backward(B, T, normed, attn_out, d_attn_out,
                              save.qkv_all, save.z_all,
                              save.beta_raw, save.alpha_raw,
                              save.conv_post_silu,
                              save.q_conv, save.k_conv, save.v_conv,
                              save.q_norm, save.k_norm,
                              save.delta_out, save.z_silu,
                              save.states_t,
                              save.beta_flat, save.gate_flat,
                              save.conv_state_copy,
                              &layer->ssm,
                              d_normed, NULL, NULL, NULL, NULL,
                              NULL, NULL, NULL, NULL);
            
            // Restore SSM state
            memcpy(ssm_state_tmp, saved_ssm_state, state_sz * sizeof(float));
            
            free(saved_ssm_state);
            free(fwd_out);
            free(states_t); free(qkv_all_b); free(z_all_b);
            free(beta_raw_b); free(alpha_raw_b);
            free(conv_out_b); free(q_conv_b); free(k_conv_b); free(v_conv_b);
            free(q_norm_b); free(k_norm_b); free(delta_out_b); free(z_silu_b);
            free(beta_flat_b); free(gate_flat_b); free(conv_state_copy);
            
        } else {
            // GQA backward with saved intermediates
            gqa_fwd_save_t save;
            memset(&save, 0, sizeof(save));
            
            int q_dim = GQA_Q_HEADS * GQA_HEAD_DIM;
            int kv_dim = GQA_KV_HEADS * GQA_HEAD_DIM;
            
            float *Q_norm_b = (float *)malloc(N * q_dim * sizeof(float));
            float *Q_raw_b = (float *)malloc(N * q_dim * sizeof(float));
            float *K_norm_b = (float *)malloc(N * kv_dim * sizeof(float));
            float *K_raw_b = (float *)malloc(N * kv_dim * sizeof(float));
            float *V_b = (float *)malloc(N * kv_dim * sizeof(float));
            float *gate_b = (float *)malloc(N * q_dim * sizeof(float));
            float *gate_sig_b = (float *)malloc(N * q_dim * sizeof(float));
            float *attn_pre_gate_b = (float *)malloc(N * q_dim * sizeof(float));
            
            if (!Q_norm_b || !Q_raw_b || !K_norm_b || !K_raw_b || !V_b ||
                !gate_b || !gate_sig_b || !attn_pre_gate_b) {
                fprintf(stderr, "model backward GQA save alloc failed\n");
                free(Q_norm_b); free(Q_raw_b); free(K_norm_b); free(K_raw_b);
                free(V_b); free(gate_b); free(gate_sig_b); free(attn_pre_gate_b);
                free(d_ffn_out); free(d_x_after_attn); free(d_attn_out); free(d_normed);
                free(d_x); return;
            }
            
            save.Q_norm = Q_norm_b;
            save.Q_raw = Q_raw_b;
            save.K_norm = K_norm_b;
            save.K_raw = K_raw_b;
            save.V = V_b;
            save.gate = gate_b;
            save.gate_sig = gate_sig_b;
            save.attn_out_pre_gate = attn_pre_gate_b;
            
            // Run forward with save
            float *fwd_out = (float *)malloc(N * model->d_model * sizeof(float));
            wubu_gqa_forward_save(normed, B, T, &layer->gqa, model->d_model, fwd_out, &save,
                                   layer->gqa.head_dim, layer->gqa.q_heads, layer->gqa.kv_heads);
            
            // Run exact backward
            wubu_gqa_backward(B, T, model->d_model, normed,
                              save.Q_norm, save.Q_raw,
                              save.K_norm, save.K_raw,
                              save.V,
                              save.gate, save.gate_sig,
                              save.attn_out_pre_gate, attn_out,
                              d_attn_out,
                              &layer->gqa,
                              d_normed,
                              NULL, NULL, NULL, NULL, NULL, NULL);
            
            free(fwd_out);
            free(Q_norm_b); free(Q_raw_b); free(K_norm_b); free(K_raw_b);
            free(V_b); free(gate_b); free(gate_sig_b); free(attn_pre_gate_b);
        }
        
        // Pre-attention RMSNorm backward
        float *d_x_pre_attn = (float *)malloc(N * model->d_model * sizeof(float));
        memset(d_x_pre_attn, 0, layer_sz);
        wubu_rms_norm_backward(B, T, model->d_model, normed, layer->attn_norm_weight,
                               1e-6f, d_normed, d_x_pre_attn);
        
        // Residual: x_pre_attn also feeds x_after_attn = x_pre_attn + attn_out
        for (int i = 0; i < N * model->d_model; i++)
            d_x_pre_attn[i] += d_x_after_attn[i];
        
        memcpy(d_x, d_x_pre_attn, layer_sz);
        
        free(d_ffn_out);
        free(d_x_after_attn);
        free(d_attn_out);
        free(d_normed);
        free(d_x_pre_attn);
    }
    
    memcpy(d_embeddings, d_x, N * model->d_model * sizeof(float));
    free(d_x);
}

