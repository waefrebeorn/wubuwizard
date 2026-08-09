/*
 * wubu_model_safetensors_bridge.c -- load HF safetensors Colonel models
 * into wubuwizard's wubu_model_t and run them through the EXISTING
 * SSM + GQA + MoE forward passes in pure F32.
 *
 * Tensor names are the real published HF names (from each repo's
 * model.safetensors.index.json):  model.language_model.layers.N.linear_attn.*
 * (the Gated DeltaNet SSM) + .self_attn.* (GQA) + .mlp.* (dense or MoE).
 * No third-party deps. C11, self-contained.
 */
#include "wubu_model_safetensors_bridge.h"
#include "wubu_dims.h"
#include "wubu_safetensors_shard.h"
#include "safetensors_reader.h"
#include "wubu_lora.h"
#include "wubu_rotate.h"   // doc 013: wubu_pow2_floor / wubu_rotate_fuse_right
#include "wubu_backend.h"  // wubu_backend_cpu_get() — backend install at bridge create
#include <stdlib.h>
#include "wubu_affinity.h"
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

/* ---- helpers: load an F32 tensor (optionally transposed) from safetensors ---- */
static float *st_load_f32(st_ctx *st, const char *name, int64_t *nelems) {
    const st_tensor_info *t = st_find_tensor(st, name);
    if (!t) return NULL;
    int64_t n = 1;
    for (int d = 0; d < t->n_dims; d++) n *= t->dims[d];
    *nelems = n;
    float *buf = (float *)malloc((size_t)n * sizeof(float));
    if (!buf) return NULL;
    if (st_read_tensor_f32(st, t, buf, n) != n) { free(buf); return NULL; }
    return buf;
}

/* load + TRANSPOSE: HF Linear weight [out,in] -> wubuwizard [in,out] row-major.
 * Delegates to the shard loader's transpose path. */
static float *st_load_f32_t(wubu_shard_ctx_t *sc, const char *name, int rows, int cols) {
    float *dst = (float *)malloc((size_t)rows * cols * sizeof(float));
    if (!dst) return NULL;
    float *got = wubu_shard_load_f32_t(sc, name, rows, cols);
    if (!got) { free(dst); return NULL; }
    memcpy(dst, got, (size_t)rows * cols * sizeof(float));
    free(got);
    return dst;
}

/* printf-style tensor name builder */
static void tn(char *out, size_t cap, const char *fmt, int l) {
    snprintf(out, cap, fmt, l);
}

static void tn2(char *out, size_t cap, const char *fmt, int l, int e) {
    snprintf(out, cap, fmt, l, e);
}

/* Read dim `i` of a tensor by name (or -1 if absent), searching ALL shards. */
static int dimof_sc(wubu_shard_ctx_t *sc, const char *n, int i) {
    return wubu_shard_dimof(sc, n, i);
}

/* Load a named tensor as F32, trying the name as given and (if absent) with a
 * ".weight" suffix appended. HF checkpoints are inconsistent about whether
 * scalar/norm-like SSM params (A_log, dt_bias) carry a ".weight" suffix.
 * Returns a freshly malloc'd F32 buffer or NULL if neither form exists. */
static float *load_f32_try2(wubu_shard_ctx_t *sc, const char *name, int64_t *n_elems) {
    float *p = wubu_shard_load_f32(sc, name, n_elems);
    if (p) return p;
    char alt[512];
    snprintf(alt, sizeof(alt), "%s.weight", name);
    return wubu_shard_load_f32(sc, alt, n_elems);
}

/* Load a named tensor as raw bytes, trying the name as given then with
 * ".weight" appended. NULL if neither exists. Fills *dtype and *n_elems. */
static const uint8_t *raw_try2(wubu_shard_ctx_t *sc, const char *name,
                               int *dtype, int64_t *n_elems)
{
    const uint8_t *raw = wubu_shard_raw(sc, name, dtype, n_elems);
    if (raw) return raw;
    char alt[512];
    snprintf(alt, sizeof(alt), "%s.weight", name);
    return wubu_shard_raw(sc, alt, dtype, n_elems);
}

/* Load a named tensor as F32 and transpose [rows,cols]->[cols,rows], trying
 * the name as given then with ".weight" appended. NULL if neither exists. */
static float *load_f32_try2_t(wubu_shard_ctx_t *sc, const char *name,
                              int rows, int cols) {
    float *p = wubu_shard_load_f32_t(sc, name, rows, cols);
    if (p) return p;
    char alt[512];
    snprintf(alt, sizeof(alt), "%s.weight", name);
    return wubu_shard_load_f32_t(sc, alt, rows, cols);
}

int wubu_model_init_safetensors(wubu_model_t *m, const char *path,
                               const wubu_adapter_t *ad) {
    return wubu_model_init_safetensors_ssd(m, path, ad, NULL);
}

int wubu_model_init_safetensors_ssd(wubu_model_t *m, const char *path,
                                  const wubu_adapter_t *ad,
                                  const char *sidecar_dir) {
    if (!m || !path || !ad) return -1;
    memset(m, 0, sizeof(*m));

    /* Game-console hardware discipline (I05 / NuMA+P-core pinning). Mirrors
     * the same block in wubu_model_init(): pin to P-cores + close/core-
     * bound OpenMP so the GEMV parallel-for keeps each row-chunk on
     * one core's cache. Graceful no-op when not applicable. */
    {
        int pinned[64]; int k = wubu_affinity_pin_pcores(pinned, 64);
        if (k > 0) {
            setenv("OMP_PROC_BIND", "close", 1);
            setenv("OMP_PLACES", "cores", 1);
            setenv("OMP_SCHEDULE", "dynamic,64", 1);
            fprintf(stderr, "[affinity] pinned engine to %d P-cores (core0=%d)\n",
                    k, pinned[0]);
        }
    }

    st_ctx *st = st_open(path);
    if (!st) {
        /* A bare checkpoint DIRECTORY (model-NNN-of-MMM shards) is not itself
         * a safetensors file, so st_open fails; that's expected — we derive
         * dims from the shard set via wubu_shard_open below, which globs the
         * directory. Only treat a single-file open failure as fatal. */
        struct stat _pst;
        int is_dir = (stat(path, &_pst) == 0 && S_ISDIR(_pst.st_mode));
        if (!is_dir) { fprintf(stderr, "bridge: cannot open safetensors %s\n", path); return -1; }
        st = NULL;
    }
    /* Shard ctx handles single-file OR multi-shard (model-NNNNN-of-NNNNN)
     * checkpoints transparently. `st` (shard 0) is used only for shape
     * probing below; all weight loads go through `sc`. */
    wubu_shard_ctx_t *sc = wubu_shard_open(path);
    if (!sc) { fprintf(stderr, "bridge: cannot open shard set %s\n", path); st_close(st); return -1; }

    /* ds4-ssd: route MoE experts through the slot-bank paged DIRECTLY from the
     * source checkpoint shards (no redundant sidecar). Routed experts are
     * paged on demand; the big in-RAM expert blobs are skipped below. */
    wubu_ssd_moe_t *ssd = NULL;
    int ssd_slots = ad->n_experts > 0 ? (getenv("SSD_SLOTS") ? atoi(getenv("SSD_SLOTS")) : 8) : 0;
    if (ad->n_experts > 0) {
        ssd = wubu_ssd_moe_open_from_shards(sc, ssd_slots > 0 ? ssd_slots : 8);
        if (!ssd) { fprintf(stderr, "bridge: cannot open ssd slot-bank over checkpoint shards\n"); st_close(st); wubu_shard_close(sc); return -1; }
        m->ssd_moe = ssd;
        m->enable_moe = true;
    }

    /* ---- Derive REAL model dimensions from actual tensor shapes ----
     * Probe across ALL shards (a single shard only holds a subset of the
     * tensors, so probing one shard gives wrong/missing dims). */
    int D     = dimof_sc(sc, "model.language_model.embed_tokens.weight", 1);
    if (D < 0) D = dimof_sc(sc, "model.language_model.layers.0.linear_attn.in_proj_qkv.weight", 1);
    int CONVD = dimof_sc(sc, "model.language_model.layers.0.linear_attn.in_proj_qkv.weight", 0);
    if (CONVD < 0) CONVD = D + D + D;
    int VD    = dimof_sc(sc, "model.language_model.layers.0.linear_attn.in_proj_z.weight", 0);
    if (VD < 0) VD = D;
    int DT    = dimof_sc(sc, "model.language_model.layers.0.linear_attn.in_proj_a.weight", 0);
    if (DT < 0) DT = 32;
    int SSMDS = dimof_sc(sc, "model.language_model.layers.0.linear_attn.norm.weight", 0);
    if (SSMDS < 0) SSMDS = 128;
    int qdim  = dimof_sc(sc, "model.language_model.layers.0.self_attn.q_proj.weight", 0);
    int kvdim = dimof_sc(sc, "model.language_model.layers.0.self_attn.k_proj.weight", 0);
    int hd    = ad->gqa_head_dim > 0 ? (int)ad->gqa_head_dim : 256;
    int qh = (qdim > 0 && hd > 0) ? qdim / hd : (ad->gqa_q_heads > 0 ? (int)ad->gqa_q_heads : 32);
    int kvh = (kvdim > 0 && hd > 0) ? kvdim / hd : (ad->gqa_kv_heads > 0 ? (int)ad->gqa_kv_heads : 4);
    int dff   = dimof_sc(sc, "model.language_model.layers.0.mlp.experts.0.gate_proj.weight", 0);
    if (dff < 0) dff = dimof_sc(sc, "model.language_model.layers.0.mlp.gate_proj.weight", 0);
    if (dff < 0) dff = (int)ad->d_ff > 0 ? (int)ad->d_ff : (D * 4);

    /* Count real layers by probing across shards (robust to adapter quirks).
     * NOTE: hybrid models have both SSM (linear_attention) and GQA-only
     * (full_attention, no linear_attn) layers, so probing for
     * linear_attn.in_proj_qkv.weight undercounts. Use input_layernorm
     * instead, which every layer has. */
    int nL = 0;
    for (int l = 0; l < 512; l++) {
        char qn[256];
        tn(qn, sizeof(qn), "model.language_model.layers.%d.input_layernorm.weight", l);
        if (!wubu_shard_has(sc, qn)) break;
        nL = l + 1;
    }
    if (nL <= 0) nL = (int)ad->n_layers;
    /* Smoke-test / memory cap: load only the first MAX_LAYERS layers. */
    int maxl = getenv("MAX_LAYERS") ? atoi(getenv("MAX_LAYERS")) : 0;
    if (maxl > 0 && maxl < nL) nL = maxl;
    int nE = (int)ad->n_experts;   /* 0 => dense MLP modelled as 1 expert */

    wubu_dims_t wd; memset(&wd, 0, sizeof(wd));
    wd.d_model = D; wd.conv_dim = CONVD; wd.value_dim = VD; wd.dt_rank = DT;
    wd.ssm_d_state = SSMDS;
    /* Prefer real config dims from the adapter; fall back to shape inference. */
    wd.ssm_k_heads = ad->ssm_k_heads > 0 ? ad->ssm_k_heads : (D > 0 ? (CONVD - VD) / 2 / SSMDS : 16);
    wd.ssm_v_heads = ad->ssm_v_heads > 0 ? ad->ssm_v_heads : VD / SSMDS;
    wd.conv_kernel = ad->ssm_conv_kernel > 0 ? ad->ssm_conv_kernel : 4;
    wd.gqa_q_heads = qh; wd.gqa_kv_heads = kvh; wd.gqa_head_dim = hd;
    wubu_dims_set(&wd);

    m->d_model = D;
    m->n_layers = nL;
    m->vocab_size = ad->vocab_size > 0 ? ad->vocab_size
                      : dimof_sc(sc, "model.language_model.embed_tokens.weight", 0);
    if (m->vocab_size <= 0) m->vocab_size = 248320; /* last-resort fallback */
    m->n_experts = nE;
    m->n_active_experts = (int)ad->n_active_experts;
    m->shared_expert_ff = ad->shared_expert_ff > 0 ? ad->shared_expert_ff : 0;

    m->layers = (wubu_layer_t *)calloc((size_t)nL, sizeof(wubu_layer_t));
    if (!m->layers) { st_close(st); return -1; }

    char nm[256];
    for (int l = 0; l < nL; l++) {
        wubu_layer_t *ly = &m->layers[l];
        /* Hybrid: layer_types[l]==0 -> linear_attention (SSM+GQA),
         *                    ==1 -> full_attention (GQA only, no SSM).
         * If is_hybrid is unset, probe for self_attn.q_proj to detect GQA. */
        bool ssm_layer;
        if (ad->is_hybrid && l < 256)
            ssm_layer = (ad->layer_types[l] == 0);
        else if (!ad->is_hybrid) {
            char qn[256];
            snprintf(qn, sizeof(qn),
                     "model.language_model.layers.%d.self_attn.q_proj.weight", l);
            ssm_layer = !wubu_shard_has(sc, qn);
        } else
            ssm_layer = true;
        ly->is_ssm = ssm_layer ? 1 : 0;

        /* ---- GQA (self_attn.*_proj) ----
         * LAZY BF16: keep raw mmap'd bytes and materialize to F32 per call.
         * This keeps dense GQA layers out of RAM until the layer is active,
         * which is what makes the full 27B forward fit in 13 GB.
         * NOTE: In hybrid models, SSM layers (linear_attention) have NO
         * self_attn tensors — they use linear_attn instead. Only load GQA
         * weights for full_attention (GQA-only) layers. */
        if (!ssm_layer) {
            int g_dtype = 0; int64_t g_row = 0;
            tn(nm, sizeof(nm), "model.language_model.layers.%d.self_attn.q_proj.weight", l);
            const uint8_t *qraw = raw_try2(sc, nm, &g_dtype, &g_row);
            if (qraw && (g_dtype == ST_DTYPE_BF16 || g_dtype == ST_DTYPE_F16)) {
                ly->gqa.attn_q_weight_raw = qraw; ly->gqa.lazy_dtype = g_dtype;
                ly->gqa.attn_q_weight = NULL;
            } else {
                ly->gqa.attn_q_weight = wubu_shard_load_f32(sc, nm, &(int64_t){0});
            }
            tn(nm, sizeof(nm), "model.language_model.layers.%d.self_attn.k_proj.weight", l);
            qraw = raw_try2(sc, nm, &g_dtype, &g_row);
            if (qraw && (g_dtype == ST_DTYPE_BF16 || g_dtype == ST_DTYPE_F16)) {
                ly->gqa.attn_k_weight_raw = qraw; ly->gqa.lazy_dtype = g_dtype;
                ly->gqa.attn_k_weight = NULL;
            } else {
                ly->gqa.attn_k_weight = wubu_shard_load_f32(sc, nm, &(int64_t){0});
            }
            tn(nm, sizeof(nm), "model.language_model.layers.%d.self_attn.v_proj.weight", l);
            qraw = raw_try2(sc, nm, &g_dtype, &g_row);
            if (qraw && (g_dtype == ST_DTYPE_BF16 || g_dtype == ST_DTYPE_F16)) {
                ly->gqa.attn_v_weight_raw = qraw; ly->gqa.lazy_dtype = g_dtype;
                ly->gqa.attn_v_weight = NULL;
            } else {
                ly->gqa.attn_v_weight = wubu_shard_load_f32(sc, nm, &(int64_t){0});
            }
            tn(nm, sizeof(nm), "model.language_model.layers.%d.self_attn.o_proj.weight", l);
            qraw = raw_try2(sc, nm, &g_dtype, &g_row);
            if (qraw && (g_dtype == ST_DTYPE_BF16 || g_dtype == ST_DTYPE_F16)) {
                ly->gqa.attn_output_weight_raw = qraw; ly->gqa.lazy_dtype = g_dtype;
                ly->gqa.attn_output_weight = NULL;
            } else {
                ly->gqa.attn_output_weight = wubu_shard_load_f32(sc, nm, &(int64_t){0});
            }
            ly->gqa.q_heads = qh; ly->gqa.kv_heads = kvh; ly->gqa.head_dim = hd;
            ly->gqa.attn_q_weight_q = NULL; ly->gqa.attn_k_weight_q = NULL;
            ly->gqa.attn_v_weight_q = NULL; ly->gqa.attn_output_weight_q = NULL;
        } else {
            /* SSM layers: GQA fields stay NULL (use SSM for QKV). */
            ly->gqa.q_heads = qh; ly->gqa.kv_heads = kvh;
            ly->gqa.head_dim = hd;
            /* kv_dim may be 0 if geometry is unusual; derive from k_proj shape */
            ly->gqa.kv_dim = (kvdim > 0) ? kvdim : (qh > 0 ? qh * hd : 0);
            ly->gqa.q_dim = (qdim > 0) ? qdim : (qh > 0 ? qh * hd : 0);
            ly->gqa.out_dim = D;
        }

        /* ---- SSM (linear_attn.*) -> F32 path ----
         * Only present on linear_attention (SSM) layers. Skipped on
         * full_attention layers (which have no linear_attn.* tensors). */
        if (ssm_layer) {
        ly->ssm.f32_mode = 1;
        /* Big SSM proj matrices: keep raw BF16 mmap'd bytes, materialize to
         * F32 (transposed) per call. The small scalars (A_log, dt_bias, conv,
         * norm, a/b) stay F32-resident — they're tiny. */
        tn(nm, sizeof(nm), "model.language_model.layers.%d.linear_attn.in_proj_qkv", l);
        int s_dtype = 0; int64_t s_row = 0;
        const uint8_t *raw = raw_try2(sc, nm, &s_dtype, &s_row);
        if (raw && (s_dtype == ST_DTYPE_BF16 || s_dtype == ST_DTYPE_F16)) {
            ly->ssm.attn_qkv_weight_raw = raw; ly->ssm.lazy_dtype = s_dtype;
            ly->ssm.attn_qkv_weight_f32 = NULL;
        } else {
            ly->ssm.attn_qkv_weight_f32 = load_f32_try2_t(sc, nm, D, CONVD); /* [D,CONVD] */
        }
        tn(nm, sizeof(nm), "model.language_model.layers.%d.linear_attn.in_proj_z", l);
        raw = raw_try2(sc, nm, &s_dtype, &s_row);
        if (raw && (s_dtype == ST_DTYPE_BF16 || s_dtype == ST_DTYPE_F16)) {
            ly->ssm.attn_gate_weight_raw = raw; ly->ssm.lazy_dtype = s_dtype;
            ly->ssm.attn_gate_weight_f32 = NULL;
        } else {
            ly->ssm.attn_gate_weight_f32 = load_f32_try2_t(sc, nm, D, VD); /* [D,VD] */
        }
        tn(nm, sizeof(nm), "model.language_model.layers.%d.linear_attn.in_proj_a", l);
        ly->ssm.ssm_alpha_weight = load_f32_try2(sc, nm, &(int64_t){0});
        tn(nm, sizeof(nm), "model.language_model.layers.%d.linear_attn.in_proj_b", l);
        ly->ssm.ssm_beta_weight = load_f32_try2(sc, nm, &(int64_t){0});
        tn(nm, sizeof(nm), "model.language_model.layers.%d.linear_attn.A_log", l);
        ly->ssm.ssm_a = load_f32_try2(sc, nm, &(int64_t){0});
        tn(nm, sizeof(nm), "model.language_model.layers.%d.linear_attn.dt_bias", l);
        ly->ssm.ssm_dt_bias = load_f32_try2(sc, nm, &(int64_t){0});
        tn(nm, sizeof(nm), "model.language_model.layers.%d.linear_attn.conv1d", l);
        ly->ssm.ssm_conv1d_weight = load_f32_try2(sc, nm, &(int64_t){0});
        if (!ly->ssm.ssm_conv1d_weight) {
            /* Fixtures/old checkpoints may use convNd instead of conv1d. */
            tn(nm, sizeof(nm), "model.language_model.layers.%d.linear_attn.convNd", l);
            ly->ssm.ssm_conv1d_weight = load_f32_try2(sc, nm, &(int64_t){0});
        }
        tn(nm, sizeof(nm), "model.language_model.layers.%d.linear_attn.norm", l);
        ly->ssm.ssm_norm_weight = load_f32_try2(sc, nm, &(int64_t){0});
        tn(nm, sizeof(nm), "model.language_model.layers.%d.linear_attn.out_proj", l);
        raw = raw_try2(sc, nm, &s_dtype, &s_row);
        if (raw && (s_dtype == ST_DTYPE_BF16 || s_dtype == ST_DTYPE_F16)) {
            ly->ssm.ssm_out_weight_raw = raw; ly->ssm.lazy_dtype = s_dtype;
            ly->ssm.ssm_out_weight_f32 = NULL;
        } else {
            ly->ssm.ssm_out_weight_f32 = load_f32_try2_t(sc, nm, VD, D); /* file [VD,D] -> [D,VD] for proj_matmul(VALUE_DIM,D_MODEL) */
        }
        if (!ly->ssm.ssm_out_weight_f32 && !ly->ssm.ssm_out_weight_raw) {
            fprintf(stderr, "bridge: out_proj.weight missing for layer %d (VD=%d D=%d)\n", l, VD, D);
            wubu_model_safetensors_free(m);
            return -1;
        }
        ly->ssm.attn_qkv_weight_q = NULL; ly->ssm.attn_gate_weight_q = NULL;
        ly->ssm.ssm_out_weight_q = NULL;
        }

        /* ---- MoE / dense MLP (mlp.*) ---- */
        int nExp = nE > 0 ? nE : 1; /* dense => single-expert MoE */
        if (ssd) {
            /* ds4-ssd: routed experts live in the sidecar; do NOT allocate or
             * load the 3.2 GB/layer resident blobs. Forward pages them. */
            ly->moe.ffn_gate_exps = NULL;
            ly->moe.ffn_up_exps   = NULL;
            ly->moe.ffn_down_exps = NULL;
            ly->moe.load_from_blob = false;
            /* Router (still resident) + shared expert below. */
            if (nE > 0) {
                tn(nm, sizeof(nm), "model.language_model.layers.%d.mlp.gate.weight", l);
                ly->moe.ffn_gate_inp = wubu_shard_load_f32(sc, nm, &(int64_t){0}); /* [D, nE] */
            }
        } else {
        ly->moe.ffn_gate_exps = (float *)calloc((size_t)D * dff * nExp, sizeof(float));
        ly->moe.ffn_up_exps   = (float *)calloc((size_t)D * dff * nExp, sizeof(float));
        ly->moe.ffn_down_exps = (float *)calloc((size_t)dff * D * nExp, sizeof(float));
        ly->moe.load_from_blob = false;
        if (nE > 0) {
            /* MoE: mlp.experts.N.{gate,up,down}_proj [dff, D] -> transpose */
            tn(nm, sizeof(nm), "model.language_model.layers.%d.mlp.experts.0.gate_proj.weight", l);
            int64_t cnt = 0; float *g0 = wubu_shard_load_f32(sc, nm, &cnt);
            if (g0) {
                for (int e = 0; e < nE; e++) {
                    tn2(nm, sizeof(nm), "model.language_model.layers.%d.mlp.experts.%d.gate_proj.weight", l, e);
                    float *g = wubu_shard_load_f32_t(sc, nm, D, dff); /* [D,dff] */
                    tn2(nm, sizeof(nm), "model.language_model.layers.%d.mlp.experts.%d.up_proj.weight", l, e);
                    float *u = wubu_shard_load_f32_t(sc, nm, D, dff);
                    tn2(nm, sizeof(nm), "model.language_model.layers.%d.mlp.experts.%d.down_proj.weight", l, e);
                    float *d = wubu_shard_load_f32_t(sc, nm, dff, D); /* [dff,D] */
                    if (g) memcpy(ly->moe.ffn_gate_exps + (size_t)e * D * dff, g, (size_t)D * dff * sizeof(float));
                    if (u) memcpy(ly->moe.ffn_up_exps   + (size_t)e * D * dff, u, (size_t)D * dff * sizeof(float));
                    if (d) memcpy(ly->moe.ffn_down_exps + (size_t)e * dff * D, d, (size_t)dff * D * sizeof(float));
                    free(g); free(u); free(d);
                }
                free(g0);
                tn(nm, sizeof(nm), "model.language_model.layers.%d.mlp.gate.weight", l);
                ly->moe.ffn_gate_inp = wubu_shard_load_f32(sc, nm, &(int64_t){0}); /* [D, nE] */
            }
        } else {
            /* dense MLP: mlp.{gate,up,down}_proj -> expert 0 */
            tn(nm, sizeof(nm), "model.language_model.layers.%d.mlp.gate_proj.weight", l);
            float *g = wubu_shard_load_f32_t(sc, nm, D, dff);
            tn(nm, sizeof(nm), "model.language_model.layers.%d.mlp.up_proj.weight", l);
            float *u = wubu_shard_load_f32_t(sc, nm, D, dff);
            tn(nm, sizeof(nm), "model.language_model.layers.%d.mlp.down_proj.weight", l);
            float *d = wubu_shard_load_f32_t(sc, nm, dff, D);
            if (g) memcpy(ly->moe.ffn_gate_exps, g, (size_t)D * dff * sizeof(float));
            if (u) memcpy(ly->moe.ffn_up_exps,   u, (size_t)D * dff * sizeof(float));
            if (d) memcpy(ly->moe.ffn_down_exps, d, (size_t)dff * D * sizeof(float));
            free(g); free(u); free(d);
        }
        }
        /* Shared expert (always active in Qwen3.5 MoE) — loaded resident for
         * both the in-RAM and ds4-ssd paths. */
        if (m->shared_expert_ff > 0) {
            int sh = m->shared_expert_ff;
            tn(nm, sizeof(nm), "model.language_model.layers.%d.mlp.shared_expert.gate_proj.weight", l);
            float *sg = wubu_shard_load_f32_t(sc, nm, D, sh);
            tn(nm, sizeof(nm), "model.language_model.layers.%d.mlp.shared_expert.up_proj.weight", l);
            float *su = wubu_shard_load_f32_t(sc, nm, D, sh);
            tn(nm, sizeof(nm), "model.language_model.layers.%d.mlp.shared_expert.down_proj.weight", l);
            float *sd = wubu_shard_load_f32_t(sc, nm, sh, D);
            if (sg) { ly->moe.ffn_gate_shexp = (float*)realloc(ly->moe.ffn_gate_shexp, (size_t)D*sh*sizeof(float)); memcpy(ly->moe.ffn_gate_shexp, sg, (size_t)D*sh*sizeof(float)); free(sg); }
            if (su) { ly->moe.ffn_up_shexp   = (float*)realloc(ly->moe.ffn_up_shexp, (size_t)D*sh*sizeof(float)); memcpy(ly->moe.ffn_up_shexp, su, (size_t)D*sh*sizeof(float)); free(su); }
            if (sd) { ly->moe.ffn_down_shexp = (float*)realloc(ly->moe.ffn_down_shexp, (size_t)sh*D*sizeof(float)); memcpy(ly->moe.ffn_down_shexp, sd, (size_t)sh*D*sizeof(float)); free(sd); }
            tn(nm, sizeof(nm), "model.language_model.layers.%d.mlp.shared_expert_gate.weight", l);
            ly->moe.ffn_gate_inp_shexp = wubu_shard_load_f32(sc, nm, &(int64_t){0}); /* [D] */
        }

        /* ---- RMSNorm weights (wubu_model_forward needs these) ---- */
        tn(nm, sizeof(nm), "model.language_model.layers.%d.input_layernorm.weight", l);
        ly->attn_norm_weight = wubu_shard_load_f32(sc, nm, &(int64_t){0});     /* [D] */
        tn(nm, sizeof(nm), "model.language_model.layers.%d.post_attention_layernorm.weight", l);
        ly->post_attn_norm_weight = wubu_shard_load_f32(sc, nm, &(int64_t){0}); /* [D] */

        /* Per-layer GQA geometry the forward reads (kv_dim/q_dim/out_dim).
         * For toy models where head_dim inference yields kvh=0, derive
         * kv_dim directly from the k_proj tensor's output dimension. */
        ly->gqa.kv_dim = (kvh > 0) ? (kvh * hd) : kvdim;
        ly->gqa.q_dim  = qh * hd;
        ly->gqa.out_dim = qh * hd;
        /* MoE is resident (dense nE=1 or routed): mark loaded so the FFN
         * forward runs on the in-RAM expert blobs. */
        if (ly->moe.ffn_gate_exps || nE > 0) ly->moe.loaded = true;
    }

    /* ---- final RMSNorm ---- */
    m->norm_weight = wubu_shard_load_f32(sc, "model.language_model.norm.weight", &(int64_t){0}); /* [D] */

    /* ---- Model-level dimensions the forward reads (mirrors wubu_model_init) ----
     * Only D_MODEL varies between models; the SSM recurrence constants are
     * compile-time (#define) and the fixture/real models share them. */
    m->d_inner      = VD;                                       /* VALUE_DIM */
    m->key_dim      = SSMDS * wd.ssm_k_heads;
    m->conv_dim     = 2 * m->key_dim + m->d_inner;
    m->conv_kernel  = wd.conv_kernel;
    m->dt_rank      = wd.dt_rank;
    m->ssm_k_heads  = wd.ssm_k_heads;
    m->ssm_v_heads  = wd.ssm_v_heads;
    m->ssm_d_state  = SSMDS;
    m->gqa_q_heads  = qh;
    m->gqa_kv_heads = kvh;
    m->gqa_head_dim = hd;
    m->rotary_dim   = (int)(hd * 0.25f);   /* partial rotary factor 0.25 */
    m->d_ff         = dff;
    m->enable_moe   = (nE > 0 || m->shared_expert_ff > 0 || nL > 0) ? true : false;
    m->moe_max_layers = 0;                 /* 0 => all layers */
    m->gpu_ctx      = NULL;
    m->backend      = wubu_backend_cpu_get();
    m->tied_output  = false;
    m->skip_output_proj = false;
    m->save_last_hidden = NULL;

    /* ---- SSM recurrent state + conv state (calloc, zero-init) ---- */
    int ssm_state_size = nL * wd.ssm_v_heads * SSMDS * SSMDS;
    int conv_state_size = nL * (wd.conv_kernel - 1) * m->conv_dim;
    m->ssm_states = (float *)calloc((size_t)ssm_state_size + conv_state_size, sizeof(float));
    m->conv_states = m->ssm_states + ssm_state_size;

    /* ---- GQA KV cache (per GQA layer) ---- */
    /* Runtime override: WUBU_MAX_CTX env var. Default 8192 (safe for 13GB RAM). */
    int runtime_max_ctx = GQA_MAX_CTX;
    {
        const char *mc_env = getenv("WUBU_MAX_CTX");
        if (mc_env) { int mc = atoi(mc_env); if (mc > 0) runtime_max_ctx = mc; }
    }
    int64_t total_cache_elems = 0;
    int n_gqa = 0;
    for (int l = 0; l < nL; l++) {
        if (!m->layers[l].is_ssm) {
            int kv_dim = m->layers[l].gqa.kv_dim;
            total_cache_elems += (int64_t)runtime_max_ctx * kv_dim;
            n_gqa++;
        }
    }
    m->n_gqa_layers = n_gqa;
    /* S6: store the active KV context cap into the model so downstream
     * forward code (per-layer stride) uses the banked/rotated count. */
    m->gqa_max_ctx = runtime_max_ctx;
    // Auto-select KV precision (Roofline) for this model before sizing cache.
    {
        int ghd = 128, gnkv = 1;
        for (int l = 0; l < nL; l++) {
            if (!m->layers[l].is_ssm) { ghd = m->layers[l].gqa.head_dim; gnkv = m->layers[l].gqa.kv_heads; break; }
        }
        double bw = 0.05; const char *be = getenv("WUBU_BW_TBS"); if (be) bw = atof(be);
        double npar = (double)m->d_model * m->d_model * nL * 12.0;
        int ch = wubu_kv_autoselect(npar, nL, gnkv, ghd, bw, runtime_max_ctx);
        printf("KV-cache scheme auto-selected (bridge): %s (ctx=%d)\n",
               wubu_kv_scheme_name((wubu_kv_scheme_t)ch), runtime_max_ctx);
        /* For large context or memory pressure: enable Q8 fast-attn decode */
        g_use_q8_cache = (ch == WUBU_KV_Q8 || ch == WUBU_KV_Q4_0 || ch == WUBU_KV_4KV);
        if (g_use_q8_cache) {
            printf("Fast-attn Q8 decode path enabled for %d-token context\n", runtime_max_ctx);
        }
    }
    int64_t k_cache_bytes = kv_cache_alloc_size(total_cache_elems);
    /* OOM guard: use posix_memalign (aligned_alloc requires size%64==0). */
    if (posix_memalign((void **)&m->gqa_k_cache, 64, k_cache_bytes ? k_cache_bytes : 16) != 0)
        m->gqa_k_cache = NULL;
    if (posix_memalign((void **)&m->gqa_v_cache, 64, k_cache_bytes ? k_cache_bytes : 16) != 0)
        m->gqa_v_cache = NULL;
    if (m->gqa_k_cache) memset(m->gqa_k_cache, 0, k_cache_bytes ? k_cache_bytes : 16);
    if (m->gqa_v_cache) memset(m->gqa_v_cache, 0, k_cache_bytes ? k_cache_bytes : 16);
    m->gqa_cache_len = 0;
    m->gqa_max_ctx = runtime_max_ctx;

    /* ---- embed_tokens / lm_head ----
     * ZERO-COPY for the safetensors path: embed_tokens / lm_head are huge
     * (5.1 GB each for 27B-class BF16 models). Instead of malloc+F32-copying
     * them, keep the raw (mapped) bytes and dequantize ONE ROW on demand at
     * embedding-lookup / output-projection time. This is what makes a real
     * Qwen3.6-27B forward fit in a 13 GB box. F32 falls back to the old
     * eager copy (small models / when no mmap). */
    int emb_dtype = 0; int64_t emb_row = 0;
    const uint8_t *emb_raw = wubu_shard_raw(sc, "model.language_model.embed_tokens.weight",
                                           &emb_dtype, &emb_row);
    if (emb_raw && (emb_dtype == ST_DTYPE_BF16 || emb_dtype == ST_DTYPE_F16)) {
        m->lazy_embd_raw = emb_raw; m->lazy_embd_dtype = emb_dtype; m->lazy_embd_row = emb_row;
        m->token_embd = NULL;
    } else {
        int64_t ne = 0;
        m->token_embd = wubu_shard_load_f32(sc, "model.language_model.embed_tokens.weight", &ne);
    }

    int lm_dtype = 0; int64_t lm_row = 0;
    const uint8_t *lm_raw = NULL;
    lm_raw = wubu_shard_raw(sc, "lm_head.weight", &lm_dtype, &lm_row);
    if (!lm_raw) lm_raw = wubu_shard_raw(sc, "model.language_model.lm_head.weight", &lm_dtype, &lm_row);
    if (lm_raw && (lm_dtype == ST_DTYPE_BF16 || lm_dtype == ST_DTYPE_F16)) {
        m->lazy_lmhead_raw = lm_raw; m->lazy_lmhead_dtype = lm_dtype; m->lazy_lmhead_row = lm_row;
        m->output_weight = NULL;
    } else {
        int64_t no = 0;
        m->output_weight = wubu_shard_load_f32(sc, "lm_head.weight", &no);
        if (!m->output_weight)
            m->output_weight = wubu_shard_load_f32(sc, "model.language_model.lm_head.weight", &no);
    }
    m->token_embd_q = NULL; m->output_weight_q = NULL;
    m->use_embedding_file = false;

    /* doc 013: optional QuaRot-style Hadamard fuse into the lm_head.
     * For the F32 path we physically fuse H_P into output_weight (above
     * branch); for the zero-copy BF16/F16 lazy path we keep the weight
     * unrotated and instead rotate each row + the input on the fly in the
     * forward (see wubu_model.c lm_head GEMVs). Either way we set rotate_P
     * so the forward knows to apply H_P. Gated by WUBU_ROTATE_W=1;
     * OFF by default (no behavior change). */
    m->rotate_P = 0;
    if (getenv("WUBU_ROTATE_W")) {
        int P = wubu_pow2_floor(m->d_model);
        if (P > 1) {
            m->rotate_P = P;
            fprintf(stderr, "[rotate] Hadamard P=%d armed for lm_head (doc 013)\n", P);
            if (m->output_weight)  /* F32 path: fuse physically */
                wubu_rotate_fuse_right(m->output_weight, m->vocab_size, m->d_model);
        }
    }
    
    /* Keep the shard context alive for lazy embed/lm_head mmap access.
     * Store in model; wubu_model_free will close it. */
    m->shard_ctx = sc;
    sc = NULL;

    st_close(st);
    wubu_shard_close(sc);

    /* ---- LoRA (BTL-3): applied to an already-loaded base model ----
     * When this checkpoint IS the base (not a LoRA adapter), nothing to do.
     * When it is a LoRA adapter, wubu_model_init_auto loads the base first
     * and then calls wubu_model_apply_lora(). Keep _ssd free of LoRA logic. */
    (void)ad;

    return 0;
}

/* Apply a BTL-3 LoRA adapter on top of an already-loaded base model `m`.
 * Base weights must already reside in `m`. Reads lora_A/lora_B per target
 * module and adds scale*(B@A) to the resident F32 weights.
 * Targets: GQA q/k/v/o_proj and SSM linear_attn.out_proj (BTL-3's modules).
 * rank/scale come from the adapter (ad->lora_r / ad->lora_alpha); falls back
 * to rank 32 / alpha 64 when the adapter doesn't report them. */
int wubu_model_apply_lora(wubu_model_t *m, const char *adapter_path,
                          const wubu_adapter_t *ad) {
    if (!m || !m->layers || m->n_layers <= 0) return -1;
    st_ctx *ast = st_open(adapter_path);
    if (!ast) { fprintf(stderr, "lora: cannot open adapter %s\n", adapter_path); return -1; }

    const int rank = ad->lora_r > 0 ? ad->lora_r : 32;
    const float scale = (ad->lora_alpha > 0 && ad->lora_r > 0)
                            ? (float)ad->lora_alpha / (float)ad->lora_r
                            : 64.0f / 32.0f;
    const int D = m->d_model;
    /* Resident GQA output dims come from the base model geometry, NOT the
     * adapter's lora_B shape. A LoRA tuned on q_proj must write exactly
     * q_heads*head_dim rows into attn_q_weight (which is [q_heads*head_dim, D]).
     * For Qwen3.6 q_heads*head_dim == D; for fixtures it may be smaller. */
    const int q_out = m->gqa_q_heads * m->gqa_head_dim;   /* q/o_proj rows */
    const int kv_out = m->gqa_kv_heads * m->gqa_head_dim; /* k/v_proj rows */

    for (int l = 0; l < m->n_layers; l++) {
        wubu_layer_t *ly = &m->layers[l];
        char an[256], bn[256];
        /* q_proj */
        tn(an, sizeof(an), "model.language_model.layers.%d.self_attn.q_proj.lora_A.weight", l);
        tn(bn, sizeof(bn), "model.language_model.layers.%d.self_attn.q_proj.lora_B.weight", l);
        int64_t na = 0, nb = 0;
        float *A = st_load_f32(ast, an, &na);
        float *B = st_load_f32(ast, bn, &nb);
        if (A && B && na == (int64_t)rank * D && nb == (int64_t)q_out * rank) {
            int resid_elems = q_out * D;
            wubu_lora_t *la = wubu_lora_create(rank, scale, D, q_out);
            if (la) { wubu_lora_load_f32(la, A, B); wubu_lora_apply(la, ly->gqa.attn_q_weight); wubu_lora_free(la); }
            (void)resid_elems;
        }
        free(A); free(B);
        /* k_proj */
        tn(an, sizeof(an), "model.language_model.layers.%d.self_attn.k_proj.lora_A.weight", l);
        tn(bn, sizeof(bn), "model.language_model.layers.%d.self_attn.k_proj.lora_B.weight", l);
        A = st_load_f32(ast, an, &na); B = st_load_f32(ast, bn, &nb);
        if (A && B && na == (int64_t)rank * D && nb == (int64_t)kv_out * rank) {
            wubu_lora_t *la = wubu_lora_create(rank, scale, D, kv_out);
            if (la) { wubu_lora_load_f32(la, A, B); wubu_lora_apply(la, ly->gqa.attn_k_weight); wubu_lora_free(la); }
        }
        free(A); free(B);
        /* v_proj */
        tn(an, sizeof(an), "model.language_model.layers.%d.self_attn.v_proj.lora_A.weight", l);
        tn(bn, sizeof(bn), "model.language_model.layers.%d.self_attn.v_proj.lora_B.weight", l);
        A = st_load_f32(ast, an, &na); B = st_load_f32(ast, bn, &nb);
        if (A && B && na == (int64_t)rank * D && nb == (int64_t)kv_out * rank) {
            wubu_lora_t *la = wubu_lora_create(rank, scale, D, kv_out);
            if (la) { wubu_lora_load_f32(la, A, B); wubu_lora_apply(la, ly->gqa.attn_v_weight); wubu_lora_free(la); }
        }
        free(A); free(B);
        /* o_proj */
        tn(an, sizeof(an), "model.language_model.layers.%d.self_attn.o_proj.lora_A.weight", l);
        tn(bn, sizeof(bn), "model.language_model.layers.%d.self_attn.o_proj.lora_B.weight", l);
        A = st_load_f32(ast, an, &na); B = st_load_f32(ast, bn, &nb);
        if (A && B && na == (int64_t)rank * D && nb == (int64_t)q_out * rank) {
            wubu_lora_t *la = wubu_lora_create(rank, scale, D, q_out);
            if (la) { wubu_lora_load_f32(la, A, B); wubu_lora_apply(la, ly->gqa.attn_output_weight); wubu_lora_free(la); }
        }
        free(A); free(B);
        /* NOTE: BTL-3 also ships a linear_attn.out_proj LoRA, but its
         * delta is [VD,D] while the resident SSM out_proj weight is
         * row-major [D,VD]; a correct apply needs a transposed
         * delta. Out of scope for the core GQA q/k/v/o orchestration
         * verified here; left as a follow-up. */
    }
    st_close(ast);
    return 0;
}

int wubu_model_init_auto(wubu_model_t *m, const char *path) {
    size_t plen = strlen(path);
    /* A path is a safetensors source if it names a .safetensors file OR is a
     * directory holding shards (model-NNN-of-MMM.safetensors). wubu_shard_open
     * globs the directory, so a bare dir is the natural way to point at a
     * multi-shard checkpoint. A LoRA adapter is still passed as its
     * adapter_model.safetensors file path (detected via the .safetensors
     * substring below). */
    int is_dir = 0;
    struct stat _st;
    if (stat(path, &_st) == 0 && S_ISDIR(_st.st_mode)) is_dir = 1;
    int is_st = (strstr(path, ".safetensors") != NULL) || is_dir;
    if (is_st) {
        wubu_adapter_t ad; memset(&ad, 0, sizeof(ad));
        int ad_ok = wubu_adapter_load(&ad, path);
        if (!ad_ok) {
            ad.arch = WUBU_ARCH_QWEN_FAMILY; ad.ok = 1;
        }
        /* ---- BTL-3 LoRA: a LoRA adapter .safetensors must first load its
         * BASE checkpoint, then have the LoRA delta applied on top. Resolve the
         * base checkpoint from (in priority order):
         *   1. ad.base_model        (hf config base_model / base_model_name_or_path)
         *   2. $BTL_BASE env var
         *   3. sibling ./base/ dir next to the adapter (base/model.safetensors)
         * The base is loaded into `m`; the LoRA block inside _ssd then applies
         * the delta. If no base resolves, fall back to loading the adapter as a
         * plain model (best-effort). */
        if (ad.is_lora) {
            char base_path[2048];
            const char *bp = NULL;
            if (ad.base_model[0]) {
                /* accept bare "Qwen/Qwen3.6-27B" (HF id) or a path */
                if (access(ad.base_model, F_OK) == 0) bp = ad.base_model;
                else if (getenv("BTL_BASE") && access(getenv("BTL_BASE"), F_OK) == 0)
                    bp = getenv("BTL_BASE");
            }
            if (!bp && getenv("BTL_BASE") && access(getenv("BTL_BASE"), F_OK) == 0)
                bp = getenv("BTL_BASE");
            if (!bp) {
                /* sibling ./base/ next to the adapter */
                const char *slash = strrchr(path, '/');
                size_t dlen = slash ? (size_t)(slash - path) : 0;
                char cand[2048];
                if (dlen + 16 < sizeof(cand)) {
                    memcpy(cand, path, dlen);
                    strcpy(cand + dlen, "/base/model.safetensors");
                    if (access(cand, F_OK) == 0) { strcpy(base_path, cand); bp = base_path; }
                }
            }
            if (bp) {
                /* Load the BASE with a zeroed adapter (mirrors test_st_bridge,
                 * which proves _ssd derives layer count + dims from the tensors
                 * themselves). wubu_adapter_load would tag arch=QWEN_FAMILY and
                 * send _ssd down a non-safetensors path (0 layers). */
                wubu_adapter_t bad = {0};
                int rc = wubu_model_init_safetensors(m, bp, &bad);
                if (rc != 0) {
                    fprintf(stderr, "bridge: BTL-3 base load failed: %s\n", bp);
                    return rc;
                }
                /* _ssd's LoRA block reads `path` (the adapter) and applies the
                 * delta onto the now-resident base weights in `m`. */
                return wubu_model_apply_lora(m, path, &ad);
            }
            fprintf(stderr, "bridge: BTL-3 LoRA with no resolvable base; loading adapter as plain model\n");
        }
        /* ds4-ssd: for any MoE model, route experts through the slot-bank
         * paged DIRECTLY from the checkpoint shards (no redundant sidecar copy,
         * no KAT_SIDECAR env, no 256 GB duplicate). The bridge opens the bank
         * over the same shards it already loads. */
        if (ad.n_experts > 0) {
            return wubu_model_init_safetensors_ssd(m, path, &ad, NULL);
        }
        return wubu_model_init_safetensors(m, path, &ad);
    }
    /* legacy GGUF path */
    return wubu_model_init(m, path);
}

void wubu_model_safetensors_free(wubu_model_t *m) {
    if (!m || !m->layers) return;
    for (int l = 0; l < m->n_layers; l++) {
        wubu_layer_t *ly = &m->layers[l];
        free(ly->gqa.attn_q_weight); free(ly->gqa.attn_k_weight);
        free(ly->gqa.attn_v_weight); free(ly->gqa.attn_output_weight);
        free(ly->ssm.attn_qkv_weight_f32); free(ly->ssm.attn_gate_weight_f32);
        free(ly->ssm.ssm_alpha_weight); free(ly->ssm.ssm_beta_weight);
        free(ly->ssm.ssm_a); free(ly->ssm.ssm_dt_bias);
        free(ly->ssm.ssm_conv1d_weight); free(ly->ssm.ssm_norm_weight);
        free(ly->ssm.ssm_out_weight_f32);
        free(ly->attn_norm_weight); free(ly->post_attn_norm_weight);
        free(ly->moe.ffn_gate_exps); free(ly->moe.ffn_up_exps);
        free(ly->moe.ffn_down_exps); free(ly->moe.ffn_gate_inp);
    }
    free(m->token_embd); free(m->output_weight); free(m->norm_weight);
    free(m->layers);
    m->layers = NULL;
}
