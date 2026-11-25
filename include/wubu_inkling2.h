/*
 * wubu_inkling2.h -- Inkling-2 GGUF architecture probe (EG-118, step 1 of 6).
 *
 * The engine has no Inkling support at all; this header is the contract for the
 * first step: recognise `general.architecture = inkling2`, read every key the
 * forward pass will need, and map the 734 tensors into named slots with explicit
 * dimensions so nothing downstream has to re-derive them from the file layout.
 * See docs/research/inkling2-gguf-inventory.md for the measured inventory.
 */
#ifndef WUBU_INKLING2_H
#define WUBU_INKLING2_H

#include "gguf_reader.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Reason codes: probe fails closed instead of filling the config with defaults. */
enum {
    WUBU_INKLING2_OK = 0,
    WUBU_INKLING2_ERR_ARCH = -1,      /* not an inkling2 file */
    WUBU_INKLING2_ERR_MISSING = -2,   /* a required inkling2.* key is absent */
    WUBU_INKLING2_ERR_SHAPE = -3,     /* dims of a required tensor are wrong */
    WUBU_INKLING2_ERR_MISSING_TENSOR = -4
};

typedef struct {
    int n_layer;           /* LOGICAL layers (block_count, 48 for this file) */
    int n_layer_phys;      /* physical blocks with tensors (24: 6 + 12 + 6) */
    int d_model;
    int d_ff;
    int n_head;            /* 3072 / 128 */
    int head_dim;
    int n_kv_head;         /* 1024 / 128 */
    int n_expert;
    int n_expert_used;
    int d_ff_expert;
    int n_loops;
    int loop_core_start;
    int loop_core_end;
    int loop_gate_rank;
    int swa_window;
    int conv_lcache;       /* 4-tap short convs */
    int attn_res_block;    /* attention-residual query every N layers */
    int vocab_size;
    int vocab_size_unpadded;
    int rope_dim;          /* 0 for Inkling: relative-bias attention */
    float rms_eps;
    float expert_weights_scale;
    float situ_beta;
    float situ_linear_beta;
    float logit_softcap;
    /* These four were hardcoded in the forward pass. They happen to match this checkpoint exactly, but a
     * hardcoded constant that agrees with the file is indistinguishable from a correct one until a second
     * file disagrees - so they are read from the metadata now. */
    float log_scale_alpha;   /* inkling2.attention.log_scale_alpha   (0.1 here)  */
    int   log_scale_floor;   /* inkling2.attention.log_scale_floor   (8192 here)  */
    float qk_clip;           /* inkling2.attention.qk_clip           (16*sqrt2)   */
    float rel_clip;          /* inkling2.attention.relative_clip     (8.0 here)   */
    int   rel_extent;        /* inkling2.attention.relative_extent   (1024 here)  */
    int   rel_length;        /* inkling2.attention.relative_length   (16 here)    */
} wubu_inkling2_cfg_t;

/* Fills cfg from the GGUF's inkling2.* keys. 0 on success, negative enum on error. */
int wubu_inkling2_probe(gguf_ctx *ctx, wubu_inkling2_cfg_t *cfg);

/* Logical layer index -> physical block index. The 12-layer core runs n_loops
 * times, so logical 0..5 map to 0..5, logical 6..41 map to 6..17 three times, and
 * logical 42..47 map to 18..23. Returns -1 when il is out of range. */
int wubu_inkling2_phys_block(const wubu_inkling2_cfg_t *cfg, int il);

/* Name of the layer tensor for `family` (e.g. "attn_q.weight") at layer `il`.
 * Writes into buf and returns 1 when the tensor exists with the expected rank,
 * 0 when it does not (dynamic-tap convs exist only on full layers, etc.). */
int wubu_inkling2_layer_tensor(gguf_ctx *ctx, int il, const char *family, char *buf, size_t cap);

/* Global (non-layer) tensors: "token_embd.weight", "output.weight", ... */
int wubu_inkling2_global_tensor(gguf_ctx *ctx, const char *name, gguf_tensor_info **out);

/* Read `name` as F32 into out (caller sizes it from the tensor dims). Returns 0 on
 * success, WUBU_INKLING2_ERR_UNKNOWN_TYPE when the tensor uses a quant type this
 * engine has no dequantiser for (ids 30/40/42 in this file), other negative codes
 * on I/O failure. */
enum { WUBU_INKLING2_ERR_UNKNOWN_TYPE = -5, WUBU_INKLING2_ERR_IO = -6,
       WUBU_INKLING2_ERR_NO_LOOP = -7 };

int wubu_inkling2_tensor_f32(gguf_ctx *ctx, const char *name, float *out,
                             int64_t max_elems, int64_t *n_out);

/* True when every tensor the forward needs is present with the right dims.
 * Returns the number of problems, and prints the first 8 of them. */
int wubu_inkling2_validate(gguf_ctx *ctx, const wubu_inkling2_cfg_t *cfg);

/* ------------------------------------------------------------------------- */
/* Step 2: the runtime model.                                                */
/*                                                                           */
/* The 8.5 GB data blob is not mapped (gguf_buffer_data refuses it), so a    */
/* tensor here is "where it lives in the file", not a pointer to RAM. That   */
/* is deliberate: it is the SSD->RAM->VRAM staging the engine wants anyway,  */
/* and it keeps the loader honest about how much is resident.                */
/* ------------------------------------------------------------------------- */

typedef struct {
    gguf_tensor_info *info;   /* NULL when the family is absent for this layer */
    int64_t file_offset;      /* absolute offset of the tensor's data           */
    int required;             /* 1 = the forward pass cannot run without it     */
} wubu_in2_tensor_t;

typedef struct {
    wubu_in2_tensor_t attn_norm, ffn_norm;
    wubu_in2_tensor_t wq, wk, wv, wo, wgate, wrel, rel_bias;
    wubu_in2_tensor_t q_clip, rel_clip, sink;
    wubu_in2_tensor_t k_conv, v_conv, attn_post_conv, ffn_post_conv;
    wubu_in2_tensor_t attn_res_q, ffn_res_q;
    wubu_in2_tensor_t gate_up, down, gate_up_scale, down_scale, router, exp_bias;
    /* full-attention layers only */
    wubu_in2_tensor_t q_inp_norm, k_inp_norm, v_inp_norm, k_conv_proj, v_conv_proj;
    /* loop layers only: 4 control slots x {down, gain, forget} */
    wubu_in2_tensor_t loop_down[4], loop_gain[4], loop_forget[4];
    int is_full_attn;
    int is_loop_layer;
} wubu_in2_layer_t;

typedef struct {
    wubu_in2_tensor_t tok_embd, out_w, out_norm, out_res_q;
    wubu_in2_tensor_t msg_query;      /* [loop_layers, 4, hidden] learned loop queries */
    wubu_in2_tensor_t final_res_q;
    wubu_in2_layer_t layer[64];       /* physical blocks, 24 in this file       */
    int n_layer;
    int n_missing_required;
    int n_optional_absent;
} wubu_inkling2_model_t;

/* Resolves every tensor the forward needs. Returns WUBU_INKLING2_OK, or
 * WUBU_INKLING2_ERR_MISSING_TENSOR when a required family is absent (the model
 * struct is still filled so a caller can print the census). */
int  wubu_inkling2_load(gguf_ctx *ctx, const wubu_inkling2_cfg_t *cfg,
                        wubu_inkling2_model_t *m);
void wubu_inkling2_model_print(const wubu_inkling2_model_t *m, const wubu_inkling2_cfg_t *cfg);

/* Reads a tensor's raw bytes. Used by the dequantising readers below. */
int  wubu_inkling2_read(gguf_ctx *ctx, const wubu_in2_tensor_t *t, void *dst, int64_t nbytes);

/* Reads a tensor as F32 (dequantising if needed) into a caller buffer. */
int  wubu_inkling2_read_f32(gguf_ctx *ctx, const wubu_in2_tensor_t *t, float *dst, int64_t max_elems);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_INKLING2_H */
