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

/* True when every tensor the forward needs is present with the right dims.
 * Returns the number of problems, and prints the first 8 of them. */
int wubu_inkling2_validate(gguf_ctx *ctx, const wubu_inkling2_cfg_t *cfg);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_INKLING2_H */
