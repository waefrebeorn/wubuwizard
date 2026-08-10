/**
 * wubu_siglip.h — SigLIP vision encoder (SmolVLM2-2.2B mmproj)
 *
 * Loads the mmproj GGUF's `v.*` SigLIP tower + `mm.model.fc` projector
 * and produces text-aligned image embeddings [B, 81, 2048].
 */
#ifndef WUBU_SIGLIP_H
#define WUBU_SIGLIP_H

#include "gguf_reader.h"
#include <stdbool.h>

#define SIGLIP_HIDDEN       1152
#define SIGLIP_INTERMEDIATE 4304
#define SIGLIP_HEADS        16
#define SIGLIP_HEAD_DIM     (SIGLIP_HIDDEN / SIGLIP_HEADS)   /* 72 */
#define SIGLIP_N_BLOCKS     27
#define SIGLIP_N_POS        729       /* 27x27 grid (378x378 @14px) */
#define SIGLIP_PROJ         2048
#define SIGLIP_N_TOK        81        /* 729 / (3x3 merge) */

typedef struct {
    float *ln1_w, *ln1_b;
    float *q_w, *q_b, *k_w, *k_b, *v_w, *v_b, *o_w, *o_b;
    float *ln2_w, *ln2_b;
    float *ffn_up_w, *ffn_up_b, *ffn_down_w, *ffn_down_b;
} wubu_siglip_block_t;

typedef struct {
    float *patch_w;    // [14,14,3,1152] conv kernel (KH,KW,CIN,COUT)
    float *patch_b;    // [1152]
    float *pos_w;      // [1152, 729]
    float *post_ln_w;  // [1152]
    float *post_ln_b;  // [1152]
    wubu_siglip_block_t *blocks;  // [27]
    float *proj_w;     // [10368, 2048]
} wubu_siglip_t;

// Load all weights from the mmproj GGUF. Returns false on any missing
// tensor (prints which one).
bool wubu_siglip_init(wubu_siglip_t *enc, const char *path);
void wubu_siglip_free(wubu_siglip_t *enc);

// Forward: input [B, C, H, W] floats in [0,1] (caller applies the
// model's mean/std normalization). H,W must give a 27x27 patch grid
// (e.g. 378x378). Output [B, 81, 2048] projected embeddings.
void wubu_siglip_forward(wubu_siglip_t *enc, const float *input, int B, int C,
                         int H, int W, float *output);

#endif /* WUBU_SIGLIP_H */
