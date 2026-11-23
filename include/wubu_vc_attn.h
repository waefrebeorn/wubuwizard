#ifndef WUBU_VC_ATTN_H
#define WUBU_VC_ATTN_H

/*
 * wubu_vc_attn.h -- EG-77: V stored as a residual around a per-128-tile mean.
 * FlashInfer VC-Attention: restore the tile mean inside the attention pass so
 * V need not be fully dequantized first. This gate checks the quant/dequant
 * contract and that the restored V yields the same weighted sum as the full
 * values. The 4-bit residual is a stand-in for E4M3, not its bit layout.
 */

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_VC_TILE 128
#define WUBU_VC_MAX_V 1024

/* n must be a multiple of 128. Returns number of tiles, -1 on bad input. */
int wubu_vc_quant(const float *v, int n, unsigned char *codes,
                  float *tile_means, float *tile_scales);

int wubu_vc_dequant(const unsigned char *codes, const float *tile_means,
                    const float *tile_scales, int n, float *out);

/* out = softmax(logits) . v. 0/-1. */
int wubu_vc_attention(const float *logits, const float *v, int n, float *out);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_VC_ATTN_H */
