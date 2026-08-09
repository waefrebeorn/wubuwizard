/* wubu_sd_clip.h -- CLIP text encoder (ViT-L/14 style, 12 layers, 768d,
 * 8 heads, 512 ctx, quick_gelu) + byte-level BPE tokenizer for the WuBu
 * stable-diffusion engine. C11, opaque structs, self-contained.
 */
#ifndef WUBU_SD_CLIP_H
#define WUBU_SD_CLIP_H

#include <stddef.h>

typedef struct wubu_sd_clip wubu_sd_clip_t;

/* Load CLIP from a GGUF ctx (weights stay quantized in the blob).
 * Returns NULL on failure. */
wubu_sd_clip_t *wubu_sd_clip_load(void *gguf_ctx);

/* release weight cache (keep tokenizer) — phase-separated pipelines */
void wubu_sd_clip_clear_cache(wubu_sd_clip_t *c);

void wubu_sd_clip_free(wubu_sd_clip_t *c);

/* Encode a prompt: writes the full 77x768 CLIP sequence (UNet cross-attn
 * K/V) into seq_out, and the pooled (EOS) embedding into pooled_out
 * (either may be NULL). Returns 0 on success. */
int wubu_sd_clip_encode(wubu_sd_clip_t *c, const char *prompt,
                        float *seq_out /* [77][768] */,
                        float *pooled_out /* [768] */);

#endif /* WUBU_SD_CLIP_H */
