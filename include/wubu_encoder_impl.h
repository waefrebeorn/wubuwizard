/* wubu_encoder_impl.h -- OUR OWN encoder (WB06: the single-encoder
 * modality-agnostic base, made not imported).
 *
 * The KV-FS doctrine: "all files are data, all inputs are encoded" —
 * ONE shared embedding space for every modality. We DO NOT import an
 * encoder (no SigLIP, no external weights): every front-end is ours,
 * written C11 from scratch:
 *
 *   image -> wubu_imgenc  (ViT patch embedding, CC01)
 *   audio -> wubu_audio   (mel-spectrogram + real FFT, CC02)
 *   text  -> the tokenizer embeddings (wubu_tokenizer_hf)
 *
 * Each front-end produces tokens in its native space; a shared
 * projection head maps every modality into ONE embedding space (the
 * shared dim) so the model sees all inputs alike. The implementations
 * register into the wubu_encoder slot — consumers probe by name.
 *
 * C11. Depends on the wubu_encoder slot + our own front-ends.
 */
#ifndef WUBU_ENCODER_IMPL_H
#define WUBU_ENCODER_IMPL_H

#include <stdint.h>
#include <stddef.h>

#include "wubu_encoder.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The shared embedding space: every modality lands here. Chosen to be
 * the imgenc native dim (128) so the image path needs no extra head;
 * audio/text project up into it. */
#define WUBU_ENC_SHARED_DIM 128

/* Register our own encoders into the slot:
 *   "our-image" (modality IMAGE) — wubu_imgenc ViT patch embedding
 *   "our-audio" (modality AUDIO) — wubu_audio mel-spectrogram + a
 *                learned linear head into the shared dim
 *   "our-text"  (modality TEXT)  — deterministic hash projection of
 *                token ids into the shared dim (the tokenizer path;
 *                the real embedding table plugs in at load)
 * Returns 0 on success. */
int wubu_encoder_register_ours(void);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_ENCODER_IMPL_H */
