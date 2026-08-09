/* wubu_encoder.h -- THE ENCODER SLOT (WB06/AN06: the single-encoder
 * modality-agnostic base).
 *
 * The KV-FS doctrine: "all files are data, all inputs are encoded" —
 * ONE encoder space for every modality (image, audio, text, video).
 * Gemma 3 12B SigLIP is the closest base (WB06); this module is the
 * SEAM (Revolver: the slot precedes the module) that any encoder
 * implementation registers into.
 *
 * The slot contract:
 *   encode(x, n) -> embedding[dim]   (the shared space)
 *   decode(e, n) -> x                (back out, where supported)
 *   dims probed at load (never assumed — Revolver)
 *
 * C11, opaque. The registry reuses the kernel-registry pattern
 * (wubu_kernel_register): name -> ops + ctx, probe by name.
 */
#ifndef WUBU_ENCODER_H
#define WUBU_ENCODER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The encoder modality kinds. */
typedef enum {
    WUBU_MODALITY_TEXT = 0,
    WUBU_MODALITY_IMAGE,
    WUBU_MODALITY_AUDIO,
    WUBU_MODALITY_VIDEO,
    WUBU_MODALITY_COUNT
} wubu_modality_t;

/* The vtable (the contract every encoder implements). */
typedef struct wubu_encoder wubu_encoder_t;

struct wubu_encoder {
    const char *name;             /* "siglip", "whisper", "t5", ... */
    wubu_modality_t modality;     /* what this encoder eats */
    int  dim;                     /* the shared embedding dim (probed) */
    void *ctx;                    /* the encoder instance (opaque) */

    /* Encode raw input x [n floats] into the shared embedding space,
     * out [dim]. Returns 0 on success. */
    int  (*encode)(void *ctx, const float *x, size_t n, float *out);

    /* Decode an embedding back to the raw space (where supported;
     * NULL for encoders without a decoder). Returns 0 on success. */
    int  (*decode)(void *ctx, const float *e, size_t n, float *out);

    /* Destructor (may be NULL). */
    void (*free)(void *ctx);
};

/* Register an encoder (idempotent: replacing a name frees the old ctx).
 * Returns 0 on success. */
int wubu_encoder_register(wubu_encoder_t *enc);

/* Look up an encoder by name, or the first encoder for a modality. */
wubu_encoder_t *wubu_encoder_get(const char *name);
wubu_encoder_t *wubu_encoder_for_modality(wubu_modality_t mod);

/* Convenience: encode through a named encoder (NULL-safe). */
int wubu_encoder_encode_named(const char *name, const float *x, size_t n,
                              float *out);

/* How many encoders are registered (for probing). */
size_t wubu_encoder_count(void);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_ENCODER_H */
