/* wubu_encoder.c -- the encoder registry (THE ENCODER SLOT, WB06/AN06).
 *
 * One slot, many encoders: text/image/audio/video encoders register by
 * name + modality; consumers probe. The shared embedding space is the
 * KV-FS doctrine's "all inputs encoded" — the seam is here, the
 * implementations (SigLIP, whisper, t5) register into it.
 *
 * C11, self-contained (stdlib only).
 */
#include "wubu_encoder.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    wubu_encoder_t *slots;
    size_t count;
    size_t cap;
} enc_registry_t;

static enc_registry_t g_registry = { NULL, 0, 0 };

int wubu_encoder_register(wubu_encoder_t *enc) {
    if (!enc || !enc->name || !enc->encode || enc->dim < 1) return -1;

    /* Replace an existing name: free the old ctx, keep the slot. */
    for (size_t i = 0; i < g_registry.count; i++) {
        if (strcmp(g_registry.slots[i].name, enc->name) == 0) {
            if (g_registry.slots[i].free && g_registry.slots[i].ctx)
                g_registry.slots[i].free(g_registry.slots[i].ctx);
            g_registry.slots[i] = *enc;
            return 0;
        }
    }

    /* Append. Grow by doubling (no fixed capacity — Revolver). */
    if (g_registry.count == g_registry.cap) {
        size_t nc = g_registry.cap ? g_registry.cap * 2 : 4;
        wubu_encoder_t *ns = (wubu_encoder_t *)realloc(g_registry.slots,
                                                       nc * sizeof(*ns));
        if (!ns) return -1;
        g_registry.slots = ns;
        g_registry.cap = nc;
    }
    g_registry.slots[g_registry.count++] = *enc;
    return 0;
}

wubu_encoder_t *wubu_encoder_get(const char *name) {
    if (!name) return NULL;
    for (size_t i = 0; i < g_registry.count; i++)
        if (strcmp(g_registry.slots[i].name, name) == 0)
            return &g_registry.slots[i];
    return NULL;
}

wubu_encoder_t *wubu_encoder_for_modality(wubu_modality_t mod) {
    for (size_t i = 0; i < g_registry.count; i++)
        if (g_registry.slots[i].modality == mod)
            return &g_registry.slots[i];
    return NULL;
}

int wubu_encoder_encode_named(const char *name, const float *x, size_t n,
                              float *out) {
    wubu_encoder_t *e = wubu_encoder_get(name);
    if (!e || !e->encode) return -1;
    return e->encode(e->ctx, x, n, out);
}

size_t wubu_encoder_count(void) {
    return g_registry.count;
}
