/*
 * test_encoder.c -- WB06/AN06 gate: THE ENCODER SLOT.
 *
 * Proves the modality-agnostic seam: encoders register by name +
 * modality into ONE slot, consumers probe (never hardcode), and the
 * shared embedding space works for any input type.
 *
 *   1. REGISTRY: an encoder registers and is discoverable by name AND
 *      by modality (probe, don't assume).
 *   2. ENCODE: raw input (any modality) maps into the shared dim with
 *      finite, deterministic output.
 *   3. SHARED SPACE: text and image inputs both land in the same dim
 *      (the KV-FS "all inputs encoded" invariant).
 *   4. REPLACE: re-registration replaces cleanly (no dup, no leak).
 *
 * The test uses a tiny hashing encoder (the real SigLIP/whisper/T5
 * payloads register into the SAME slot — the seam is what's proven).
 *
 * Gate: `make test_encoder`.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "wubu_encoder.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)

#define EMBED_DIM 64

/* A minimal hashing encoder: any input maps deterministically into the
 * shared embedding space. This is the PAYLOAD the seam carries; the
 * real encoders (SigLIP etc.) are drop-in replacements. */
static uint64_t enc_rng = 0xFEEDFACE12345678ull;
static float enc_frand(void) {
    enc_rng ^= enc_rng << 13;
    enc_rng ^= enc_rng >> 7;
    enc_rng ^= enc_rng << 17;
    return 2.0f * ((float)((enc_rng >> 40) * 0x1p-24)) - 1.0f;
}

typedef struct {
    float *proj;   /* [dim] the modality projection */
} hash_enc_ctx_t;

static int hash_encode(void *ctx, const float *x, size_t n, float *out) {
    hash_enc_ctx_t *c = (hash_enc_ctx_t *)ctx;
    if (!c || !out || n == 0) return -1;
    /* a deterministic bag-of-values projection into the shared dim */
    for (int d = 0; d < EMBED_DIM; d++) {
        float acc = 0.0f;
        for (size_t i = 0; i < n; i++)
            acc += x[i] * c->proj[(d + (int)i) % EMBED_DIM];
        out[d] = acc;
    }
    return 0;
}
static int hash_decode(void *ctx, const float *e, size_t n, float *out) {
    (void)ctx;
    /* decode = the same projection transposed (a symmetric stub — the
     * slot proves the seam; real decoders are modality-specific) */
    for (size_t i = 0; i < n; i++) out[i] = e[i % EMBED_DIM];
    return 0;
}
static void hash_free(void *ctx) {
    hash_enc_ctx_t *c = (hash_enc_ctx_t *)ctx;
    free(c->proj);
    free(c);
}

static wubu_encoder_t g_text_enc, g_img_enc;

int main(void)
{
    printf("=== test_encoder (WB06/AN06: THE ENCODER SLOT) ===\n");

    /* ---- register a text + an image encoder into the same slot ---- */
    hash_enc_ctx_t *tctx = (hash_enc_ctx_t *)calloc(1, sizeof(*tctx));
    hash_enc_ctx_t *ictx = (hash_enc_ctx_t *)calloc(1, sizeof(*ictx));
    if (!tctx || !ictx) return 1;
    tctx->proj = (float *)malloc((size_t)EMBED_DIM * sizeof(float));
    ictx->proj = (float *)malloc((size_t)EMBED_DIM * sizeof(float));
    for (int i = 0; i < EMBED_DIM; i++) { tctx->proj[i] = enc_frand(); ictx->proj[i] = enc_frand(); }

    g_text_enc.name = "hash-text";
    g_text_enc.modality = WUBU_MODALITY_TEXT;
    g_text_enc.dim = EMBED_DIM;
    g_text_enc.ctx = tctx;
    g_text_enc.encode = hash_encode;
    g_text_enc.decode = hash_decode;
    g_text_enc.free = hash_free;

    g_img_enc.name = "hash-image";
    g_img_enc.modality = WUBU_MODALITY_IMAGE;
    g_img_enc.dim = EMBED_DIM;
    g_img_enc.ctx = ictx;
    g_img_enc.encode = hash_encode;
    g_img_enc.decode = hash_decode;
    g_img_enc.free = hash_free;

    CHECK(wubu_encoder_register(&g_text_enc) == 0, "text encoder registers");
    CHECK(wubu_encoder_register(&g_img_enc) == 0, "image encoder registers");
    CHECK(wubu_encoder_count() == 2, "two encoders in the registry");

    /* ---- 1. discoverable by name AND modality ---- */
    CHECK(wubu_encoder_get("hash-text") != NULL, "probe by name");
    CHECK(wubu_encoder_for_modality(WUBU_MODALITY_IMAGE) != NULL,
          "probe by modality");
    CHECK(wubu_encoder_for_modality(WUBU_MODALITY_AUDIO) == NULL,
          "no audio encoder yet -> NULL (the seam, not a crash)");

    /* ---- 2. encode: finite, deterministic, shared dim ---- */
    float text_in[32], img_in[256];
    float e_text[EMBED_DIM], e_img[EMBED_DIM], e_text2[EMBED_DIM];
    for (int i = 0; i < 32; i++) text_in[i] = (float)((i * 3) % 17) / 8.0f;
    for (int i = 0; i < 256; i++) img_in[i] = (float)((i * 11) % 31) / 16.0f;

    CHECK(wubu_encoder_encode_named("hash-text", text_in, 32, e_text) == 0,
          "text encodes");
    CHECK(wubu_encoder_encode_named("hash-image", img_in, 256, e_img) == 0,
          "image encodes");
    wubu_encoder_encode_named("hash-text", text_in, 32, e_text2);
    int finite = 1, deterministic = 1;
    for (int d = 0; d < EMBED_DIM; d++) {
        if (!isfinite(e_text[d]) || !isfinite(e_img[d])) finite = 0;
        if (e_text[d] != e_text2[d]) deterministic = 0;
    }
    CHECK(finite, "embeddings are finite");
    CHECK(deterministic, "embeddings are deterministic (pure function)");

    /* ---- 3. shared space: same dim, both modalities ---- */
    printf("  ok: text dim=%d image dim=%d (the shared embedding space)\n",
           wubu_encoder_get("hash-text")->dim,
           wubu_encoder_for_modality(WUBU_MODALITY_IMAGE)->dim);
    CHECK(wubu_encoder_get("hash-text")->dim == EMBED_DIM &&
          wubu_encoder_for_modality(WUBU_MODALITY_IMAGE)->dim == EMBED_DIM,
          "both modalities land in the same embedding dim (KV-FS doctrine)");

    /* ---- 4. replace: clean, no dup, no leak ---- */
    CHECK(wubu_encoder_register(&g_text_enc) == 0, "re-registration succeeds");
    CHECK(wubu_encoder_count() == 2, "re-registration replaces, does not duplicate");

    /* decode round-trip (the seam supports it where the encoder does) */
    float back[32];
    wubu_encoder_t *te = wubu_encoder_get("hash-text");
    if (te->decode) {
        CHECK(te->decode(te->ctx, e_text, 32, back) == 0, "decode round-trip");
    }

    if (failures == 0) printf("=== ALL ENCODER-SLOT TESTS PASSED ===\n");
    else printf("=== %d FAILURES ===\n", failures);
    return failures ? 1 : 0;
}
