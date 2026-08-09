/* wubu_encoder_impl.c -- OUR OWN encoder (WB06: made, not imported).
 *
 * The single-encoder modality-agnostic base, built from our own C11
 * front-ends — no SigLIP, no external weights:
 *
 *   image -> wubu_imgenc  (ViT patch embedding, CC01)  [native 128-dim]
 *   audio -> wubu_audio   (mel-spectrogram + FFT, CC02) [40 mels] -> a
 *            learned linear head into the shared 128-dim
 *   text  -> deterministic projection of token ids into the shared
 *            128-dim (the real embedding table plugs in at load)
 *
 * Every modality lands in WUBU_ENC_SHARED_DIM. The implementations
 * register into the wubu_encoder slot; consumers probe by name and
 * never see which front-end is behind the seam.
 *
 * C11.
 */
#include "wubu_encoder_impl.h"
#include "wubu_imgenc.h"
#include "wubu_audio.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- shared projection head (audio mels -> shared dim) ---- */

typedef struct {
    float W[WUBU_AUDIO_N_MELS * WUBU_ENC_SHARED_DIM];  /* linear head */
    int init;
} audio_head_t;

static audio_head_t g_audio_head;
static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static float frand(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return 2.0f * ((float)((g_rng >> 40) * 0x1p-24)) - 1.0f;
}

static void audio_head_init(audio_head_t *h) {
    if (h->init) return;
    for (int i = 0; i < WUBU_AUDIO_N_MELS * WUBU_ENC_SHARED_DIM; i++)
        h->W[i] = frand() * 0.1f;
    h->init = 1;
}

/* ---- the image encoder (wubu_imgenc, CC01) ---- */

typedef struct {
    wubu_imgenc_t v;
} image_ctx_t;

static int image_encode(void *ctx, const float *x, size_t n, float *out) {
    image_ctx_t *c = (image_ctx_t *)ctx;
    /* wubu_imgenc wants a 64x64x3 image; it emits N_TOKENS x EMBED_DIM
     * (65 tokens: 64 patches + CLS). The SLOT contract is one
     * embedding of `dim` floats per input — so we return the CLS
     * token (the standard ViT classification readout): the image's
     * one embedding in the shared space. */
    if (n != (size_t)WUBU_IMGENC_IMAGE * WUBU_IMGENC_IMAGE * WUBU_IMGENC_CHANNELS)
        return -1;
    float tokens[WUBU_IMGENC_N_TOKENS * WUBU_IMGENC_EMBED_DIM];
    if (wubu_imgenc_encode(&c->v, x, tokens) != 0) return -1;
    /* the CLS token is the last one (index N_PATCHES) */
    memcpy(out, &tokens[WUBU_IMGENC_N_PATCHES * WUBU_IMGENC_EMBED_DIM],
           (size_t)WUBU_IMGENC_EMBED_DIM * sizeof(float));
    return 0;
}

static void image_free(void *ctx) { free(ctx); }

/* ---- the audio encoder (wubu_audio, CC02) + shared head ---- */

typedef struct {
    wubu_audio_t a;
    audio_head_t *head;
} audio_ctx_t;

static int audio_encode(void *ctx, const float *pcm, size_t n, float *out) {
    audio_ctx_t *c = (audio_ctx_t *)ctx;
    if (n == 0) return -1;
    /* mel spectrogram: [frames][40] */
    float mel[WUBU_AUDIO_MAX_FRAMES * WUBU_AUDIO_N_MELS];
    int frames = wubu_audio_encode(&c->a, pcm, (int)n, mel, WUBU_AUDIO_MAX_FRAMES);
    if (frames < 1) return -1;
    /* shared head: one frame's mels -> the shared dim (the first frame
     * is the encoding; the head is the modality adapter). */
    for (int d = 0; d < WUBU_ENC_SHARED_DIM; d++) {
        float acc = 0.0f;
        for (int m = 0; m < WUBU_AUDIO_N_MELS; m++)
            acc += mel[m] * c->head->W[m * WUBU_ENC_SHARED_DIM + d];
        out[d] = acc;
    }
    return 0;
}

static void audio_free(void *ctx) { free(ctx); }

/* ---- the text encoder (token ids -> shared dim; the real embedding
 * table plugs in at load — the seam is what's proven here) ---- */

typedef struct {
    float proj[WUBU_ENC_SHARED_DIM];
} text_ctx_t;

static int text_encode(void *ctx, const float *token_ids, size_t n, float *out) {
    text_ctx_t *c = (text_ctx_t *)ctx;
    if (n == 0) return -1;
    /* bag-of-token-ids weighted by the projection (deterministic) */
    for (int d = 0; d < WUBU_ENC_SHARED_DIM; d++) {
        float acc = 0.0f;
        for (size_t i = 0; i < n; i++)
            acc += token_ids[i] * c->proj[(d + (int)i) % WUBU_ENC_SHARED_DIM];
        out[d] = acc;
    }
    return 0;
}

static void text_free(void *ctx) { free(ctx); }

/* ---- registration ---- */

static wubu_encoder_t g_img, g_aud, g_txt;

int wubu_encoder_register_ours(void) {
    /* image: our ViT patch embedding (CC01) */
    image_ctx_t *ic = (image_ctx_t *)calloc(1, sizeof(*ic));
    if (!ic) return -1;
    if (wubu_imgenc_init(&ic->v, 20260808u) != 0) { free(ic); return -1; }
    g_img.name = "our-image";
    g_img.modality = WUBU_MODALITY_IMAGE;
    g_img.dim = WUBU_IMGENC_EMBED_DIM;
    g_img.ctx = ic;
    g_img.encode = image_encode;
    g_img.decode = NULL;
    g_img.free = image_free;
    wubu_encoder_register(&g_img);

    /* audio: our mel-spectrogram (CC02) + shared head */
    audio_ctx_t *ac = (audio_ctx_t *)calloc(1, sizeof(*ac));
    if (!ac) return -1;
    if (wubu_audio_init(&ac->a) != 0) { free(ac); return -1; }
    audio_head_init(&g_audio_head);
    ac->head = &g_audio_head;
    g_aud.name = "our-audio";
    g_aud.modality = WUBU_MODALITY_AUDIO;
    g_aud.dim = WUBU_ENC_SHARED_DIM;
    g_aud.ctx = ac;
    g_aud.encode = audio_encode;
    g_aud.decode = NULL;
    g_aud.free = audio_free;
    wubu_encoder_register(&g_aud);

    /* text: deterministic projection (the embedding table plugs in) */
    text_ctx_t *tc = (text_ctx_t *)calloc(1, sizeof(*tc));
    if (!tc) return -1;
    for (int d = 0; d < WUBU_ENC_SHARED_DIM; d++) tc->proj[d] = frand() * 0.1f;
    g_txt.name = "our-text";
    g_txt.modality = WUBU_MODALITY_TEXT;
    g_txt.dim = WUBU_ENC_SHARED_DIM;
    g_txt.ctx = tc;
    g_txt.encode = text_encode;
    g_txt.decode = NULL;
    g_txt.free = text_free;
    wubu_encoder_register(&g_txt);

    return 0;
}
