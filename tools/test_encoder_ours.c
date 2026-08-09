/*
 * test_encoder_ours.c -- WB06 gate: OUR OWN encoder (made, not imported).
 *
 * The single-encoder modality-agnostic base built from our own C11
 * front-ends — no SigLIP, no external weights:
 *
 *   1. OWN FRONT-ENDS: "our-image" runs wubu_imgenc (ViT patch
 *      embedding from scratch, CC01); "our-audio" runs wubu_audio
 *      (mel-spectrogram + real FFT from scratch, CC02); "our-text"
 *      projects token ids deterministically.
 *   2. SHARED SPACE: all three modalities land in the SAME embedding
 *      dim (WUBU_ENC_SHARED_DIM) — the KV-FS "all inputs are encoded"
 *      invariant, with OUR code behind the seam.
 *   3. FINITE + DETERMINISTIC: every modality encodes to finite,
 *      reproducible embeddings (a pure function of the input).
 *   4. MODALITY-AGNOSTIC: the registry probes by modality — the
 *      consumer never knows which front-end answers.
 *
 * Gate: `make test_encoder_ours`.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "wubu_encoder.h"
#include "wubu_encoder_impl.h"
#include "wubu_imgenc.h"
#include "wubu_audio.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)

int main(void)
{
    printf("=== test_encoder_ours (WB06: OUR OWN encoder, made not imported) ===\n");

    if (wubu_encoder_register_ours() != 0) {
        printf("  FAIL: register our encoders\n");
        return 1;
    }

    /* ---- 1. our three front-ends are registered ---- */
    CHECK(wubu_encoder_get("our-image") != NULL, "our-image registered");
    CHECK(wubu_encoder_get("our-audio") != NULL, "our-audio registered");
    CHECK(wubu_encoder_get("our-text") != NULL, "our-text registered");
    CHECK(wubu_encoder_for_modality(WUBU_MODALITY_IMAGE) != NULL,
          "modality probe finds the image encoder");
    CHECK(wubu_encoder_for_modality(WUBU_MODALITY_AUDIO) != NULL,
          "modality probe finds the audio encoder");

    /* ---- 2. image: a synthetic 64x64x3 frame through our ViT ---- */
    {
        float img[WUBU_IMGENC_IMAGE * WUBU_IMGENC_IMAGE * WUBU_IMGENC_CHANNELS];
        for (int i = 0; i < (int)(sizeof(img)/sizeof(img[0])); i++)
            img[i] = (float)((i * 2654435761u) % 1000) / 1000.0f;  /* grayscale-ish */
        float e1[WUBU_IMGENC_EMBED_DIM], e2[WUBU_IMGENC_EMBED_DIM];
        CHECK(wubu_encoder_encode_named("our-image", img,
              (size_t)WUBU_IMGENC_IMAGE*WUBU_IMGENC_IMAGE*WUBU_IMGENC_CHANNELS,
              e1) == 0, "image encodes through our ViT");
        wubu_encoder_encode_named("our-image", img,
              (size_t)WUBU_IMGENC_IMAGE*WUBU_IMGENC_IMAGE*WUBU_IMGENC_CHANNELS, e2);
        int finite = 1, det = 1;
        for (int i = 0; i < WUBU_IMGENC_EMBED_DIM; i++) {
            if (!isfinite(e1[i])) finite = 0;
            if (e1[i] != e2[i]) det = 0;
        }
        CHECK(finite, "image embedding is finite");
        CHECK(det, "image embedding is deterministic");
        printf("  ok: image -> 1 x %d-dim embedding (CLS readout of our ViT, CC01)\n",
               WUBU_IMGENC_EMBED_DIM);
    }

    /* ---- 3. audio: PCM through our mel-spectrogram + shared head ---- */
    {
        float pcm[WUBU_AUDIO_SAMPLE_RATE];  /* 1 second */
        for (int i = 0; i < WUBU_AUDIO_SAMPLE_RATE; i++)
            pcm[i] = 0.1f * sinf(2.0f * 3.14159f * 440.0f * (float)i /
                                 (float)WUBU_AUDIO_SAMPLE_RATE);
        float e1[WUBU_ENC_SHARED_DIM], e2[WUBU_ENC_SHARED_DIM];
        CHECK(wubu_encoder_encode_named("our-audio", pcm,
              (size_t)WUBU_AUDIO_SAMPLE_RATE, e1) == 0,
              "audio encodes through our mel-spectrogram");
        wubu_encoder_encode_named("our-audio", pcm,
              (size_t)WUBU_AUDIO_SAMPLE_RATE, e2);
        int finite = 1, det = 1;
        for (int i = 0; i < WUBU_ENC_SHARED_DIM; i++) {
            if (!isfinite(e1[i])) finite = 0;
            if (e1[i] != e2[i]) det = 0;
        }
        CHECK(finite, "audio embedding is finite");
        CHECK(det, "audio embedding is deterministic");
        printf("  ok: audio (440Hz) -> %d-dim shared (our FFT+mel, CC02)\n",
               WUBU_ENC_SHARED_DIM);
    }

    /* ---- 4. text: token ids -> shared dim ---- */
    {
        float ids[16];
        for (int i = 0; i < 16; i++) ids[i] = (float)(i * 7 + 3);
        float e1[WUBU_ENC_SHARED_DIM], e2[WUBU_ENC_SHARED_DIM];
        CHECK(wubu_encoder_encode_named("our-text", ids, 16, e1) == 0,
              "text encodes");
        wubu_encoder_encode_named("our-text", ids, 16, e2);
        int det = 1;
        for (int i = 0; i < WUBU_ENC_SHARED_DIM; i++)
            if (e1[i] != e2[i]) det = 0;
        CHECK(det, "text embedding is deterministic");
        printf("  ok: text -> %d-dim shared (tokenizer path)\n",
               WUBU_ENC_SHARED_DIM);
    }

    /* ---- 5. shared space: all three in the same dim ---- */
    int img_dim = wubu_encoder_get("our-image")->dim;
    int aud_dim = wubu_encoder_get("our-audio")->dim;
    int txt_dim = wubu_encoder_get("our-text")->dim;
    printf("  ok: shared dims — image=%d audio=%d text=%d\n",
           img_dim, aud_dim, txt_dim);
    CHECK(aud_dim == WUBU_ENC_SHARED_DIM && txt_dim == WUBU_ENC_SHARED_DIM,
          "audio + text land in the shared 128-dim (KV-FS doctrine)");
    CHECK(img_dim == WUBU_ENC_SHARED_DIM,
          "image native dim IS the shared dim (no extra head)");

    if (failures == 0) printf("=== ALL OUR-ENCODER TESTS PASSED ===\n");
    else printf("=== %d FAILURES ===\n", failures);
    return failures ? 1 : 0;
}
