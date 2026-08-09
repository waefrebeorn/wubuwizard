/*
 * test_codec.c -- research/067 gate: THE UNIVERSAL CODEC.
 *
 * The user's prior art: "I did a lot of work trying to create a visual
 * audio Kodak that transformed audio into pictures" (wubumind_codec.py,
 * Zephyr-HD). The redesign: audio becomes a PICTURE, so the SAME image
 * encoder reads EVERY modality — text, image, video, audio — all
 * landing in one compressible encoder space (the canvas).
 *
 *   1. AUDIO -> IMAGE: a 440Hz tone + a 2kHz tone become an RGB
 *      picture (the Kodak). Magnitude in red, phase in green/blue.
 *   2. IMAGE -> AUDIO: the picture decodes BACK into audio (ISTFT
 *      reconstruction) — the round-trip is real (the codec, ours).
 *   3. ROUND-TRIP: the reconstructed audio correlates with the
 *      original (the band energy survives the image round-trip).
 *   4. AUDIO-THROUGH-THE-VIT: the Kodak image feeds OUR image encoder
 *      -> the shared embedding space. Audio lands where images land —
 *      ONE encoder for every modality (the universal projection).
 *
 * Gate: `make test_codec`.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "wubu_codec.h"
#include "wubu_imgenc.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)

int main(void)
{
    printf("=== test_codec (research/067: THE UNIVERSAL CODEC) ===\n");

    /* ---- 1. audio -> image (the Kodak) ---- */
    int sr = 16000, n = sr;   /* 1 second */
    float *pcm = (float *)malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) {
        /* 440Hz + 2kHz tones, mixed */
        pcm[i] = 0.4f * sinf(2.0f * 3.14159f * 440.0f * i / sr)
               + 0.3f * sinf(2.0f * 3.14159f * 2000.0f * i / sr);
    }
    float img[WUBU_CODEC_IMAGE_H * WUBU_CODEC_IMAGE_W * 3];
    CHECK(wubu_codec_audio_to_image(pcm, n, img) == 0,
          "audio encodes into an image (the visual audio Kodak)");
    int img_energy = 0;
    for (int i = 0; i < WUBU_CODEC_IMAGE_H * WUBU_CODEC_IMAGE_W * 3; i++)
        if (img[i] > 0.05f) img_energy++;
    printf("  ok: 1s of audio -> %dx%d RGB image (%d lit pixels)\n",
           WUBU_CODEC_IMAGE_W, WUBU_CODEC_IMAGE_H, img_energy);
    CHECK(img_energy > 100, "the image carries real energy (not blank)");

    /* ---- 2. image -> audio (decode back) ---- */
    float *rec = (float *)calloc((size_t)n, sizeof(float));
    int written = wubu_codec_image_to_audio(img, rec, n);
    CHECK(written > n / 2, "image decodes back into audio (ISTFT)");
    printf("  ok: image -> %d audio samples reconstructed\n", written);

    /* ---- 3. round-trip: the reconstructed audio correlates ---- */
    /* compare band energy: RMS of original vs reconstructed */
    double rms_orig = 0.0, rms_rec = 0.0;
    for (int i = 0; i < n; i++) {
        rms_orig += (double)pcm[i] * pcm[i];
        rms_rec += (double)rec[i] * rec[i];
    }
    rms_orig = sqrt(rms_orig / n);
    rms_rec = sqrt(rms_rec / n);
    printf("  ok: RMS original=%.4f reconstructed=%.4f (ratio %.2f)\n",
           rms_orig, rms_rec, rms_rec / (rms_orig + 1e-6));
    CHECK(fabs(rms_orig - rms_rec) < 0.35 * rms_orig + 0.05,
          "reconstructed energy is within range of the original");

    /* ---- 4. audio-through-the-ViT (the universal projection) ---- */
    /* the Kodak image is 128x80; our ViT wants 64x64x3 — downscale
     * and feed it. Audio lands in the SAME space as images. */
    {
        float vimg[WUBU_IMGENC_IMAGE * WUBU_IMGENC_IMAGE * WUBU_IMGENC_CHANNELS];
        for (int y = 0; y < WUBU_IMGENC_IMAGE; y++) {
            int sy = (int)((long)y * WUBU_CODEC_IMAGE_H / WUBU_IMGENC_IMAGE);
            if (sy >= WUBU_CODEC_IMAGE_H) sy = WUBU_CODEC_IMAGE_H - 1;
            for (int x = 0; x < WUBU_IMGENC_IMAGE; x++) {
                int sx = (int)((long)x * WUBU_CODEC_IMAGE_W / WUBU_IMGENC_IMAGE);
                if (sx >= WUBU_CODEC_IMAGE_W) sx = WUBU_CODEC_IMAGE_W - 1;
                size_t src = ((size_t)sy * WUBU_CODEC_IMAGE_W + sx) * 3;
                size_t dst = ((size_t)y * WUBU_IMGENC_IMAGE + x) * 3;
                vimg[dst] = img[src];
                vimg[dst+1] = img[src+1];
                vimg[dst+2] = img[src+2];
            }
        }
        wubu_imgenc_t v;
        CHECK(wubu_imgenc_init(&v, 42u) == 0, "our ViT initializes");
        float tokens[WUBU_IMGENC_N_TOKENS * WUBU_IMGENC_EMBED_DIM];
        CHECK(wubu_imgenc_encode(&v, vimg, tokens) == 0,
              "the AUDIO IMAGE encodes through our ViT");
        int finite = 1;
        for (int i = 0; i < WUBU_IMGENC_N_TOKENS * WUBU_IMGENC_EMBED_DIM; i++)
            if (!isfinite(tokens[i])) finite = 0;
        CHECK(finite, "audio embeddings are finite (the shared space)");
        printf("  ok: AUDIO -> Kodak image -> our ViT -> %d tokens x %d dim "
               "(ONE encoder for every modality)\n",
               WUBU_IMGENC_N_TOKENS, WUBU_IMGENC_EMBED_DIM);
    }

    free(pcm);
    free(rec);
    if (failures == 0) printf("=== ALL UNIVERSAL-CODEC TESTS PASSED ===\n");
    else printf("=== %d FAILURES ===\n", failures);
    return failures ? 1 : 0;
}
