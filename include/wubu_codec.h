/* wubu_codec.h -- THE UNIVERSAL CODEC (research/067): the Zephyr-HD
 * audio<->image transform in C11.
 *
 * The user's prior art (wubumind_codec.py — "the visual audio Kodak"):
 * audio transformed INTO a picture, reversibly. 5 perceptual bands
 * (Bass/Mids/Presence/Treble/Harmonics) map to image rows, magnitude
 * -> red channel, phase sin/cos -> green/blue. The decoder reads the
 * image back and reconstructs the audio (ISTFT).
 *
 * Why it matters: with audio-as-image, the SAME image encoder (our
 * ViT) reads EVERY modality — text chunks, images, video frames, AND
 * audio all land in ONE shared space (the universal projection), and
 * the canvas (wubu_canvas) is the compressible field.
 *
 * C11, self-contained (radix-2 FFT, ours).
 */
#ifndef WUBU_CODEC_H
#define WUBU_CODEC_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The Kodak image size: 5 bands x N columns x 3 channels (RGB). */
#define WUBU_CODEC_BANDS   5
#define WUBU_CODEC_COLS    128      /* time columns */
#define WUBU_CODEC_IMAGE_H (WUBU_CODEC_BANDS * 16)  /* 80 */
#define WUBU_CODEC_IMAGE_W (WUBU_CODEC_COLS)         /* 128 */

/* Encode audio PCM into an RGB image (the Kodak). `pcm` is n_samples
 * floats; out is IMAGE_H*IMAGE_W*3 floats in [0,1]. Returns 0. */
int wubu_codec_audio_to_image(const float *pcm, int n_samples, float *out);

/* Decode the image back into audio (ISTFT reconstruction). out holds
 * n_samples floats (the caller sizes it; returns samples written). */
int wubu_codec_image_to_audio(const float *img, float *out, int out_cap);

/* The 5-band frequency split (used by both directions). */
void wubu_codec_band_split(int n_fft, int sample_rate,
                           int *band_bins);   /* 6 boundaries */

#ifdef __cplusplus
}
#endif

#endif /* WUBU_CODEC_H */
