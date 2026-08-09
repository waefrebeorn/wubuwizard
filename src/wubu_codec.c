/* wubu_codec.c -- THE UNIVERSAL CODEC (research/067): the Zephyr-HD
 * audio<->image transform in C11.
 *
 * The "visual audio Kodak", ours: STFT -> 5 perceptual bands -> RGB
 * image (magnitude in red, phase sin/cos in green/blue); decode reads
 * the image back and reconstructs the audio with overlap-add ISTFT.
 * With audio-as-image, ONE image encoder reads every modality.
 *
 * C11.
 */
#include "wubu_codec.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

#define FFT_N      256
#define HOP        64
#define N_FREQ     (FFT_N / 2 + 1)

/* ---- radix-2 FFT (ours) ---- */
static void fft_radix2(float *re, float *im, int n, int inverse) {
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float tr = re[i]; re[i] = re[j]; re[j] = tr;
            float ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * 3.14159265358979f / len * (inverse ? -1 : 1);
        float wpr = cosf(ang), wpi = sinf(ang);
        for (int i = 0; i < n; i += len) {
            float wr = 1.0f, wi = 0.0f;
            for (int j = 0; j < len / 2; j++) {
                int a = i + j, b = i + j + len / 2;
                float ur = re[b], ui = im[b];
                float vr = ur * wr - ui * wi;
                float vi = ur * wi + ui * wr;
                re[b] = re[a] - vr; im[b] = im[a] - vi;
                re[a] += vr;        im[a] += vi;
                float nwr = wr * wpr - wi * wpi;
                wi = wr * wpi + wi * wpr;
                wr = nwr;
            }
        }
    }
    if (inverse)
        for (int i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}

/* Hann window */
static void hann(float *w, int n) {
    for (int i = 0; i < n; i++)
        w[i] = 0.5f * (1.0f - cosf(2.0f * 3.14159265358979f * i / (n - 1)));
}

void wubu_codec_band_split(int n_fft, int sample_rate, int *band_bins) {
    static const int cross[4] = {300, 4000, 10000, 16000};
    band_bins[0] = 0;
    for (int b = 0; b < 4; b++)
        band_bins[b + 1] = (int)((long)cross[b] * n_fft / sample_rate);
    band_bins[5] = n_fft / 2 + 1;
    for (int b = 1; b < 6; b++)
        if (band_bins[b] > band_bins[b-1] + 1 &&
            band_bins[b] > n_fft / 2 + 1)
            band_bins[b] = n_fft / 2 + 1;
}

int wubu_codec_audio_to_image(const float *pcm, int n_samples, float *out) {
    if (!pcm || !out || n_samples < FFT_N) return -1;

    float win[FFT_N];
    hann(win, FFT_N);

    int n_cols = WUBU_CODEC_COLS;
    /* the frame stride: spread n_samples over the columns */
    int max_frames = (n_samples - FFT_N) / HOP + 1;
    int stride = max_frames > n_cols ? (max_frames / n_cols) : 1;
    int n_frames = max_frames / stride;
    if (n_frames > n_cols) n_frames = n_cols;

    int band_bins[6];
    wubu_codec_band_split(FFT_N, 16000, band_bins);

    /* per-band value ranges for normalization */
    float mag_max[WUBU_CODEC_BANDS];
    for (int b = 0; b < WUBU_CODEC_BANDS; b++) mag_max[b] = 1e-6f;

    /* pass 1: magnitude maxima per band */
    for (int c = 0; c < n_frames; c++) {
        int f0 = c * stride * HOP;
        float re[FFT_N], im[FFT_N];
        for (int i = 0; i < FFT_N; i++) {
            re[i] = pcm[f0 + i] * win[i];
            im[i] = 0.0f;
        }
        fft_radix2(re, im, FFT_N, 0);
        for (int b = 0; b < WUBU_CODEC_BANDS; b++) {
            float m = 0.0f;
            for (int k = band_bins[b]; k < band_bins[b+1]; k++)
                if (re[k]*re[k] + im[k]*im[k] > m)
                    m = re[k]*re[k] + im[k]*im[k];
            if (m > mag_max[b]) mag_max[b] = m;
        }
    }
    for (int b = 0; b < WUBU_CODEC_BANDS; b++) mag_max[b] = sqrtf(mag_max[b]) + 1e-6f;

    /* pass 2: draw the image. band b -> rows b*16..b*16+15;
     * within a band, row r = bin (r/16 fraction of the band).
     * red = normalized magnitude, green = sin(phase), blue = cos(phase). */
    for (int b = 0; b < WUBU_CODEC_BANDS; b++) {
        int nbins = band_bins[b+1] - band_bins[b];
        if (nbins < 1) continue;
        for (int c = 0; c < n_frames; c++) {
            int f0 = c * stride * HOP;
            float re[FFT_N], im[FFT_N];
            for (int i = 0; i < FFT_N; i++) {
                re[i] = pcm[f0 + i] * win[i];
                im[i] = 0.0f;
            }
            fft_radix2(re, im, FFT_N, 0);
            for (int r = 0; r < 16; r++) {
                int bin = band_bins[b] + (int)((long)r * nbins / 16);
                if (bin >= band_bins[b+1]) bin = band_bins[b+1] - 1;
                float mag = sqrtf(re[bin]*re[bin] + im[bin]*im[bin]);
                float phase = atan2f(im[bin], re[bin]);
                size_t px = ((size_t)(b * 16 + r) * WUBU_CODEC_IMAGE_W + c) * 3;
                out[px]   = mag / mag_max[b];           /* red: magnitude */
                out[px+1] = (sinf(phase) + 1.0f) * 0.5f; /* green: sin(phase) */
                out[px+2] = (cosf(phase) + 1.0f) * 0.5f; /* blue: cos(phase) */
            }
        }
    }
    /* zero-fill unused columns (n_frames < n_cols) */
    for (int c = n_frames; c < n_cols; c++)
        for (int y = 0; y < WUBU_CODEC_IMAGE_H; y++) {
            size_t px = ((size_t)y * WUBU_CODEC_IMAGE_W + c) * 3;
            out[px] = out[px+1] = out[px+2] = 0.0f;
        }

    /* METADATA: store the global magnitude peak in the last pixel's
     * red channel (the Zephyr sidebar trick — the absolute scale must
     * survive the round-trip). peak/10 -> [0,1] for typical peaks. */
    {
        float gpeak = 1e-6f;
        for (int i = 0; i < WUBU_CODEC_IMAGE_H * WUBU_CODEC_IMAGE_W; i++) {
            float m = out[(size_t)i * 3];   /* normalized magnitude */
            if (m > gpeak) gpeak = m;
        }
        /* find the actual peak magnitude (pre-normalization) by
         * scanning the bands' mag_max: the max normalized value is 1.0
         * at the peak bin, so scale = the largest mag_max. */
        float scale = 1e-6f;
        for (int b = 0; b < WUBU_CODEC_BANDS; b++)
            if (mag_max[b] > scale) scale = mag_max[b];
        size_t last = ((size_t)WUBU_CODEC_IMAGE_H * WUBU_CODEC_IMAGE_W - 1) * 3;
        float meta = scale / 10.0f;
        if (meta > 1.0f) meta = 1.0f;
        out[last] = meta;
        (void)gpeak;
    }
    return 0;
}

int wubu_codec_image_to_audio(const float *img, float *out, int out_cap) {
    if (!img || !out || out_cap < FFT_N) return -1;

    int band_bins[6];
    wubu_codec_band_split(FFT_N, 16000, band_bins);

    /* build the STFT frames back: red->mag (max 1), green/blue->phase */
    int n_frames = WUBU_CODEC_COLS;
    int max_frames = (out_cap - FFT_N) / HOP + 1;
    int stride = max_frames > n_frames ? (max_frames / n_frames) : 1;
    if (stride < 1) stride = 1;

    float win[FFT_N];
    hann(win, FFT_N);
    float *accum = (float *)calloc((size_t)out_cap, sizeof(float));
    float *wsum = (float *)calloc((size_t)out_cap, sizeof(float));
    if (!accum || !wsum) { free(accum); free(wsum); return -1; }

    /* read the metadata peak (the Zephyr sidebar trick) */
    float scale = 10.0f;   /* default if metadata absent */
    {
        size_t last = ((size_t)WUBU_CODEC_IMAGE_H * WUBU_CODEC_IMAGE_W - 1) * 3;
        if (img[last] > 1e-6f) scale = img[last] * 10.0f;
    }

    for (int c = 0; c < n_frames; c++) {
        float re[FFT_N], im[FFT_N];
        memset(re, 0, sizeof(re));
        memset(im, 0, sizeof(im));
        for (int b = 0; b < WUBU_CODEC_BANDS; b++) {
            int nbins = band_bins[b+1] - band_bins[b];
            if (nbins < 1) continue;
            for (int r = 0; r < 16; r++) {
                int bin = band_bins[b] + (int)((long)r * nbins / 16);
                if (bin >= band_bins[b+1]) bin = band_bins[b+1] - 1;
                size_t px = ((size_t)(b * 16 + r) * WUBU_CODEC_IMAGE_W + c) * 3;
                float mag = img[px];                    /* [0,1] */
                float sp = img[px+1] * 2.0f - 1.0f;     /* sin(phase) */
                float cp = img[px+2] * 2.0f - 1.0f;     /* cos(phase) */
                float phase = atan2f(sp, cp);
                re[bin] = mag * cosf(phase);
                im[bin] = mag * sinf(phase);
            }
        }
        fft_radix2(re, im, FFT_N, 1);   /* inverse */
        int f0 = c * stride * HOP;
        if (f0 + FFT_N > out_cap) break;
        for (int i = 0; i < FFT_N; i++) {
            accum[f0 + i] += re[i] * win[i];
            wsum[f0 + i] += win[i] * win[i];
        }
    }
    int written = 0;
    for (int i = 0; i < out_cap; i++) {
        float w = wsum[i] > 1e-6f ? wsum[i] : 1.0f;
        out[i] = accum[i] / w * scale;
        if (wsum[i] > 1e-6f) written++;
    }
    free(accum);
    free(wsum);
    return written;
}
