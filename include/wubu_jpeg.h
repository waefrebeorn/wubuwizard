/* wubu_jpeg.h -- JPEG baseline decoder (ours, Huffman + IDCT from spec).
 *
 * Public format (ITU-T T.81): SOI/APPn/DQT/SOF0/DHT/SOS/EOI, Huffman-
 * coded entropy data, dequantize, 8x8 IDCT, YCbCr->RGB. Decoded from
 * spec — no codec licensing. Also the frame codec for MJPEG video.
 *
 * C11, self-contained (no zlib needed — JPEG is not compressed).
 */
#ifndef WUBU_JPEG_H
#define WUBU_JPEG_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Decode a baseline JPEG into RGB floats [0,1], row-major (w*h*3).
 * Returns 0 on success. Caller frees *out with free(). */
int wubu_jpeg_decode(const unsigned char *data, size_t n,
                     float **out, int *w, int *h);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_JPEG_H */
