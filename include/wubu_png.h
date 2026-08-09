/* wubu_png.h -- PNG decoder (ours, zlib inflate + filter reconstruction).
 *
 * The user's directive: "we will have to use whatever containers have
 * information available because I am not paying for the licensing of an
 * encoder... We need to ingest all file types."
 *
 * PNG is a public format (RFC 2083): 8-byte signature, IHDR/PLTE/IDAT/
 * IEND chunks, zlib-compressed scanlines with per-row filters
 * (None/Sub/Up/Average/Paeth). We decode it OURSELVES on top of zlib's
 * inflate (a system library, not a paid codec). Output: RGB float
 * pixels in [0,1] — the exact input our ViT image encoder eats.
 *
 * C11, self-contained (zlib only).
 */
#ifndef WUBU_PNG_H
#define WUBU_PNG_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Decode a PNG file into RGB floats [0,1], row-major (w*h*3).
 * Returns 0 on success. Caller frees *out with free(). */
int wubu_png_decode(const unsigned char *data, size_t n,
                    float **out, int *w, int *h);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_PNG_H */
