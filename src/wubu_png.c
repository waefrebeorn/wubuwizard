/* wubu_png.c -- PNG decoder (ours, zlib inflate + filter reconstruction).
 *
 * Public format (RFC 2083), decoded from spec — no codec licensing.
 * zlib is a system library; everything else is ours.
 *
 *   signature: 89 50 4E 47 0D 0A 1A 0A
 *   chunks:    IHDR (w,h,bitdepth,colortype,...), IDAT (zlib stream),
 *              IEND. Filters per scanline: 0=None 1=Sub 2=Up
 *              3=Average 4=Paeth.
 *   colortypes: 0 gray, 2 RGB, 3 palette, 4 gray+alpha, 6 RGBA.
 *   We decode 2/6 (RGB/RGBA) and 0/4 (gray/gray+alpha); palette (3)
 *   via PLTE. Output RGB floats [0,1].
 *
 * C11.
 */
#include "wubu_png.h"

#include <stdlib.h>
#include <string.h>
#include <zlib.h>

static uint32_t rd_be32(const unsigned char *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

static int paeth(int a, int b, int c) {
    int p = a + b - c;
    int pa = abs(p - a), pb = abs(p - b), pc = abs(p - c);
    if (pa <= pb && pa <= pc) return a;
    if (pb <= pc) return b;
    return c;
}

int wubu_png_decode(const unsigned char *data, size_t n,
                    float **out, int *w, int *h) {
    if (!data || !out || !w || !h || n < 33) return -1;
    static const unsigned char sig[8] = {0x89, 0x50, 0x4E, 0x47,
                                         0x0D, 0x0A, 0x1A, 0x0A};
    if (memcmp(data, sig, 8) != 0) return -1;

    size_t pos = 8;
    int width = 0, height = 0, bitdepth = 0, colortype = 0;
    unsigned char *palette = NULL;   /* 3 bytes per entry */
    int palette_entries = 0;
    size_t idat_off = 0, idat_len = 0;
    int have_idat = 0;

    while (pos + 12 <= n) {
        uint32_t len = rd_be32(data + pos);
        uint32_t type = rd_be32(data + pos + 4);
        size_t chunk_start = pos + 8;
        if (chunk_start + len > n) break;
        const unsigned char *cdata = data + chunk_start;

        if (type == 0x49484452u && len >= 13) {       /* IHDR */
            width = (int)rd_be32(cdata);
            height = (int)rd_be32(cdata + 4);
            bitdepth = cdata[8];
            colortype = cdata[9];
            if (width < 1 || height < 1 || width > 16384 || height > 16384)
                return -1;
            if (bitdepth != 8) return -1;             /* 8-bit only */
            if (colortype != 0 && colortype != 2 &&
                colortype != 3 && colortype != 4 && colortype != 6)
                return -1;
        } else if (type == 0x504C5445u) {             /* PLTE */
            palette = (unsigned char *)malloc(len);
            if (!palette) return -1;
            memcpy(palette, cdata, len);
            palette_entries = (int)(len / 3);
        } else if (type == 0x49444154u) {             /* IDAT */
            if (!have_idat) { idat_off = chunk_start; idat_len = len; }
            else { idat_len += len; }                 /* concatenated */
            have_idat = 1;
        } else if (type == 0x49454E44u) {             /* IEND */
            break;
        }
        pos = chunk_start + len + 4;                  /* + CRC */
    }
    if (!have_idat || width < 1) { free(palette); return -1; }

    /* inflate the concatenated IDAT stream */
    z_stream zs;
    memset(&zs, 0, sizeof(zs));
    if (inflateInit(&zs) != Z_OK) { free(palette); return -1; }
    size_t raw_cap = (size_t)(width * 4 + 1) * height + 64;
    unsigned char *raw = (unsigned char *)malloc(raw_cap);
    if (!raw) { inflateEnd(&zs); free(palette); return -1; }
    zs.next_in = (Bytef *)(data + idat_off);
    zs.avail_in = (uInt)idat_len;
    zs.next_out = raw;
    zs.avail_out = (uInt)raw_cap;
    int zr = inflate(&zs, Z_FINISH);
    size_t raw_len = raw_cap - zs.avail_out;
    inflateEnd(&zs);
    if (zr != Z_STREAM_END) { free(raw); free(palette); return -1; }

    /* channels per pixel */
    int chans = 3;   /* RGB default */
    if (colortype == 0 || colortype == 4) chans = 1;
    if (colortype == 4 || colortype == 6) chans = 4;
    if (colortype == 3) chans = 1;   /* palette indices */

    int stride = width * chans;
    size_t expect = (size_t)(stride + 1) * height;
    if (raw_len < expect) { free(raw); free(palette); return -1; }

    /* unfilter into a clean RGB buffer */
    unsigned char *px = (unsigned char *)malloc((size_t)width * height * 3);
    if (!px) { free(raw); free(palette); return -1; }
    const unsigned char *prev = NULL;
    for (int y = 0; y < height; y++) {
        const unsigned char *row = raw + (size_t)y * (stride + 1);
        unsigned char filt = row[0];
        const unsigned char *src = row + 1;
        unsigned char *dst = px + (size_t)y * width * 3;
        unsigned char *prevrow = (unsigned char *)prev;
        for (int x = 0; x < stride; x++) {
            int a = (x >= chans) ? dst[x - chans] : 0;
            int b = prevrow ? prevrow[x] : 0;
            int c = (prevrow && x >= chans) ? prevrow[x - chans] : 0;
            int v = src[x];
            switch (filt) {
                case 0: break;
                case 1: v += a; break;
                case 2: v += b; break;
                case 3: v += (a + b) / 2; break;
                case 4: v += paeth(a, b, c); break;
                default: free(px); free(raw); free(palette); return -1;
            }
            dst[x] = (unsigned char)(v & 0xFF);
        }
        prev = (const unsigned char *)dst;
    }
    free(raw);

    /* convert to RGB floats [0,1] */
    float *rgb = (float *)malloc((size_t)width * height * 3 * sizeof(float));
    if (!rgb) { free(px); free(palette); return -1; }
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            size_t pi = (size_t)y * width * 3 + (size_t)x * 3;
            size_t si = (size_t)y * stride + (size_t)x * chans;
            unsigned char r, g, b;
            if (colortype == 2) {           /* RGB */
                r = px[si]; g = px[si+1]; b = px[si+2];
            } else if (colortype == 6) {    /* RGBA */
                r = px[si]; g = px[si+1]; b = px[si+2];
            } else if (colortype == 0) {    /* gray */
                r = g = b = px[si];
            } else if (colortype == 4) {    /* gray+alpha */
                r = g = b = px[si];
            } else {                        /* palette */
                int idx = px[si];
                if (idx >= 0 && idx < palette_entries) {
                    r = palette[idx*3]; g = palette[idx*3+1]; b = palette[idx*3+2];
                } else { r = g = b = 0; }
            }
            rgb[pi] = r / 255.0f;
            rgb[pi+1] = g / 255.0f;
            rgb[pi+2] = b / 255.0f;
        }
    }
    free(px);
    free(palette);
    *out = rgb;
    *w = width;
    *h = height;
    return 0;
}
