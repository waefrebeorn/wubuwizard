/* wubu_jpeg.c -- JPEG baseline decoder (ours: Huffman + IDCT from spec).
 *
 * ITU-T T.81 baseline sequential, decoded from the public spec. This is
 * the classic compact decoder: marker parse, quant tables, Huffman
 * tables, bit reader, MCU loop, 8x8 IDCT, YCbCr->RGB. No codec
 * licensing — the format is public and we wrote every line.
 *
 * C11.
 */
#include "wubu_jpeg.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_COMP 4

typedef struct {
    int id, h, v, tq;
    int dc_pred;
} comp_t;

typedef struct {
    uint8_t bits[17];
    uint16_t huffval[256];
    int nvals;
    /* fast lookup: max code length 16 -> build canonical tree walk */
    int mincode[17], maxcode[17], valptr[17];
} huff_t;

typedef struct {
    int data[64];
} quant_t;

typedef struct {
    const unsigned char *buf;
    size_t len, pos;
    uint64_t bitbuf;
    int bitcnt;
} bitreader_t;

static int rd_be16(const unsigned char *p) { return (p[0] << 8) | p[1]; }

/* zigzag reorder (T.81 Annex F.1.2.2) */
static const int kZig[64] = {
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
};
static int zigzag(int k) { return kZig[k & 63]; }

/* ---- bit reader ---- */
static int br_fill(bitreader_t *br) {
    while (br->bitcnt <= 24 && br->pos < br->len) {
        br->bitbuf = (br->bitbuf << 8) | br->buf[br->pos++];
        br->bitcnt += 8;
    }
    return br->bitcnt > 0;
}
static int br_getbits(bitreader_t *br, int n) {
    if (!br_fill(br)) return 0;
    int v = (int)((br->bitbuf >> (br->bitcnt - n)) & ((1u << n) - 1));
    br->bitcnt -= n;
    return v;
}
static int br_getbyte_align(bitreader_t *br) {
    br->bitcnt = 0;
    if (br->pos < br->len) return br->buf[br->pos++];
    return 0;
}
/* decode one Huffman symbol; returns -1 on error */
static int br_decode_huff(bitreader_t *br, const huff_t *h) {
    if (!br_fill(br)) return -1;
    int code = 0, k = 0;
    for (int i = 1; i <= 16; i++) {
        code = (code << 1) | br_getbits(br, 1);
        if (code <= h->maxcode[i]) {
            int idx = h->valptr[i] + (code - h->mincode[i]);
            if (idx >= 0 && idx < h->nvals) { k = h->huffval[idx]; break; }
            return -1;
        }
    }
    return k;
}
static int br_receive(bitreader_t *br, int ssss) {
    if (ssss == 0) return 0;
    int v = br_getbits(br, ssss);
    if (v < (1 << (ssss - 1))) v -= (1 << ssss) - 1;
    return v;
}

/* ---- IDCT (8x8, separable, fixed point) ---- */
static void idct_8x8(int *b) {
    int tmp[64];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            double s = 0.0;
            for (int k = 0; k < 8; k++) {
                double c = (k == 0) ? 0.7071067811865476 : 1.0;
                s += c * b[i * 8 + k] *
                     cos((2.0 * j + 1.0) * k * 3.141592653589793 / 16.0);
            }
            tmp[i * 8 + j] = (int)llround(s * 0.3535533905932738);
        }
    }
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            double s = 0.0;
            for (int k = 0; k < 8; k++) {
                double c = (k == 0) ? 0.7071067811865476 : 1.0;
                s += c * tmp[k * 8 + j] *
                     cos((2.0 * i + 1.0) * k * 3.141592653589793 / 16.0);
            }
            int v = (int)llround(s * 0.3535533905932738);
            if (v < -128) v = -128;
            if (v > 127) v = 127;
            b[i * 8 + j] = v;
        }
    }
}

int wubu_jpeg_decode(const unsigned char *data, size_t n,
                     float **out, int *w, int *h) {
    if (!data || !out || !w || !h || n < 4) return -1;
    if (data[0] != 0xFF || data[1] != 0xD8) return -1;   /* SOI */

    quant_t qtab[4];
    huff_t hdc[4], hac[4];
    memset(qtab, 0, sizeof(qtab));
    memset(hdc, 0, sizeof(hdc));
    memset(hac, 0, sizeof(hac));
    int have_q[4] = {0}, have_dc[4] = {0}, have_ac[4] = {0};
    comp_t comps[MAX_COMP];
    int ncomp = 0, width = 0, height = 0;
    int scan_comp[4], scan_ncomp = 0;

    size_t pos = 2;
    while (pos + 2 <= n) {
        if (data[pos] != 0xFF) { pos++; continue; }
        int marker = data[pos+1];
        if (marker == 0xD9 || marker == 0xDA) break;   /* EOI / SOS */
        if (marker >= 0xD0 && marker <= 0xD7) { pos += 2; continue; }
        if (pos + 4 > n) break;
        int len = rd_be16(data + pos + 2);
        if (len < 2) break;
        const unsigned char *seg = data + pos + 4;
        int seglen = len - 2;

        if (marker == 0xDB) {               /* DQT */
            int i = 0;
            while (i < seglen) {
                int pq_tq = seg[i];
                int tq = pq_tq & 15;
                if (tq >= 4) return -1;
                int *dst = (int *)qtab[tq].data;
                int p = 0;
                i++;
                for (int k = 0; k < 64; k++) {
                    if (pq_tq & 0xF0) {     /* 16-bit */
                        if (i + 2 > seglen) return -1;
                        dst[k] = (seg[i] << 8) | seg[i+1];
                        i += 2;
                    } else {
                        if (i + 1 > seglen) return -1;
                        dst[k] = seg[i++];
                    }
                }
                have_q[tq] = 1;
                (void)p;
            }
        } else if (marker == 0xC0) {        /* SOF0: precision(1),
                                            * height(2), width(2),
                                            * ncomp(1), then comps */
            height = (seg[1] << 8) | seg[2];
            width = (seg[3] << 8) | seg[4];
            ncomp = seg[5];
            if (ncomp > MAX_COMP || width < 1 || height < 1) return -1;
            for (int c = 0; c < ncomp; c++) {
                comps[c].id = seg[6 + c * 3];
                comps[c].h = seg[7 + c * 3] >> 4;
                comps[c].v = seg[7 + c * 3] & 15;
                comps[c].tq = seg[8 + c * 3];
                comps[c].dc_pred = 0;
            }
        } else if (marker == 0xC4) {        /* DHT */
            int i = 0;
            while (i < seglen) {
                int tc_th = seg[i];
                int tc = tc_th >> 4, th = tc_th & 15;
                huff_t *h = tc == 0 ? &hdc[th] : &hac[th];
                memset(h->bits, 0, sizeof(h->bits));
                memset(h->huffval, 0, sizeof(h->huffval));
                int cnt = 0;
                i++;
                for (int k = 1; k <= 16; k++) { h->bits[k] = seg[i++]; cnt += h->bits[k]; }
                for (int k = 0; k < cnt; k++) h->huffval[k] = seg[i++];
                h->nvals = cnt;
                /* build canonical code lookup */
                int code = 0, k2 = 0;
                h->mincode[1] = 0; h->maxcode[1] = -1; h->valptr[1] = 0;
                for (int l = 1; l <= 16; l++) {
                    if (h->bits[l] == 0) {
                        h->mincode[l] = -1;
                        h->maxcode[l] = -1;
                    } else {
                        h->valptr[l] = k2;
                        h->mincode[l] = code;
                        code += h->bits[l];
                        h->maxcode[l] = code - 1;
                        k2 += h->bits[l];
                    }
                    code <<= 1;
                    if (l < 16) {
                        h->mincode[l+1] = 0; h->maxcode[l+1] = -1; h->valptr[l+1] = 0;
                    }
                }
                if (tc == 0) have_dc[th] = 1; else have_ac[th] = 1;
            }
        }
        pos += 4 + seglen;
    }

    /* SOS is the next marker after the loop break — re-scan from pos */
    /* (the loop above breaks at SOS; find it) */
    size_t sos_pos = pos;
    while (sos_pos + 2 <= n) {
        if (data[sos_pos] == 0xFF && data[sos_pos+1] == 0xDA) break;
        if (data[sos_pos] == 0xFF && data[sos_pos+1] == 0xD9) return -1;
        if (data[sos_pos] != 0xFF) { sos_pos++; continue; }
        int m2 = data[sos_pos+1];
        if (m2 >= 0xD0 && m2 <= 0xD7) { sos_pos += 2; continue; }
        if (sos_pos + 4 > n) return -1;
        sos_pos += 4 + rd_be16(data + sos_pos + 2);
    }
    if (sos_pos + 4 > n) return -1;
    int slen = rd_be16(data + sos_pos + 2);
    const unsigned char *sseg = data + sos_pos + 4;
    scan_ncomp = sseg[0];
    for (int c = 0; c < scan_ncomp; c++) {
        scan_comp[c] = sseg[1 + c * 2];
        /* dc/ac selector at sseg[2+c*2]: we use component tables */
    }
    /* entropy data starts after the SOS segment: marker(2) + len(2) +
     * (slen-2) payload = sos_pos + slen + 2 */
    bitreader_t br;
    br.buf = data; br.len = n; br.pos = sos_pos + (size_t)slen + 2;
    br.bitbuf = 0; br.bitcnt = 0;

    if (ncomp < 1 || width < 1 || height < 1) return -1;
    int mcu_w = comps[0].h * 8, mcu_h = comps[0].v * 8;
    int mcus_x = (width + mcu_w - 1) / mcu_w;
    int mcus_y = (height + mcu_h - 1) / mcu_h;

    float *rgb = (float *)calloc((size_t)width * height * 3, sizeof(float));
    if (!rgb) return -1;

    /* per-component block buffers + MCU accumulation */
    int block[64];
    int Y[8][8], Cb[8][8], Cr[8][8];    /* 4:2:0 assumed for 3-comp */

    for (int my = 0; my < mcus_y; my++) {
        for (int mx = 0; mx < mcus_x; mx++) {
            /* decode each component's block(s); handle 4:2:0 3-comp and
             * gray 1-comp; general h/v with block layout */
            if (ncomp == 3) {
                /* Y: h*v blocks (typically 1x1 for 4:2:0) */
                for (int by = 0; by < comps[0].v; by++)
                    for (int bx = 0; bx < comps[0].h; bx++) {
                        /* decode block */
                        for (int k = 0; k < 64; k++) block[k] = 0;
                        int rs = br_decode_huff(&br, &hdc[comps[0].tq]);
                        if (rs < 0) { free(rgb); return -1; }
                        int diff = br_receive(&br, rs);
                        comps[0].dc_pred += diff;
                        block[0] = comps[0].dc_pred * (int)qtab[comps[0].tq].data[0];
                        int k = 1;
                        while (k < 64) {
                            int r = br_decode_huff(&br, &hac[comps[0].tq]);
                            if (r < 0) { free(rgb); return -1; }
                            int rrrr = r >> 4, ssss = r & 15;
                            if (ssss == 0) {
                                if (rrrr == 15) k += 16;
                                else break;
                            } else {
                                k += rrrr;
                                if (k >= 64) break;
                                block[zigzag(k)] = br_receive(&br, ssss) *
                                    (int)qtab[comps[0].tq].data[k];
                                k++;
                            }
                        }
                        /* copy into Y at block position */
                        int ox = (mx * comps[0].h + bx) * 8, oy = (my * comps[0].v + by) * 8;
                        idct_8x8(block);
                        for (int yy = 0; yy < 8; yy++)
                            for (int xx = 0; xx < 8; xx++)
                                if (oy + yy < height && ox + xx < width)
                                    Y[yy][xx] = block[yy*8+xx];
                        comps[0].dc_pred = comps[0].dc_pred;
                    }
                /* Cb, Cr (4:2:0: one block each) */
                for (int ci = 1; ci < 3; ci++) {
                    for (int k = 0; k < 64; k++) block[k] = 0;
                    int rs = br_decode_huff(&br, &hdc[comps[ci].tq]);
                    if (rs < 0) { free(rgb); return -1; }
                    comps[ci].dc_pred += br_receive(&br, rs);
                    block[0] = comps[ci].dc_pred * (int)qtab[comps[ci].tq].data[0];
                    int k = 1;
                    while (k < 64) {
                        int r = br_decode_huff(&br, &hac[comps[ci].tq]);
                        if (r < 0) { free(rgb); return -1; }
                        int rrrr = r >> 4, ssss = r & 15;
                        if (ssss == 0) {
                            if (rrrr == 15) k += 16;
                            else break;
                        } else {
                            k += rrrr;
                            if (k >= 64) break;
                            block[zigzag(k)] = br_receive(&br, ssss) *
                                (int)qtab[comps[ci].tq].data[k];
                            k++;
                        }
                    }
                    idct_8x8(block);
                    for (int yy = 0; yy < 8; yy++)
                        for (int xx = 0; xx < 8; xx++) {
                            if (ci == 1) Cb[yy][xx] = block[yy*8+xx];
                            else Cr[yy][xx] = block[yy*8+xx];
                        }
                }
                /* upsample 4:2:0 + YCbCr->RGB */
                for (int yy = 0; yy < 8; yy++)
                    for (int xx = 0; xx < 8; xx++) {
                        int px = mx * 8 + xx, py = my * 8 + yy;
                        if (px >= width || py >= height) continue;
                        int yv = Y[yy][xx] + 128;
                        int cb = Cb[yy/2][xx/2] + 128;
                        int cr = Cr[yy/2][xx/2] + 128;
                        int r = yv + 1.402f * (cr - 128);
                        int g = yv - 0.344136f * (cb - 128) - 0.714136f * (cr - 128);
                        int b = yv + 1.772f * (cb - 128);
                        if (r < 0) r = 0; if (r > 255) r = 255;
                        if (g < 0) g = 0; if (g > 255) g = 255;
                        if (b < 0) b = 0; if (b > 255) b = 255;
                        size_t o = ((size_t)py * width + px) * 3;
                        rgb[o] = r / 255.0f;
                        rgb[o+1] = g / 255.0f;
                        rgb[o+2] = b / 255.0f;
                    }
            } else if (ncomp == 1) {
                /* grayscale: one block per MCU */
                for (int k = 0; k < 64; k++) block[k] = 0;
                int rs = br_decode_huff(&br, &hdc[comps[0].tq]);
                if (rs < 0) { free(rgb); return -1; }
                comps[0].dc_pred += br_receive(&br, rs);
                block[0] = comps[0].dc_pred * (int)qtab[comps[0].tq].data[0];
                int k = 1;
                while (k < 64) {
                    int r = br_decode_huff(&br, &hac[comps[0].tq]);
                    if (r < 0) { free(rgb); return -1; }
                    int rrrr = r >> 4, ssss = r & 15;
                    if (ssss == 0) {
                        if (rrrr == 15) k += 16;
                        else break;
                    } else {
                        k += rrrr;
                        if (k >= 64) break;
                        block[zigzag(k)] = br_receive(&br, ssss) *
                            (int)qtab[comps[0].tq].data[k];
                        k++;
                    }
                }
                idct_8x8(block);
                for (int yy = 0; yy < 8; yy++)
                    for (int xx = 0; xx < 8; xx++) {
                        int px = mx * 8 + xx, py = my * 8 + yy;
                        if (px >= width || py >= height) continue;
                        int v = block[yy*8+xx] + 128;
                        if (v < 0) v = 0; if (v > 255) v = 255;
                        size_t o = ((size_t)py * width + px) * 3;
                        rgb[o] = rgb[o+1] = rgb[o+2] = v / 255.0f;
                    }
            } else {
                free(rgb); return -1;   /* 4-comp CMYK: later */
            }
        }
    }
    *out = rgb;
    *w = width;
    *h = height;
    return 0;
}
