/* wubu_pdf.c -- PDF text extraction (ours: object scan + FlateDecode).
 *
 * PDF = objects; stream objects carry compressed content streams.
 * We scan the file for "stream"/"endstream" pairs, inflate FlateDecode
 * streams, and pull text from the content operators:
 *   (string) Tj      show text
 *   [(a) (b)] TJ     show text array
 *   (string) '       show + move
 * Parenthesized strings with escapes (\n \r \t \b \f \( \) \\ \ooo).
 * Written from spec (ISO 32000) — no licensing.
 *
 * C11.
 */
#include "wubu_pdf.h"

#include <stdlib.h>
#include <string.h>
#include <zlib.h>

/* inflate a raw (FlateDecode) stream */
static unsigned char *inflate_raw(const unsigned char *src, size_t n,
                                  size_t *outlen) {
    /* growable output */
    size_t cap = n * 4 + 1024;
    unsigned char *out = (unsigned char *)malloc(cap);
    if (!out) return NULL;
    z_stream zs;
    memset(&zs, 0, sizeof(zs));
    if (inflateInit(&zs) != Z_OK) { free(out); return NULL; }
    zs.next_in = (Bytef *)src;
    zs.avail_in = (uInt)n;
    zs.next_out = out;
    zs.avail_out = (uInt)cap;
    int zr = inflate(&zs, Z_FINISH);
    while (zr == Z_OK && zs.avail_out == 0) {
        size_t used = zs.next_out - out;
        size_t ncap = cap * 2;
        unsigned char *nout = (unsigned char *)realloc(out, ncap);
        if (!nout) { inflateEnd(&zs); free(out); return NULL; }
        out = nout;
        cap = ncap;
        zs.next_out = out + used;
        zs.avail_out = (uInt)(cap - used);
        zr = inflate(&zs, Z_FINISH);
    }
    inflateEnd(&zs);
    if (zr != Z_STREAM_END) { free(out); return NULL; }
    *outlen = (size_t)(zs.total_out);
    return out;
}

/* decode a PDF parenthesized string into out; returns chars written */
static size_t pdf_string(const unsigned char *s, size_t n, char *out,
                         size_t outcap) {
    size_t k = 0, i = 0;
    while (i < n && k + 1 < outcap) {
        unsigned char c = s[i];
        if (c == '\\' && i + 1 < n) {
            unsigned char e = s[i+1];
            switch (e) {
                case 'n': out[k++] = '\n'; i += 2; break;
                case 'r': out[k++] = '\r'; i += 2; break;
                case 't': out[k++] = '\t'; i += 2; break;
                case 'b': out[k++] = '\b'; i += 2; break;
                case 'f': out[k++] = '\f'; i += 2; break;
                case '(': out[k++] = '('; i += 2; break;
                case ')': out[k++] = ')'; i += 2; break;
                case '\\': out[k++] = '\\'; i += 2; break;
                default:
                    if (e >= '0' && e <= '7' && i + 3 < n) {
                        int v = (e - '0') * 64 + (s[i+2]-'0') * 8 + (s[i+3]-'0');
                        out[k++] = (char)(v & 0xFF);
                        i += 4;
                    } else { out[k++] = (char)e; i += 2; }
            }
        } else if (c == ')') {
            break;
        } else if (c == '(') {
            /* nested — keep going (rare) */
            i++;
        } else {
            out[k++] = (char)c;
            i++;
        }
    }
    return k;
}

/* extract text operators from one content stream */
static void content_text(const unsigned char *cs, size_t n,
                         char *out, size_t outcap, size_t *outlen) {
    size_t i = 0;
    while (i < n && *outlen + 4 < outcap) {
        if (cs[i] == '(') {
            /* find the closing paren (respecting escapes) */
            size_t depth = 1, j = i + 1;
            while (j < n && depth > 0) {
                if (cs[j] == '\\') { j += 2; continue; }
                if (cs[j] == '(') depth++;
                if (cs[j] == ')') depth--;
                j++;
            }
            size_t slen = (j > 0 && j <= n) ? j - i - 2 : 0;
            if (depth == 0 && slen > 0) {
                /* look ahead for Tj or ' (show-text operators) */
                size_t k = j;
                while (k < n && (cs[k] == ' ' || cs[k] == '\n' ||
                                 cs[k] == '\r' || cs[k] == '\t')) k++;
                if (k + 2 <= n && cs[k] == 'T' && cs[k+1] == 'j') {
                    size_t w = pdf_string(cs + i + 1, slen,
                                          out + *outlen, outcap - *outlen);
                    *outlen += w;
                    i = k + 2;
                    continue;
                }
                if (k < n && cs[k] == '\'') {
                    size_t w = pdf_string(cs + i + 1, slen,
                                          out + *outlen, outcap - *outlen);
                    *outlen += w;
                    if (*outlen + 2 < outcap) { out[(*outlen)++] = '\n'; }
                    i = k + 1;
                    continue;
                }
            }
        } else if (cs[i] == '[') {
            /* a TJ array: [ (a) -10 (b) ] TJ */
            size_t j = i + 1, depth = 1;
            size_t str_start = (size_t)-1;
            while (j < n && depth > 0) {
                if (cs[j] == '[') depth++;
                if (cs[j] == ']') depth--;
                j++;
            }
            /* scan for TJ after the bracket */
            size_t k = j;
            while (k < n && (cs[k] == ' ' || cs[k] == '\n' ||
                             cs[k] == '\r' || cs[k] == '\t')) k++;
            if (k + 2 <= n && cs[k] == 'T' && cs[k+1] == 'J') {
                /* extract all (strings) inside the array */
                size_t p = i + 1;
                (void)str_start;
                while (p + 1 < j) {
                    if (cs[p] == '(') {
                        size_t d2 = 1, q = p + 1;
                        while (q < n && d2 > 0) {
                            if (cs[q] == '\\') { q += 2; continue; }
                            if (cs[q] == '(') d2++;
                            if (cs[q] == ')') d2--;
                            q++;
                        }
                        size_t sl = q - p - 2;
                        size_t w = pdf_string(cs + p + 1, sl,
                                              out + *outlen, outcap - *outlen);
                        *outlen += w;
                        p = q;
                    } else p++;
                }
                if (*outlen + 2 < outcap) { out[(*outlen)++] = ' '; }
                i = k + 2;
                continue;
            }
        }
        i++;
    }
}

char *wubu_pdf_extract_text(const unsigned char *data, size_t n) {
    if (!data || n < 8) return NULL;
    if (memcmp(data, "%PDF-", 5) != 0) return NULL;

    size_t cap = 128 * 1024;
    char *text = (char *)calloc(cap, 1);
    if (!text) return NULL;
    size_t outlen = 0;

    /* scan for stream/endstream pairs */
    size_t i = 0;
    while (i + 6 <= n && outlen + 16 < cap) {
        /* find "stream" (end of line before it) */
        if (i + 7 <= n && memcmp(data + i, "stream", 6) == 0 &&
            (i == 0 || data[i-1] == '\n' || data[i-1] == '\r')) {
            /* stream data starts after the EOL */
            size_t sd = i + 6;
            if (sd < n && data[sd] == '\r') sd++;
            if (sd < n && data[sd] == '\n') sd++;
            /* find endstream */
            size_t e = sd;
            while (e + 9 <= n && memcmp(data + e, "endstream", 9) != 0) e++;
            if (e + 9 > n) break;
            /* trim trailing EOL before endstream */
            size_t send = e;
            while (send > sd && (data[send-1] == '\n' || data[send-1] == '\r'))
                send--;
            if (send > sd) {
                /* try raw inflate first (FlateDecode), else literal */
                size_t ilen = 0;
                unsigned char *plain = inflate_raw(data + sd, send - sd, &ilen);
                if (plain) {
                    content_text(plain, ilen, text, cap, &outlen);
                    free(plain);
                } else {
                    /* maybe a literal (uncompressed) stream */
                    content_text(data + sd, send - sd, text, cap, &outlen);
                }
            }
            i = e + 9;
            continue;
        }
        i++;
    }
    if (outlen == 0) { free(text); return NULL; }
    text[outlen] = '\0';
    return text;
}
