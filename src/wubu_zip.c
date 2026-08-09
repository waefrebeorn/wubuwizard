/* wubu_zip.c -- ZIP reader (ours: central directory + zlib inflate).
 *
 * Public container (PKWARE APPNOTE): EOCD -> central directory ->
 * entries (name, sizes, method) -> per-entry local header for data
 * offset -> inflate with zlib. Written from spec — no licensing.
 *
 * C11.
 */
#include "wubu_zip.h"

#include <stdlib.h>
#include <string.h>
#include <zlib.h>

struct wubu_zip {
    wubu_zip_entry_t *entries;
    size_t count;
};

static uint32_t rd_le32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint16_t rd_le16(const unsigned char *p) {
    return (uint16_t)(p[0] | (p[1] << 8));
}

wubu_zip_t *wubu_zip_open(const unsigned char *data, size_t n) {
    if (!data || n < 22) return NULL;
    /* find the EOCD (scan back from the end for PK\x05\x06) */
    size_t eocd = n >= 65557 ? n - 65557 : 0;
    size_t eocd_pos = (size_t)-1;
    for (size_t i = n - 22; i >= eocd && i + 22 <= n; i--) {
        if (data[i] == 'P' && data[i+1] == 'K' &&
            data[i+2] == 0x05 && data[i+3] == 0x06) {
            eocd_pos = i;
            break;
        }
        if (i == 0) break;
    }
    if (eocd_pos == (size_t)-1) return NULL;
    const unsigned char *e = data + eocd_pos;
    uint32_t cd_size = rd_le32(e + 12);
    uint32_t cd_off = rd_le32(e + 16);
    uint16_t n_entries = rd_le16(e + 10);
    if (cd_off + cd_size > n || n_entries == 0) return NULL;

    wubu_zip_t *z = (wubu_zip_t *)calloc(1, sizeof(*z));
    if (!z) return NULL;
    z->entries = (wubu_zip_entry_t *)calloc(n_entries, sizeof(*z->entries));
    if (!z->entries) { free(z); return NULL; }
    z->count = n_entries;

    size_t p = cd_off;
    for (uint16_t i = 0; i < n_entries; i++) {
        if (p + 46 > n) { wubu_zip_free(z); return NULL; }
        if (!(data[p] == 'P' && data[p+1] == 'K' &&
              data[p+2] == 0x01 && data[p+3] == 0x02)) {
            wubu_zip_free(z); return NULL;
        }
        wubu_zip_entry_t *en = &z->entries[i];
        uint16_t method = rd_le16(data + p + 10);
        uint16_t nlen = rd_le16(data + p + 28);
        uint16_t xlen = rd_le16(data + p + 30);
        uint16_t clen = rd_le16(data + p + 32);
        en->method = method;
        en->crc32 = rd_le32(data + p + 16);
        en->comp_size = rd_le32(data + p + 20);
        en->uncomp_size = rd_le32(data + p + 24);
        en->local_off = rd_le32(data + p + 42);
        if (p + 46 + nlen > n) { wubu_zip_free(z); return NULL; }
        size_t namelen = nlen < sizeof(en->name) - 1 ? nlen : sizeof(en->name) - 1;
        memcpy(en->name, data + p + 46, namelen);
        en->name[namelen] = '\0';
        en->is_dir = namelen > 0 && en->name[namelen-1] == '/';
        p += 46 + nlen + xlen + clen;
    }
    return z;
}

size_t wubu_zip_count(const wubu_zip_t *z) { return z ? z->count : 0; }

const wubu_zip_entry_t *wubu_zip_entry(const wubu_zip_t *z, size_t i) {
    if (!z || i >= z->count) return NULL;
    return &z->entries[i];
}

int wubu_zip_find(const wubu_zip_t *z, const char *name) {
    if (!z || !name) return -1;
    for (size_t i = 0; i < z->count; i++)
        if (strcmp(z->entries[i].name, name) == 0) return (int)i;
    return -1;
}

int wubu_zip_extract(const wubu_zip_t *z, const unsigned char *data,
                     size_t n, size_t i, unsigned char **out, size_t *outlen) {
    if (!z || !data || !out || !outlen || i >= z->count) return -1;
    const wubu_zip_entry_t *en = &z->entries[i];
    /* local header at en->local_off: PK\x03\x04 + fixed 30 bytes,
     * then name len + extra len */
    size_t lo = en->local_off;
    if (lo + 30 > n) return -1;
    if (!(data[lo] == 'P' && data[lo+1] == 'K' &&
          data[lo+2] == 0x03 && data[lo+3] == 0x04)) return -1;
    uint16_t nlen = rd_le16(data + lo + 26);
    uint16_t xlen = rd_le16(data + lo + 28);
    size_t data_off = lo + 30 + nlen + xlen;
    if (data_off + en->comp_size > n) return -1;

    unsigned char *buf = (unsigned char *)malloc(en->uncomp_size ? en->uncomp_size : 1);
    if (!buf) return -1;
    if (en->method == 0) {                    /* stored */
        memcpy(buf, data + data_off, en->uncomp_size);
        *out = buf;
        *outlen = en->uncomp_size;
        return 0;
    }
    if (en->method == 8) {                    /* deflate */
        z_stream zs;
        memset(&zs, 0, sizeof(zs));
        if (inflateInit2(&zs, -MAX_WBITS) != Z_OK) { free(buf); return -1; }
        zs.next_in = (Bytef *)(data + data_off);
        zs.avail_in = (uInt)en->comp_size;
        zs.next_out = buf;
        zs.avail_out = (uInt)en->uncomp_size;
        int zr = inflate(&zs, Z_FINISH);
        inflateEnd(&zs);
        if (zr != Z_STREAM_END) { free(buf); return -1; }
        *out = buf;
        *outlen = en->uncomp_size;
        return 0;
    }
    free(buf);
    return -1;   /* unsupported method (bzip2/lzma: later) */
}

void wubu_zip_free(wubu_zip_t *z) {
    if (!z) return;
    free(z->entries);
    free(z);
}
