/*
 * wubu_ssd_row_reuse.c -- EG-07 sliding-window row reuse. Pure C11;
 * the backing store is an in-memory image and reads are accounted, not issued.
 */
#include "wubu_ssd_row_reuse.h"

#include <stdlib.h>
#include <string.h>

struct wubu_ssd_row_reader {
    const uint8_t *data;
    size_t size;
    size_t row_bytes;
    int    window_rows;
    size_t coalesce_bytes;
    uint8_t *window;
    int    have_window;
    int    start;
    size_t reads;
    uint64_t bytes;
    int    deltas;
};

wubu_ssd_row_reader_t *wubu_ssd_row_reader_create(const uint8_t *data,
                                                   size_t size,
                                                   size_t row_bytes,
                                                   int window_rows,
                                                   size_t coalesce_bytes) {
    wubu_ssd_row_reader_t *r;
    if (!data || row_bytes == 0 || window_rows <= 0 || coalesce_bytes == 0)
        return NULL;
    if ((size_t)window_rows > size / row_bytes) return NULL;
    r = (wubu_ssd_row_reader_t *)calloc(1, sizeof(*r));
    if (!r) return NULL;
    r->window = (uint8_t *)malloc((size_t)window_rows * row_bytes);
    if (!r->window) { free(r); return NULL; }
    r->data = data;
    r->size = size;
    r->row_bytes = row_bytes;
    r->window_rows = window_rows;
    r->coalesce_bytes = coalesce_bytes;
    r->start = -1;
    return r;
}

void wubu_ssd_row_reader_free(wubu_ssd_row_reader_t *r) {
    if (!r) return;
    free(r->window);
    free(r);
}

/* Full refill: copy start..start+k rows in coalesced chunks. */
static const uint8_t *refill(wubu_ssd_row_reader_t *r, int start) {
    size_t total = (size_t)r->window_rows * r->row_bytes;
    size_t off = 0;
    const uint8_t *src = r->data + (size_t)start * r->row_bytes;
    while (off < total) {
        size_t chunk = r->coalesce_bytes;
        if (chunk > total - off) chunk = total - off;
        memcpy(r->window + off, src + off, chunk);
        r->reads++;
        r->bytes += chunk;
        off += chunk;
    }
    r->start = start;
    r->have_window = 1;
    return r->window;
}

const uint8_t *wubu_ssd_row_window(wubu_ssd_row_reader_t *r, int start) {
    if (!r || start < 0) return NULL;
    if ((size_t)start + (size_t)r->window_rows > r->size / r->row_bytes)
        return NULL;

    if (!r->have_window) return refill(r, start);

    if (start == r->start + 1) {
        /* Incremental s_agg(k+1)-s_agg(k): drop the oldest row, read one. */
        size_t keep = (size_t)(r->window_rows - 1) * r->row_bytes;
        memmove(r->window, r->window + r->row_bytes, keep);
        memcpy(r->window + keep,
               r->data + (size_t)(start + r->window_rows - 1) * r->row_bytes,
               r->row_bytes);
        r->reads++;
        r->bytes += r->row_bytes;
        r->deltas++;
        r->start = start;
        return r->window;
    }

    if (start == r->start) return r->window; /* same window, zero I/O */
    return refill(r, start);                 /* jump: full coalesced refill */
}

size_t wubu_ssd_row_reader_reads(const wubu_ssd_row_reader_t *r) {
    return r ? r->reads : 0;
}

uint64_t wubu_ssd_row_reader_bytes(const wubu_ssd_row_reader_t *r) {
    return r ? r->bytes : 0;
}

int wubu_ssd_row_reader_delta_updates(const wubu_ssd_row_reader_t *r) {
    return r ? r->deltas : -1;
}
