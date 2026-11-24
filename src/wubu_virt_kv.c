/* wubu_virt_kv.c -- EG-107 contiguous virtual KV gate. Strict C11. */
#include "wubu_virt_kv.h"

#include <stdlib.h>

struct wubu_virt_kv {
    int batch, max_len, page_size, physical_pages, used;
    int request[WUBU_VIRT_MAX_BATCH];
    unsigned char mapped[WUBU_VIRT_MAX_BATCH][WUBU_VIRT_MAX_BATCH];
    int mapped_count;
};

wubu_virt_kv_t *wubu_virt_kv_create(int batch, int max_len,
                                    int page_size, int physical_pages)
{
    wubu_virt_kv_t *p;
    if (batch < 1 || batch > WUBU_VIRT_MAX_BATCH || max_len < 1 ||
        page_size < 1 || physical_pages < 1) return NULL;
    p = (wubu_virt_kv_t *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->batch = batch;
    p->max_len = max_len;
    p->page_size = page_size;
    p->physical_pages = physical_pages;
    for (int i = 0; i < batch; ++i) p->request[i] = -1;
    return p;
}

void wubu_virt_kv_free(wubu_virt_kv_t *p) { free(p); }

size_t wubu_virt_kv_virtual_tokens(const wubu_virt_kv_t *p)
{
    return p ? (size_t)p->batch * p->max_len : 0;
}

int wubu_virt_kv_alloc_req(wubu_virt_kv_t *p, int request_id)
{
    if (!p || request_id < 0) return -1;
    for (int i = 0; i < p->batch; ++i)
        if (p->request[i] == request_id) return -1;
    if (p->used >= p->batch) return -1;
    for (int i = 0; i < p->batch; ++i) {
        if (p->request[i] == -1) {
            p->request[i] = request_id;
            ++p->used;
            return i;
        }
    }
    return -1;
}

int wubu_virt_kv_step(wubu_virt_kv_t *p, const int *seq_len, int n)
{
    int need = 0;
    if (!p || !seq_len || n != p->batch) return -1;
    for (int b = 0; b < p->batch; ++b) {
        if (seq_len[b] < 0 || seq_len[b] > p->max_len) return -1;
        int pages = (seq_len[b] + p->page_size - 1) / p->page_size;
        for (int k = 0; k < pages; ++k)
            if (!p->mapped[b][k]) ++need;
    }
    if (p->mapped_count + need > p->physical_pages) return -1;
    for (int b = 0; b < p->batch; ++b) {
        int pages = (seq_len[b] + p->page_size - 1) / p->page_size;
        for (int k = 0; k < pages; ++k)
            if (!p->mapped[b][k]) { p->mapped[b][k] = 1; ++p->mapped_count; }
    }
    return 0;
}

long wubu_virt_kv_offset(const wubu_virt_kv_t *p, int batch_idx, int token)
{
    if (!p || batch_idx < 0 || batch_idx >= p->batch ||
        token < 0 || token >= p->max_len) return -1;
    return (long)batch_idx * p->max_len + token;
}

size_t wubu_virt_kv_mapped_pages(const wubu_virt_kv_t *p)
{
    return p ? (size_t)p->mapped_count : 0;
}
