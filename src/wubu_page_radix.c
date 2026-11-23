/*
 * wubu_page_radix.c -- EG-38 page-owning radix reuse. Pure C11.
 */
#include "wubu_page_radix.h"

#include <stdlib.h>
#include <string.h>

wubu_page_radix_t *wubu_page_radix_create(int page_size, int n_pages) {
    wubu_page_radix_t *r;
    if (page_size <= 0 || n_pages <= 0 || n_pages > WUBU_PAGE_RADIX_MAX_PAGES)
        return NULL;
    r = (wubu_page_radix_t *)calloc(1, sizeof(*r));
    if (r) { r->page_size = page_size; r->n_pages = n_pages; }
    return r;
}

void wubu_page_radix_free(wubu_page_radix_t *r) { free(r); }

int wubu_page_radix_insert(wubu_page_radix_t *r, const int *tokens, int n,
                           const int *pages, int n_pages) {
    wubu_page_entry_t *e;
    if (!r || !tokens || !pages || n <= 0 || n > WUBU_PAGE_RADIX_MAX_TOKENS ||
        n_pages <= 0 || n_pages > WUBU_PAGE_RADIX_MAX_PAGES ||
        r->n_entries >= WUBU_PAGE_RADIX_MAX_ENTRIES) return -1;
    for (int i = 0; i < n_pages; i++)
        if (pages[i] < 0 || pages[i] >= r->n_pages) return -1;

    e = &r->entries[r->n_entries];
    memcpy(e->tokens, tokens, (size_t)n * sizeof(int));
    e->n_tokens = n;
    memcpy(e->pages, pages, (size_t)n_pages * sizeof(int));
    e->n_pages = n_pages;
    e->active = 1;
    return r->n_entries++;
}

int wubu_page_radix_match(wubu_page_radix_t *r, const int *tokens, int n,
                          int *matched_tokens, int *pages_out,
                          int *n_pages_out) {
    int best = -1, best_len = 0;
    if (!r || !tokens || n <= 0) return -1;

    for (int i = 0; i < r->n_entries; i++) {
        const wubu_page_entry_t *e = &r->entries[i];
        int k = 0;
        if (!e->active) continue;
        while (k < n && k < e->n_tokens && tokens[k] == e->tokens[k]) k++;
        if (k > best_len) { best_len = k; best = i; }
    }
    if (pages_out && n_pages_out) {
        int need = (best_len + r->page_size - 1) / r->page_size;
        int copied = 0;
        const wubu_page_entry_t *e = best >= 0 ? &r->entries[best] : NULL;
        for (int j = 0; e && j < e->n_pages && copied < need; j++) {
            int pg = e->pages[j];
            r->refcount[pg]++;
            pages_out[copied++] = pg;
        }
        *n_pages_out = copied;
    }
    if (matched_tokens) *matched_tokens = best_len;
    return best;
}

int wubu_page_radix_release(wubu_page_radix_t *r, int entry) {
    const wubu_page_entry_t *e;
    if (!r || entry < 0 || entry >= r->n_entries) return -1;
    e = &r->entries[entry];
    for (int i = 0; i < e->n_pages; i++) {
        int pg = e->pages[i];
        if (r->refcount[pg] > 0) r->refcount[pg]--;
    }
    return 0;
}

void wubu_page_radix_snapshot(wubu_page_radix_t *r) {
    if (r) memcpy(r->snapshot, r->refcount, sizeof(r->refcount));
}

void wubu_page_radix_restore(wubu_page_radix_t *r) {
    if (r) memcpy(r->refcount, r->snapshot, sizeof(r->refcount));
}

int wubu_page_radix_refcount(const wubu_page_radix_t *r, int page) {
    return r && page >= 0 && page < r->n_pages ? r->refcount[page] : -1;
}
