#ifndef WUBU_PAGE_RADIX_H
#define WUBU_PAGE_RADIX_H

/*
 * wubu_page_radix.h -- EG-38: logical radix prefix tree with page ownership,
 * decode-side reuse and draft-state snapshot/restore (SGLang HiCache lineage).
 * Multi-rank sharding is out of scope. Pure C11.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_PAGE_RADIX_MAX_ENTRIES 32
#define WUBU_PAGE_RADIX_MAX_TOKENS 64
#define WUBU_PAGE_RADIX_MAX_PAGES 32

typedef struct {
    int tokens[WUBU_PAGE_RADIX_MAX_TOKENS];
    int n_tokens;
    int pages[WUBU_PAGE_RADIX_MAX_PAGES];
    int n_pages;
    int active;
} wubu_page_entry_t;

typedef struct {
    int page_size;
    int n_pages;
    int refcount[WUBU_PAGE_RADIX_MAX_PAGES];
    int snapshot[WUBU_PAGE_RADIX_MAX_PAGES];
    wubu_page_entry_t entries[WUBU_PAGE_RADIX_MAX_ENTRIES];
    int n_entries;
} wubu_page_radix_t;

wubu_page_radix_t *wubu_page_radix_create(int page_size, int n_pages);
void wubu_page_radix_free(wubu_page_radix_t *r);

/* Register a complete sequence with its owned pages. Returns entry id or -1. */
int wubu_page_radix_insert(wubu_page_radix_t *r, const int *tokens, int n,
                           const int *pages, int n_pages);

/* Longest-prefix lookup; borrows (increments refcounts of) matched pages.
 * Returns entry id or -1. */
int wubu_page_radix_match(wubu_page_radix_t *r, const int *tokens, int n,
                          int *matched_tokens, int *pages_out,
                          int *n_pages_out);

/* Return borrowed pages of a previously matched entry. Returns 0 or -1. */
int wubu_page_radix_release(wubu_page_radix_t *r, int entry);

/* Draft-state rollback: snapshot/restore page refcounts. */
void wubu_page_radix_snapshot(wubu_page_radix_t *r);
void wubu_page_radix_restore(wubu_page_radix_t *r);

int wubu_page_radix_refcount(const wubu_page_radix_t *r, int page);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_PAGE_RADIX_H */
