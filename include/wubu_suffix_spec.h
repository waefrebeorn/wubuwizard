#ifndef WUBU_SUFFIX_SPEC_H
#define WUBU_SUFFIX_SPEC_H

/*
 * wubu_suffix_spec.h -- EG-73: suffix decoding from a finished-request corpus.
 * Lookup keys are the last 2 and last 4 prompt tokens, extended to the longest
 * suffix match in any finished response; corpus eviction is oldest-first.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_SUFFIX_MAX_SEQ 64
#define WUBU_SUFFIX_MAX_LEN 128

typedef struct wubu_suffix wubu_suffix_t;

wubu_suffix_t *wubu_suffix_create(int corpus_capacity, int max_len);
void wubu_suffix_free(wubu_suffix_t *p);

int wubu_suffix_ingest(wubu_suffix_t *p, const int *tokens, int n);
int wubu_suffix_corpus_count(const wubu_suffix_t *p);

/* Longest prompt suffix that is a suffix of a corpus sequence; writes the
 * matched tokens and returns its length (0 when there is no match). */
int wubu_suffix_match(const wubu_suffix_t *p, const int *prompt, int n,
                      int *out_tokens, int out_cap);

/* Fill unmatched draft slots with the last sampled token. 0/-1. */
int wubu_suffix_fill(int *out, int len, int total, int last_token);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_SUFFIX_SPEC_H */
