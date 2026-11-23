/* wubu_suffix.c -- EG-73 suffix decoding corpus gate. Strict C11. */
#include "wubu_suffix_spec.h"

#include <stdlib.h>
#include <string.h>

struct wubu_suffix {
    int capacity, max_len, n;
    int seq[WUBU_SUFFIX_MAX_SEQ][WUBU_SUFFIX_MAX_LEN];
    int len[WUBU_SUFFIX_MAX_SEQ];
};

wubu_suffix_t *wubu_suffix_create(int corpus_capacity, int max_len)
{
    wubu_suffix_t *p;
    if (corpus_capacity < 1 || corpus_capacity > WUBU_SUFFIX_MAX_SEQ ||
        max_len < 2 || max_len > WUBU_SUFFIX_MAX_LEN) return NULL;
    p = (wubu_suffix_t *)calloc(1, sizeof(*p));
    if (p) { p->capacity = corpus_capacity; p->max_len = max_len; }
    return p;
}

void wubu_suffix_free(wubu_suffix_t *p) { free(p); }

int wubu_suffix_ingest(wubu_suffix_t *p, const int *tokens, int n)
{
    int at, keep;
    if (!p || !tokens || n < 2) return -1;
    keep = n < p->max_len ? n : p->max_len;
    if (p->n == p->capacity) {
        memmove(p->seq[0], p->seq[1],
                (size_t)(p->n - 1) * WUBU_SUFFIX_MAX_LEN * sizeof(int));
        memmove(p->len, p->len + 1, (size_t)(p->n - 1) * sizeof(int));
        --p->n;
    }
    at = p->n++;
    memcpy(p->seq[at], tokens + (n - keep), (size_t)keep * sizeof(int));
    p->len[at] = keep;
    return 0;
}

int wubu_suffix_corpus_count(const wubu_suffix_t *p)
{
    return p ? p->n : -1;
}

int wubu_suffix_match(const wubu_suffix_t *p, const int *prompt, int n,
                      int *out_tokens, int out_cap)
{
    int limit;
    if (!p || !prompt || !out_tokens || n < 2 || out_cap < 2) return -1;
    limit = n < p->max_len ? n : p->max_len;
    for (int L = limit; L >= 2; --L) {
        const int *suffix = prompt + (n - L);
        for (int s = 0; s < p->n; ++s) {
            if (p->len[s] < L) continue;
            if (memcmp(p->seq[s] + (p->len[s] - L), suffix,
                       (size_t)L * sizeof(int)) == 0) {
                memcpy(out_tokens, suffix, (size_t)L * sizeof(int));
                return L;
            }
        }
    }
    return 0;
}

int wubu_suffix_fill(int *out, int len, int total, int last_token)
{
    if (!out || len < 0 || total < len) return -1;
    for (int i = len; i < total; ++i) out[i] = last_token;
    return 0;
}
