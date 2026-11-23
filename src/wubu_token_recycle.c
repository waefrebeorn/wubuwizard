/* wubu_token_recycle.c -- EG-25b token recycling gate. Strict C11. */
#include "wubu_token_recycle.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    int anchor, len;
    int token[WUBU_RECYCLE_MAX_LEN];
} recycle_slot_t;

struct wubu_token_recycle {
    int slots, max_len, n, next;
    recycle_slot_t slot[WUBU_RECYCLE_MAX_SLOTS];
};

wubu_token_recycle_t *wubu_token_recycle_create(int slots, int max_len)
{
    wubu_token_recycle_t *p;
    if (slots < 1 || slots > WUBU_RECYCLE_MAX_SLOTS ||
        max_len < 1 || max_len > WUBU_RECYCLE_MAX_LEN) return NULL;
    p = (wubu_token_recycle_t *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->slots = slots;
    p->max_len = max_len;
    for (int i = 0; i < slots; ++i) p->slot[i].anchor = -1;
    return p;
}

void wubu_token_recycle_free(wubu_token_recycle_t *p) { free(p); }

int wubu_token_recycle_record(wubu_token_recycle_t *p,
                              const int *tokens, int n)
{
    int keep, at = -1;
    if (!p || !tokens || n < 2) return -1;
    keep = n - 1 < p->max_len ? n - 1 : p->max_len;
    for (int i = 0; i < p->n; ++i)
        if (p->slot[i].anchor == tokens[0]) { at = i; break; }
    if (at < 0) {
        at = p->next;
        p->next = (p->next + 1) % p->slots;
        if (p->n < p->slots) ++p->n;
    }
    p->slot[at].anchor = tokens[0];
    p->slot[at].len = keep;
    memcpy(p->slot[at].token, tokens + 1, (size_t)keep * sizeof(int));
    return 0;
}

int wubu_token_recycle_fetch(const wubu_token_recycle_t *p,
                             int last_token, int *out, int cap)
{
    if (!p || !out || cap < 1) return -1;
    for (int i = 0; i < p->n; ++i) {
        if (p->slot[i].anchor == last_token) {
            int k = p->slot[i].len < cap ? p->slot[i].len : cap;
            memcpy(out, p->slot[i].token, (size_t)k * sizeof(int));
            return k;
        }
    }
    return 0;
}

int wubu_token_recycle_count(const wubu_token_recycle_t *p)
{
    return p ? p->n : -1;
}
