/* wubu_adapt_len.c -- EG-88 adaptive candidate length gate. Strict C11. */
#include "wubu_adapt_len.h"

#include <stdlib.h>

struct wubu_adapt_len {
    int block_size;
    float threshold;
};

wubu_adapt_len_t *wubu_adapt_len_create(int block_size, float threshold)
{
    wubu_adapt_len_t *p;
    if (block_size < WUBU_ADAPT_LEN_MIN || threshold < 0.0f || threshold > 1.0f)
        return NULL;
    p = (wubu_adapt_len_t *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->block_size = block_size;
    p->threshold = threshold;
    return p;
}

void wubu_adapt_len_free(wubu_adapt_len_t *p) { free(p); }

int wubu_adapt_len_choose(const wubu_adapt_len_t *p, const float *probs, int n)
{
    double cum = 1.0;
    int limit, length;
    if (!p || !probs || n < WUBU_ADAPT_LEN_MIN) return -1;
    limit = n < p->block_size ? n : p->block_size;
    length = limit;
    for (int j = 1; j < limit; ++j) {
        double v = probs[j];
        cum *= (v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v));
        if (1.0 - cum > (double)p->threshold) {
            length = j + 1;
            break;
        }
    }
    return length < WUBU_ADAPT_LEN_MIN ? WUBU_ADAPT_LEN_MIN : length;
}

int wubu_adapt_len_accept(const int *draft, const int *target, int len)
{
    int accepted = 0;
    if (!draft || !target || len < WUBU_ADAPT_LEN_MIN) return -1;
    for (int i = 1; i < len; ++i) {
        if (draft[i] != target[i - 1]) break;
        ++accepted;
    }
    return accepted;
}

int wubu_adapt_len_kv_rows(int accepted)
{
    return accepted < 0 ? 0 : accepted + 1;
}
