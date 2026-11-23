/* wubu_batch_h2d.c -- EG-98 portable batched H2D planner. Strict C11. */
#include "wubu_batch_h2d.h"

#include <stdlib.h>
#include <string.h>

struct h2d_blob {
    const unsigned char *src;
    size_t dst, bytes;
    int pinned;
};

struct wubu_batch_h2d {
    int mode, capacity, n, groups, fail_at;
    struct h2d_blob *blobs;
};

wubu_batch_h2d_t *wubu_batch_h2d_create(int mode, int capacity)
{
    wubu_batch_h2d_t *p;
    if ((mode < 0 || mode > 2) || capacity < 1) return NULL;
    p = (wubu_batch_h2d_t *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->blobs = (struct h2d_blob *)calloc((size_t)capacity, sizeof(*p->blobs));
    if (!p->blobs) { free(p); return NULL; }
    p->mode = mode;
    p->capacity = capacity;
    p->fail_at = -1;
    return p;
}

void wubu_batch_h2d_free(wubu_batch_h2d_t *p)
{
    if (!p) return;
    free(p->blobs);
    free(p);
}

int wubu_batch_h2d_add(wubu_batch_h2d_t *p, const unsigned char *src,
                       size_t dst_offset, size_t bytes, int pinned)
{
    struct h2d_blob *b;
    if (!p || !src || bytes == 0 || p->n >= p->capacity) return -1;
    b = &p->blobs[p->n++];
    b->src = src;
    b->dst = dst_offset;
    b->bytes = bytes;
    b->pinned = pinned;
    return 0;
}

int wubu_batch_h2d_issue(wubu_batch_h2d_t *p,
                         unsigned char *dev, size_t dev_bytes)
{
    int groups;
    if (!p || !dev || p->n == 0) return -1;
    groups = p->mode == 0 ? p->n
                          : (p->n + WUBU_H2D_MAX_GROUP - 1) / WUBU_H2D_MAX_GROUP;
    for (int i = 0; i < p->n; ++i) {
        const struct h2d_blob *b = &p->blobs[i];
        if (!b->pinned || b->bytes > dev_bytes || b->dst > dev_bytes - b->bytes)
            return -1;
        for (int j = 0; j < i; ++j) {
            const struct h2d_blob *o = &p->blobs[j];
            if (b->dst < o->dst + o->bytes && o->dst < b->dst + b->bytes)
                return -1;              /* destinations must be disjoint */
        }
    }
    if (p->fail_at >= 0 && p->fail_at < groups) return -1; /* abort, no commit */
    for (int i = 0; i < p->n; ++i) {
        const struct h2d_blob *b = &p->blobs[i];
        memcpy(dev + b->dst, b->src, b->bytes);
    }
    p->groups = groups;
    return p->groups;
}

int wubu_batch_h2d_groups(const wubu_batch_h2d_t *p)
{
    return p ? p->groups : -1;
}

int wubu_batch_h2d_set_fail_at(wubu_batch_h2d_t *p, int group)
{
    if (!p || group < -1) return -1;
    p->fail_at = group;
    return 0;
}
