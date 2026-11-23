/*
 * wubu_sleep_alloc.c -- EG-16 tagged sleep/wake allocator. Pure C11.
 */
#include "wubu_sleep_alloc.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    char  *name;
    unsigned tag;
    size_t offset, bytes;
    unsigned char *host;   /* non-NULL while offloaded */
} block_t;

struct wubu_sleep_alloc {
    unsigned char *device;
    size_t capacity, dev_used, host_bytes, pos;
    block_t *blocks;
    int n, cap;
};

wubu_sleep_alloc_t *wubu_sleep_alloc_create(size_t device_capacity) {
    wubu_sleep_alloc_t *a = (wubu_sleep_alloc_t *)calloc(1, sizeof(*a));
    if (!a) return NULL;
    if (device_capacity == 0) device_capacity = 1;
    a->device = (unsigned char *)malloc(device_capacity);
    if (!a->device) { free(a); return NULL; }
    a->capacity = device_capacity;
    a->cap = 8;
    a->blocks = (block_t *)calloc((size_t)a->cap, sizeof(block_t));
    if (!a->blocks) { free(a->device); free(a); return NULL; }
    return a;
}

void wubu_sleep_alloc_free(wubu_sleep_alloc_t *a) {
    int i;
    if (!a) return;
    for (i = 0; i < a->n; i++) free(a->blocks[i].host);
    free(a->blocks);
    free(a->device);
    free(a);
}

void *wubu_sleep_alloc_push(wubu_sleep_alloc_t *a, const char *name,
                            unsigned tag, size_t bytes) {
    block_t *b;
    if (!a || bytes == 0 || a->n >= a->cap) return NULL;
    if (a->pos + bytes > a->capacity) return NULL;
    if (a->n == a->cap) {
        int nc = a->cap * 2;
        block_t *nb = (block_t *)realloc(a->blocks, (size_t)nc * sizeof(block_t));
        if (!nb) return NULL;
        a->blocks = nb;
        a->cap = nc;
    }
    b = &a->blocks[a->n++];
    b->name = name ? (char *)name : (char *)"";
    b->tag = tag;
    b->offset = a->pos;
    b->bytes = bytes;
    b->host = NULL;
    a->pos += bytes;
    a->dev_used += bytes;
    memset(a->device + b->offset, 0, bytes);
    return a->device + b->offset;
}

int wubu_sleep_alloc_sleep(wubu_sleep_alloc_t *a, unsigned tag_mask) {
    int i, count = 0;
    if (!a || tag_mask == 0) return -1;
    for (i = 0; i < a->n; i++) {
        block_t *b = &a->blocks[i];
        if (b->host || !(b->tag & tag_mask)) continue;
        b->host = (unsigned char *)malloc(b->bytes);
        if (!b->host) return -1;
        memcpy(b->host, a->device + b->offset, b->bytes);
        memset(a->device + b->offset, 0, b->bytes); /* device bytes released */
        a->dev_used -= b->bytes;
        a->host_bytes += b->bytes;
        count++;
    }
    return count;
}

int wubu_sleep_alloc_wake(wubu_sleep_alloc_t *a, unsigned tag_mask) {
    int i, count = 0;
    if (!a || tag_mask == 0) return -1;
    for (i = 0; i < a->n; i++) {
        block_t *b = &a->blocks[i];
        if (!b->host || !(b->tag & tag_mask)) continue;
        if (a->dev_used + b->bytes > a->capacity) return -1;
        memcpy(a->device + b->offset, b->host, b->bytes);
        a->dev_used += b->bytes;
        a->host_bytes -= b->bytes;
        free(b->host);
        b->host = NULL;
        count++;
    }
    return count;
}

size_t wubu_sleep_alloc_device_used(const wubu_sleep_alloc_t *a) {
    return a ? a->dev_used : 0;
}

size_t wubu_sleep_alloc_host_bytes(const wubu_sleep_alloc_t *a) {
    return a ? a->host_bytes : 0;
}

int wubu_sleep_alloc_resident(const wubu_sleep_alloc_t *a, const char *name) {
    int i;
    if (!a || !name) return 0;
    for (i = 0; i < a->n; i++)
        if (strcmp(a->blocks[i].name, name) == 0)
            return a->blocks[i].host == NULL ? 1 : 0;
    return 0;
}

int wubu_sleep_alloc_nblocks(const wubu_sleep_alloc_t *a) {
    return a ? a->n : 0;
}
