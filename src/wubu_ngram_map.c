/* wubu_ngram_map.c -- EG-72 k4v n-gram map. Strict C11. */
#include "wubu_ngram_map.h"

#include <stdlib.h>
#include <string.h>

#define MAP_MAX_N 16

typedef struct {
    uint64_t key;
    int values[WUBU_NGRAM_K4V_MAX];
    int counts[WUBU_NGRAM_K4V_MAX];
    int n;
} map_entry;

struct wubu_ngram_map {
    int capacity, ngram, used;
    map_entry *entries;
};

static uint64_t map_key(const int *tokens, int n)
{
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < n; ++i) {
        uint64_t v = (uint64_t)(unsigned)tokens[i];
        h ^= v; h *= 1099511628211ULL;
        h ^= v >> 32; h *= 1099511628211ULL;
    }
    return h;
}

wubu_ngram_map_t *wubu_ngram_map_create(int capacity, int n)
{
    wubu_ngram_map_t *p;
    if (capacity < 1 || n < 1 || n > MAP_MAX_N) return NULL;
    p = (wubu_ngram_map_t *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->entries = (map_entry *)calloc((size_t)capacity, sizeof(map_entry));
    if (!p->entries) { free(p); return NULL; }
    p->capacity = capacity;
    p->ngram = n;
    return p;
}

void wubu_ngram_map_free(wubu_ngram_map_t *p)
{
    if (!p) return;
    free(p->entries);
    free(p);
}

int wubu_ngram_map_observe(wubu_ngram_map_t *p, const int *tokens, int n,
                           int next_token)
{
    uint64_t key;
    map_entry *e = NULL;
    if (!p || !tokens || n != p->ngram || next_token < 0) return -1;
    key = map_key(tokens, n);
    for (int i = 0; i < p->used; ++i)
        if (p->entries[i].key == key) { e = &p->entries[i]; break; }
    if (!e) {
        if (p->used >= p->capacity) {
            /* Evict the least-hit entry; ties choose the oldest. */
            int worst = 0, total = p->entries[0].counts[0];
            for (int i = 1; i < p->used; ++i) {
                int s = 0;
                for (int k = 0; k < p->entries[i].n; ++k) s += p->entries[i].counts[k];
                if (s < total) { total = s; worst = i; }
            }
            e = &p->entries[worst];
        } else {
            e = &p->entries[p->used++];
        }
        e->key = key;
        e->n = 0;
    }
    for (int i = 0; i < e->n; ++i)
        if (e->values[i] == next_token) { ++e->counts[i]; return 0; }
    if (e->n < WUBU_NGRAM_K4V_MAX) {
        e->values[e->n] = next_token;
        e->counts[e->n] = 1;
        ++e->n;
        return 0;
    }
    {
        int worst = 0;
        for (int i = 1; i < e->n; ++i)
            if (e->counts[i] < e->counts[worst]) worst = i;
        /* Shift left so the surviving values keep their insertion order. */
        for (int i = worst; i < e->n - 1; ++i) {
            e->values[i] = e->values[i + 1];
            e->counts[i] = e->counts[i + 1];
        }
        e->values[e->n - 1] = next_token;
        e->counts[e->n - 1] = 1;
    }
    return 0;
}

int wubu_ngram_map_values(const wubu_ngram_map_t *p, const int *tokens, int n,
                          int *out_values, int *out_counts)
{
    uint64_t key;
    const map_entry *e = NULL;
    if (!p || !tokens || !out_values || !out_counts || n != p->ngram) return -1;
    key = map_key(tokens, n);
    for (int i = 0; i < p->used; ++i)
        if (p->entries[i].key == key) { e = &p->entries[i]; break; }
    if (!e) return 0;
    for (int i = 0; i < e->n; ++i) {
        out_values[i] = e->values[i];
        out_counts[i] = e->counts[i];
    }
    for (int i = 1; i < e->n; ++i) {       /* stable count-desc sort */
        int v = out_values[i], c = out_counts[i], j = i - 1;
        while (j >= 0 && out_counts[j] < c) {
            out_counts[j + 1] = out_counts[j];
            out_values[j + 1] = out_values[j];
            --j;
        }
        out_counts[j + 1] = c;
        out_values[j + 1] = v;
    }
    return e->n;
}

/* Wire layout: magic[8] | capacity | ngram | used | entries[44] | FNV-64. */
#define MAP_HEADER 20
#define MAP_ENTRY  44
#define MAP_SUFFIX 8

static uint64_t map_checksum(const unsigned char *buf, size_t n)
{
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) { h ^= buf[i]; h *= 1099511628211ULL; }
    return h;
}

size_t wubu_ngram_map_encoded_size(const wubu_ngram_map_t *p)
{
    return p ? MAP_HEADER + (size_t)p->used * MAP_ENTRY + MAP_SUFFIX : 0;
}

int wubu_ngram_map_encode(const wubu_ngram_map_t *p,
                          unsigned char *buf, size_t cap)
{
    size_t need, off = MAP_HEADER;
    uint64_t sum;
    int32_t v;
    if (!p || !buf) return -1;
    need = wubu_ngram_map_encoded_size(p);
    if (cap < need) return -1;
    memcpy(buf, "WUK4V001", 8);
    v = p->capacity; memcpy(buf + 8, &v, 4);
    v = p->ngram;    memcpy(buf + 12, &v, 4);
    v = p->used;     memcpy(buf + 16, &v, 4);
    for (int i = 0; i < p->used; ++i) {
        const map_entry *e = &p->entries[i];
        memcpy(buf + off, &e->key, 8); off += 8;
        v = e->n; memcpy(buf + off, &v, 4); off += 4;
        for (int k = 0; k < WUBU_NGRAM_K4V_MAX; ++k) {
            v = e->values[k]; memcpy(buf + off, &v, 4); off += 4;
        }
        for (int k = 0; k < WUBU_NGRAM_K4V_MAX; ++k) {
            v = e->counts[k]; memcpy(buf + off, &v, 4); off += 4;
        }
    }
    sum = map_checksum(buf, off);
    memcpy(buf + off, &sum, 8);
    return (int)need;
}

int wubu_ngram_map_decode(const unsigned char *buf, size_t bytes,
                          wubu_ngram_map_t **out)
{
    wubu_ngram_map_t *p;
    uint64_t sum;
    int32_t cap, ng, used, ecount;
    size_t off = MAP_HEADER;
    if (!buf || !out || bytes < MAP_HEADER + MAP_SUFFIX ||
        memcmp(buf, "WUK4V001", 8) != 0) return -1;
    memcpy(&sum, buf + bytes - MAP_SUFFIX, 8);
    if (map_checksum(buf, bytes - MAP_SUFFIX) != sum) return -1;
    memcpy(&cap, buf + 8, 4);
    memcpy(&ng, buf + 12, 4);
    memcpy(&used, buf + 16, 4);
    if (cap < 1 || ng < 1 || ng > MAP_MAX_N || used < 0 || used > cap ||
        bytes != MAP_HEADER + (size_t)used * MAP_ENTRY + MAP_SUFFIX) return -1;
    p = wubu_ngram_map_create(cap, ng);
    if (!p) return -1;
    p->used = used;
    for (int i = 0; i < used; ++i) {
        map_entry *e = &p->entries[i];
        memcpy(&e->key, buf + off, 8); off += 8;
        memcpy(&ecount, buf + off, 4); off += 4;
        if (ecount < 0 || ecount > WUBU_NGRAM_K4V_MAX) { wubu_ngram_map_free(p); return -1; }
        e->n = ecount;
        for (int k = 0; k < WUBU_NGRAM_K4V_MAX; ++k) {
            memcpy(&e->values[k], buf + off, 4); off += 4;
        }
        for (int k = 0; k < WUBU_NGRAM_K4V_MAX; ++k) {
            memcpy(&e->counts[k], buf + off, 4); off += 4;
            if (e->counts[k] < 0) { wubu_ngram_map_free(p); return -1; }
        }
    }
    *out = p;
    return 0;
}

int wubu_ngram_map_merge(wubu_ngram_map_t *p, const wubu_ngram_map_t *other)
{
    if (!p || !other || p->ngram != other->ngram) return -1;
    for (int i = 0; i < other->used; ++i) {
        const map_entry *src = &other->entries[i];
        map_entry *dst = NULL;
        for (int k = 0; k < p->used; ++k)
            if (p->entries[k].key == src->key) { dst = &p->entries[k]; break; }
        if (!dst) {
            if (p->used >= p->capacity) {
                int worst = 0, total = 0;
                for (int k = 0; k < WUBU_NGRAM_K4V_MAX; ++k) total += p->entries[0].counts[k];
                for (int k = 1; k < p->used; ++k) {
                    int s = 0;
                    for (int j = 0; j < WUBU_NGRAM_K4V_MAX; ++j) s += p->entries[k].counts[j];
                    if (s < total) { total = s; worst = k; }
                }
                dst = &p->entries[worst];
            } else {
                dst = &p->entries[p->used++];
            }
            dst->key = src->key;
            dst->n = 0;
        }
        for (int k = 0; k < src->n; ++k) {
            int slot = -1;
            for (int j = 0; j < dst->n; ++j)
                if (dst->values[j] == src->values[k]) { slot = j; break; }
            if (slot >= 0) {
                dst->counts[slot] += src->counts[k];
            } else if (dst->n < WUBU_NGRAM_K4V_MAX) {
                dst->values[dst->n] = src->values[k];
                dst->counts[dst->n] = src->counts[k];
                ++dst->n;
            } else {
                int low = 0;
                for (int j = 1; j < dst->n; ++j)
                    if (dst->counts[j] < dst->counts[low]) low = j;
                dst->values[low] = src->values[k];
                dst->counts[low] = src->counts[k];
            }
        }
    }
    return 0;
}
