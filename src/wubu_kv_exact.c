/* wubu_kv_exact.c -- atomic exact-ID KV page allocation (LP-05). */
#include "wubu_kv_exact.h"

#include <stdlib.h>
#include <string.h>

#define WUBU_KV_EXACT_MAGIC UINT64_C(0x4b56455841435421) /* "KVEXACT!" */

struct wubu_kv_exact {
    uint64_t       magic;
    int            capacity;   /* total slots, ids in [0, capacity) */
    int            quota;      /* max ids granted by one alloc_exact call */
    int            free_count; /* entries currently in free_ids */
    int            free_cap;   /* allocated length of free_ids */
    int           *free_ids;   /* the free-ID list */
    unsigned char *used;       /* used[id] != 0 iff slot id is allocated */
};

/* Shape checks only: pointer, range, sign, uniqueness. Pure / no allocation. */
static wubu_kv_exact_status_t
validate_ids(const wubu_kv_exact_t *a, const int *ids, int n)
{
    int i, j;

    if (!a || a->magic != WUBU_KV_EXACT_MAGIC) return WUBU_KV_EXACT_ERR_ARG;
    if (n < 0)  return WUBU_KV_EXACT_ERR_ARG;
    if (n == 0) return WUBU_KV_EXACT_OK;
    if (!ids)   return WUBU_KV_EXACT_ERR_ARG;

    for (i = 0; i < n; ++i) {
        if (ids[i] < 0 || ids[i] >= a->capacity)
            return WUBU_KV_EXACT_ERR_BAD_ID;
    }
    /* O(n^2) keeps this query allocation-free; n is bounded by quota. */
    for (i = 0; i < n; ++i) {
        for (j = i + 1; j < n; ++j) {
            if (ids[i] == ids[j]) return WUBU_KV_EXACT_ERR_DUP_ID;
        }
    }
    return WUBU_KV_EXACT_OK;
}

wubu_kv_exact_t *wubu_kv_exact_create(int capacity, int quota)
{
    wubu_kv_exact_t *a;
    int i;

    if (capacity <= 0 || quota <= 0) return NULL;
    a = (wubu_kv_exact_t *)calloc(1, sizeof(*a));
    if (!a) return NULL;
    a->free_ids = (int *)malloc(sizeof(int) * (size_t)capacity);
    a->used = (unsigned char *)calloc((size_t)capacity, 1);
    if (!a->free_ids || !a->used) {
        free(a->free_ids);
        free(a->used);
        free(a);
        return NULL;
    }
    for (i = 0; i < capacity; ++i) a->free_ids[i] = i;
    a->magic      = WUBU_KV_EXACT_MAGIC;
    a->capacity   = capacity;
    a->quota      = quota;
    a->free_count = capacity;
    a->free_cap   = capacity;
    return a;
}

void wubu_kv_exact_free(wubu_kv_exact_t *a)
{
    if (!a || a->magic != WUBU_KV_EXACT_MAGIC) return;
    a->magic = 0;
    free(a->free_ids);
    free(a->used);
    free(a);
}

wubu_kv_exact_status_t
wubu_kv_exact_can_alloc_exact(const wubu_kv_exact_t *a, const int *ids, int n)
{
    wubu_kv_exact_status_t st = validate_ids(a, ids, n);
    int i;

    if (st != WUBU_KV_EXACT_OK) return st;
    if (n == 0) return WUBU_KV_EXACT_OK;
    if (n > a->quota)      return WUBU_KV_EXACT_ERR_QUOTA;
    if (n > a->free_count) return WUBU_KV_EXACT_ERR_NO_CAPACITY;

    /* "All available": every named slot must currently be free. */
    for (i = 0; i < n; ++i) {
        if (a->used[ids[i]] != 0) return WUBU_KV_EXACT_ERR_NOT_FREE;
    }
    return WUBU_KV_EXACT_OK;
}

wubu_kv_exact_status_t
wubu_kv_exact_alloc_exact(wubu_kv_exact_t *a, const int *ids, int n)
{
    wubu_kv_exact_status_t st = wubu_kv_exact_can_alloc_exact(a, ids, n);
    int *new_free;
    unsigned char *new_used;
    int i, j = 0;

    /* Every failure path returns here before the object is touched. */
    if (st != WUBU_KV_EXACT_OK) return st;
    if (n == 0) return WUBU_KV_EXACT_OK;

    /* Build the complete replacement state in scratch memory first. */
    new_free = (int *)malloc(sizeof(int) * (size_t)a->free_count);
    if (!new_free) return WUBU_KV_EXACT_ERR_OOM;
    new_used = (unsigned char *)malloc((size_t)a->capacity);
    if (!new_used) {
        free(new_free);
        return WUBU_KV_EXACT_ERR_OOM;
    }

    memcpy(new_used, a->used, (size_t)a->capacity);
    for (i = 0; i < n; ++i) new_used[ids[i]] = 1;
    for (i = 0; i < a->free_count; ++i) {
        int id = a->free_ids[i];
        if (new_used[id] == 0) new_free[j++] = id; /* remove granted ids */
    }
    /* j == free_count - n; all checks already guaranteed that. */

    /* Commit point: nothing above mutated `a`. */
    free(a->free_ids);
    free(a->used);
    a->free_ids   = new_free;
    a->used       = new_used;
    a->free_count = j;
    a->free_cap   = j;
    return WUBU_KV_EXACT_OK;
}

wubu_kv_exact_status_t
wubu_kv_exact_release(wubu_kv_exact_t *a, const int *ids, int n)
{
    wubu_kv_exact_status_t st = validate_ids(a, ids, n);
    int *new_free;
    unsigned char *new_used;
    int used_count, i, j;

    if (st != WUBU_KV_EXACT_OK) return st;
    if (n == 0) return WUBU_KV_EXACT_OK;

    used_count = a->capacity - a->free_count;
    if (n > used_count) return WUBU_KV_EXACT_ERR_NO_CAPACITY;
    for (i = 0; i < n; ++i) {
        if (a->used[ids[i]] == 0) return WUBU_KV_EXACT_ERR_NOT_IN_USE;
    }

    new_free = (int *)malloc(sizeof(int) * (size_t)(a->free_count + n));
    if (!new_free) return WUBU_KV_EXACT_ERR_OOM;
    new_used = (unsigned char *)malloc((size_t)a->capacity);
    if (!new_used) {
        free(new_free);
        return WUBU_KV_EXACT_ERR_OOM;
    }

    memcpy(new_free, a->free_ids, (size_t)a->free_count * sizeof(int));
    memcpy(new_used, a->used, (size_t)a->capacity);
    j = a->free_count;
    for (i = 0; i < n; ++i) {
        new_used[ids[i]] = 0;
        new_free[j++] = ids[i];
    }

    free(a->free_ids);
    free(a->used);
    a->free_ids   = new_free;
    a->used       = new_used;
    a->free_count = j;
    a->free_cap   = j;
    return WUBU_KV_EXACT_OK;
}

int wubu_kv_exact_capacity(const wubu_kv_exact_t *a)
{
    return (a && a->magic == WUBU_KV_EXACT_MAGIC) ? a->capacity : 0;
}

int wubu_kv_exact_quota(const wubu_kv_exact_t *a)
{
    return (a && a->magic == WUBU_KV_EXACT_MAGIC) ? a->quota : 0;
}

int wubu_kv_exact_free_count(const wubu_kv_exact_t *a)
{
    return (a && a->magic == WUBU_KV_EXACT_MAGIC) ? a->free_count : 0;
}

int wubu_kv_exact_is_free(const wubu_kv_exact_t *a, int id)
{
    if (!a || a->magic != WUBU_KV_EXACT_MAGIC) return 0;
    if (id < 0 || id >= a->capacity) return 0;
    return a->used[id] == 0;
}

size_t wubu_kv_exact_snapshot_size(const wubu_kv_exact_t *a)
{
    if (!a || a->magic != WUBU_KV_EXACT_MAGIC) return 0;
    return 3u * sizeof(int)
         + (size_t)a->capacity
         + (size_t)a->free_count * sizeof(int);
}

void wubu_kv_exact_snapshot(const wubu_kv_exact_t *a, unsigned char *out)
{
    size_t off;

    if (!a || a->magic != WUBU_KV_EXACT_MAGIC || !out) return;
    off = 0;
    memcpy(out + off, &a->capacity,   sizeof(int)); off += sizeof(int);
    memcpy(out + off, &a->quota,      sizeof(int)); off += sizeof(int);
    memcpy(out + off, &a->free_count, sizeof(int)); off += sizeof(int);
    if (a->capacity > 0) {
        memcpy(out + off, a->used, (size_t)a->capacity);
        off += (size_t)a->capacity;
    }
    if (a->free_count > 0) {
        memcpy(out + off, a->free_ids,
               (size_t)a->free_count * sizeof(int));
    }
}
