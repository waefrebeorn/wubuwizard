/* wubu_kvfs.c — KV namespace layer (G1 implementation)
 *
 * Path-addressable KV cache. Every mount is a directory
 * entry mapping a path prefix to a contiguous block range
 * in the flat KV tensor. Read and write route through
 * the mount table — the same address translation as
 * PagedAttention (SOSP'23), just exposed as a filesystem.
 *
 * Speed-kernel design (why it's fast):
 *   - The mount table is an FNV-1a open-addressing hash table,
 *     so path resolution is O(1) average — not a linear scan.
 *   - Longest-prefix lookup walks parent segments, one hash
 *     probe per segment (path depth, not mount count).
 *   - wubu_kvfs_open() resolves once into a handle that carries
 *     the precomputed absolute float offset + capacity; the hot
 *     read/write is a bounds check + memcpy. Zero string ops.
 *   - abs_offset/abs_limit are computed at mount time and stored
 *     in the (cold) mount struct — no multiply per I/O.
 *
 * C11, opaque struct, minimal includes. No third-party deps.
 */
#define _POSIX_C_SOURCE 200809L
#include "wubu_kvfs.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define KVFS_SLOT_EMPTY 0u
#define KVFS_SLOT_LIVE  1u
#define KVFS_SLOT_TOMB  2u

/* ---- namespace handle ---- */
struct wubu_kvfs {
    wubu_kvfs_mount_t *mounts;     /* mount slots (some may be dead) */
    int                 n_mounts;  /* slots used in mounts[] */
    int                 cap_mounts;
    int                 n_live;    /* live mounts (mount_count) */
    uint32_t            block_size;
    uint32_t            total_blocks;
    uint32_t            used_blocks;
    /* open-addressing hash: path FNV-1a -> mount index */
    uint64_t           *hash_keys;
    int32_t            *hash_vals;  /* index into mounts[], -1 unused */
    uint8_t            *hash_state; /* EMPTY / LIVE / TOMB */
    int                 n_slots;
};

/* A resolved handle: precomputed absolute offset + capacity.
 * Nothing else — the hot path derefs two size_t and memcpys. */
struct wubu_kvfs_handle {
    size_t abs_offset;   /* float offset into the flat KV tensor */
    size_t capacity;     /* floats available from abs_offset */
};

/* ---- FNV-1a 64-bit ---- */
static uint64_t fnv1a(const char *s) {
    uint64_t h = UINT64_C(14695981039346656037);
    while (*s) {
        h ^= (unsigned char)*s++;
        h *= UINT64_C(1099511628211);
    }
    return h;
}

/* Probe the hash table for `path`. Returns the mount index
 * (>= 0) or -1. Skips tombstones; stops at empty slot. */
static int hash_find(const struct wubu_kvfs *fs, const char *path) {
    uint64_t h = fnv1a(path);
    uint32_t mask = (uint32_t)fs->n_slots - 1;
    for (uint32_t i = (uint32_t)h & mask; ; i = (i + 1) & mask) {
        uint8_t st = fs->hash_state[i];
        if (st == KVFS_SLOT_EMPTY) return -1;
        if (st == KVFS_SLOT_LIVE && fs->hash_keys[i] == h &&
            strcmp(fs->mounts[fs->hash_vals[i]].path, path) == 0)
            return fs->hash_vals[i];
    }
}

/* Insert `m_idx` under `path`. Returns 0, or -1 if the path is
 * already live (duplicate mount). Reuses the first tombstone or
 * empty slot found by probing. */
static int hash_insert(struct wubu_kvfs *fs, const char *path, int m_idx) {
    uint64_t h = fnv1a(path);
    uint32_t mask = (uint32_t)fs->n_slots - 1;
    int first_tomb = -1;
    for (uint32_t i = (uint32_t)h & mask; ; i = (i + 1) & mask) {
        uint8_t st = fs->hash_state[i];
        if (st == KVFS_SLOT_LIVE && fs->hash_keys[i] == h &&
            strcmp(fs->mounts[fs->hash_vals[i]].path, path) == 0)
            return -1; /* duplicate */
        if (st == KVFS_SLOT_TOMB) {
            if (first_tomb < 0) first_tomb = (int)i;
            continue;
        }
        if (st == KVFS_SLOT_EMPTY) {
            uint32_t slot = (first_tomb >= 0) ? (uint32_t)first_tomb : i;
            fs->hash_keys[slot]   = h;
            fs->hash_vals[slot]   = m_idx;
            fs->hash_state[slot]  = KVFS_SLOT_LIVE;
            return 0;
        }
    }
}

/* Remove `path`. Returns the removed mount index, or -1. */
static int hash_remove(struct wubu_kvfs *fs, const char *path) {
    uint64_t h = fnv1a(path);
    uint32_t mask = (uint32_t)fs->n_slots - 1;
    for (uint32_t i = (uint32_t)h & mask; ; i = (i + 1) & mask) {
        uint8_t st = fs->hash_state[i];
        if (st == KVFS_SLOT_EMPTY) return -1;
        if (st == KVFS_SLOT_LIVE && fs->hash_keys[i] == h &&
            strcmp(fs->mounts[fs->hash_vals[i]].path, path) == 0) {
            fs->hash_state[i] = KVFS_SLOT_TOMB;
            return fs->hash_vals[i];
        }
    }
}

/* Rehash everything into a new table (doubled). */
static int hash_grow(struct wubu_kvfs *fs) {
    int new_slots = fs->n_slots * 2;
    uint64_t *nk = (uint64_t *)calloc((size_t)new_slots, sizeof(*nk));
    int32_t  *nv = (int32_t  *)malloc((size_t)new_slots * sizeof(*nv));
    uint8_t  *ns = (uint8_t  *)calloc((size_t)new_slots, sizeof(*ns));
    if (!nk || !nv || !ns) { free(nk); free(nv); free(ns); return -1; }
    /* stash old, install new */
    uint64_t *ok = fs->hash_keys;  int32_t *ov = fs->hash_vals;
    uint8_t  *os = fs->hash_state; int n_old = fs->n_slots;
    fs->hash_keys  = nk; fs->hash_vals = nv; fs->hash_state = ns;
    fs->n_slots = new_slots;
    for (int i = 0; i < n_old; i++) {
        if (os[i] == KVFS_SLOT_LIVE) {
            int idx = ov[i];
            if (hash_insert(fs, fs->mounts[idx].path, idx) < 0) {
                /* shouldn't happen: rehash of live keys, no dups */
            }
        }
    }
    free(ok); free(ov); free(os);
    return 0;
}

wubu_kvfs_t *wubu_kvfs_create(uint32_t block_size, uint32_t total_blocks) {
    wubu_kvfs_t *fs = (wubu_kvfs_t *)calloc(1, sizeof(*fs));
    if (!fs) return NULL;
    fs->block_size   = block_size;
    fs->total_blocks = total_blocks;
    fs->cap_mounts   = 8;
    fs->mounts = (wubu_kvfs_mount_t *)calloc(fs->cap_mounts, sizeof(*fs->mounts));
    if (!fs->mounts) { free(fs); return NULL; }
    fs->n_slots = 16; /* load factor capped at 50% before growth */
    fs->hash_keys  = (uint64_t *)calloc((size_t)fs->n_slots, sizeof(*fs->hash_keys));
    fs->hash_vals  = (int32_t  *)malloc((size_t)fs->n_slots * sizeof(*fs->hash_vals));
    fs->hash_state = (uint8_t  *)calloc((size_t)fs->n_slots, sizeof(*fs->hash_state));
    if (!fs->hash_keys || !fs->hash_vals || !fs->hash_state) {
        free(fs->hash_keys); free(fs->hash_vals); free(fs->hash_state);
        free(fs->mounts); free(fs);
        return NULL;
    }
    return fs;
}

static int grow_mounts(wubu_kvfs_t *fs) {
    int new_cap = fs->cap_mounts * 2;
    wubu_kvfs_mount_t *tmp = (wubu_kvfs_mount_t *)realloc(fs->mounts,
                                    new_cap * sizeof(*tmp));
    if (!tmp) return -1;
    fs->mounts = tmp;
    memset(fs->mounts + fs->cap_mounts, 0,
           (new_cap - fs->cap_mounts) * sizeof(*tmp));
    fs->cap_mounts = new_cap;
    return 0;
}

int wubu_kvfs_mount(wubu_kvfs_t *fs, const char *path,
                      uint32_t start_block, uint32_t n_blocks) {
    if (!fs || !path || !*path) return -1;
    if (n_blocks == 0) return -1;
    if ((uint64_t)start_block + n_blocks > fs->total_blocks) return -1;
    /* Reject paths that won't fit in the mount struct — snprintf
     * would silently truncate, creating a ghost mount whose path
     * is unresolvable via find_mount() but whose blocks are still
     * counted in used_blocks. */
    if (strlen(path) >= sizeof(((wubu_kvfs_mount_t *)0)->path)) return -1;
    if (hash_find(fs, path) >= 0) return -1; /* duplicate */

    /* keep load factor under 50% — grow BEFORE touching the slot so a
     * failed grow cannot leave a written-but-unregistered ghost mount. */
    if (fs->n_live + 1 > fs->n_slots / 2) {
        if (hash_grow(fs) < 0) return -1;
    }

    /* find a free slot (reuse a dead mount) or append */
    int m_idx = -1;
    for (int i = 0; i < fs->n_mounts; i++) {
        if (fs->mounts[i].path[0] == '\0') { m_idx = i; break; }
    }
    if (m_idx < 0) {
        if (fs->n_mounts >= fs->cap_mounts) {
            if (grow_mounts(fs) < 0) return -1;
        }
        m_idx = fs->n_mounts++;
    }

    wubu_kvfs_mount_t *m = &fs->mounts[m_idx];
    memset(m, 0, sizeof(*m));
    snprintf(m->path, sizeof(m->path), "%s", path);
    m->start_block = start_block;
    m->n_blocks    = n_blocks;
    m->block_size  = fs->block_size;
    /* precompute the absolute float range — the hot path multiplies
     * nothing; it only bounds-checks against these two values. */
    m->abs_offset  = (size_t)start_block * fs->block_size;
    m->abs_limit   = (size_t)(start_block + n_blocks) * fs->block_size;

    if (hash_insert(fs, path, m_idx) < 0) {
        /* roll back the slot write — no ghost mount, no leaked slot */
        memset(m, 0, sizeof(*m));
        if (m_idx == fs->n_mounts - 1) fs->n_mounts--; /* un-append */
        return -1;
    }
    fs->n_live++;
    fs->used_blocks += n_blocks;
    return 0;
}

int wubu_kvfs_unmount(wubu_kvfs_t *fs, const char *path) {
    if (!fs || !path) return -1;
    int m_idx = hash_remove(fs, path);
    if (m_idx < 0) return -1;
    fs->used_blocks -= fs->mounts[m_idx].n_blocks;
    fs->n_live--;
    fs->mounts[m_idx].path[0] = '\0'; /* mark dead; slot reused later */
    return 0;
}

/* Find the mount that owns this path: exact hash hit first, then
 * walk up parent segments (one hash probe per segment). This is
 * the same longest-prefix semantics as before, but O(path depth)
 * instead of O(mount count) — no linear scan, no strncmp storms. */
static const wubu_kvfs_mount_t *find_mount(const wubu_kvfs_t *fs,
                                               const char *path) {
    if (!fs || !path || !*path) return NULL;
    char buf[256];
    size_t plen = strlen(path);
    if (plen >= sizeof(buf)) return NULL;
    memcpy(buf, path, plen + 1);
    for (;;) {
        int m_idx = hash_find(fs, buf);
        if (m_idx >= 0 && fs->mounts[m_idx].path[0])
            return &fs->mounts[m_idx];
        char *slash = strrchr(buf, '/');
        if (!slash) return NULL;
        if (slash == buf) {
            if (buf[1] == '\0') return NULL; /* "/" not mounted */
            buf[1] = '\0';                    /* try root */
            continue;
        }
        *slash = '\0';
    }
}

int wubu_kvfs_lookup(const wubu_kvfs_t *fs, const char *path,
                       uint32_t *out_block, size_t *out_offset) {
    const wubu_kvfs_mount_t *m = find_mount(fs, path);
    if (!m) return -1;
    if (out_block) *out_block = m->start_block;
    if (out_offset) *out_offset = 0;
    return 0;
}

/* ---- handle API: resolve once, use many ---- */
wubu_kvfs_handle_t *wubu_kvfs_open(const wubu_kvfs_t *fs, const char *path) {
    if (!fs || !path) return NULL;
    const wubu_kvfs_mount_t *m = find_mount(fs, path);
    if (!m) return NULL;
    wubu_kvfs_handle_t *h = (wubu_kvfs_handle_t *)malloc(sizeof(*h));
    if (!h) return NULL;
    h->abs_offset = m->abs_offset;
    h->capacity   = m->abs_limit - m->abs_offset;
    return h;
}

int wubu_kvfs_handle_read(const wubu_kvfs_handle_t *h,
                          const float *kv_base, float *dst, size_t n_floats) {
    if (!h || !kv_base || !dst || n_floats > h->capacity) return -1;
    memcpy(dst, kv_base + h->abs_offset, n_floats * sizeof(float));
    return 0;
}

int wubu_kvfs_handle_write(const wubu_kvfs_handle_t *h,
                           float *kv_base, const float *src, size_t n_floats) {
    if (!h || !kv_base || !src || n_floats > h->capacity) return -1;
    memcpy(kv_base + h->abs_offset, src, n_floats * sizeof(float));
    return 0;
}

size_t wubu_kvfs_handle_offset(const wubu_kvfs_handle_t *h) {
    return h ? h->abs_offset : 0;
}

size_t wubu_kvfs_handle_capacity(const wubu_kvfs_handle_t *h) {
    return h ? h->capacity : 0;
}

void wubu_kvfs_handle_close(wubu_kvfs_handle_t *h) {
    free(h);
}

uint32_t wubu_kvfs_block_size(const wubu_kvfs_t *fs) {
    return fs ? fs->block_size : 0;
}

/* ---- the path-based convenience API ---- */
int wubu_kvfs_read(const wubu_kvfs_t *fs, const char *path,
                       const float *kv_base, float *dst, size_t n_floats) {
    if (!fs || !path || !kv_base || !dst || n_floats == 0) return -1;
    const wubu_kvfs_mount_t *m = find_mount(fs, path);
    if (!m) return -1;
    /* Subtraction form: abs_offset + n_floats > abs_limit
     * is equivalent to n_floats > abs_limit - abs_offset,
     * but avoids overflow when abs_offset is large. */
    if (n_floats > m->abs_limit - m->abs_offset) return -1;
    memcpy(dst, kv_base + m->abs_offset, n_floats * sizeof(float));
    return 0;
}

int wubu_kvfs_write(wubu_kvfs_t *fs, const char *path,
                        float *kv_base, const float *src, size_t n_floats) {
    if (!fs || !path || !kv_base || !src || n_floats == 0) return -1;
    const wubu_kvfs_mount_t *m = find_mount(fs, path);
    if (!m) return -1;
    /* Subtraction form avoids overflow vs. addition form. */
    if (n_floats > m->abs_limit - m->abs_offset) return -1;
    memcpy(kv_base + m->abs_offset, src, n_floats * sizeof(float));
    return 0;
}

char *wubu_kvfs_snapshot_json(const wubu_kvfs_t *fs, size_t *out_len) {
    if (!fs) return NULL;
    /* Worst-case: every mount is a 255-char path + full JSON
     * key/value overhead ≈ 120 bytes per mount. */
    size_t buf_cap = 4096 + fs->n_live * 400;
    char *buf = (char *)malloc(buf_cap);
    if (!buf) return NULL;
    size_t pos = 0;
    pos += (size_t)snprintf(buf + pos, buf_cap - pos,
            "{\"block_size\":%u,\"total_blocks\":%u,"
            "\"used_blocks\":%u,\"registered\":%d,\"mounts\":[",
            fs->block_size, fs->total_blocks,
            fs->used_blocks, fs->n_live);
    int first = 1;
    for (int i = 0; i < fs->n_mounts; i++) {
        const wubu_kvfs_mount_t *m = &fs->mounts[i];
        if (m->path[0] == '\0') continue; /* dead */
        if (!first) pos += (size_t)snprintf(buf + pos, buf_cap - pos, ",");
        first = 0;
        pos += (size_t)snprintf(buf + pos, buf_cap - pos,
                "{\"path\":\"%s\",\"start_block\":%u,\"n_blocks\":%u,"
                "\"abs_offset\":%zu,\"abs_limit\":%zu}",
                m->path, m->start_block, m->n_blocks,
                m->abs_offset, m->abs_limit);
    }
    pos += (size_t)snprintf(buf + pos, buf_cap - pos, "]}");
    if (out_len) *out_len = pos;
    return buf;
}

int wubu_kvfs_mount_count(const wubu_kvfs_t *fs) {
    return fs ? fs->n_live : 0;
}

void wubu_kvfs_free(wubu_kvfs_t *fs) {
    if (!fs) return;
    free(fs->mounts);
    free(fs->hash_keys);
    free(fs->hash_vals);
    free(fs->hash_state);
    free(fs);
}
