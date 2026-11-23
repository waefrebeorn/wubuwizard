/* wubu_conv_store.c -- EG-99 content-addressed conversation blocks. C11. */
#include "wubu_conv_store.h"

#include <stdlib.h>

struct wubu_conv_store {
    size_t block_bytes;
    int turn_states, n, cp_n, root_kept, turn_n;
    uint64_t keys[WUBU_CONV_MAX_FILES];
    int cp[WUBU_CONV_MAX_CP];
};

static uint64_t fnv(uint64_t h, const unsigned char *p, size_t n)
{
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

static uint64_t block_key(uint64_t prefix, const unsigned char *data, size_t n)
{
    unsigned char p[8];
    for (int i = 0; i < 8; ++i) p[i] = (unsigned char)(prefix >> (8 * i));
    return fnv(fnv(1469598103934665603ULL, p, 8), data, n);
}

wubu_conv_store_t *wubu_conv_store_create(size_t block_bytes, int turn_states)
{
    wubu_conv_store_t *p;
    if (block_bytes == 0 || turn_states < 1 || turn_states >= WUBU_CONV_MAX_CP)
        return NULL;
    p = (wubu_conv_store_t *)calloc(1, sizeof(*p));
    if (p) { p->block_bytes = block_bytes; p->turn_states = turn_states; }
    return p;
}

void wubu_conv_store_free(wubu_conv_store_t *p) { free(p); }

int wubu_conv_store_write(wubu_conv_store_t *p, uint64_t prefix_id,
                          const unsigned char *data, size_t bytes)
{
    size_t off = 0;
    int added = 0;
    if (!p || !data || (bytes && bytes < p->block_bytes)) return -1;
    while (off < bytes) {
        size_t n = p->block_bytes < bytes - off ? p->block_bytes : bytes - off;
        uint64_t key = block_key(prefix_id, data + off, n);
        int found = 0;
        for (int i = 0; i < p->n; ++i) if (p->keys[i] == key) { found = 1; break; }
        if (!found) {
            if (p->n >= WUBU_CONV_MAX_FILES) return -1;
            p->keys[p->n++] = key;
            ++added;
        }
        off += n;
    }
    return added;
}

int wubu_conv_store_file_count(const wubu_conv_store_t *p)
{
    return p ? p->n : -1;
}

int wubu_conv_store_has_block(const wubu_conv_store_t *p, uint64_t prefix_id,
                              const unsigned char *data, size_t bytes)
{
    uint64_t key;
    if (!p || !data || bytes == 0) return -1;
    key = block_key(prefix_id, data, bytes);
    for (int i = 0; i < p->n; ++i) if (p->keys[i] == key) return 1;
    return 0;
}

int wubu_conv_store_checkpoint(wubu_conv_store_t *p, int is_root, int token_pos)
{
    if (!p || token_pos < 0) return -1;
    if (is_root) {
        p->root_kept = 1;
        p->cp[0] = token_pos;
    } else {
        if (p->turn_n == p->turn_states) {
            for (int i = 1; i < p->turn_n; ++i) p->cp[i] = p->cp[i + 1];
            --p->turn_n;
        }
        p->cp[1 + p->turn_n++] = token_pos;
    }
    return 0;
}

int wubu_conv_store_checkpoint_count(const wubu_conv_store_t *p)
{
    return p ? (p->root_kept ? 1 : 0) + p->turn_n : -1;
}
