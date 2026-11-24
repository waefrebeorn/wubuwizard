/* wubu_kv_connector.c -- EG-108 KV connector gate. Strict C11. */
#include "wubu_kv_connector.h"

#include <stdlib.h>

typedef struct {
    int hash, index, present, pinned, compressed;
} chunk_entry_t;

struct wubu_kv_connector {
    int chunk, block;
    chunk_entry_t entry[WUBU_KV_CONN_MAX_CHUNKS];
};

wubu_kv_connector_t *wubu_kv_connector_create(int chunk_tokens, int block_size)
{
    wubu_kv_connector_t *p;
    if (chunk_tokens < 1 || block_size < 1 ||
        chunk_tokens % block_size != 0) return NULL;
    p = (wubu_kv_connector_t *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->chunk = chunk_tokens;
    p->block = block_size;
    return p;
}

void wubu_kv_connector_free(wubu_kv_connector_t *p) { free(p); }

int wubu_kv_connector_chunk_tokens(const wubu_kv_connector_t *p)
{
    return p ? p->chunk : -1;
}

int wubu_kv_connector_block_size(const wubu_kv_connector_t *p)
{
    return p ? p->block : -1;
}

int wubu_kv_connector_blocks_per_chunk(const wubu_kv_connector_t *p)
{
    return p ? p->chunk / p->block : -1;
}

int wubu_kv_connector_chunk_count(const wubu_kv_connector_t *p, int tokens)
{
    if (!p || tokens < 0) return -1;
    return tokens / p->chunk;
}

int wubu_kv_connector_hash(const wubu_kv_connector_t *p, int n_tokens)
{
    if (!p || n_tokens < 0) return -1;
    return (n_tokens / p->chunk) * 31 + 7;
}

static chunk_entry_t *find(wubu_kv_connector_t *p, int hash, int index)
{
    for (int i = 0; i < WUBU_KV_CONN_MAX_CHUNKS; ++i)
        if (p->entry[i].present && p->entry[i].hash == hash &&
            p->entry[i].index == index) return &p->entry[i];
    return NULL;
}

int wubu_kv_connector_store_chunk(wubu_kv_connector_t *p, int hash, int index)
{
    if (!p || hash < 0 || index < 0) return -1;
    for (int i = 0; i < WUBU_KV_CONN_MAX_CHUNKS; ++i) {
        chunk_entry_t *e = &p->entry[i];
        if (!e->present || (e->hash == hash && e->index == index)) {
            e->hash = hash;
            e->index = index;
            e->present = 1;
            return 0;
        }
    }
    return -1;
}

int wubu_kv_connector_lookup(const wubu_kv_connector_t *p, int hash,
                             int start_index, int *out_count)
{
    int count = 0;
    if (!p || hash < 0 || start_index < 0 || !out_count) return -1;
    for (int i = start_index; i < WUBU_KV_CONN_MAX_CHUNKS; ++i)
        if (find((wubu_kv_connector_t *)p, hash, i)) ++count;
        else break;
    *out_count = count;
    return count;
}

int wubu_kv_connector_pin(wubu_kv_connector_t *p, int hash)
{
    int found = 0;
    if (!p || hash < 0) return -1;
    for (int i = 0; i < WUBU_KV_CONN_MAX_CHUNKS; ++i)
        if (p->entry[i].present && p->entry[i].hash == hash) {
            p->entry[i].pinned = 1;
            found = 1;
        }
    return found ? 0 : -1;
}

int wubu_kv_connector_is_pinned(const wubu_kv_connector_t *p, int hash)
{
    if (!p || hash < 0) return -1;
    for (int i = 0; i < WUBU_KV_CONN_MAX_CHUNKS; ++i)
        if (p->entry[i].present && p->entry[i].hash == hash)
            return p->entry[i].pinned ? 1 : 0;
    return 0;
}

int wubu_kv_connector_compress(wubu_kv_connector_t *p, int hash)
{
    int found = 0;
    if (!p || hash < 0) return -1;
    for (int i = 0; i < WUBU_KV_CONN_MAX_CHUNKS; ++i)
        if (p->entry[i].present && p->entry[i].hash == hash) {
            p->entry[i].compressed = 1;
            found = 1;
        }
    return found ? 0 : -1;
}

int wubu_kv_connector_is_compressed(const wubu_kv_connector_t *p, int hash)
{
    if (!p || hash < 0) return -1;
    for (int i = 0; i < WUBU_KV_CONN_MAX_CHUNKS; ++i)
        if (p->entry[i].present && p->entry[i].hash == hash)
            return p->entry[i].compressed ? 1 : 0;
    return 0;
}

int wubu_kv_connector_evict_unpinned(wubu_kv_connector_t *p)
{
    int evicted = 0;
    if (!p) return -1;
    for (int i = 0; i < WUBU_KV_CONN_MAX_CHUNKS; ++i)
        if (p->entry[i].present && !p->entry[i].pinned) {
            p->entry[i].present = 0;
            ++evicted;
        }
    return evicted;
}
