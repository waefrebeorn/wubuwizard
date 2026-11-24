#ifndef WUBU_KV_CONNECTOR_H
#define WUBU_KV_CONNECTOR_H

/*
 * wubu_kv_connector.h -- EG-108: LMCache-style engine-agnostic KV connector.
 * Transfer granularity is a chunk (often 256 tokens) that is a multiple of the
 * engine's native block size. Chunks are keyed by a token hash; the connector
 * exposes lookup, pin, compress and unpinned-eviction control operations.
 * This gates the connector contract, not transfer bandwidth or compression
 * quality.
 */

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_KV_CONN_MAX_CHUNKS 64

typedef struct wubu_kv_connector wubu_kv_connector_t;

wubu_kv_connector_t *wubu_kv_connector_create(int chunk_tokens, int block_size);
void wubu_kv_connector_free(wubu_kv_connector_t *p);

int wubu_kv_connector_chunk_tokens(const wubu_kv_connector_t *p);
int wubu_kv_connector_block_size(const wubu_kv_connector_t *p);
int wubu_kv_connector_blocks_per_chunk(const wubu_kv_connector_t *p);
int wubu_kv_connector_chunk_count(const wubu_kv_connector_t *p, int tokens);

int wubu_kv_connector_hash(const wubu_kv_connector_t *p, int n_tokens);

int wubu_kv_connector_store_chunk(wubu_kv_connector_t *p, int hash, int index);
int wubu_kv_connector_lookup(const wubu_kv_connector_t *p, int hash,
                             int start_index, int *out_count);

int wubu_kv_connector_pin(wubu_kv_connector_t *p, int hash);
int wubu_kv_connector_is_pinned(const wubu_kv_connector_t *p, int hash);
int wubu_kv_connector_compress(wubu_kv_connector_t *p, int hash);
int wubu_kv_connector_is_compressed(const wubu_kv_connector_t *p, int hash);
int wubu_kv_connector_evict_unpinned(wubu_kv_connector_t *p);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_KV_CONNECTOR_H */
