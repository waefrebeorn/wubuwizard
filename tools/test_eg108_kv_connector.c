/* test_eg108_kv_connector.c -- EG-108 gate: LMCache-style KV connector. */
#include <stdio.h>
#include <string.h>

#include "wubu_kv_connector.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    wubu_kv_connector_t *p = wubu_kv_connector_create(256, 16);
    int hash = 0;
    int blocks;
    CHECK(p != NULL);
    CHECK(wubu_kv_connector_chunk_tokens(p) == 256);
    CHECK(wubu_kv_connector_block_size(p) == 16);
    CHECK(wubu_kv_connector_blocks_per_chunk(p) == 16);

    hash = wubu_kv_connector_hash(p, 600);             /* 2 chunks + 88 tail */
    CHECK(wubu_kv_connector_chunk_count(p, 600) == 2);
    CHECK(wubu_kv_connector_store_chunk(p, hash, 0) == 0);
    CHECK(wubu_kv_connector_store_chunk(p, hash, 1) == 0);
    CHECK(wubu_kv_connector_lookup(p, hash, 0, &blocks) == 2);
    CHECK(blocks == 2);
    CHECK(wubu_kv_connector_pin(p, hash) == 0);
    CHECK(wubu_kv_connector_is_pinned(p, hash) == 1);
    CHECK(wubu_kv_connector_compress(p, hash) == 0);
    CHECK(wubu_kv_connector_is_compressed(p, hash) == 1);
    CHECK(wubu_kv_connector_evict_unpinned(p) == 0);   /* pinned chunk survives */
    CHECK(wubu_kv_connector_lookup(p, hash, 0, &blocks) == 2);
    printf("EG-108 chunk=256 block=16 hash=%d lookup=2 pinned+compressed survives evict\n", hash);

    CHECK(wubu_kv_connector_create(10, 16) == NULL);   /* chunk % block != 0 */
    CHECK(wubu_kv_connector_lookup(p, -1, 0, &blocks) == -1);
    CHECK(wubu_kv_connector_hash(NULL, 10) == -1);

    wubu_kv_connector_free(p);
    if (failures) { fprintf(stderr, "test_eg108_kv_connector: FAIL (%d)\n", failures); return 1; }
    printf("test_eg108_kv_connector: PASS\n");
    return 0;
}
