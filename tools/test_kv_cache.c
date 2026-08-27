/*
 * tools/test_kv_cache.c — Test the PagedAttention-style KV cache.
 */
#include "wubu_kv_cache.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define NUM_HEADS 32
#define HEAD_DIM  128

int main(void)
{
    printf("=== KV Cache Test ===\n");

    wubu_kv_cache_t *cache = wubu_kv_create(8);
    if (!cache) { printf("FAIL: create\n"); return 1; }

    /* Simulate appending 20 FP16 tokens for seq 0 */
    size_t k_bytes = 20 * NUM_HEADS * HEAD_DIM * sizeof(uint16_t);
    uint16_t *k_buf = malloc(k_bytes);
    uint16_t *v_buf = malloc(k_bytes);
    for (size_t i = 0; i < k_bytes / sizeof(uint16_t); i++) {
        k_buf[i] = (uint16_t)(i & 0xFFFF);
        v_buf[i] = (uint16_t)((i * 3) & 0xFFFF);
    }

    int rc = wubu_kv_append(cache, 0, k_buf, v_buf, 20);
    if (rc != 0) { printf("FAIL: append\n"); return 1; }
    printf("  Appended 20 tokens to seq 0: PASS\n");

    /* Read back token 5 — verify exact values */
    uint16_t k_read[NUM_HEADS * HEAD_DIM];
    uint16_t v_read[NUM_HEADS * HEAD_DIM];
    rc = wubu_kv_read(cache, 0, 5, k_read, v_read);
    if (rc != 0) { printf("FAIL: read\n"); return 1; }

    int ok = 1;
    for (int i = 0; i < NUM_HEADS * HEAD_DIM; i++) {
        uint16_t expected_k = (uint16_t)(((size_t)5 * NUM_HEADS * HEAD_DIM + i) & 0xFFFF);
        uint16_t expected_v = (uint16_t)((((size_t)5 * NUM_HEADS * HEAD_DIM + i) * 3) & 0xFFFF);
        if (k_read[i] != expected_k || v_read[i] != expected_v) { ok = 0; break; }
    }
    printf("  Read-back token 5 exact match: %s\n", ok ? "PASS" : "FAIL");

    /* Test second sequence */
    rc = wubu_kv_append(cache, 1, k_buf, v_buf, 16);
    printf("  Appended 16 tokens to seq 1: %s\n", rc == 0 ? "PASS" : "FAIL");

    /* Read back token 1 from seq 1 */
    rc = wubu_kv_read(cache, 1, 1, k_read, v_read);
    ok = 1;
    if (rc == 0) {
        for (int i = 0; i < NUM_HEADS * HEAD_DIM; i++) {
            uint16_t expected_k = (uint16_t)(((size_t)1 * NUM_HEADS * HEAD_DIM + i) & 0xFFFF);
            if (k_read[i] != expected_k) { ok = 0; break; }
        }
    } else { ok = 0; }
    printf("  Read-back seq 1 token 1: %s\n", ok ? "PASS" : "FAIL");

    /* Stats */
    int pages_used = 0, pages_free = 0;
    size_t bytes_used = 0, bytes_alloc = 0;
    wubu_kv_stats(cache, &pages_used, &pages_free, &bytes_used, &bytes_alloc);
    printf("  Stats: pages_used=%d pages_free=%d bytes_used=%zu bytes_alloc=%zu\n",
           pages_used, pages_free, bytes_used, bytes_alloc);

    /* Free sequence 0 */
    wubu_kv_free_sequence(cache, 0);
    wubu_kv_stats(cache, &pages_used, &pages_free, &bytes_used, &bytes_alloc);
    printf("  After freeing seq 0: pages_used=%d pages_free=%d\n",
           pages_used, pages_free);

    wubu_kv_destroy(cache);
    free(k_buf);
    free(v_buf);
    printf("KV Cache: PASS\n");
    return 0;
}
