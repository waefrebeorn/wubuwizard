/* test_eg16_sleep.c -- EG-16 gate: tagged sleep/wake with partial restore. */
#include <stdio.h>
#include <string.h>

#include "wubu_sleep_alloc.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

enum { TAG_WEIGHTS = 1u, TAG_KV = 2u, TAG_SCRATCH = 4u, TAG_PERSIST = 8u };

static void fill(unsigned char *p, size_t n, unsigned char seed) {
    for (size_t i = 0; i < n; i++) p[i] = (unsigned char)(seed + i * 7u);
}
static int same(const unsigned char *p, size_t n, unsigned char seed) {
    for (size_t i = 0; i < n; i++)
        if (p[i] != (unsigned char)(seed + i * 7u)) return 0;
    return 1;
}

int main(void) {
    wubu_sleep_alloc_t *a = wubu_sleep_alloc_create(16384);
    unsigned char *w, *kv, *sc, *pe;
    CHECK(a != NULL);
    if (!a) return 1;

    w  = (unsigned char *)wubu_sleep_alloc_push(a, "weights", TAG_WEIGHTS, 1024);
    kv = (unsigned char *)wubu_sleep_alloc_push(a, "kv", TAG_KV, 2048);
    sc = (unsigned char *)wubu_sleep_alloc_push(a, "scratch", TAG_SCRATCH, 4096);
    pe = (unsigned char *)wubu_sleep_alloc_push(a, "persist", TAG_PERSIST, 512);
    CHECK(w && kv && sc && pe);
    CHECK(wubu_sleep_alloc_nblocks(a) == 4);
    CHECK(wubu_sleep_alloc_device_used(a) == 7680);
    fill(w, 1024, 11); fill(kv, 2048, 31);
    fill(sc, 4096, 71); fill(pe, 512, 97);

    /* Sleep only KV + scratch; weights and persistent state stay resident. */
    CHECK(wubu_sleep_alloc_sleep(a, TAG_KV | TAG_SCRATCH) == 2);
    CHECK(wubu_sleep_alloc_device_used(a) == 1536);
    CHECK(wubu_sleep_alloc_host_bytes(a) == 6144);
    CHECK(wubu_sleep_alloc_resident(a, "weights") == 1);
    CHECK(wubu_sleep_alloc_resident(a, "persist") == 1);
    CHECK(wubu_sleep_alloc_resident(a, "kv") == 0);
    CHECK(wubu_sleep_alloc_resident(a, "scratch") == 0);

    /* Simulate device memory reuse while blocks are offloaded. */
    memset(kv, 0xAA, 2048);
    memset(sc, 0xAA, 4096);
    CHECK(same(w, 1024, 11));
    CHECK(same(pe, 512, 97));

    /* Restore KV only. Scratch stays offloaded; data must be byte-exact. */
    CHECK(wubu_sleep_alloc_wake(a, TAG_KV) == 1);
    CHECK(wubu_sleep_alloc_device_used(a) == 3584);
    CHECK(wubu_sleep_alloc_host_bytes(a) == 4096);
    CHECK(same(kv, 2048, 31));
    CHECK(wubu_sleep_alloc_resident(a, "scratch") == 0);

    CHECK(wubu_sleep_alloc_wake(a, TAG_SCRATCH) == 1);
    CHECK(wubu_sleep_alloc_device_used(a) == 7680);
    CHECK(wubu_sleep_alloc_host_bytes(a) == 0);
    CHECK(same(sc, 4096, 71));

    /* No-op masks and bad arguments. */
    CHECK(wubu_sleep_alloc_sleep(a, 0) == -1);
    CHECK(wubu_sleep_alloc_wake(a, 0) == -1);
    CHECK(wubu_sleep_alloc_sleep(a, 0x8000u) == 0);
    CHECK(wubu_sleep_alloc_wake(a, 0x8000u) == 0);
    CHECK(wubu_sleep_alloc_sleep(NULL, 1u) == -1);
    CHECK(wubu_sleep_alloc_device_used(NULL) == 0);

    printf("EG-16 device_used=7680 host=0 after partial wake; blocks=%d\n",
           wubu_sleep_alloc_nblocks(a));
    wubu_sleep_alloc_free(a);
    if (failures) {
        fprintf(stderr, "test_eg16_sleep: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg16_sleep: PASS\n");
    return 0;
}
