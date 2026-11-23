/* test_eg29_blob.c -- EG-29 gate: SHA-256 blob store + manifest. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wubu_blob_store.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    const char *abc = "abc";
    const char *expected = "ba7816bf8f01cfea414140de5dae2223"
                           "b00361a396177a9cb410ff61f20015ad";
    const char *payload = "hello world";
    const char *dir = getenv("TMPDIR");
    char store_dir[512], path[640], back[128], hex[65];
    wubu_digest_t d1, d2;
    wubu_blob_store_t *s;
    long n;
    size_t bytes = 0;
    FILE *f;

    CHECK(wubu_sha256(abc, 3, &d1) == 0);
    printf("EG-29 sha256(abc)=%s\n", d1.hex);
    CHECK(strcmp(d1.hex, expected) == 0);

    snprintf(store_dir, sizeof(store_dir), "%s", dir && *dir ? dir : ".");
    s = wubu_blob_store_create(store_dir);
    CHECK(s != NULL);
    if (!s) return 1;

    CHECK(wubu_blob_put(s, payload, strlen(payload), &d1) == 0);
    CHECK(wubu_blob_put(s, payload, strlen(payload), &d2) == 0); /* dedup */
    CHECK(strcmp(d1.hex, d2.hex) == 0);
    printf("EG-29 blob=%s\n", d1.hex);

    memset(back, 0, sizeof(back));
    n = wubu_blob_get(s, d1.hex, back, sizeof(back));
    CHECK(n == (long)strlen(payload));
    CHECK(strcmp(back, payload) == 0);

    CHECK(wubu_blob_manifest_add(s, "weights/q4_0.bin", d1.hex,
                                 strlen(payload)) == 0);
    memset(hex, 0, sizeof(hex));
    CHECK(wubu_blob_manifest_get(s, "weights/q4_0.bin", hex, sizeof(hex),
                                 &bytes) == 0);
    CHECK(strcmp(hex, d1.hex) == 0 && bytes == strlen(payload));
    CHECK(wubu_blob_manifest_get(s, "missing", hex, sizeof(hex), &bytes) == -1);

    /* Corrupt the blob on disk; reads must reject it by digest. */
    snprintf(path, sizeof(path), "%s/%s.blob", store_dir, d1.hex);
    f = fopen(path, "wb");
    CHECK(f != NULL);
    if (f) { fputc('X', f); fclose(f); }
    CHECK(wubu_blob_get(s, d1.hex, back, sizeof(back)) == -1);
    CHECK(wubu_blob_get(s, "not-a-digest", back, sizeof(back)) == -1);
    CHECK(wubu_blob_put(s, NULL, 0, &d1) == -1);

    wubu_blob_store_free(s);
    remove(path);
    if (failures) {
        fprintf(stderr, "test_eg29_blob: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg29_blob: PASS\n");
    return 0;
}
