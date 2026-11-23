#ifndef WUBU_BLOB_STORE_H
#define WUBU_BLOB_STORE_H

/*
 * wubu_blob_store.h -- EG-29: content-addressed SHA-256 blob store with a
 * small manifest (Ollama manifest/blob layout lineage), reimplemented C11.
 *
 * Blobs are named by their SHA-256 hex digest. Writes go to a temporary file
 * and are renamed into place, so a partial write can be resumed safely and
 * identical content is stored once. Reads re-hash the bytes and reject a
 * corrupted blob.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    unsigned char bytes[32];
    char hex[65];
} wubu_digest_t;

typedef struct wubu_blob_store wubu_blob_store_t;

int wubu_sha256(const void *data, size_t len, wubu_digest_t *out);

wubu_blob_store_t *wubu_blob_store_create(const char *dir);
void wubu_blob_store_free(wubu_blob_store_t *s);

/* Store bytes; returns 0 and fills digest. Existing identical blob is a no-op. */
int wubu_blob_put(wubu_blob_store_t *s, const void *data, size_t len,
                  wubu_digest_t *digest);

/* Read and verify a blob by hex digest. Returns bytes or -1. */
long wubu_blob_get(wubu_blob_store_t *s, const char *hex,
                   void *buf, size_t cap);

/* Add/look up a manifest entry (virtual path -> digest + byte count). */
int wubu_blob_manifest_add(wubu_blob_store_t *s, const char *path,
                           const char *hex, size_t bytes);
int wubu_blob_manifest_get(wubu_blob_store_t *s, const char *path,
                           char *hex, size_t hex_cap, size_t *bytes);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_BLOB_STORE_H */
