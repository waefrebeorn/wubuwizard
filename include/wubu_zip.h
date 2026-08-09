/* wubu_zip.h -- ZIP archive reader (ours: central directory + zlib).
 *
 * The key that unlocks ALL office formats: docx/xlsx/pptx/odt are ZIP
 * archives of XML parts. We parse the central directory ourselves and
 * inflate entries with zlib (system library). The user's directive:
 * "use whatever containers have information available" — the ZIP
 * container is public (APPNOTE), no licensing.
 *
 * C11, self-contained (zlib only).
 */
#ifndef WUBU_ZIP_H
#define WUBU_ZIP_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* A single ZIP entry. */
typedef struct {
    char name[256];          /* entry path, e.g. "word/document.xml" */
    uint32_t crc32;
    size_t comp_size;
    size_t uncomp_size;
    size_t local_off;        /* local header offset (for data) */
    int method;              /* 0 = stored, 8 = deflate */
    int is_dir;
} wubu_zip_entry_t;

typedef struct wubu_zip wubu_zip_t;

/* Open a ZIP archive from memory. Lists entries (central directory).
 * Returns NULL on parse failure. */
wubu_zip_t *wubu_zip_open(const unsigned char *data, size_t n);

/* How many entries. */
size_t wubu_zip_count(const wubu_zip_t *z);

/* Get entry i. */
const wubu_zip_entry_t *wubu_zip_entry(const wubu_zip_t *z, size_t i);

/* Find an entry by name (exact). Returns index or -1. */
int wubu_zip_find(const wubu_zip_t *z, const char *name);

/* Extract entry i: returns malloc'd buffer (caller frees) + size.
 * Returns 0 on success. */
int wubu_zip_extract(const wubu_zip_t *z, const unsigned char *data,
                     size_t n, size_t i, unsigned char **out, size_t *outlen);

/* Free the archive listing. */
void wubu_zip_free(wubu_zip_t *z);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_ZIP_H */
