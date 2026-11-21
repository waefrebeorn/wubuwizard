/*
 * wubu_state_version.h -- versioned opaque state-file boundary (LP-16).
 *
 * Problem: opaque state files and session serialisation drift across builds.
 * A newer reader silently misreads an older blob (or vice versa) and produces
 * garbage. This module puts a small, fixed, endian-stable header in front of
 * every state file:
 *
 *     magic[8] | version(u32 LE) | size(u32 LE) | crc32(u32 LE) | payload
 *
 * The reader validates magic, size, and CRC32 before it will hand back an
 * object -- there is no partial load on corruption. A file whose version is
 * older than the current one is refused with a *named* reason and the reader
 * reports a *named migration target* so callers can say "this is v1, migrate
 * to v2" instead of failing opaquely.
 *
 * Strict C11, opaque state at the seam, status-as-value, no third-party deps.
 */
#ifndef WUBU_STATE_VERSION_H
#define WUBU_STATE_VERSION_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque state object: version + owned payload. Never exposed across the seam. */
typedef struct wubu_state_version wubu_state_version_t;

/* Version this build writes and reads. Bump with a matching entry in the
 * migration table in the implementation. */
#define WUBU_SV_CURRENT_VERSION 2u

/* Hard cap on payload bytes accepted from a file (guards a forged size field). */
#define WUBU_SV_MAX_PAYLOAD (16u * 1024u * 1024u)

/* Named outcome codes. Values are stable ABI; append only. */
typedef enum {
    WUBU_SV_OK = 0,
    WUBU_SV_ERR_ARG = 1,
    WUBU_SV_ERR_IO = 2,
    WUBU_SV_ERR_BAD_MAGIC = 3,
    WUBU_SV_ERR_TRUNCATED = 4,
    WUBU_SV_ERR_CRC = 5,
    WUBU_SV_ERR_SIZE = 6,
    WUBU_SV_ERR_STALE = 7,
    WUBU_SV_ERR_NEWER = 8
} wubu_state_version_status_t;

/* Result of a load (also serves as a version inspection). Plain value type,
 * safe to copy. `reason` is the stable named reason for `code`; on a stale
 * file `migration_target` names the version to migrate to (e.g. "v2"). */
typedef struct wubu_sv_result {
    wubu_state_version_status_t code;
    char reason[32];
    uint32_t file_version;
    uint32_t current_version;
    char migration_target[24];
} wubu_sv_result_t;

/* Canonical named reason for a status code (never NULL). */
const char *wubu_state_version_status_name(wubu_state_version_status_t code);

/* Initialise a result to WUBU_SV_OK / empty strings. */
void wubu_state_version_result_init(wubu_sv_result_t *out);

/* Version metadata. */
uint32_t wubu_state_version_current(void);
const char *wubu_state_version_current_name(void);

/* Build a state object with an explicit version and a copied payload.
 * version 0 is invalid. Returns NULL on bad args / OOM. Use the explicit
 * version to author a legacy-format file (e.g. version 1) for migration
 * testing; normal callers should use wubu_state_version_create_current(). */
wubu_state_version_t *wubu_state_version_create(uint32_t version,
                                                 const void *payload,
                                                 size_t len);
wubu_state_version_t *wubu_state_version_create_current(const void *payload,
                                                        size_t len);
void wubu_state_version_free(wubu_state_version_t *s);

uint32_t wubu_state_version_get_version(const wubu_state_version_t *s);
/* Borrowed pointer; *out_len set to payload length. */
const void *wubu_state_version_payload(const wubu_state_version_t *s,
                                       size_t *out_len);

/* Write the versioned header + CRC-protected payload. 0 on success. */
int wubu_state_version_save(const wubu_state_version_t *s, const char *path);

/* Load and fully validate a state file. Returns NULL on ANY failure with no
 * partial object. `out` (if non-NULL) receives the named outcome; on a stale
 * file it carries WUBU_SV_ERR_STALE, reason "stale version", and a named
 * migration_target. */
wubu_state_version_t *wubu_state_version_load(const char *path,
                                               wubu_sv_result_t *out);

/* Inspect only the header of a file without loading/validating the payload.
 * Useful to report a migration target before deciding whether to load. */
wubu_state_version_status_t wubu_state_version_inspect(const char *path,
                                                       wubu_sv_result_t *out);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_STATE_VERSION_H */
