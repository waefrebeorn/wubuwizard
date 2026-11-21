/*
 * wubu_state_version.c -- versioned opaque state-file boundary (LP-16).
 * C11, self-contained, status-as-value. See include/wubu_state_version.h.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wubu_state_version.h"

/* Fixed on-disk header: 8-byte magic + three little-endian u32 fields. */
#define WUBU_SV_MAGIC        "WBSV001\0"
#define WUBU_SV_MAGIC_LEN    8u
#define WUBU_SV_HDR_LEN      (WUBU_SV_MAGIC_LEN + 12u)  /* 20 bytes */

struct wubu_state_version {
    uint32_t version;
    unsigned char *payload;
    size_t len;
};

/* Named migration table: a stale version maps to a named target. Append only;
 * keep in sync with WUBU_SV_CURRENT_VERSION. */
typedef struct {
    uint32_t from_version;
    const char *target_name;
} wubu_sv_migration_t;

static const wubu_sv_migration_t WUBU_SV_MIGRATIONS[] = {
    { 1u, "v2" }   /* v1 -> v2: header gained a CRC32 field */
};
#define WUBU_SV_NMIGRATIONS \
    (sizeof(WUBU_SV_MIGRATIONS) / sizeof(WUBU_SV_MIGRATIONS[0]))

/* ---- endian helpers (explicit little-endian, host independent) --------- */
static void put32(unsigned char *p, uint32_t v)
{
    p[0] = (unsigned char)(v & 0xffu);
    p[1] = (unsigned char)((v >> 8) & 0xffu);
    p[2] = (unsigned char)((v >> 16) & 0xffu);
    p[3] = (unsigned char)((v >> 24) & 0xffu);
}

static uint32_t get32(const unsigned char *p)
{
    return (uint32_t)p[0] |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

/* CRC-32 (IEEE 802.3, reflected, poly 0xEDB88320). Bitwise and stateless --
 * no lookup table, no init flag, no thread-safety hazard, no third party. */
static uint32_t crc32_bytes(const unsigned char *data, size_t len)
{
    uint32_t crc = 0xffffffffu;
    size_t i;
    for (i = 0; i < len; ++i) {
        int k;
        crc ^= (uint32_t)data[i];
        for (k = 0; k < 8; ++k) {
            uint32_t mask = (crc & 1u) ? 0xedb88320u : 0u;
            crc = (crc >> 1) ^ mask;
        }
    }
    return crc ^ 0xffffffffu;
}

/* ---- status names ------------------------------------------------------ */
const char *wubu_state_version_status_name(wubu_state_version_status_t code)
{
    switch (code) {
    case WUBU_SV_OK:           return "ok";
    case WUBU_SV_ERR_ARG:      return "invalid argument";
    case WUBU_SV_ERR_IO:       return "file i/o error";
    case WUBU_SV_ERR_BAD_MAGIC:return "bad magic";
    case WUBU_SV_ERR_TRUNCATED:return "truncated file";
    case WUBU_SV_ERR_CRC:      return "crc32 mismatch";
    case WUBU_SV_ERR_SIZE:     return "payload size out of range";
    case WUBU_SV_ERR_STALE:    return "stale version";
    case WUBU_SV_ERR_NEWER:    return "version newer than reader";
    default:                   return "unknown status";
    }
}

static void set_result(wubu_sv_result_t *out, wubu_state_version_status_t code,
                       uint32_t file_version)
{
    if (!out) return;
    out->code = code;
    snprintf(out->reason, sizeof(out->reason), "%s",
             wubu_state_version_status_name(code));
    out->file_version = file_version;
    out->current_version = WUBU_SV_CURRENT_VERSION;
    out->migration_target[0] = '\0';
}

/* Name the migration target for a stale version: an explicit table entry
 * wins, otherwise fall back to the current version's canonical name. */
static void set_migration_target(wubu_sv_result_t *out, uint32_t from_version)
{
    size_t i;
    if (!out) return;
    for (i = 0; i < WUBU_SV_NMIGRATIONS; ++i) {
        if (WUBU_SV_MIGRATIONS[i].from_version == from_version) {
            snprintf(out->migration_target, sizeof(out->migration_target),
                     "%s", WUBU_SV_MIGRATIONS[i].target_name);
            return;
        }
    }
    snprintf(out->migration_target, sizeof(out->migration_target),
             "v%u", (unsigned)WUBU_SV_CURRENT_VERSION);
}

void wubu_state_version_result_init(wubu_sv_result_t *out)
{
    if (!out) return;
    set_result(out, WUBU_SV_OK, 0u);
}

uint32_t wubu_state_version_current(void) { return WUBU_SV_CURRENT_VERSION; }
const char *wubu_state_version_current_name(void) { return "v2"; }

/* ---- lifecycle --------------------------------------------------------- */
wubu_state_version_t *wubu_state_version_create(uint32_t version,
                                                 const void *payload,
                                                 size_t len)
{
    wubu_state_version_t *s;
    if (version == 0u || len > WUBU_SV_MAX_PAYLOAD) return NULL;
    if (len > 0u && !payload) return NULL;
    s = (wubu_state_version_t *)calloc(1, sizeof(*s));
    if (!s) return NULL;
    if (len > 0u) {
        s->payload = (unsigned char *)malloc(len);
        if (!s->payload) { free(s); return NULL; }
        memcpy(s->payload, payload, len);
    }
    s->version = version;
    s->len = len;
    return s;
}

wubu_state_version_t *wubu_state_version_create_current(const void *payload,
                                                        size_t len)
{
    return wubu_state_version_create(WUBU_SV_CURRENT_VERSION, payload, len);
}

void wubu_state_version_free(wubu_state_version_t *s)
{
    if (!s) return;
    free(s->payload);
    free(s);
}

uint32_t wubu_state_version_get_version(const wubu_state_version_t *s)
{
    return s ? s->version : 0u;
}

const void *wubu_state_version_payload(const wubu_state_version_t *s,
                                       size_t *out_len)
{
    if (!s) {
        if (out_len) *out_len = 0u;
        return NULL;
    }
    if (out_len) *out_len = s->len;
    return s->payload;
}

/* ---- save -------------------------------------------------------------- */
int wubu_state_version_save(const wubu_state_version_t *s, const char *path)
{
    unsigned char hdr[WUBU_SV_HDR_LEN];
    FILE *f;
    int ok = 1;

    if (!s || !path || s->version == 0u || s->len > WUBU_SV_MAX_PAYLOAD)
        return -1;

    memcpy(hdr, WUBU_SV_MAGIC, WUBU_SV_MAGIC_LEN);
    put32(hdr + WUBU_SV_MAGIC_LEN, s->version);
    put32(hdr + WUBU_SV_MAGIC_LEN + 4u, (uint32_t)s->len);
    put32(hdr + WUBU_SV_MAGIC_LEN + 8u,
          crc32_bytes(s->payload, s->len));

    f = fopen(path, "wb");
    if (!f) return -1;
    if (fwrite(hdr, 1, sizeof(hdr), f) != sizeof(hdr)) ok = 0;
    if (ok && s->len > 0u &&
        fwrite(s->payload, 1, s->len, f) != s->len) ok = 0;
    if (fclose(f) != 0) ok = 0;
    return ok ? 0 : -1;
}

/* ---- load helpers ------------------------------------------------------ */
/* Read an entire bounded file into a fresh buffer. Returns NULL on any I/O
 * problem or if the file is larger than the header + max payload. */
static unsigned char *read_file_bounded(const char *path, size_t *out_n)
{
    FILE *f;
    long end;
    size_t n;
    unsigned char *buf;

    *out_n = 0u;
    f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0L, SEEK_END) != 0) { fclose(f); return NULL; }
    end = ftell(f);
    if (end < 0 || (size_t)end > (size_t)WUBU_SV_HDR_LEN + WUBU_SV_MAX_PAYLOAD) {
        fclose(f);
        return NULL;
    }
    n = (size_t)end;
    if (fseek(f, 0L, SEEK_SET) != 0) { fclose(f); return NULL; }
    buf = (unsigned char *)malloc(n > 0u ? n : 1u);
    if (!buf) { fclose(f); return NULL; }
    if (n > 0u && fread(buf, 1, n, f) != n) {
        free(buf); fclose(f); return NULL;
    }
    fclose(f);
    *out_n = n;
    return buf;
}

/* Parse and fully validate a file. On success builds the opaque object unless
 * payload_only (inspect); on any failure returns NULL with `out` naming why.
 * The object is allocated only after every check passes: no partial load. */
static wubu_state_version_t *load_impl(const char *path, wubu_sv_result_t *out,
                                       int payload_only)
{
    unsigned char *buf = NULL;
    size_t n = 0u;
    uint32_t version, size, want_crc, got_crc;
    wubu_sv_result_t local;
    wubu_sv_result_t *res = out ? out : &local;
    wubu_state_version_t *s = NULL;

    if (!res) res = &local;
    wubu_state_version_result_init(res);
    res->current_version = WUBU_SV_CURRENT_VERSION;

    if (!path) { set_result(res, WUBU_SV_ERR_ARG, 0u); return NULL; }
    buf = read_file_bounded(path, &n);
    if (!buf) {
        FILE *probe = fopen(path, "rb");
        set_result(res, probe ? WUBU_SV_ERR_SIZE : WUBU_SV_ERR_IO, 0u);
        if (probe) fclose(probe);
        return NULL;
    }

    if (n < WUBU_SV_HDR_LEN) {
        set_result(res, WUBU_SV_ERR_TRUNCATED, 0u);
        goto done;
    }
    if (memcmp(buf, WUBU_SV_MAGIC, WUBU_SV_MAGIC_LEN) != 0) {
        set_result(res, WUBU_SV_ERR_BAD_MAGIC, 0u);
        goto done;
    }

    version = get32(buf + WUBU_SV_MAGIC_LEN);
    size = get32(buf + WUBU_SV_MAGIC_LEN + 4u);
    want_crc = get32(buf + WUBU_SV_MAGIC_LEN + 8u);
    set_result(res, WUBU_SV_OK, version);

    if (version == 0u) { set_result(res, WUBU_SV_ERR_SIZE, version); goto done; }
    if (size > WUBU_SV_MAX_PAYLOAD ||
        n - WUBU_SV_HDR_LEN != (size_t)size) {
        set_result(res, WUBU_SV_ERR_TRUNCATED, version);
        goto done;
    }
    got_crc = crc32_bytes(buf + WUBU_SV_HDR_LEN, size);
    if (got_crc != want_crc) {
        set_result(res, WUBU_SV_ERR_CRC, version);
        goto done;
    }

    /* File is structurally valid. Version policy: */
    if (version < WUBU_SV_CURRENT_VERSION) {
        set_result(res, WUBU_SV_ERR_STALE, version);
        set_migration_target(res, version);
        goto done;
    }
    if (version > WUBU_SV_CURRENT_VERSION) {
        set_result(res, WUBU_SV_ERR_NEWER, version);
        goto done;
    }

    if (payload_only) {
        res->code = WUBU_SV_OK; /* inspect: valid current-version header */
        goto done;
    }

    /* Build the opaque object only after every check passed: no partial load. */
    s = wubu_state_version_create(version,
                                  size > 0u ? buf + WUBU_SV_HDR_LEN : NULL,
                                  (size_t)size);
    if (!s) set_result(res, WUBU_SV_ERR_IO, version);

done:
    free(buf);
    return s;
}

wubu_state_version_t *wubu_state_version_load(const char *path,
                                               wubu_sv_result_t *out)
{
    return load_impl(path, out, 0);
}

wubu_state_version_status_t wubu_state_version_inspect(const char *path,
                                                       wubu_sv_result_t *out)
{
    wubu_sv_result_t local;
    wubu_sv_result_t *res = out ? out : &local;
    (void)load_impl(path, res, 1);
    return res->code;
}
