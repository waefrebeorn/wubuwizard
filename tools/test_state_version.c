/* test_state_version.c -- gate for LP-16 (state/ABI semver boundary).
 *
 * Covers: (1) current-version round-trip with CRC-protected payload,
 * (2) stale version refused with a named reason, (3) truncated file refused
 * with no partial object, (4) named migration target reported in status.
 * Also checks CRC corruption and bad magic are refused with no partial load.
 *
 * Build (from repo root):
 *   gcc ... tools/test_state_version.c src/wubu_state_version.c ... \
 *        -o test_state_version.exe && ./test_state_version.exe
 * Prints "test_state_version: PASS" and exits 0 on success.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wubu_state_version.h"

#define BASE "C:/Users/eman5/AppData/Local/hermes/cache/scratch/"
#define P_GOOD  BASE "state_version_good.bin"
#define P_STALE BASE "state_version_stale.bin"
#define P_TRUNC BASE "state_version_trunc.bin"
#define P_CORR  BASE "state_version_corr.bin"
#define P_MAGIC BASE "state_version_magic.bin"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

/* Copy `src` to `dst`, then cut `cut` bytes off the tail (the declared size
 * in the header stays larger than the bytes actually on disk). */
static int truncate_copy(const char *src, const char *dst, long cut)
{
    FILE *in = fopen(src, "rb");
    FILE *out;
    unsigned char buf[4096];
    size_t r, total = 0, keep;
    long end;
    if (!in) return -1;
    if (fseek(in, 0L, SEEK_END) != 0) { fclose(in); return -1; }
    end = ftell(in);
    if (end <= cut) { fclose(in); return -1; }
    keep = (size_t)end - (size_t)cut;
    if (fseek(in, 0L, SEEK_SET) != 0) { fclose(in); return -1; }
    out = fopen(dst, "wb");
    if (!out) { fclose(in); return -1; }
    while (total < keep && (r = fread(buf, 1, sizeof(buf), in)) > 0) {
        size_t w = r;
        if (total + w > keep) w = keep - total;
        if (fwrite(buf, 1, w, out) != w) { fclose(in); fclose(out); return -1; }
        total += w;
    }
    fclose(in);
    return fclose(out) == 0 ? 0 : -1;
}

int main(void)
{
    static const unsigned char payload[] =
        "wuBu-state-v2-payload:0123456789:abcdefghijklmnopqrstuvwxyz";
    const size_t plen = sizeof(payload) - 1u; /* no trailing NUL */
    wubu_sv_result_t r;
    size_t got_len = 0;
    const void *got;

    /* ---- 1. round-trip -------------------------------------------------- */
    {
        wubu_state_version_t *s =
            wubu_state_version_create_current(payload, plen);
        CHECK(s != NULL);
        CHECK(wubu_state_version_get_version(s) == WUBU_SV_CURRENT_VERSION);
        CHECK(wubu_state_version_save(s, P_GOOD) == 0);
        wubu_state_version_free(s);

        memset(&r, 0xAA, sizeof(r));
        wubu_state_version_t *r2 = wubu_state_version_load(P_GOOD, &r);
        CHECK(r2 != NULL);
        CHECK(r.code == WUBU_SV_OK);
        CHECK(strcmp(r.reason, "ok") == 0);
        CHECK(r.file_version == WUBU_SV_CURRENT_VERSION);
        CHECK(r.current_version == WUBU_SV_CURRENT_VERSION);
        got = wubu_state_version_payload(r2, &got_len);
        CHECK(got != NULL);
        CHECK(got_len == plen);
        CHECK(got && memcmp(got, payload, plen) == 0);
        wubu_state_version_free(r2);
    }

    /* ---- 2. stale version rejected with a named reason ------------------ */
    {
        wubu_state_version_t *legacy =
            wubu_state_version_create(1u, payload, plen);
        CHECK(legacy != NULL);
        CHECK(wubu_state_version_save(legacy, P_STALE) == 0);
        wubu_state_version_free(legacy);

        memset(&r, 0, sizeof(r));
        wubu_state_version_t *r3 = wubu_state_version_load(P_STALE, &r);
        CHECK(r3 == NULL); /* stale is refused, never half-loaded */
        CHECK(r.code == WUBU_SV_ERR_STALE);
        CHECK(strcmp(r.reason, "stale version") == 0);
        CHECK(r.file_version == 1u);
        wubu_state_version_free(r3);
    }

    /* ---- 3. truncated file refused, no partial load --------------------- */
    {
        CHECK(truncate_copy(P_GOOD, P_TRUNC, 7L) == 0);
        memset(&r, 0, sizeof(r));
        wubu_state_version_t *r4 = wubu_state_version_load(P_TRUNC, &r);
        CHECK(r4 == NULL);
        CHECK(r.code == WUBU_SV_ERR_TRUNCATED);
        CHECK(strcmp(r.reason, "truncated file") == 0);
        wubu_state_version_free(r4);
    }

    /* ---- 4. named migration target in status ---------------------------- */
    {
        wubu_sv_result_t ins;
        memset(&ins, 0, sizeof(ins));
        CHECK(wubu_state_version_inspect(P_STALE, &ins) == WUBU_SV_ERR_STALE);
        CHECK(strcmp(ins.reason, "stale version") == 0);
        CHECK(strcmp(ins.migration_target,
                     wubu_state_version_current_name()) == 0);
        CHECK(strcmp(ins.migration_target, "v2") == 0);
        CHECK(ins.file_version == 1u);
        CHECK(ins.current_version == WUBU_SV_CURRENT_VERSION);
    }

    /* ---- 5. corruption refused (no partial load on CRC/magic failure) --- */
    {
        FILE *f;

        /* CRC corruption of an otherwise current-version file. */
        {
            FILE *in = fopen(P_GOOD, "rb"), *out = fopen(P_CORR, "wb");
            int c;
            CHECK(in != NULL && out != NULL);
            if (in && out) {
                size_t i = 0;
                while ((c = fgetc(in)) != EOF) {
                    if (i == 20u) c ^= 0xff; /* first payload byte */
                    fputc(c, out);
                    ++i;
                }
                fclose(in); fclose(out);
            }
        }
        memset(&r, 0, sizeof(r));
        CHECK(wubu_state_version_load(P_CORR, &r) == NULL);
        CHECK(r.code == WUBU_SV_ERR_CRC);
        CHECK(strcmp(r.reason, "crc32 mismatch") == 0);

        /* Bad magic. */
        f = fopen(P_MAGIC, "wb");
        CHECK(f != NULL);
        if (f) { fputs("NOTWUBU!nonsense-header-payload", f); fclose(f); }
        memset(&r, 0, sizeof(r));
        CHECK(wubu_state_version_load(P_MAGIC, &r) == NULL);
        CHECK(r.code == WUBU_SV_ERR_BAD_MAGIC);
    }

    /* ---- 6. name/status plumbing sanity --------------------------------- */
    CHECK(strcmp(wubu_state_version_status_name(WUBU_SV_OK), "ok") == 0);
    CHECK(strcmp(wubu_state_version_status_name(WUBU_SV_ERR_STALE),
                 "stale version") == 0);
    CHECK(wubu_state_version_current() == 2u);

    if (failures == 0) {
        printf("test_state_version: PASS\n");
        return 0;
    }
    fprintf(stderr, "test_state_version: FAIL (%d)\n", failures);
    return 1;
}
