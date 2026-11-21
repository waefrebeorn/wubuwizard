/*
 * wubu_kernel_spec.c -- LP-10 compile-time kernel family specialization. C11.
 *
 * Model families are data rows, not branches inside a kernel body. A row maps
 * a family key to a kernel-body descriptor; the descriptor selects the
 * specialized body. There is exactly one generic identity routine here with
 * no `if (family == ...)` anywhere -- kernel bodies never branch on family.
 *
 * Identity is FNV-1a 64-bit over a fixed endian serialization of the body
 * descriptor fields (version tag, attention, rotary pairs, window tokens,
 * MoE route, stop-token count, each stop token). It deliberately excludes
 * the family key, row position, and registry size, so adding a new family row
 * leaves existing kernel-body identities byte-identical.
 *
 * WaefreBeorn Umbrella License v3.0
 */
#include "wubu_kernel_spec.h"
#include <stdlib.h>
#include <string.h>

/* ---- opaque registry ---- */
struct wubu_kernel_spec {
    wubu_kernel_row_t rows[WUBU_KERNEL_SPEC_MAX_ROWS];
    size_t            n;
};

/* ---- FNV-1a 64-bit over a deterministic little-endian byte stream ---- */
#define WUBU_FNV64_OFFSET 1469598103934665603ULL
#define WUBU_FNV64_PRIME  1099511628211ULL
#define WUBU_BODY_VERSION 1u  /* bump only on an intentional ABI change */

static uint64_t fnv64_bytes(uint64_t h, const unsigned char *p, size_t n)
{
    size_t i;
    for (i = 0; i < n; ++i) {
        h ^= (uint64_t)p[i];
        h *= WUBU_FNV64_PRIME;
    }
    return h;
}

static void put_u32le(unsigned char out[4], uint32_t v)
{
    out[0] = (unsigned char)(v & 0xFFu);
    out[1] = (unsigned char)((v >> 8)  & 0xFFu);
    out[2] = (unsigned char)((v >> 16) & 0xFFu);
    out[3] = (unsigned char)((v >> 24) & 0xFFu);
}

uint64_t wubu_kernel_body_identity(const wubu_kernel_body_t *body)
{
    unsigned char b[4];
    uint64_t h = WUBU_FNV64_OFFSET;
    size_t i;

    if (!body) return 0ULL;

    /* domain/version tag */
    put_u32le(b, (uint32_t)WUBU_BODY_VERSION);
    h = fnv64_bytes(h, b, 4);

    /* specialization descriptor --- no family key, index, or row count */
    put_u32le(b, (uint32_t)body->attn);          h = fnv64_bytes(h, b, 4);
    put_u32le(b, (uint32_t)body->rotary_pairs);  h = fnv64_bytes(h, b, 4);
    put_u32le(b, (uint32_t)body->window_tokens); h = fnv64_bytes(h, b, 4);
    put_u32le(b, (uint32_t)body->moe);           h = fnv64_bytes(h, b, 4);
    put_u32le(b, (uint32_t)body->n_stop_tokens); h = fnv64_bytes(h, b, 4);
    for (i = 0; i < body->n_stop_tokens; ++i) {
        put_u32le(b, (uint32_t)body->stop_tokens[i]);
        h = fnv64_bytes(h, b, 4);
    }
    return h ? h : 1ULL;  /* keep the zero value reserved for "unknown" */
}

/* ---- registry lifecycle ---- */
wubu_kernel_spec_t *wubu_kernel_spec_create(void)
{
    return (wubu_kernel_spec_t *)calloc(1, sizeof(wubu_kernel_spec_t));
}

void wubu_kernel_spec_free(wubu_kernel_spec_t *spec)
{
    free(spec);
}

static int find_family(const wubu_kernel_spec_t *spec, const char *family)
{
    size_t i;
    if (!spec || !family) return -1;
    for (i = 0; i < spec->n; ++i)
        if (strcmp(spec->rows[i].family, family) == 0) return (int)i;
    return -1;
}

int wubu_kernel_spec_register(wubu_kernel_spec_t *spec,
                              const wubu_kernel_row_t *row)
{
    size_t len;
    if (!spec || !row) return 0;
    if (spec->n >= WUBU_KERNEL_SPEC_MAX_ROWS) return 0;

    len = strlen(row->family);
    if (len == 0 || len >= WUBU_KERNEL_SPEC_MAX_FAMILY) return 0;
    if (row->body.n_stop_tokens > WUBU_KERNEL_SPEC_MAX_STOP) return 0;
    if (find_family(spec, row->family) >= 0) return 0;  /* duplicate key */

    /* value copy: the whole row, including the fixed stop-token list, lives
     * inside the registry. Caller needs no lifetime guarantees. */
    spec->rows[spec->n] = *row;
    spec->n++;
    return 1;
}

const wubu_kernel_row_t *wubu_kernel_spec_select(const wubu_kernel_spec_t *spec,
                                                  const char *family)
{
    int idx = find_family(spec, family);
    if (idx < 0) return NULL;
    return &spec->rows[(size_t)idx];
}

size_t wubu_kernel_spec_count(const wubu_kernel_spec_t *spec)
{
    return spec ? spec->n : 0;
}

uint64_t wubu_kernel_spec_identity(const wubu_kernel_spec_t *spec,
                                   const char *family)
{
    const wubu_kernel_row_t *row = wubu_kernel_spec_select(spec, family);
    return row ? wubu_kernel_body_identity(&row->body) : 0ULL;
}
