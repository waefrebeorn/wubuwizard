/*
 * wubu_preempt.c — Preemption for long-running inference.
 *
 * Enables interrupt + resume of decode/prefill on memory pressure:
 * - Checkpoint: serialize KV cache + generation state to disk
 * - Restore: reload checkpoint and resume generation
 * - Eviction policy: LRU over active sequences
 *
 * C11, self-contained.
 */

#include "wubu_preempt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdalign.h>

#define PREEMPT_MAGIC  0x5055454D  /* "PREEM" */
#define PREEMPT_VERSION 1

/* ---- Checkpoint header ---- */
typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint32_t version;
    uint32_t seq_id;
    uint32_t n_tokens;
    uint32_t kv_pages;
    uint32_t pos;               /* next token position */
    float    temperature;
    int      top_k;
    float    top_p;
    uint32_t crc32;              /* header checksum */
} preempt_header_t;

/* ---- Preempt context ---- */
struct wubu_preempt {
    char     path[256];         /* checkpoint directory */
    uint8_t  *work_buf;         /* scratch buffer */
    size_t   work_cap;
};

/* Simple CRC32 */
static uint32_t crc32(const uint8_t *buf, size_t n)
{
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < n; i++) {
        crc ^= buf[i];
        for (int j = 0; j < 8; j++)
            crc = (crc >> 1) ^ (0xEDB88320 & -(int)(crc & 1));
    }
    return crc ^ 0xFFFFFFFF;
}

wubu_preempt_t *wubu_preempt_create(const char *checkpoint_dir)
{
    wubu_preempt_t *p = calloc(1, sizeof(*p));
    if (!p) return NULL;
    snprintf(p->path, sizeof(p->path), "%s", checkpoint_dir);
    p->work_buf = malloc(65536);
    p->work_cap = 65536;
    return p;
}

void wubu_preempt_destroy(wubu_preempt_t *p)
{
    if (!p) return;
    free(p->work_buf);
    free(p);
}

/* Serialize KV cache via callback */
int wubu_preempt_save(wubu_preempt_t *p, uint32_t seq_id,
                      int n_tokens, int n_kv_pages, int pos,
                      float temperature, int top_k, float top_p,
                      wubu_preempt_kv_reader_t reader, void *userdata)
{
    if (!p || !reader) return -1;

    char fname[512];
    snprintf(fname, sizeof(fname), "%s/seq_%u.preempt", p->path, seq_id);
    FILE *f = fopen(fname, "wb");
    if (!f) return -1;

    /* Header */
    preempt_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = PREEMPT_MAGIC;
    hdr.version = PREEMPT_VERSION;
    hdr.seq_id = seq_id;
    hdr.n_tokens = (uint32_t)n_tokens;
    hdr.kv_pages = (uint32_t)n_kv_pages;
    hdr.pos = (uint32_t)pos;
    hdr.temperature = temperature;
    hdr.top_k = (uint32_t)top_k;
    hdr.top_p = top_p;

    /* Write header placeholder (CRC computed later) */
    fwrite(&hdr, sizeof(hdr), 1, f);

    /* Serialize KV pages via reader callback */
    size_t bytes_per_page = reader(userdata, 0, NULL, 0);
    for (int i = 0; i < n_kv_pages; i++) {
        size_t need = reader(userdata, i, p->work_buf, p->work_cap);
        if (!need) { fclose(f); return -1; }
        if (need > p->work_cap) {
            p->work_buf = realloc(p->work_buf, need);
            p->work_cap = need;
            need = reader(userdata, i, p->work_buf, p->work_cap);
            if (!need) { fclose(f); return -1; }
        }
        fwrite(p->work_buf, 1, need, f);
    }

    /* Finalize CRC */
    long end_pos = ftell(f);
    fseek(f, 0, SEEK_SET);
    size_t hdr_bytes = fread(&hdr, 1, sizeof(hdr), f);
    hdr.crc32 = crc32((const uint8_t *)&hdr + 4, sizeof(hdr) - 4);
    fseek(f, 0, SEEK_SET);
    fwrite(&hdr, sizeof(hdr), 1, f);
    fclose(f);
    return 0;
}

/* Restore from checkpoint */
int wubu_preempt_restore(wubu_preempt_t *p, uint32_t seq_id,
                         wubu_preempt_kv_writer_t writer, void *userdata)
{
    if (!p || !writer) return -1;

    char fname[512];
    snprintf(fname, sizeof(fname), "%s/seq_%u.preempt", p->path, seq_id);
    FILE *f = fopen(fname, "rb");
    if (!f) return -1;

    preempt_header_t hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) { fclose(f); return -1; }

    if (hdr.magic != PREEMPT_MAGIC || hdr.version != PREEMPT_VERSION) {
        fclose(f); return -1;
    }

    /* Verify CRC */
    uint32_t stored_crc = hdr.crc32;
    hdr.crc32 = 0;
    uint32_t calc = crc32((const uint8_t *)&hdr + 4, sizeof(hdr) - 4);
    if (stored_crc != calc) { fclose(f); return -1; }
    hdr.crc32 = stored_crc;

    /* Read KV pages via writer callback */
    for (uint32_t i = 0; i < hdr.kv_pages; i++) {
        size_t need = writer(userdata, i, NULL, 0);
        if (!need) continue;
        if (need > p->work_cap) {
            p->work_buf = realloc(p->work_buf, need);
            p->work_cap = need;
        }
        if (fread(p->work_buf, 1, need, f) != need) {
            fclose(f); return -1;
        }
        writer(userdata, i, p->work_buf, need);
    }

    fclose(f);
    return 0;
}

/* Check if checkpoint exists for seq_id */
int wubu_preempt_exists(const wubu_preempt_t *p, uint32_t seq_id)
{
    char fname[512];
    snprintf(fname, sizeof(fname), "%s/seq_%u.preempt", p->path, seq_id);
    FILE *f = fopen(fname, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}

/* Delete checkpoint after successful restore */
int wubu_preempt_delete(wubu_preempt_t *p, uint32_t seq_id)
{
    char fname[512];
    snprintf(fname, sizeof(fname), "%s/seq_%u.preempt", p->path, seq_id);
    return remove(fname);
}
