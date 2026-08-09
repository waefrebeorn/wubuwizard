/* wubu_userfs.c -- USER SPACE IN THE NAMESPACE (the user's files ARE
 * the training data — every user action trains at every time).
 *
 * RESEARCH/065 upgrades: paragraph-group chunking (content-aware
 * chunks double retrieval, arXiv:2603.06976) + the usage ledger
 * (implicit feedback, arXiv:2606.20482). Every chunk is its own
 * embedding file; every usage event is a preference annotation.
 *
 * C11.
 */
#include "wubu_userfs.h"
#include "wubu_audio.h"   /* WUBU_AUDIO_* caps for the PCM buffer */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/stat.h>
#include <dirent.h>

struct wubu_userfs {
    wubu_kvfs_t *fs;
    float *kv_base;
    wubu_encoder_t *text_enc;
    wubu_encoder_t *audio_enc;
    int dim;
    uint32_t emb_start, emb_blocks;
    uint32_t emb_next;          /* next free embedding block (a file IS a
                                 * mount region — each chunk is its own
                                 * mount, its own path, its own blocks) */
    char data_path[64], emb_path[64], meta_path[64], index_path[96];
};

/* ---- tiny helpers ---- */

static const char *ext_of(const char *path) {
    const char *dot = strrchr(path, '.');
    return dot ? dot + 1 : "";
}
static const char *base_of(const char *path) {
    const char *slash = strrchr(path, '/');
    return slash ? slash + 1 : path;
}
static int is_text_ext(const char *ext) {
    static const char *text_exts[] = {
        "txt", "md", "markdown", "c", "h", "py", "json", "log",
        "cfg", "toml", "yaml", "yml", "sh", "asm", "tex", "csv",
    };
    for (size_t i = 0; i < sizeof(text_exts)/sizeof(text_exts[0]); i++)
        if (strcmp(ext, text_exts[i]) == 0) return 1;
    return 0;
}

/* text bytes -> deterministic token ids (byte value; the real HF
 * tokenizer plugs in at load — the seam is what's proven here) */
static size_t text_to_ids(const unsigned char *data, size_t n,
                          float *ids, size_t cap) {
    size_t k = 0;
    for (size_t i = 0; i < n && k < cap; i++)
        ids[k++] = (float)data[i];
    return k;
}

/* ---- WAV reader (ours, C11): RIFF -> fmt (PCM) -> data -> floats ---- */
static long wav_pcm_floats(const unsigned char *buf, size_t n,
                           float *out, size_t cap) {
    if (n < 44) return -1;
    if (memcmp(buf, "RIFF", 4) != 0 || memcmp(buf + 8, "WAVE", 4) != 0)
        return -1;
    size_t i = 12;
    int bits = 16, channels = 1;
    long data_len = -1;
    const unsigned char *data_ptr = NULL;
    while (i + 8 <= n) {
        unsigned chunk = (unsigned)buf[i+0] | ((unsigned)buf[i+1] << 8) |
                         ((unsigned)buf[i+2] << 16) | ((unsigned)buf[i+3] << 24);
        unsigned clen = (unsigned)buf[i+4] | ((unsigned)buf[i+5] << 8) |
                        ((unsigned)buf[i+6] << 16) | ((unsigned)buf[i+7] << 24);
        if (chunk == 0x20746D66u /* fmt */ && clen >= 16) {
            /* fmt payload: audioFormat +8, numChannels +10,
             * sampleRate +12, byteRate +16, blockAlign +20,
             * bitsPerSample +22 */
            bits = (int)buf[i+22] | ((int)buf[i+23] << 8);
            channels = (int)buf[i+10] | ((int)buf[i+11] << 8);
        } else if (chunk == 0x61746164u /* data */) {
            data_ptr = buf + i + 8;
            data_len = (long)clen;
        }
        i += 8 + clen + (clen & 1);
    }
    if (data_len < 0 || !data_ptr) return -1;
    long n_samples = data_len / (bits / 8) / (channels ? channels : 1);
    if (n_samples < 1) return -1;
    long k = 0;
    for (long s = 0; s < n_samples && k < (long)cap; s++) {
        long o = (long)(s * (bits / 8) * channels);
        if (bits == 16) {
            int16_t v = (int16_t)((uint16_t)data_ptr[o] |
                                  ((uint16_t)data_ptr[o+1] << 8));
            out[k++] = (float)v / 32768.0f;
        } else if (bits == 8) {
            out[k++] = ((float)data_ptr[o] - 128.0f) / 128.0f;
        } else {
            return -1;
        }
    }
    return k;
}

/* ---- the namespace layout ---- */

wubu_userfs_t *wubu_userfs_mount(wubu_kvfs_t *fs, float *kv_base,
                                 wubu_encoder_t *text_enc,
                                 wubu_encoder_t *audio_enc,
                                 uint32_t data_start, uint32_t data_blocks,
                                 uint32_t emb_start, uint32_t emb_blocks,
                                 uint32_t meta_start, uint32_t meta_blocks) {
    if (!fs || !kv_base || !text_enc) return NULL;
    if (wubu_kvfs_mount(fs, WUBU_USERFS_DATA_PATH, data_start, data_blocks) != 0)
        return NULL;
    if (wubu_kvfs_mount(fs, WUBU_USERFS_EMB_PATH, emb_start, emb_blocks) != 0) {
        wubu_kvfs_unmount(fs, WUBU_USERFS_DATA_PATH);
        return NULL;
    }
    if (meta_blocks > 0 &&
        wubu_kvfs_mount(fs, WUBU_USERFS_META_PATH, meta_start, meta_blocks) != 0) {
        wubu_kvfs_unmount(fs, WUBU_USERFS_DATA_PATH);
        wubu_kvfs_unmount(fs, WUBU_USERFS_EMB_PATH);
        return NULL;
    }
    wubu_userfs_t *m = (wubu_userfs_t *)calloc(1, sizeof(*m));
    if (!m) {
        wubu_kvfs_unmount(fs, WUBU_USERFS_DATA_PATH);
        wubu_kvfs_unmount(fs, WUBU_USERFS_EMB_PATH);
        if (meta_blocks > 0) wubu_kvfs_unmount(fs, WUBU_USERFS_META_PATH);
        return NULL;
    }
    m->fs = fs;
    m->kv_base = kv_base;
    m->text_enc = text_enc;
    m->audio_enc = audio_enc;
    m->dim = text_enc->dim;
    m->emb_start = emb_start;
    m->emb_blocks = emb_blocks;
    m->emb_next = 0;   /* chunk files start at the emb region start */
    snprintf(m->data_path, sizeof(m->data_path), "%s", WUBU_USERFS_DATA_PATH);
    snprintf(m->emb_path, sizeof(m->emb_path), "%s", WUBU_USERFS_EMB_PATH);
    snprintf(m->meta_path, sizeof(m->meta_path), "%s", WUBU_USERFS_META_PATH);
    snprintf(m->index_path, sizeof(m->index_path), "%s/%s",
             WUBU_USERFS_DATA_PATH, WUBU_USERFS_INDEX_NAME);
    return m;
}

/* ---- the INDEX ledger: "name|chunks|size|path" lines ----
 * The namespace stores FLOATS (the KV cache is float memory). Text
 * ledgers are packed byte-per-float: val = (float)byte, 0 = terminator.
 * This keeps the ledger float-honest with the KVFS API (no byte/float
 * aliasing, no overflow). */

/* pack a text buffer into a float array (byte per float) */
static size_t pack_text_into_floats(const char *text, float *out, size_t cap) {
    size_t n = strlen(text);
    if (n >= cap) n = cap - 1;
    for (size_t i = 0; i < n; i++) out[i] = (float)(unsigned char)text[i];
    out[n] = 0.0f;
    return n + 1;
}

/* unpack a float array back into text (stops at 0 or non-byte) */
static void unpack_floats_to_text(const float *in, size_t n, char *out,
                                  size_t out_cap) {
    size_t k = 0;
    for (size_t i = 0; i < n && k + 1 < out_cap; i++) {
        float v = in[i];
        if (v == 0.0f || v < 1.0f || v > 255.0f) break;
        out[k++] = (char)(unsigned char)v;
    }
    out[k] = '\0';
}

/* read the whole INDEX region as floats (the region holds up to
 * region_cap_floats) */
static int index_read(wubu_userfs_t *m, float *out, size_t cap_floats) {
    return wubu_kvfs_read(m->fs, m->index_path, m->kv_base, out, cap_floats);
}
static int index_write(wubu_userfs_t *m, const float *src, size_t n_floats) {
    return wubu_kvfs_write(m->fs, m->index_path, m->kv_base, src, n_floats);
}

#define INDEX_FLOATS 2048   /* 2048 floats = up to ~2047 text bytes */

static int index_contains(wubu_userfs_t *m, const char *name, long size) {
    float raw[INDEX_FLOATS];
    if (index_read(m, raw, INDEX_FLOATS) < 0) return 0;
    char idx[INDEX_FLOATS];
    unpack_floats_to_text(raw, INDEX_FLOATS, idx, sizeof(idx));
    /* ledger lines are "name|chunks|size|path\n" — scan line by line */
    char sizebuf[32];
    snprintf(sizebuf, sizeof(sizebuf), "%ld", size);
    char *line = idx;
    while (*line) {
        char *nl = strchr(line, '\n');
        size_t len = nl ? (size_t)(nl - line) : strlen(line);
        if (len > 0) {
            char copy[600];
            if (len >= sizeof(copy)) len = sizeof(copy) - 1;
            memcpy(copy, line, len);
            copy[len] = '\0';
            /* split on '|': field 0 = name, field 1 = chunks,
             * field 2 = size */
            char *f0 = copy;
            char *f1 = strchr(f0, '|');
            if (f1) {
                *f1 = '\0';
                char *f2 = strchr(f1 + 1, '|');
                if (f2) {
                    *f2 = '\0';
                    char *f3 = strchr(f2 + 1, '|');
                    if (f3) *f3 = '\0';
                    if (strcmp(f0, name) == 0 &&
                        strcmp(f2 + 1, sizebuf) == 0)
                        return 1;
                }
            }
        }
        if (!nl) break;
        line = nl + 1;
    }
    return 0;
}

static int index_append(wubu_userfs_t *m, const char *name,
                        int n_chunks, long size, const char *emb_path) {
    char line[512];
    snprintf(line, sizeof(line), "%s|%d|%ld|%s\n",
             name, n_chunks, size, emb_path);
    /* read current, append, write back */
    float raw[INDEX_FLOATS];
    if (index_read(m, raw, INDEX_FLOATS) < 0) return -1;
    char idx[INDEX_FLOATS];
    unpack_floats_to_text(raw, INDEX_FLOATS, idx, sizeof(idx));
    size_t cur = strlen(idx);
    size_t add = strlen(line);
    if (cur + add >= sizeof(idx)) return -1;
    memcpy(idx + cur, line, add + 1);
    float out[INDEX_FLOATS];
    pack_text_into_floats(idx, out, INDEX_FLOATS);
    return index_write(m, out, INDEX_FLOATS);
}

/* ---- chunk files: a file IS a mount region (the KVFS doctrine) ----
 * Each chunk gets its own path (/kv/user/emb/<name>.p<k>) and its own
 * block region inside the emb mount's span. Region-addressed reads and
 * writes are per-file: no aliasing between chunks. */

/* blocks needed for one embedding file (dim floats at block_size) */
static uint32_t emb_blocks_for(const wubu_userfs_t *m, uint32_t block_size) {
    uint32_t need = (uint32_t)m->dim / block_size;
    if (m->dim % block_size) need++;
    return need;
}

/* drop the previous chunk files of a name (re-ingest of a changed
 * file: the old chunk mounts must go before new ones mount). Returns
 * the first freed block offset (emb region-relative), or emb_blocks
 * if nothing was freed. */
static uint32_t chunk_unmount_old(wubu_userfs_t *m, const char *base,
                                  uint32_t block_size) {
    uint32_t first_free = m->emb_blocks;   /* nothing freed yet */
    for (int k = 0; k < 64; k++) {
        char path[512];
        snprintf(path, sizeof(path), "%s/%s.p%d", m->emb_path, base, k);
        wubu_kvfs_handle_t *h = wubu_kvfs_open(m->fs, path);
        if (h) {
            size_t abs = wubu_kvfs_handle_offset(h);
            wubu_kvfs_handle_close(h);
            uint32_t rel = (uint32_t)((abs - (size_t)m->emb_start * block_size)
                                      / block_size);
            if (rel < first_free) first_free = rel;
        }
        wubu_kvfs_unmount(m->fs, path);
    }
    return first_free;
}

/* write one chunk embedding as its own file; returns 0 on success */
static int chunk_write(wubu_userfs_t *m, const char *base, int idx,
                       const float *emb, uint32_t block_size) {
    uint32_t need = emb_blocks_for(m, block_size);
    if (m->emb_next + need > m->emb_blocks) return -1;   /* region full */
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.p%d", m->emb_path, base, idx);
    /* the file is a mount: its own region inside the emb span */
    if (wubu_kvfs_mount(m->fs, path, m->emb_start + m->emb_next, need) != 0)
        return -1;
    if (wubu_kvfs_write(m->fs, path, m->kv_base, emb, (size_t)m->dim) != 0)
        return -1;
    m->emb_next += need;
    return 0;
}

/* ---- chunked text ingest (research/065 #4) ----
 * Split the text on blank lines into paragraph groups; every group
 * (capped at WUBU_USERFS_CHUNK_CAP ids) becomes its own embedding
 * FILE (its own mount region). Retrieval targets the chunk. */
static int ingest_text_chunked(wubu_userfs_t *m, const unsigned char *data,
                               size_t n, const char *base,
                               uint32_t block_size) {
    size_t pos = 0;
    int chunk_idx = 0;
    int written = 0;
    while (pos < n && chunk_idx < 64) {
        /* collect one paragraph group: up to CHUNK_CAP ids */
        float ids[WUBU_USERFS_CHUNK_CAP];
        size_t k = 0;
        int saw_para = 0;
        while (pos < n && k < WUBU_USERFS_CHUNK_CAP) {
            size_t start = pos;
            while (pos < n && data[pos] != '\n') pos++;
            size_t linelen = pos - start;
            if (linelen == 0 || (linelen == 1 && data[start] == '\r')) {
                pos++;   /* blank line = paragraph boundary */
                if (saw_para) break;
                continue;
            }
            for (size_t i = 0; i < linelen && k < WUBU_USERFS_CHUNK_CAP; i++)
                ids[k++] = (float)data[start + i];
            if (linelen < 120) ids[k-1] = (float)'\n';  /* line sep */
            saw_para = 1;
            pos++;
            if (k >= WUBU_USERFS_CHUNK_CAP / 2 && pos < n &&
                data[pos] == '\n') break;
        }
        if (k == 0) { pos++; continue; }

        float emb[4096];
        if (m->dim > (int)(sizeof(emb) / sizeof(emb[0]))) return -1;
        if (m->text_enc->encode(m->text_enc->ctx, ids, k, emb) != 0) break;
        if (chunk_write(m, base, chunk_idx, emb, block_size) != 0) break;
        written++;
        chunk_idx++;
    }
    return written;
}

/* ---- ingest one file ---- */

int wubu_userfs_ingest(wubu_userfs_t *m, const char *real_path) {
    if (!m || !real_path) return -1;
    const char *base = base_of(real_path);
    const char *ext = ext_of(real_path);
    if (!*base) return -1;

    FILE *fp = fopen(real_path, "rb");
    if (!fp) return -1;
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (fsize <= 0 || fsize > 16 * 1024 * 1024) { fclose(fp); return -1; }
    unsigned char *data = (unsigned char *)malloc((size_t)fsize);
    if (!data) { fclose(fp); return -1; }
    if (fread(data, 1, (size_t)fsize, fp) != (size_t)fsize) {
        free(data); fclose(fp); return -1;
    }
    fclose(fp);

    /* already interpreted + unchanged? skip (incremental training) */
    if (index_contains(m, base, fsize)) { free(data); return 1; }

    /* re-ingest of a changed file: drop its old chunk files and reuse
     * their blocks (the freed span starts the new chunk files) */
    uint32_t block_size0 = wubu_kvfs_block_size(m->fs);
    if (block_size0 == 0) block_size0 = 128;
    uint32_t freed = chunk_unmount_old(m, base, block_size0);
    if (freed < m->emb_next) m->emb_next = freed;

    /* the KV block size: the namespace's own geometry */
    uint32_t block_size = wubu_kvfs_block_size(m->fs);
    if (block_size == 0) block_size = 128;   /* defensive */

    float emb[4096];
    const char *type = "other";
    int rc = -1;
    int n_chunks = 0;
    if (is_text_ext(ext)) {
        n_chunks = ingest_text_chunked(m, data, (size_t)fsize, base, block_size);
        rc = (n_chunks > 0) ? 0 : -1;
        type = "text";
    } else if (strcmp(ext, "wav") == 0 && m->audio_enc) {
        float pcm[WUBU_AUDIO_MAX_FRAMES * WUBU_AUDIO_HOP + WUBU_AUDIO_FRAME_SIZE];
        long n = wav_pcm_floats(data, (size_t)fsize, pcm,
                                (size_t)(WUBU_AUDIO_MAX_FRAMES * WUBU_AUDIO_HOP +
                                         WUBU_AUDIO_FRAME_SIZE));
        if (n > 0 && m->audio_enc->encode)
            rc = m->audio_enc->encode(m->audio_enc->ctx, pcm, (int)n, emb);
        type = "audio";
        if (rc == 0) {
            rc = chunk_write(m, base, 0, emb, block_size);
            n_chunks = 1;
        }
    }
    free(data);
    if (rc != 0) return 1;   /* unsupported/undeccodable */

    /* ledger it */
    char embp[512];
    snprintf(embp, sizeof(embp), "%s/%s.p0", m->emb_path, base);
    if (index_append(m, base, n_chunks, fsize, embp) != 0) return -1;
    return 0;
}

/* ---- the usage ledger (research/065 #3: implicit feedback) ---- */

int wubu_userfs_record_usage(wubu_userfs_t *m, const char *name,
                             const char *event, float weight) {
    if (!m || !name || !event) return -1;
    if (strchr(name, '/')) return -1;
    if (strchr(event, '/')) return -1;
    if (!m->meta_path[0]) return -1;   /* no meta mount */

    /* /kv/user/meta/<name>: append "event|weight|tick" lines */
    char metap[512];
    snprintf(metap, sizeof(metap), "%s/%s", m->meta_path, name);
    float raw[2048];
    int rc = wubu_kvfs_read(m->fs, metap, m->kv_base, raw, 2048);
    char buf[2048];
    if (rc == 0) unpack_floats_to_text(raw, 2048, buf, sizeof(buf));
    else buf[0] = '\0';
    size_t cur = strlen(buf);

    char line[128];
    static unsigned tick = 0;
    snprintf(line, sizeof(line), "%s|%.3f|%u\n", event, weight, ++tick);
    size_t add = strlen(line);
    if (cur + add >= sizeof(buf)) return -1;
    memcpy(buf + cur, line, add + 1);
    float out[2048];
    pack_text_into_floats(buf, out, 2048);
    return wubu_kvfs_write(m->fs, metap, m->kv_base, out, 2048);
}

/* ---- ingest a whole directory (incremental) ---- */

int wubu_userfs_ingest_dir(wubu_userfs_t *m, const char *dir_path) {
    if (!m || !dir_path) return -1;
    DIR *d = opendir(dir_path);
    if (!d) return -1;
    int ingested = 0;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (e->d_name[0] == '.') continue;
        char full[1024];
        snprintf(full, sizeof(full), "%s/%s", dir_path, e->d_name);
        struct stat st;
        if (stat(full, &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) continue;
        int rc = wubu_userfs_ingest(m, full);
        if (rc == 0) ingested++;
    }
    closedir(d);
    return ingested;
}

int wubu_userfs_read_index(wubu_userfs_t *m, char *buf, size_t buflen) {
    if (!m || !buf || buflen < 1) return -1;
    float raw[INDEX_FLOATS];
    if (index_read(m, raw, INDEX_FLOATS) < 0) return 0;
    unpack_floats_to_text(raw, INDEX_FLOATS, buf, buflen);
    int n = 0;
    for (char *p = buf; *p; p++)
        if (*p == '\n') n++;
    return n;
}

int wubu_userfs_count(wubu_userfs_t *m) {
    if (!m) return -1;
    char idx[INDEX_FLOATS];
    float raw[INDEX_FLOATS];
    if (index_read(m, raw, INDEX_FLOATS) < 0) return 0;
    unpack_floats_to_text(raw, INDEX_FLOATS, idx, sizeof(idx));
    int n = 0;
    for (char *p = idx; *p; p++)
        if (*p == '\n') n++;
    return n;
}

int wubu_userfs_dim(const wubu_userfs_t *m) {
    return m ? m->dim : -1;
}

void wubu_userfs_free(wubu_userfs_t *m) {
    if (!m) return;
    if (m->fs) {
        wubu_kvfs_unmount(m->fs, WUBU_USERFS_DATA_PATH);
        wubu_kvfs_unmount(m->fs, WUBU_USERFS_EMB_PATH);
        if (m->meta_path[0]) wubu_kvfs_unmount(m->fs, WUBU_USERFS_META_PATH);
    }
    free(m);
}
