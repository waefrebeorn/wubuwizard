/* wubu_enc_fs.c -- THE ENCODER IS A MOUNT (AN16 G3 + research/064).
 *
 * The encoder does not re-implement storage: it mounts into the KV
 * namespace and lets the filesystem lift the heavy work. Write raw
 * input at /kv/enc/<id> (the bytes land in the KV tensor region),
 * the boundary encodes through the registered encoder, and the
 * embedding is a FILE at /kv/emb/<id> — tiered, evictable, paged,
 * persistent, mirrored, exactly like the KV cache itself.
 *
 * C11.
 */
#include "wubu_enc_fs.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

struct wubu_enc_fs {
    wubu_kvfs_t *fs;
    float *kv_base;            /* the flat KV tensor (the memory) */
    wubu_encoder_t *enc;       /* the transform at the boundary */
    int dim;                   /* embedding dim (file size in floats) */
    uint32_t raw_blocks;
    uint32_t emb_blocks;
};

wubu_enc_fs_t *wubu_enc_fs_mount(wubu_kvfs_t *fs, float *kv_base,
                                 wubu_encoder_t *enc,
                                 uint32_t raw_start, uint32_t raw_blocks,
                                 uint32_t emb_start, uint32_t emb_blocks) {
    if (!fs || !enc || !enc->encode || enc->dim < 1) return NULL;
    if (wubu_kvfs_mount(fs, WUBU_ENC_FS_RAW_PATH, raw_start, raw_blocks) != 0)
        return NULL;
    if (wubu_kvfs_mount(fs, WUBU_ENC_FS_EMB_PATH, emb_start, emb_blocks) != 0) {
        wubu_kvfs_unmount(fs, WUBU_ENC_FS_RAW_PATH);
        return NULL;
    }
    wubu_enc_fs_t *m = (wubu_enc_fs_t *)calloc(1, sizeof(*m));
    if (!m) {
        wubu_kvfs_unmount(fs, WUBU_ENC_FS_RAW_PATH);
        wubu_kvfs_unmount(fs, WUBU_ENC_FS_EMB_PATH);
        return NULL;
    }
    m->fs = fs;
    m->kv_base = kv_base;
    m->enc = enc;
    m->dim = enc->dim;
    m->raw_blocks = raw_blocks;
    m->emb_blocks = emb_blocks;
    return m;
}

int wubu_enc_fs_encode(wubu_enc_fs_t *m, const char *id,
                       const float *raw, size_t n_raw) {
    if (!m || !id || !raw || n_raw == 0) return -1;
    if (strchr(id, '/')) return -1;   /* one path segment per datum */

    char embp[512];
    snprintf(embp, sizeof(embp), "%s/%s", WUBU_ENC_FS_EMB_PATH, id);

    /* 1. the boundary transform: raw -> shared embedding (OUR encoder,
     * no imports). */
    float emb[4096];
    if (m->dim > (int)(sizeof(emb) / sizeof(emb[0]))) return -1;
    if (m->enc->encode(m->enc->ctx, raw, n_raw, emb) != 0) return -1;

    /* 2. the embedding is a FILE in the KV tensor region. The write is
     * bounds-checked against the mount by the namespace; from here on
     * the heavy lifting (tiering, eviction, paging, persistence,
     * mirroring) is ALREADY lifted — this is a KV cache write. */
    if (wubu_kvfs_write(m->fs, embp, m->kv_base, emb,
                        (size_t)m->dim) != 0)
        return -1;
    return 0;
}

int wubu_enc_fs_read_embedding(wubu_enc_fs_t *m, const char *id, float *out) {
    if (!m || !id || !out) return -1;
    if (strchr(id, '/')) return -1;
    char embp[512];
    snprintf(embp, sizeof(embp), "%s/%s", WUBU_ENC_FS_EMB_PATH, id);
    /* pure KVFS read — the namespace lifts everything. */
    return wubu_kvfs_read(m->fs, embp, m->kv_base, out,
                          (size_t)m->dim);
}

int wubu_enc_fs_dim(const wubu_enc_fs_t *m) {
    return m ? m->dim : -1;
}

uint32_t wubu_enc_fs_raw_blocks(const wubu_enc_fs_t *m) {
    return m ? m->raw_blocks : 0;
}
uint32_t wubu_enc_fs_emb_blocks(const wubu_enc_fs_t *m) {
    return m ? m->emb_blocks : 0;
}

void wubu_enc_fs_free(wubu_enc_fs_t *m) {
    if (!m) return;
    if (m->fs) {
        wubu_kvfs_unmount(m->fs, WUBU_ENC_FS_RAW_PATH);
        wubu_kvfs_unmount(m->fs, WUBU_ENC_FS_EMB_PATH);
    }
    free(m);
}
