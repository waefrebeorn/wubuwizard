/* wubu_enc_fs.h -- THE ENCODER IS A MOUNT (the universal file-system
 * space, AN16 G3 + research/064).
 *
 * "We are already lifting heavy work." The KV cache IS a file system
 * (ADR-003, wubu_kvfs): path -> (block, offset) in a flat KV tensor,
 * hot I/O is a bounds-checked memcpy, and the namespace already lifts
 * tiering, eviction, persistence, paging, and mirroring. So the
 * encoder does NOT re-implement storage: it MOUNTS into the space.
 *
 *   write raw input  -> /kv/enc/<id>   (the mount boundary encodes)
 *   read embedding   -> /kv/emb/<id>   (pure KVFS read — the heavy
 *                                        work is already lifted)
 *
 * The encode transform (our own encoder: imgenc/audio/patches) runs at
 * the mount boundary; the embedding lands as a FILE in the KV tensor
 * region. Every datum is a file with a path — including the inputs AND
 * the embeddings. The model's memory IS the filesystem; the encoder is
 * just another mount.
 *
 * C11, opaque. Depends on wubu_kvfs (the address layer) + the
 * wubu_encoder slot (the transform).
 */
#ifndef WUBU_ENC_FS_H
#define WUBU_ENC_FS_H

#include <stdint.h>
#include <stddef.h>

#include "wubu_kvfs.h"
#include "wubu_encoder.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_enc_fs wubu_enc_fs_t;

/* The namespace layout the mount creates:
 *   /kv/enc/<id>   — raw input region (writes land here)
 *   /kv/emb/<id>   — embedding region (the encoded file)
 */
#define WUBU_ENC_FS_RAW_PATH  "/kv/enc"
#define WUBU_ENC_FS_EMB_PATH  "/kv/emb"

/* Mount the encoder into the KV namespace. `kv_base` is the flat KV
 * tensor (the memory the namespace addresses). Blocks
 * [raw_start, raw_start+raw_blocks) become /kv/enc; [emb_start,
 * emb_start+emb_blocks) become /kv/emb. `enc` is the encoder used at
 * the boundary (its dim is the embedding file size). Returns NULL on
 * failure. */
wubu_enc_fs_t *wubu_enc_fs_mount(wubu_kvfs_t *fs, float *kv_base,
                                 wubu_encoder_t *enc,
                                 uint32_t raw_start, uint32_t raw_blocks,
                                 uint32_t emb_start, uint32_t emb_blocks);

/* Encode raw input as a file: write the raw bytes to /kv/enc/<id>,
 * run the mount boundary's encode transform, store the embedding to
 * /kv/emb/<id>. `id` is a single path segment (no slashes). Returns
 * 0 on success. */
int wubu_enc_fs_encode(wubu_enc_fs_t *m, const char *id,
                       const float *raw, size_t n_raw);

/* Read the embedding file back (pure KVFS read — tiering/eviction/
 * paging all apply). out must hold dim floats. Returns 0 on success. */
int wubu_enc_fs_read_embedding(wubu_enc_fs_t *m, const char *id,
                               float *out);

/* The embedding dim (the encoder's shared dim). */
int wubu_enc_fs_dim(const wubu_enc_fs_t *m);

/* The mount's block layout — how much of the namespace it owns. */
uint32_t wubu_enc_fs_raw_blocks(const wubu_enc_fs_t *m);
uint32_t wubu_enc_fs_emb_blocks(const wubu_enc_fs_t *m);

/* Destroy the mount (does NOT free the underlying KV namespace). */
void wubu_enc_fs_free(wubu_enc_fs_t *m);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_ENC_FS_H */
