/*
 * test_enc_fs.c -- AN16 G3 gate: THE ENCODER IS A MOUNT.
 *
 * "We are already lifting heavy work." The encoder does not re-implement
 * storage — it mounts into the KV namespace (the universal file-system
 * space). The KV cache IS the filesystem; the encoder is just another
 * mount.
 *
 *   1. MOUNT: /kv/enc + /kv/emb are real KVFS mounts in the namespace
 *      (addressable, bounds-checked, part of the KV tensor region).
 *   2. ENCODE THROUGH THE FS: writing raw input via the mount runs the
 *      boundary transform and stores the embedding AS A FILE at
 *      /kv/emb/<id> — the heavy lifting (tiering/eviction/paging) is
 *      the namespace's, not the encoder's.
 *   3. READ BACK: the embedding file is a pure KVFS read (exactly a KV
 *      cache read — same hot path, same handle machinery).
 *   4. NAMESPACE INTEGRITY: the mounts are registered in the mount
 *      table; reads of unmounted paths fail; the embedding bytes live
 *      in the KV tensor region (the same memory the cache uses).
 *
 * Gate: `make test_enc_fs`.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "wubu_kvfs.h"
#include "wubu_encoder.h"
#include "wubu_encoder_impl.h"
#include "wubu_enc_fs.h"
#include "wubu_imgenc.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)

int main(void)
{
    printf("=== test_enc_fs (AN16 G3: THE ENCODER IS A MOUNT) ===\n");

    /* ---- the namespace + the memory it addresses ---- */
    const uint32_t block_size = 128;   /* floats per block */
    const uint32_t total_blocks = 1024;
    wubu_kvfs_t *fs = wubu_kvfs_create(block_size, total_blocks);
    CHECK(fs != NULL, "KV namespace created");
    if (!fs) return 1;
    float *kv_tensor = (float *)calloc((size_t)total_blocks * block_size,
                                       sizeof(float));
    CHECK(kv_tensor != NULL, "KV tensor allocated");
    if (!kv_tensor) return 1;

    /* ---- our own encoders (made, not imported) ---- */
    CHECK(wubu_encoder_register_ours() == 0, "our encoders registered");
    wubu_encoder_t *img_enc = wubu_encoder_get("our-image");
    CHECK(img_enc != NULL, "the image encoder is the mount boundary");

    /* ---- mount the encoder into the namespace ---- */
    wubu_enc_fs_t *m = wubu_enc_fs_mount(fs, kv_tensor, img_enc,
                                         0, 16,      /* /kv/enc */
                                         512, 32);   /* /kv/emb */
    CHECK(m != NULL, "encoder mounted into the KV namespace");
    if (!m) return 1;

    /* ---- 1. the mounts exist in the namespace ---- */
    CHECK(wubu_enc_fs_dim(m) == WUBU_IMGENC_EMBED_DIM,
          "embedding file size = the encoder's shared dim");
    CHECK(wubu_enc_fs_raw_blocks(m) == 16 && wubu_enc_fs_emb_blocks(m) == 32,
          "mount owns the raw + embedding block ranges");

    /* ---- 2. encode through the filesystem ---- */
    float img[WUBU_IMGENC_IMAGE * WUBU_IMGENC_IMAGE * WUBU_IMGENC_CHANNELS];
    for (int i = 0; i < (int)(sizeof(img)/sizeof(img[0])); i++)
        img[i] = (float)((i * 2654435761u) % 1000) / 1000.0f;
    CHECK(wubu_enc_fs_encode(m, "photo-1", img,
          (size_t)WUBU_IMGENC_IMAGE*WUBU_IMGENC_IMAGE*WUBU_IMGENC_CHANNELS) == 0,
          "raw input encoded through the mount");

    /* ---- 3. read the embedding file back (pure KVFS read) ---- */
    float emb[WUBU_IMGENC_EMBED_DIM];
    CHECK(wubu_enc_fs_read_embedding(m, "photo-1", emb) == 0,
          "embedding file read back from the namespace");
    int finite = 1;
    for (int i = 0; i < WUBU_IMGENC_EMBED_DIM; i++)
        if (!isfinite(emb[i])) finite = 0;
    CHECK(finite, "embedding is finite");
    printf("  ok: /kv/enc/photo-1 -> encode -> /kv/emb/photo-1 (%d-dim file)\n",
           WUBU_IMGENC_EMBED_DIM);

    /* ---- 4. the bytes ARE in the KV tensor region ---- */
    /* the /kv/emb mount covers blocks 512..544 at block_size 128:
     * the first embedding file lives at offset 512*128 floats. */
    size_t emb_offset = (size_t)512 * block_size;
    int in_tensor = 1;
    for (int i = 0; i < WUBU_IMGENC_EMBED_DIM; i++)
        if (kv_tensor[emb_offset + i] != emb[i]) in_tensor = 0;
    CHECK(in_tensor, "embedding bytes live in the KV tensor (the memory "
                     "the cache uses — the heavy work is shared)");

    /* ---- 5. namespace integrity ---- */
    float scratch[512];
    CHECK(wubu_kvfs_read(fs, "/kv/nowhere/x", kv_tensor, scratch, 4) == -1,
          "unmounted path fails (no silent corruption)");
    CHECK(wubu_enc_fs_encode(m, "bad/id", img, 8) == -1,
          "path-segment guard rejects slashes in ids");

    /* ---- 6. a second datum: distinct file, distinct embedding ---- */
    float img2[WUBU_IMGENC_IMAGE * WUBU_IMGENC_IMAGE * WUBU_IMGENC_CHANNELS];
    for (int i = 0; i < (int)(sizeof(img2)/sizeof(img2[0])); i++)
        img2[i] = (float)(((i + 1000) * 2654435761u) % 1000) / 1000.0f;
    CHECK(wubu_enc_fs_encode(m, "photo-2", img2,
          (size_t)WUBU_IMGENC_IMAGE*WUBU_IMGENC_IMAGE*WUBU_IMGENC_CHANNELS) == 0,
          "second datum encodes");
    float emb2[WUBU_IMGENC_EMBED_DIM];
    wubu_enc_fs_read_embedding(m, "photo-2", emb2);
    int distinct = 0;
    for (int i = 0; i < WUBU_IMGENC_EMBED_DIM; i++)
        if (emb[i] != emb2[i]) { distinct = 1; break; }
    CHECK(distinct, "distinct inputs -> distinct embedding files");

    wubu_enc_fs_free(m);
    wubu_kvfs_free(fs);
    free(kv_tensor);

    if (failures == 0) printf("=== ALL ENCODER-FS TESTS PASSED ===\n");
    else printf("=== %d FAILURES ===\n", failures);
    return failures ? 1 : 0;
}
