/* wubu_userfs.h -- USER SPACE IN THE NAMESPACE (the user's files ARE
 * the training data — every user action trains at every time).
 *
 * The user's existing files (documents, code, notes, audio) are not
 * something we must go gather: they are already built. This module
 * bridges them INTO the KV namespace — the universal file-system
 * space — where they are interpreted and used:
 *
 *   real file (disk)            -> the namespace
 *   /home/u/notes/idea.md  ->   /kv/user/notes/idea.md
 *                               /kv/user/emb/idea.md.p1.emb  (chunk 1)
 *                               /kv/user/emb/idea.md.p2.emb  (chunk 2)
 *                               /kv/user/meta/idea.md        (usage)
 *                               /kv/user/INDEX               (ledger)
 *
 * RESEARCH/065 (the "make it the best" 7-hop):
 *   - PARAGRAPH-GROUP CHUNKING: content-aware chunking doubles
 *     retrieval (nDCG@5 0.459 vs fixed 0.244 — arXiv:2603.06976). A
 *     text file is split on blank lines into paragraph groups, each
 *     capped at WUBU_USERFS_CHUNK_CAP ids; EVERY chunk is its own
 *     embedding file, so retrieval targets the chunk, not the file.
 *   - USAGE LEDGER: opens/edits/reads are recorded at /kv/user/meta/<n>
 *     — the IMPLICIT FEEDBACK stream (arXiv:2606.20482: reward
 *     55%->64%, ~3x DPO gain). The RLHF loop consumes usage, not
 *     explicit labels. Every user trains at every time.
 *   - INCREMENTAL: name+size in the INDEX; a changed file re-ingests
 *     only its changed chunks. No data gathering, ever.
 *
 * Interpretation is OUR encoders behind the wubu_encoder slot (made,
 * not imported). Embeddings are FILES in the KV tensor region —
 * tiered, evictable, paged, persistent (the heavy work is already
 * lifted by wubu_kvfs).
 *
 * C11, opaque.
 */
#ifndef WUBU_USERFS_H
#define WUBU_USERFS_H

#include <stdint.h>
#include <stddef.h>

#include "wubu_kvfs.h"
#include "wubu_encoder.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_userfs wubu_userfs_t;

/* The namespace layout:
 *   /kv/user           — the user-space region (raw data files)
 *   /kv/user/emb       — the chunk-embedding region
 *   /kv/user/meta      — the usage ledger (implicit feedback)
 *   /kv/user/INDEX     — the ingest ledger (name|chunks|size|path)
 */
#define WUBU_USERFS_DATA_PATH   "/kv/user"
#define WUBU_USERFS_EMB_PATH    "/kv/user/emb"
#define WUBU_USERFS_META_PATH   "/kv/user/meta"
#define WUBU_USERFS_INDEX_NAME  "INDEX"

/* The paragraph-group chunk cap in token ids (research/065: content-
 * aware chunks, capped so the embedding stays focused). */
#define WUBU_USERFS_CHUNK_CAP   512

/* Mount the user space into the KV namespace. text_enc and audio_enc
 * are the interpreters (our encoders via the slot). Blocks
 * [data_start, +data_blocks) become /kv/user; [emb_start, +emb_blocks)
 * become /kv/user/emb; [meta_start, +meta_blocks) become
 * /kv/user/meta. Returns NULL on failure. */
wubu_userfs_t *wubu_userfs_mount(wubu_kvfs_t *fs, float *kv_base,
                                 wubu_encoder_t *text_enc,
                                 wubu_encoder_t *audio_enc,
                                 uint32_t data_start, uint32_t data_blocks,
                                 uint32_t emb_start, uint32_t emb_blocks,
                                 uint32_t meta_start, uint32_t meta_blocks);

/* Interpret one real user file into the namespace (chunked):
 *   - .txt/.md/.c/.h/.py/.json/.log  -> text: paragraph-group chunks
 *                                       -> one embedding file PER CHUNK
 *                                       (/kv/user/emb/<name>.p<k>.emb)
 *   - .wav                           -> audio: RIFF/PCM parse (ours) ->
 *                                       frames -> embedding
 *   - other                          -> classified + ledgered as
 *                                       unsupported
 * Returns 0 on success, 1 if skipped (unsupported/unchanged),
 * -1 on error. */
int wubu_userfs_ingest(wubu_userfs_t *m, const char *real_path);

/* Record a usage event on an ingested file: open/edit/read with a
 * weight (dwell seconds, edit magnitude, 1.0 for a plain read). This
 * is the IMPLICIT FEEDBACK stream — the RLHF loop (RC03) consumes it
 * instead of explicit labels. Returns 0 on success. */
int wubu_userfs_record_usage(wubu_userfs_t *m, const char *name,
                             const char *event, float weight);

/* Ingest every file in a real directory: only files whose name is not
 * in the INDEX (or whose size changed) are interpreted — incremental,
 * so every new/changed user file trains. Returns the count ingested. */
int wubu_userfs_ingest_dir(wubu_userfs_t *m, const char *dir_path);

/* Read the INDEX file back (the ledger of what the user space has
 * interpreted). Returns the number of entries. */
int wubu_userfs_read_index(wubu_userfs_t *m, char *buf, size_t buflen);

/* How many files the user space has ingested. */
int wubu_userfs_count(wubu_userfs_t *m);

/* The embedding dim (the shared space the interpreters write into). */
int wubu_userfs_dim(const wubu_userfs_t *m);

/* Destroy the mount (does NOT free the namespace). */
void wubu_userfs_free(wubu_userfs_t *m);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_USERFS_H */
