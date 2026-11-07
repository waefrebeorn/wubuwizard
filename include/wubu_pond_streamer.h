/* wubu_pond_streamer.h — real-time research-pond intake into the KV namespace
 *
 * The research ponds (research/047) are 701 MB of pure text living on disk
 * under /home/wubu/research-ponds-work/ponds/ (pond-NAME/text/). This module
 * streams them into /kv/in/ as an on-disk watcher: a poll() rescans the
 * watched roots and encodes any NEW or CHANGED file through the KV embedding
 * bridge, so the training pipeline (wubu_fs_dataset + wubu_coherence_reward)
 * sees pond knowledge appear in the filesystem namespace in near-real-time.
 *
 * Incremental: a ledger of (fullpath, mtime, size) means unchanged files are
 * skipped on every poll — only new or touched files cost an encode. Planned
 * disk budget is not preloaded; the model "browses" the ponds as they change.
 *
 * C11 (POSIX dirent/stat), opaque structs, no third-party deps.
 * Design: docs/wubu1-improvement-plan.md module #4 (AN22).
 * WaefreBeorn Umbrella License v3.0
 */
#ifndef WUBU_POND_STREAMER_H
#define WUBU_POND_STREAMER_H

#include <stddef.h>
#include <stdint.h>
#include "wubu_kv_embedding.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle */
typedef struct wubu_pond_streamer wubu_pond_streamer_t;

/* Create a streamer over an existing KV embedding bridge and a set of
 * pond root directories. Each root's basename becomes its pond name in the
 * namespace, so a root .../pond-a-kv-memory maps files under
 * /kv/in/pond-a-kv-memory/<relpath>. `kv` must outlive the streamer.
 * Returns NULL on argument error or out-of-memory. */
wubu_pond_streamer_t *wubu_pond_streamer_create(wubu_kv_embedding_t *kv,
                                                const char **pond_roots,
                                                size_t n_roots);

/* Scan every watched root and stream NEW or CHANGED regular files into the
 * KV namespace. Returns 0 on success with *out_new set to the number of
 * files streamed this poll (0 if nothing changed), or -1 on error. */
int wubu_pond_streamer_poll(wubu_pond_streamer_t *ps, int *out_new);

/* Total files currently tracked in the ledger. */
int wubu_pond_streamer_count(const wubu_pond_streamer_t *ps);

/* Cumulative number of stream events since creation (each re-encode of a
 * touched file counts). */
unsigned long wubu_pond_streamer_total_streamed(const wubu_pond_streamer_t *ps);

/* Free the streamer. Does NOT free kv (caller-owned). */
void wubu_pond_streamer_free(wubu_pond_streamer_t *ps);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_POND_STREAMER_H */
