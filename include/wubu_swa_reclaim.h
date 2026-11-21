/*
 * wubu_swa_reclaim.h -- multi-owner sliding-window-attention cell reclaim.
 *
 * Several owners share a fixed set of cache cells. Each owner has its own
 * position bound: the oldest position it still needs. Positions strictly
 * below that bound are outside the owner's live window (masked). A cell may
 * be reused only when its current content is masked for EVERY owner -- a
 * single owner that still needs the content vetoes the reuse.
 *
 * When a cell is reused, the position it held is recorded in the log of the
 * owner whose data was overwritten (the displaced owner).
 *
 * Opaque struct, status-as-value, strict C11, no third-party dependencies.
 */
#ifndef WUBU_SWA_RECLAIM_H
#define WUBU_SWA_RECLAIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_swa_reclaim wubu_swa_reclaim_t;

typedef enum {
    WUBU_SWA_OK          = 0,
    WUBU_SWA_ERR_ARG     = 1,  /* NULL object */
    WUBU_SWA_ERR_OWNER   = 2,  /* owner id out of range */
    WUBU_SWA_ERR_CELL    = 3,  /* cell id out of range */
    WUBU_SWA_ERR_POS     = 4,  /* negative position / bound */
    WUBU_SWA_ERR_UNSAFE  = 5,  /* reuse refused: a live owner still needs it */
    WUBU_SWA_ERR_CAP     = 6,  /* overwritten-log growth failed (bounds) */
    WUBU_SWA_ERR_OOM     = 7   /* allocation failed */
} wubu_swa_status_t;

/* Create a cache of n_cells shared cells used by n_owners owners. All cells
 * start empty; all bounds start at 0. Returns NULL on bad args or OOM. */
wubu_swa_reclaim_t *wubu_swa_reclaim_create(int n_cells, int n_owners);
void wubu_swa_reclaim_free(wubu_swa_reclaim_t *c);

/* Set / read an owner's position bound (oldest still-needed position). */
wubu_swa_status_t wubu_swa_reclaim_set_bound(wubu_swa_reclaim_t *c,
                                             int owner, int64_t bound);
int64_t wubu_swa_reclaim_bound(const wubu_swa_reclaim_t *c, int owner);

/* True only when `pos` is fresh enough for `owner` AND the cell's current
 * content is masked for every owner. Pure: no mutation. */
int wubu_swa_reclaim_can_reuse(const wubu_swa_reclaim_t *c, int cell,
                               int owner, int64_t pos);

/* Reuse the cell; on refusal the cache is byte-identical. On success the
 * displaced position is appended to the displaced owner's log. */
wubu_swa_status_t wubu_swa_reclaim_reuse(wubu_swa_reclaim_t *c, int cell,
                                         int owner, int64_t pos);

/* Cell contents: -1 means empty. */
int64_t wubu_swa_reclaim_cell_pos(const wubu_swa_reclaim_t *c, int cell);
int     wubu_swa_reclaim_cell_owner(const wubu_swa_reclaim_t *c, int cell);

/* Per-owner overwritten-position log (positions of that owner displaced). */
int     wubu_swa_reclaim_overwritten_count(const wubu_swa_reclaim_t *c,
                                           int owner);
int64_t wubu_swa_reclaim_overwritten_at(const wubu_swa_reclaim_t *c,
                                        int owner, int index);

/* Deterministic state serialization for byte-identical assertions. */
size_t wubu_swa_reclaim_snapshot_size(const wubu_swa_reclaim_t *c);
void wubu_swa_reclaim_snapshot(const wubu_swa_reclaim_t *c,
                               unsigned char *out);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_SWA_RECLAIM_H */
