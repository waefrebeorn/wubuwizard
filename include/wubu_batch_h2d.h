#ifndef WUBU_BATCH_H2D_H
#define WUBU_BATCH_H2D_H

/*
 * wubu_batch_h2d.h -- EG-98: portable batched expert H2D upload planner.
 * Strata BATCHED_DMA: mode 0 = per-expert copies, mode 1 = one batch call,
 * mode 2 = batch with overlap-with-compute flag. Groups capped at 128;
 * sources must be immutable pinned blobs with disjoint destinations.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_H2D_MAX_GROUP 128

typedef struct wubu_batch_h2d wubu_batch_h2d_t;

/* mode in {0,1,2}, capacity >= 1. NULL on bad input. */
wubu_batch_h2d_t *wubu_batch_h2d_create(int mode, int capacity);
void wubu_batch_h2d_free(wubu_batch_h2d_t *p);

/* Add one pinned immutable host blob targeting a device range. 0/-1. */
int wubu_batch_h2d_add(wubu_batch_h2d_t *p, const unsigned char *src,
                       size_t dst_offset, size_t bytes, int pinned);

/* Copy all blobs into dev and return the number of submission groups, or -1
 * on a rejected batch (nothing is committed on failure). */
int wubu_batch_h2d_issue(wubu_batch_h2d_t *p,
                         unsigned char *dev, size_t dev_bytes);

int wubu_batch_h2d_groups(const wubu_batch_h2d_t *p);

/* Inject a submission failure at group index (>=0), or disable with -1.
 * A failed batch commits no bytes. Returns 0/-1. */
int wubu_batch_h2d_set_fail_at(wubu_batch_h2d_t *p, int group);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_BATCH_H2D_H */
