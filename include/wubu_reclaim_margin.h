#ifndef WUBU_RECLAIM_MARGIN_H
#define WUBU_RECLAIM_MARGIN_H

/*
 * wubu_reclaim_margin.h -- EG-97: memory margin from kernel overflow class.
 * BigMoeOnEdge H8 (planner.cpp margin_frac / reclaim_is_cheap).
 *
 * The share of the residency budget left free depends on what losing a page
 * costs on this kernel, not on a fixed percentage. An unprofiled kernel keeps
 * the Unknown default; a kernel that never reclaims needs no margin.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    WUBU_OVERFLOW_COMPRESS = 0,
    WUBU_OVERFLOW_SWAP,
    WUBU_OVERFLOW_KILL,
    WUBU_OVERFLOW_NONE,
    WUBU_OVERFLOW_REFAULT,
    WUBU_OVERFLOW_UNKNOWN
} wubu_overflow_t;

typedef struct wubu_reclaim_margin wubu_reclaim_margin_t;

/* Fractions in [0,1]; min_bytes floors every nonzero class. NULL on bad input. */
wubu_reclaim_margin_t *wubu_reclaim_margin_create(double compressed,
                                                  double swapped,
                                                  double killed,
                                                  double unknown,
                                                  size_t min_bytes);
void wubu_reclaim_margin_free(wubu_reclaim_margin_t *p);

/* Refault uses the compressed fraction by default (cheap routine reclaim). */
void wubu_reclaim_margin_set_refault(wubu_reclaim_margin_t *p, double fraction);

/* max(budget * fraction(class), min_bytes); 0 for None. */
size_t wubu_reclaim_margin_bytes(const wubu_reclaim_margin_t *p,
                                 size_t budget_bytes, wubu_overflow_t cls);

/* Residency budget after reserving the margin. */
size_t wubu_reclaim_residency(const wubu_reclaim_margin_t *p,
                              size_t budget_bytes, wubu_overflow_t cls);

/* Only Compress and Refault are cheap; Unknown never inherits a verdict. */
int wubu_reclaim_is_cheap(wubu_overflow_t cls);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_RECLAIM_MARGIN_H */
