/* wubu_reclaim_margin.c -- EG-97 overflow-class memory margin. Strict C11. */
#include "wubu_reclaim_margin.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>

/* -ffast-math may elide float comparisons on NaN, so inspect the bit pattern. */
static int valid_fraction(double x)
{
    uint64_t bits;
    memcpy(&bits, &x, sizeof(bits));
    if ((bits & 0x7ff0000000000000ULL) == 0x7ff0000000000000ULL) return 0;
    return x >= 0.0 && x <= 1.0;
}

struct wubu_reclaim_margin {
    double frac[6];
    size_t min_bytes;
};

wubu_reclaim_margin_t *wubu_reclaim_margin_create(double compressed,
                                                  double swapped,
                                                  double killed,
                                                  double unknown,
                                                  size_t min_bytes)
{
    wubu_reclaim_margin_t *p;
    if (!valid_fraction(compressed) || !valid_fraction(swapped) ||
        !valid_fraction(killed) || !valid_fraction(unknown)) return NULL;
    p = (wubu_reclaim_margin_t *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->frac[WUBU_OVERFLOW_COMPRESS] = compressed;
    p->frac[WUBU_OVERFLOW_SWAP] = swapped;
    p->frac[WUBU_OVERFLOW_KILL] = killed;
    p->frac[WUBU_OVERFLOW_NONE] = 0.0;
    p->frac[WUBU_OVERFLOW_REFAULT] = compressed;
    p->frac[WUBU_OVERFLOW_UNKNOWN] = unknown;
    p->min_bytes = min_bytes;
    return p;
}

void wubu_reclaim_margin_free(wubu_reclaim_margin_t *p) { free(p); }

void wubu_reclaim_margin_set_refault(wubu_reclaim_margin_t *p, double fraction)
{
    if (!p || !valid_fraction(fraction)) return;
    p->frac[WUBU_OVERFLOW_REFAULT] = fraction;
}

size_t wubu_reclaim_margin_bytes(const wubu_reclaim_margin_t *p,
                                 size_t budget_bytes, wubu_overflow_t cls)
{
    size_t margin;
    if (!p || cls < WUBU_OVERFLOW_COMPRESS || cls > WUBU_OVERFLOW_UNKNOWN) return 0;
    if (cls == WUBU_OVERFLOW_NONE) return 0;
    margin = (size_t)(p->frac[cls] * (double)budget_bytes);
    return margin < p->min_bytes ? p->min_bytes : margin;
}

size_t wubu_reclaim_residency(const wubu_reclaim_margin_t *p,
                              size_t budget_bytes, wubu_overflow_t cls)
{
    size_t margin = wubu_reclaim_margin_bytes(p, budget_bytes, cls);
    return margin > budget_bytes ? 0 : budget_bytes - margin;
}

int wubu_reclaim_is_cheap(wubu_overflow_t cls)
{
    return cls == WUBU_OVERFLOW_COMPRESS || cls == WUBU_OVERFLOW_REFAULT;
}
