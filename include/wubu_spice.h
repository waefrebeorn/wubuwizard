#ifndef WUBU_SPICE_H
#define WUBU_SPICE_H

/*
 * wubu_spice.h -- EG-51: SPICE confidence-adaptive expert lookahead and
 * low-rank surrogate for low-confidence misses (opt-in behind accuracy gate),
 * reimplemented in C11.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    WUBU_SPICE_EXACT = 0,
    WUBU_SPICE_SURROGATE = 1,
    WUBU_SPICE_LOOKAHEAD = 2
};

typedef struct {
    int rows, cols, rank;
    float *sigma;
    float *u; /* rows * rank */
    float *v; /* cols * rank */
    float relerr;
} wubu_spice_surrogate_t;

/* Build a rank-limited approximation of M[rows*cols]; reports relerr. */
int wubu_spice_surrogate_build(const float *M, int rows, int cols, int rank,
                               wubu_spice_surrogate_t *out);
void wubu_spice_surrogate_free(wubu_spice_surrogate_t *s);

/* y = surrogate row applied to x[cols] (scalar output). */
int wubu_spice_surrogate_apply(const wubu_spice_surrogate_t *s, int row,
                               const float *x, float *y);

/* Route each expert: high confidence -> lookahead, low confidence ->
 * surrogate only when the measured relerr is within tol, else exact. */
int wubu_spice_route(const float *conf, int n, float relerr, float tol,
                     float lookahead_conf, int *actions);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_SPICE_H */
