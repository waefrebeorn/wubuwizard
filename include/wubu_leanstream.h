#ifndef WUBU_LEANSTREAM_H
#define WUBU_LEANSTREAM_H

/*
 * wubu_leanstream.h -- EG-65: speculate-and-refine from partial GPU results
 * (LeanStream lineage), reimplemented as a C11 streaming gate. One gate: peak
 * working-set reduction versus bit-parity with the full materialization.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Full path: materialize every token's hidden state, then read out. */
int wubu_leanstream_full(const float *input, int n, int dim, float *out,
                         size_t *peak_bytes);

/* Streaming path: keep only one chunk plus a refined running summary. */
int wubu_leanstream_stream(const float *input, int n, int dim, int chunk,
                           float *out, size_t *peak_bytes);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_LEANSTREAM_H */
