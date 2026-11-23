#ifndef WUBU_DFLY_H
#define WUBU_DFLY_H

/*
 * wubu_dfly.h -- EG-81: checkpoint-native drafting with shared KV.
 * Draft tokens are proposed from target hidden states and sparse indices;
 * both target and draft read the same KV store, so no separate draft KV is
 * allocated. The FP8/MXFP4 device path is out of scope for T1.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_DFLY_MAX_LAYERS 64
#define WUBU_DFLY_MAX_INDEX  64

typedef struct wubu_dfly wubu_dfly_t;

wubu_dfly_t *wubu_dfly_create(int layers, int draft_tokens,
                              size_t kv_bytes_per_token, int index_slots);
void wubu_dfly_free(wubu_dfly_t *p);

/* Bind a draft layer to the shared target KV; returns bytes avoided. */
size_t wubu_dfly_share_kv(wubu_dfly_t *p, int layer);

/* Sparse indices stored once and shared by all draft layers. */
int wubu_dfly_set_indices(wubu_dfly_t *p, const int *idx, int n);
int wubu_dfly_indices_match(const wubu_dfly_t *p, const int *idx, int n);

int wubu_dfly_separate_kv_allocations(const wubu_dfly_t *p);
size_t wubu_dfly_bytes_avoided(const wubu_dfly_t *p);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_DFLY_H */
