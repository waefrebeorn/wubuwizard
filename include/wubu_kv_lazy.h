#ifndef WUBU_KV_LAZY_H
#define WUBU_KV_LAZY_H

/*
 * wubu_kv_lazy.h -- EG-28: lazy per-layer quantized-KV conversion with a
 * VRAM budget guard (mlx-lm maybe_quantize_kv_cache lineage), reimplemented
 * in C11.
 *
 * Each layer contributes raw KV bytes, quantized KV bytes and a quality
 * sensitivity score. When usage exceeds the budget, layers are converted in
 * increasing sensitivity-per-byte-saved order until the budget is met.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_kv_lazy wubu_kv_lazy_t;

#define WUBU_KV_LAZY_MAX_LAYERS 32

typedef struct {
    size_t raw_bytes;
    size_t quant_bytes;
    float sensitivity;      /* larger = more quality lost if quantized */
    int quantized;
} wubu_kv_layer_t;

typedef struct {
    int converted;          /* layers converted by the last call */
    size_t bytes_saved;
    size_t bytes_after;
    float quality_cost;     /* sum of sensitivities of converted layers */
    int within_budget;
} wubu_kv_lazy_result_t;

wubu_kv_lazy_t *wubu_kv_lazy_create(const wubu_kv_layer_t *layers, int n,
                                    size_t budget);
void wubu_kv_lazy_free(wubu_kv_lazy_t *k);

/* Convert layers lazily until usage <= budget. Returns 0 or -1. */
int wubu_kv_lazy_apply(wubu_kv_lazy_t *k, wubu_kv_lazy_result_t *out);

/* Current per-layer state (read-only). */
const wubu_kv_layer_t *wubu_kv_lazy_layer(const wubu_kv_lazy_t *k, int i);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_KV_LAZY_H */
