/*
 * wubu_kv_lazy.c -- EG-28 lazy KV quantization. Pure C11.
 */
#include "wubu_kv_lazy.h"

#include <stdlib.h>
#include <string.h>

struct wubu_kv_lazy {
    wubu_kv_layer_t layers[WUBU_KV_LAZY_MAX_LAYERS];
    int n;
    size_t budget;
};

wubu_kv_lazy_t *wubu_kv_lazy_create(const wubu_kv_layer_t *layers, int n,
                                    size_t budget) {
    wubu_kv_lazy_t *k;
    if (!layers || n <= 0 || n > WUBU_KV_LAZY_MAX_LAYERS) return NULL;
    k = (wubu_kv_lazy_t *)calloc(1, sizeof(*k));
    if (!k) return NULL;
    memcpy(k->layers, layers, (size_t)n * sizeof(layers[0]));
    k->n = n;
    k->budget = budget;
    return k;
}

void wubu_kv_lazy_free(wubu_kv_lazy_t *k) { free(k); }

int wubu_kv_lazy_apply(wubu_kv_lazy_t *k, wubu_kv_lazy_result_t *out) {
    size_t used = 0;
    int order[WUBU_KV_LAZY_MAX_LAYERS], n_order = 0, converted = 0;
    if (!k || !out) return -1;

    for (int i = 0; i < k->n; i++)
        used += k->layers[i].quantized ? k->layers[i].quant_bytes
                                       : k->layers[i].raw_bytes;

    /* Insertion sort by sensitivity / bytes saved (cheapest first). */
    for (int i = 0; i < k->n; i++) {
        if (k->layers[i].quantized) continue;
        if (k->layers[i].raw_bytes <= k->layers[i].quant_bytes) continue;
        int j = n_order++;
        order[j] = i;
        while (j > 0) {
            const wubu_kv_layer_t *a = &k->layers[order[j]];
            const wubu_kv_layer_t *b = &k->layers[order[j - 1]];
            float ca = a->sensitivity / (float)(a->raw_bytes - a->quant_bytes);
            float cb = b->sensitivity / (float)(b->raw_bytes - b->quant_bytes);
            if (ca >= cb) break;
            int t = order[j]; order[j] = order[j - 1]; order[j - 1] = t;
            j--;
        }
    }

    out->bytes_saved = 0;
    out->quality_cost = 0.0f;
    for (int j = 0; j < n_order && used > k->budget; j++) {
        wubu_kv_layer_t *l = &k->layers[order[j]];
        size_t save = l->raw_bytes - l->quant_bytes;
        l->quantized = 1;
        used -= save;
        out->bytes_saved += save;
        out->quality_cost += l->sensitivity;
        converted++;
    }
    out->converted = converted;
    out->bytes_after = used;
    out->within_budget = used <= k->budget ? 1 : 0;
    return 0;
}

const wubu_kv_layer_t *wubu_kv_lazy_layer(const wubu_kv_lazy_t *k, int i) {
    if (!k || i < 0 || i >= k->n) return NULL;
    return &k->layers[i];
}
