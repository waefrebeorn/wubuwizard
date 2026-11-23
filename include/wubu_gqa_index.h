#ifndef WUBU_GQA_INDEX_H
#define WUBU_GQA_INDEX_H

/*
 * wubu_gqa_index.h -- EG-44b: GQA grouping for a sparse attention index.
 * G=QH/KH query heads share one KV head; scores within a group are averaged
 * and top-k is selected once for the shared index instead of once per query
 * head. Work counters are model counts, not measured timings.
 */

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_GQA_MAX_HEADS 64
#define WUBU_GQA_MAX_ITEMS 256

typedef struct wubu_gqa_index wubu_gqa_index_t;

wubu_gqa_index_t *wubu_gqa_index_create(int q_heads, int kv_heads, int n_items);
void wubu_gqa_index_free(wubu_gqa_index_t *p);

int wubu_gqa_index_group_size(const wubu_gqa_index_t *p);

/* scores is [q_heads][n_items]; select top-k for one shared KV head. */
int wubu_gqa_index_select(const wubu_gqa_index_t *p, const float *scores,
                          int n_items, int kv_head, int k, int *out);

int wubu_gqa_index_work(const wubu_gqa_index_t *p, int k);
int wubu_gqa_index_work_naive(const wubu_gqa_index_t *p, int k);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_GQA_INDEX_H */
