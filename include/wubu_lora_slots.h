#ifndef WUBU_LORA_SLOTS_H
#define WUBU_LORA_SLOTS_H

/*
 * wubu_lora_slots.h -- EG-17: multi-adapter LoRA slots with grouped apply
 * (Punica bgmv/sgmv + LoRA LRU lineage), reimplemented in portable C11.
 *
 * Each slot owns one adapter and its dimensions. A grouped apply selects a
 * set of slots for one request, validates that every selected adapter has the
 * same linear dimensions, and sums their contributions into a fresh output.
 * No adapter state is shared between requests: the output is always
 * overwritten, never accumulated across calls.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_lora_slots wubu_lora_slots_t;

wubu_lora_slots_t *wubu_lora_slots_create(int capacity);
void wubu_lora_slots_free(wubu_lora_slots_t *s);

/* Install adapter `id` (0..capacity-1). Returns 0, or -1 on bad/dup id,
 * dimension error, or OOM. Copies A/B internally. */
int wubu_lora_slots_add(wubu_lora_slots_t *s, int id, int rank, float alpha,
                        int in_f, int out_f, const float *A, const float *B);

/* Remove an adapter, freeing its weights. Returns 0 or -1. */
int wubu_lora_slots_remove(wubu_lora_slots_t *s, int id);
int wubu_lora_slots_count(const wubu_lora_slots_t *s);

/* Sum contributions of the listed adapters for x[in_f] into out[out_f].
 * Returns 0, or -1 if any id is unknown, duplicated, or dimension-mismatched.
 */
int wubu_lora_slots_apply_grouped(const wubu_lora_slots_t *s,
                                  const int *ids, int n_ids,
                                  const float *x, float *out, int out_f);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_LORA_SLOTS_H */
