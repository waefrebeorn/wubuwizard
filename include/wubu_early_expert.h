#ifndef WUBU_EARLY_EXPERT_H
#define WUBU_EARLY_EXPERT_H

/*
 * wubu_early_expert.h -- EG-96: lossless cross-layer expert pread.
 * turbo-fieldfare H13: score layer L+1's experts with L+1's router on L's
 * router input, pread the best uncached ones into staging outside the slot
 * pool, and adopt a staged buffer only when the real plan misses it.
 * The planned expert list is never changed; only the read source is swapped.
 */

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_EE_MAX_EXPERTS 64

typedef struct wubu_early_expert wubu_early_expert_t;

/* reads in [1,8], staging_slots in [1,64]. NULL on bad input. */
wubu_early_expert_t *wubu_early_expert_create(int reads, int staging_slots);
void wubu_early_expert_free(wubu_early_expert_t *p);

/* Stage the highest-scoring uncached candidates, up to the read budget and
 * staging capacity. Returns the number staged, -1 on bad input. */
int wubu_early_expert_stage(wubu_early_expert_t *p,
                            const int *ids, const double *scores, int n,
                            const unsigned char *cached, int cached_n);

int wubu_early_expert_is_staged(const wubu_early_expert_t *p, int expert_id);
int wubu_early_expert_staged(const wubu_early_expert_t *p);

/* Real next-layer plan: adopt staged buffers on a miss, skip those reads.
 * out_ids receives the plan unchanged (lossless); used_staged[i] marks an
 * adopted slot. Returns the number adopted, -1 on bad input. */
int wubu_early_expert_adopt(wubu_early_expert_t *p, const int *plan_ids, int n,
                            int *out_ids, unsigned char *used_staged);

int wubu_early_expert_adopted(const wubu_early_expert_t *p);
int wubu_early_expert_wasted(const wubu_early_expert_t *p);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_EARLY_EXPERT_H */
