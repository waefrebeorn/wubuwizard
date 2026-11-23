#ifndef WUBU_EXPERT_CACHE_H
#define WUBU_EXPERT_CACHE_H

/*
 * wubu_expert_cache.h -- EG-36b: MoE expert cache replacement policy.
 * ClassLRU protects a designated shared-expert class; within a class the
 * victim is the least-recently/frequently-used slot, scored as
 * freq / (now - last + 1). Misses are recorded as pending fetches, never as
 * synchronous CPU wait. This gates the policy, not a real weight residency.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_EXPERT_MAX_SLOTS 64

typedef struct wubu_expert_cache wubu_expert_cache_t;

wubu_expert_cache_t *wubu_expert_cache_create(int slots, int protected_class);
void wubu_expert_cache_free(wubu_expert_cache_t *p);

/* Record an expert access. Returns 1 on hit, 0 on a miss (insert/evict),
 * -1 on bad arguments. */
int wubu_expert_cache_access(wubu_expert_cache_t *p, int expert_id,
                             int class, uint64_t now);

int wubu_expert_cache_find(const wubu_expert_cache_t *p, int expert_id);
int wubu_expert_cache_count(const wubu_expert_cache_t *p);
int wubu_expert_cache_evictions(const wubu_expert_cache_t *p);
int wubu_expert_cache_protected_evictions(const wubu_expert_cache_t *p);
int wubu_expert_cache_wait_units(const wubu_expert_cache_t *p);
int wubu_expert_cache_pending_fetches(const wubu_expert_cache_t *p);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_EXPERT_CACHE_H */
