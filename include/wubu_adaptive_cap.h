#ifndef WUBU_ADAPTIVE_CAP_H
#define WUBU_ADAPTIVE_CAP_H

/*
 * wubu_adaptive_cap.h -- EG-66b: TensorFold-style adaptive draft cap.
 * A full-accept window grows the cap by one, a mostly-rejected window shrinks
 * it by one, and the result is clamped to [min, max] and the free KV slots.
 * This gates the policy arithmetic, not a measured acceptance-rate gain.
 */

#ifdef __cplusplus
extern "C" {
#endif

int wubu_adaptive_cap_update(int current, int accepted, int drafted,
                             int kv_free_slots, int min_cap, int max_cap);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_ADAPTIVE_CAP_H */
