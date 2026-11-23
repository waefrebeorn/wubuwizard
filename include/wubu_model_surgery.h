#ifndef WUBU_MODEL_SURGERY_H
#define WUBU_MODEL_SURGERY_H

/*
 * wubu_model_surgery.h -- EG-39b: Hibana-style model surgery planner.
 * Given per-layer byte costs and importance scores, drop the cheapest
 * importance-per-byte layers until the memory budget is met, refusing when
 * the resulting importance loss exceeds the configured cap. This gates the
 * selection policy, not reproduced model quality or measured throughput.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_SURGERY_MAX_LAYERS 256

typedef struct {
    uint64_t bytes;
    double importance;
} wubu_surgery_layer_t;

/* keep_out[n] receives 1/0. Returns 0 when feasible, -1 otherwise. */
int wubu_surgery_plan(const wubu_surgery_layer_t *layers, int n,
                      uint64_t budget, double max_loss_frac, int *keep_out);

uint64_t wubu_surgery_bytes_saved(const wubu_surgery_layer_t *layers,
                                  const int *keep, int n);
uint64_t wubu_surgery_remaining_bytes(const wubu_surgery_layer_t *layers,
                                      const int *keep, int n);
double wubu_surgery_loss_fraction(const wubu_surgery_layer_t *layers,
                                  const int *keep, int n);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_MODEL_SURGERY_H */
