#ifndef WUBU_LAYER_PROBE_H
#define WUBU_LAYER_PROBE_H

/*
 * wubu_layer_probe.h -- EG-100: measured real-layer probe for tier pricing.
 * BigMoeOnEdge H6: time a layer's halves at decode and prompt widths, validate
 * the device against host outputs with width-dependent NMSE bounds (1e-3 at
 * one token, 1e-2 at 128), then price tiers from the measured rate rather than
 * a static cost. Missing kernels fall back to the host rate.
 */

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_LAYER_PROBE_MAX_DIM 64

typedef struct wubu_layer_probe wubu_layer_probe_t;

/* experts in [1,64], dim in [1,64]. NULL on bad input. */
wubu_layer_probe_t *wubu_layer_probe_create(int experts, int dim);
void wubu_layer_probe_free(wubu_layer_probe_t *p);

/* Time the synthetic layer at `tokens` width for `iterations` reps.
 * Writes a measured MB/s rate. Returns 0 on success, -1 on bad input. */
int wubu_layer_probe_run(wubu_layer_probe_t *p, int tokens, int iterations,
                         double *rate_mbs);

const float *wubu_layer_probe_host_out(const wubu_layer_probe_t *p);
int wubu_layer_probe_host_out_n(const wubu_layer_probe_t *p);

/* NMSE gate against the last host result: 1e-3 at 1 token, 1e-2 at 128. */
int wubu_layer_probe_validate(const wubu_layer_probe_t *p,
                              const float *device_out, int n, int tokens);

/* Measured rate when a probe ran, otherwise the fallback host rate. */
double wubu_layer_probe_price(const wubu_layer_probe_t *p, double fallback_rate);

/* Choose the device tier only when it validates and beats the host rate. */
int wubu_layer_probe_choose(const wubu_layer_probe_t *p,
                            const float *device_out, int n, int tokens,
                            double host_rate);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_LAYER_PROBE_H */
