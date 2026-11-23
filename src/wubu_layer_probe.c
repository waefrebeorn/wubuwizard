/* wubu_layer_probe.c -- EG-100 measured layer probe gate. Strict C11. */
#include "wubu_layer_probe.h"

#include <stdlib.h>
#include <time.h>

struct wubu_layer_probe {
    int experts, dim, measured;
    float host_out[WUBU_LAYER_PROBE_MAX_DIM];
    double last_rate;
};

wubu_layer_probe_t *wubu_layer_probe_create(int experts, int dim)
{
    wubu_layer_probe_t *p;
    if (experts < 1 || experts > 64 || dim < 1 || dim > WUBU_LAYER_PROBE_MAX_DIM)
        return NULL;
    p = (wubu_layer_probe_t *)calloc(1, sizeof(*p));
    if (p) { p->experts = experts; p->dim = dim; }
    return p;
}

void wubu_layer_probe_free(wubu_layer_probe_t *p) { free(p); }

int wubu_layer_probe_run(wubu_layer_probe_t *p, int tokens, int iterations,
                         double *rate_mbs)
{
    clock_t start, elapsed;
    double bytes, seconds;
    volatile float sink = 0.0f;
    if (!p || !rate_mbs || tokens < 1 || tokens > 128 ||
        iterations < 1 || iterations > 1000000) return -1;

    start = clock();
    for (int r = 0; r < iterations; ++r) {
        for (int i = 0; i < p->dim; ++i) {
            float acc = 0.0f;
            for (int e = 0; e < p->experts; ++e)
                acc += (float)((e + 1) * (i + 1) * (tokens % 7 + 1)) * 0.001f;
            if (r == iterations - 1) p->host_out[i] = acc;
            sink += acc;
        }
    }
    elapsed = clock() - start;
    if (elapsed <= 0) elapsed = 1;
    (void)sink;
    bytes = (double)iterations * (double)tokens * (double)p->experts *
            (double)p->dim * 4.0;
    seconds = (double)elapsed / (double)CLOCKS_PER_SEC;
    p->last_rate = bytes / seconds / 1000000.0;
    p->measured = 1;
    *rate_mbs = p->last_rate;
    return 0;
}

const float *wubu_layer_probe_host_out(const wubu_layer_probe_t *p)
{
    return p ? p->host_out : NULL;
}

int wubu_layer_probe_host_out_n(const wubu_layer_probe_t *p)
{
    return p ? p->dim : -1;
}

int wubu_layer_probe_validate(const wubu_layer_probe_t *p,
                              const float *device_out, int n, int tokens)
{
    double limit, num = 0.0, den = 0.0;
    if (!p || !device_out || n != p->dim || !p->measured ||
        tokens < 1 || tokens > 128) return -1;
    limit = tokens == 1 ? 1e-3 : 1e-2;
    for (int i = 0; i < n; ++i) {
        double d = (double)device_out[i] - (double)p->host_out[i];
        num += d * d;
        den += (double)p->host_out[i] * (double)p->host_out[i];
    }
    if (den == 0.0) return num == 0.0 ? 0 : -1;
    return num / den <= limit ? 0 : -1;
}

double wubu_layer_probe_price(const wubu_layer_probe_t *p, double fallback_rate)
{
    if (!p || !p->measured) return fallback_rate;
    return p->last_rate;
}

int wubu_layer_probe_choose(const wubu_layer_probe_t *p,
                            const float *device_out, int n, int tokens,
                            double host_rate)
{
    if (!p || wubu_layer_probe_validate(p, device_out, n, tokens) != 0 ||
        !(p->last_rate > host_rate)) return 0;
    return 1;
}
