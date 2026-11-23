#ifndef WUBU_ONLINE_SD_H
#define WUBU_ONLINE_SD_H

/*
 * wubu_online_sd.h -- EG-59: online speculative decoding. The drafter is
 * updated from live verification corrections every I steps (online SD lineage),
 * reimplemented in C11. One gate: acceptance drift over a session.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int vocab, interval;
    float lr;
    float *bias;
    unsigned long long rng;
    int since_update;
} wubu_online_t;

int wubu_online_init(wubu_online_t *m, int vocab, int interval,
                     float lr, unsigned long long seed);
void wubu_online_free(wubu_online_t *m);

/* One propose/verify step. Returns 1 when the draft token was accepted, 0
 * when rejected, -1 on bad input. Updates the drafter every 'interval' steps. */
int wubu_online_step(wubu_online_t *m, const float *target);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_ONLINE_SD_H */
