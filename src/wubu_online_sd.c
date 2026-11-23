/*
 * wubu_online_sd.c -- EG-59 online drafter corrections. Pure C11.
 */
#include "wubu_online_sd.h"

#include <math.h>
#include <stdlib.h>

static double urand(unsigned long long *s) {
    unsigned long long z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    return (double)(z >> 11) * (1.0 / 9007199254740992.0);
}

int wubu_online_init(wubu_online_t *m, int vocab, int interval,
                     float lr, unsigned long long seed) {
    if (!m || vocab <= 0 || interval <= 0 || !(lr >= 0.0f)) return -1;
    m->bias = (float *)calloc((size_t)vocab, sizeof(float));
    if (!m->bias) return -1;
    m->vocab = vocab;
    m->interval = interval;
    m->lr = lr;
    m->rng = seed;
    m->since_update = 0;
    return 0;
}

void wubu_online_free(wubu_online_t *m) {
    if (!m) return;
    free(m->bias);
    m->bias = NULL;
    m->vocab = 0;
}

int wubu_online_step(wubu_online_t *m, const float *target) {
    float q[64], mx = -INFINITY, den = 0.0f, acc = 0.0f, u;
    int tok = -1, accepted;
    if (!m || !target || !m->bias || m->vocab <= 0 || m->vocab > 64) return -1;
    for (int i = 0; i < m->vocab; i++) {
        if (!(target[i] >= 0.0f)) return -1;
        q[i] = m->bias[i];
        if (q[i] > mx) mx = q[i];
    }
    for (int i = 0; i < m->vocab; i++) {
        q[i] = expf(q[i] - mx);
        den += q[i];
    }
    for (int i = 0; i < m->vocab; i++) q[i] /= den;

    u = (float)urand(&m->rng);
    for (int i = 0; i < m->vocab; i++) {
        acc += q[i];
        if (u < acc) { tok = i; break; }
    }
    if (tok < 0) tok = m->vocab - 1;
    accepted = urand(&m->rng) < (q[tok] > 0.0f && target[tok] / q[tok] < 1.0f
                                 ? target[tok] / q[tok] : 1.0f);

    if (++m->since_update >= m->interval) {
        m->since_update = 0;
        for (int i = 0; i < m->vocab; i++)
            m->bias[i] += m->lr * (target[i] - q[i]);
    }
    return accepted;
}
