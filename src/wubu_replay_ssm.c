/* wubu_replay_ssm.c -- EG-69 checkpoint+input-buffer SSM gate. Strict C11. */
#include "wubu_replay_ssm.h"

#include <stdlib.h>
#include <string.h>

struct replay_input {
    double alpha;
    float v[WUBU_REPLAY_MAX_DIM];
    float k[WUBU_REPLAY_MAX_DIM];
    float q[WUBU_REPLAY_MAX_DIM];
};

struct wubu_replay_ssm {
    int dim, cap, buffered, checkpoints;
    double state[WUBU_REPLAY_MAX_DIM][WUBU_REPLAY_MAX_DIM];
    struct replay_input buf[WUBU_REPLAY_MAX_BUF];
};

wubu_replay_ssm_t *wubu_replay_ssm_create(int dim, int buffer_cap)
{
    wubu_replay_ssm_t *p;
    if (dim < 1 || dim > WUBU_REPLAY_MAX_DIM ||
        buffer_cap < 1 || buffer_cap > WUBU_REPLAY_MAX_BUF) return NULL;
    p = (wubu_replay_ssm_t *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->dim = dim;
    p->cap = buffer_cap;
    return p;
}

void wubu_replay_ssm_free(wubu_replay_ssm_t *p) { free(p); }

/* Recompute state from the checkpoint by replaying buffered inputs. */
static void replay_state(const wubu_replay_ssm_t *p,
                         double (*out_matrix)[WUBU_REPLAY_MAX_DIM],
                         float *o)
{
    int dim = p->dim < WUBU_REPLAY_MAX_DIM ? p->dim : WUBU_REPLAY_MAX_DIM;
    double s[WUBU_REPLAY_MAX_DIM][WUBU_REPLAY_MAX_DIM] = {{0.0}};
    memcpy(s, p->state, sizeof(s));
    for (int b = 0; b < p->buffered; ++b) {
        const struct replay_input *e = &p->buf[b];
        for (int i = 0; i < dim; ++i)
            for (int j = 0; j < dim; ++j)
                s[i][j] = e->alpha * s[i][j] + (double)e->v[i] * (double)e->k[j];
    }
    if (out_matrix)
        for (int i = 0; i < WUBU_REPLAY_MAX_DIM; ++i)
            for (int j = 0; j < WUBU_REPLAY_MAX_DIM; ++j) out_matrix[i][j] = s[i][j];
    if (o && p->buffered > 0) {
        const struct replay_input *last = &p->buf[p->buffered - 1];
        for (int i = 0; i < dim; ++i) {
            double acc = 0.0;
            for (int j = 0; j < dim; ++j) acc += s[i][j] * (double)last->q[j];
            o[i] = (float)acc;
        }
    }
}

int wubu_replay_ssm_step(wubu_replay_ssm_t *p, double alpha,
                         const float *v, const float *k,
                         const float *q, float *out)
{
    struct replay_input *e;
    if (!p || !v || !k || !q || !out) return -1;
    if (p->buffered >= p->cap) {
        /* Checkpoint is written only when the input buffer fills. */
        replay_state(p, p->state, NULL);
        ++p->checkpoints;
        p->buffered = 0;
    }
    e = &p->buf[p->buffered++];
    e->alpha = alpha;
    for (int i = 0; i < p->dim; ++i) {
        e->v[i] = v[i]; e->k[i] = k[i]; e->q[i] = q[i];
    }
    replay_state(p, NULL, out);
    return 0;
}

int wubu_replay_ssm_buffered(const wubu_replay_ssm_t *p)
{
    return p ? p->buffered : -1;
}

int wubu_replay_ssm_checkpoints(const wubu_replay_ssm_t *p)
{
    return p ? p->checkpoints : -1;
}

int wubu_replay_ssm_rollback(wubu_replay_ssm_t *p, int keep)
{
    if (!p || keep < 0 || keep > p->buffered) return -1;
    p->buffered = keep;
    return 0;
}
