#ifndef WUBU_REPLAY_SSM_H
#define WUBU_REPLAY_SSM_H

/*
 * wubu_replay_ssm.h -- EG-69: ReplaySSM-style recurrent state, C11 gate.
 *
 * State is a high-precision checkpoint plus a bounded buffer of recent inputs
 * (alpha, v, k, q). Every output is computed by replaying the checkpoint with
 * the buffer; the checkpoint is written only when the buffer fills. Speculative
 * rollback therefore truncates the input buffer — no state snapshot is copied.
 * Mamba-2 recurrence: S_t = alpha_t*S_{t-1} + v_t*k_t^T, o_t = S_t*q_t.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_REPLAY_MAX_DIM 4
#define WUBU_REPLAY_MAX_BUF 32

typedef struct wubu_replay_ssm wubu_replay_ssm_t;

/* dim in [1,4], buffer capacity in [1,32]. NULL on invalid input. */
wubu_replay_ssm_t *wubu_replay_ssm_create(int dim, int buffer_cap);
void wubu_replay_ssm_free(wubu_replay_ssm_t *p);

/* Append one input, compute its output from checkpoint+buffer. 0/-1. */
int wubu_replay_ssm_step(wubu_replay_ssm_t *p, double alpha,
                         const float *v, const float *k,
                         const float *q, float *out);

/* Inputs buffered since the last checkpoint commit. -1 on bad pointer. */
int wubu_replay_ssm_buffered(const wubu_replay_ssm_t *p);

/* Number of checkpoint commits. -1 on bad pointer. */
int wubu_replay_ssm_checkpoints(const wubu_replay_ssm_t *p);

/* Drop buffered inputs past `keep` (append-only rollback). 0/-1. */
int wubu_replay_ssm_rollback(wubu_replay_ssm_t *p, int keep);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_REPLAY_SSM_H */
