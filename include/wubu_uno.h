#ifndef WUBU_UNO_H
#define WUBU_UNO_H

/*
 * wubu_uno.h -- EG-55: gated-LoRA diffusion drafter on the verifier backbone,
 * one shared KV, Psi-Spec lossless sampling (UNO lineage), reimplemented in C11.
 *
 * One gate: lossless corrected sampling plus a single shared KV allocation.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int dim, rank, n_kv;
    float *W;    /* dim x dim verifier backbone */
    float *A;    /* dim x rank */
    float *B;    /* rank x dim */
    float *kv;   /* the one shared KV buffer */
} wubu_uno_t;

int wubu_uno_init(wubu_uno_t *m, int dim, int rank, int n_kv);
void wubu_uno_free(wubu_uno_t *m);

/* y = W x + sigmoid(gate) * A (B x). Both paths read the same KV pointer. */
int wubu_uno_draft_step(const wubu_uno_t *m, const float *x, float gate,
                        float *y);

/* Standard speculative-sampling accept/reject with residual correction.
 * Returns 1 when the draft token is accepted, 0 otherwise. */
int wubu_psi_sample(const float *p, const float *q, int n, int tok,
                    float u1, float u2, int *emit);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_UNO_H */
