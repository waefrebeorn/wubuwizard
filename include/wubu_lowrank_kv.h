#ifndef WUBU_LOWRANK_KV_H
#define WUBU_LOWRANK_KV_H

/*
 * wubu_lowrank_kv.h -- EG-45a: low-rank key approximation (LowRAM lineage),
 * one KV family per gate. Deterministic subspace iteration in C11.
 */

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_LOWRANK_MAX_RANK 16

/* Approximate K[rows,dim] with rank-r factors U[rows,r], V[r,dim] and write
 * the reconstruction to out[rows,dim]. Returns 0 or -1. */
int wubu_lowrank_keys(const float *K, int rows, int dim, int rank,
                      int iters, float *U, float *V, float *out);

/* Relative Frobenius error of the reconstruction. */
float wubu_lowrank_relerr(const float *a, const float *b, int n);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_LOWRANK_KV_H */
