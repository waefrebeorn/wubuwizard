#ifndef WUBU_STREAMK_H
#define WUBU_STREAMK_H

/*
 * wubu_streamk.h -- EG-79: portable Stream-K work partition and fixup.
 * A GEMM is split across blocks of output rows; each block writes float
 * partial sums, and a fixup pass folds them into the final output. This is
 * the C11 host escape from CUDA-native tiling, not a CUDA kernel claim.
 */

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_STREAMK_MAX_M 128
#define WUBU_STREAMK_MAX_BLOCKS 64

int wubu_streamk_partition(int m, int blocks, int *lo, int *hi);

/* out[m*n] = sum over blocks of A[i,k]*B[k,j]. 0/-1. */
int wubu_streamk_matmul(const float *A, const float *B,
                        int m, int n, int k, int blocks, float *out);

/* Number of partial groups folded into out. 0 groups on bad input. */
int wubu_streamk_fixups(const float *out, int m, int n, int blocks);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_STREAMK_H */
