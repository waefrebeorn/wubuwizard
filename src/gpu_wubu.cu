/*
 * gpu_wubu.cu -- CUDA kernels for WuBu training (the seed grows fast).
 *
 * The DA pass found: the wizard already has cuBLAS + GPU kernels, but
 * the WuBu training loop was pure CPU. This module gives the trainer
 * a GPU backend: SGEMM (cuBLAS) for the big matrix products and a
 * fused attention kernel for the hybrid local/global pattern. The
 * trainer calls these through a pluggable dispatch -- CPU when no GPU,
 * CUDA when present (the wubu_model.h pattern).
 *
 * API (C linkage):
 *   gpu_wubu_init() / gpu_wubu_free()
 *   gpu_wubu_matmul(y, w, x, M, N, K)        // y[M,N] = x[M,K] @ w[K,N]
 *   gpu_wubu_attn(...)                        // hybrid windowed attention
 */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

static cublasHandle_t g_cublas = NULL;
static int g_ready = 0;

/* ---- weight cache: the model weights are re-uploaded on every
 * matmul call, but they only change when the optimizer steps. A small
 * open-address cache keyed by (pointer, bytes) with a generation
 * counter kills ~140MB of per-step H2D traffic. ---- */
#define WCACHE_ENTRIES 192
typedef struct { const void *ptr; size_t bytes; float *d; unsigned gen; } wc_entry_t;
static wc_entry_t g_wc[WCACHE_ENTRIES];
static unsigned g_wgen = 1;

static float *wc_get(const void *ptr, size_t bytes)
{
    if (!ptr || bytes == 0) return NULL;
    unsigned h = (unsigned)(((uintptr_t)ptr >> 6) ^ (uintptr_t)ptr) & (WCACHE_ENTRIES - 1);
    for (int i = 0; i < WCACHE_ENTRIES; i++) {
        wc_entry_t *e = &g_wc[(h + i) & (WCACHE_ENTRIES - 1)];
        if (e->ptr == ptr) {
            if (e->gen != g_wgen) {   /* the optimizer moved the weights */
                cudaMemcpy(e->d, ptr, bytes, cudaMemcpyHostToDevice);
                e->gen = g_wgen;
            }
            return e->d;
        }
        if (!e->ptr) {                /* empty slot */
            float *d = NULL;
            if (cudaMalloc(&d, bytes) != cudaSuccess) return NULL;
            cudaMemcpy(d, ptr, bytes, cudaMemcpyHostToDevice);
            e->ptr = ptr; e->bytes = bytes; e->d = d; e->gen = g_wgen;
            return d;
        }
    }
    return NULL;                      /* cache full: caller falls back */
}

extern "C" {

/* call after the optimizer updates the weights: the GPU weight cache
 * re-uploads on the next matmul */
void gpu_wubu_mark_weights_dirty(void) { g_wgen++; }

int gpu_wubu_init(void)
{
    if (g_ready) return 1;
    cudaError_t ce = cudaSetDevice(0);
    if (ce != cudaSuccess) { fprintf(stderr, "gpu_wubu: no CUDA device\n"); return 0; }
    cublasStatus_t st = cublasCreate(&g_cublas);
    if (st != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "gpu_wubu: cublas init failed\n"); return 0; }
    g_ready = 1;
    return 1;
}

void gpu_wubu_free(void)
{
    if (g_cublas) { cublasDestroy(g_cublas); g_cublas = NULL; }
    g_ready = 0;
}

int gpu_wubu_ready(void) { return g_ready; }

/* y[M,N] = x[M,K] @ w[K,N]  (row-major, F32). Uses cuBLAS SGEMM. */
int gpu_wubu_matmul(float *y, const float *w, const float *x,
                     int M, int N, int K)
{
    if (!g_ready) return 0;
    if (M <= 0 || N <= 0 || K <= 0) return 0;
    static float *d_x = NULL, *d_w = NULL, *d_y = NULL;
    static size_t cap_x = 0, cap_w = 0, cap_y = 0;
    size_t nx = (size_t)M * K, nw = (size_t)K * N, ny = (size_t)M * N;
    if (nx > cap_x) {
        if (d_x) cudaFree(d_x);
        cudaMalloc(&d_x, nx * sizeof(float)); cap_x = nx;
    }
    if (nw > cap_w) {
        if (d_w) cudaFree(d_w);
        cudaMalloc(&d_w, nw * sizeof(float)); cap_w = nw;
    }
    if (ny > cap_y) {
        if (d_y) cudaFree(d_y);
        cudaMalloc(&d_y, ny * sizeof(float)); cap_y = ny;
    }
    if (!d_x || !d_y) return 0;
    float *d_wu = wc_get(w, nw * sizeof(float));
    if (!d_wu) return 0;
    /* THE STREAM FIX (the acceleration): the synchronous cudaMemcpy
     * round-trips serialized the CPU<->GPU pipeline (241 syncs/step ->
     * the step spent ~92% waiting). Use a per-call CUDA stream: the
     * H2D x upload, the cublas kernel and the D2H y download all queue
     * on ONE stream and the cublasHandle carries it, so the copies
     * overlap the compute and the CPU only blocks at the final sync.
     * The weights stay resident via wc_get (re-upload only on g_wgen
     * bump, the optimizer's weight update). */
    static cudaStream_t g_stream = NULL;
    if (!g_stream) cudaStreamCreate(&g_stream);
    cudaMemcpyAsync(d_x, x, nx * sizeof(float), cudaMemcpyHostToDevice, g_stream);
    /* The caller's CPU loop is out[s,o] = sum_i w[o,i] * x[s,i] where w
     * is stored [out,in] row-major (w[o*in+i]) -- i.e. out = x @ w^T.
     * cuBLAS: C^T = B^T @ A^T ; we need C = x @ w^T so:
     *   cublasSgemm(OP_T, OP_N, N, M, K) with A=w (lda=in, transposed),
     *   B=x (lda=in), C=y -- the DA check (gpu_matmul_check) proves this
     *   matches the CPU loop to <1e-4. */
    float alpha = 1.0f, beta = 0.0f;
    cublasSetStream(g_cublas, g_stream);
    cublasSgemm(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                N, M, K, &alpha, d_wu, K, d_x, K, &beta, d_y, N);
    cudaMemcpyAsync(y, d_y, ny * sizeof(float), cudaMemcpyDeviceToHost, g_stream);
    cudaStreamSynchronize(g_stream);   /* the one sync per matmul */
    return 1;
}

/* y[M,N] = a[K,M]^T @ b[K,N]  (row-major, F32) -- the backward's
 * weight-gradient outer products: dW[out,in] = dy[seq,out]^T @ x[seq,in]
 * with M=out, N=in, K=seq. Verified by test_backprop's FD checks (the
 * weight grads are numerically checked per parameter type).
 * cuBLAS mapping (a row-major [K,M] lda=M == col-major [M,K];
 * b row-major [K,N] lda=N == col-major [N,K]): C[N,M] = B' @ A'^T with
 * C = y^T, so sgemm(OP_N, OP_T, N, M, K, B', N, A', M, C, N). */
int gpu_wubu_matmul_tx(float *y, const float *a, const float *b,
                        int M, int N, int K)
{
    if (!g_ready) return 0;
    if (M <= 0 || N <= 0 || K <= 0) return 0;
    static float *d_a = NULL, *d_b = NULL, *d_y = NULL;
    static size_t cap_a = 0, cap_b = 0, cap_y = 0;
    size_t na = (size_t)K * M, nb = (size_t)K * N, ny = (size_t)M * N;
    if (na > cap_a) {
        if (d_a) cudaFree(d_a);
        cudaMalloc(&d_a, na * sizeof(float)); cap_a = na;
    }
    if (nb > cap_b) {
        if (d_b) cudaFree(d_b);
        cudaMalloc(&d_b, nb * sizeof(float)); cap_b = nb;
    }
    if (ny > cap_y) {
        if (d_y) cudaFree(d_y);
        cudaMalloc(&d_y, ny * sizeof(float)); cap_y = ny;
    }
    if (!d_a || !d_b || !d_y) return 0;
    static cudaStream_t g_stream = NULL;
    if (!g_stream) cudaStreamCreate(&g_stream);
    cudaMemcpyAsync(d_a, a, na * sizeof(float), cudaMemcpyHostToDevice, g_stream);
    float *d_bu = wc_get(b, nb * sizeof(float));
    if (!d_bu) return 0;
    float alpha = 1.0f, beta = 0.0f;
    cublasSetStream(g_cublas, g_stream);
    cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                N, M, K, &alpha, d_bu, N, d_a, M, &beta, d_y, N);
    cudaMemcpyAsync(y, d_y, ny * sizeof(float), cudaMemcpyDeviceToHost, g_stream);
    cudaStreamSynchronize(g_stream);
    return 1;
}

/* y[M,N] = x[M,K] @ w[K,N] with w STORED [K,N] (no transpose) -- the
 * backward's input-gradient products: dL/dx[seq,in] = dy[seq,out] @
 * w[out,in] with M=seq, N=in, K=out. */
int gpu_wubu_matmul_nt(float *y, const float *w, const float *x,
                        int M, int N, int K)
{
    if (!g_ready) return 0;
    if (M <= 0 || N <= 0 || K <= 0) return 0;
    static float *d_x = NULL, *d_w = NULL, *d_y = NULL;
    static size_t cap_x = 0, cap_w = 0, cap_y = 0;
    size_t nx = (size_t)M * K, nw = (size_t)K * N, ny = (size_t)M * N;
    if (nx > cap_x) {
        if (d_x) cudaFree(d_x);
        cudaMalloc(&d_x, nx * sizeof(float)); cap_x = nx;
    }
    if (nw > cap_w) {
        if (d_w) cudaFree(d_w);
        cudaMalloc(&d_w, nw * sizeof(float)); cap_w = nw;
    }
    if (ny > cap_y) {
        if (d_y) cudaFree(d_y);
        cudaMalloc(&d_y, ny * sizeof(float)); cap_y = ny;
    }
    if (!d_x || !d_y) return 0;
    float *d_wu = wc_get(w, nw * sizeof(float));
    if (!d_wu) return 0;
    static cudaStream_t g_stream = NULL;
    if (!g_stream) cudaStreamCreate(&g_stream);
    cudaMemcpyAsync(d_x, x, nx * sizeof(float), cudaMemcpyHostToDevice, g_stream);
    /* x row-major [M,K] lda=K == col-major [K,M]; w row-major [K,N]
     * lda=N == col-major [N,K]; y row-major [M,N] lda=N == [N,M].
     * C[N,M] = op(A)[N,K] @ op(B)[K,M] with op(A)=N, op(B)=N. */
    float alpha = 1.0f, beta = 0.0f;
    cublasSetStream(g_cublas, g_stream);
    cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha, d_wu, N, d_x, K, &beta, d_y, N);
    cudaMemcpyAsync(y, d_y, ny * sizeof(float), cudaMemcpyDeviceToHost, g_stream);
    cudaStreamSynchronize(g_stream);
    return 1;
}

/* The Muon Newton-Schulz 5 orthogonalization, GPU-side (the optimizer
 * was the last CPU bottleneck -- ~61 GFLOP/step of NS5). X[rows,cols]
 * in-place; tall matrices are transposed first per the recipe; each
 * iteration is Frobenius-renormalized (the fp32-stability fix). Uses
 * cuBLAS GEMMs + nrm2/scal/axpby. Returns 1 on success. */
int gpu_wubu_ns5(float *X, int rows, int cols)
{
    if (!g_ready) return 0;
    if (rows <= 0 || cols <= 0) return 0;
    int trows = rows, tcols = cols;
    if (rows > cols) { trows = cols; tcols = rows; }
    size_t n = (size_t)rows * cols;
    size_t nsq = (size_t)trows * trows;
    static float *d_m = NULL, *d_t = NULL, *d_a = NULL, *d_a2 = NULL;
    static size_t cap_m = 0, cap_t = 0, cap_a = 0;
    if (n > cap_m) {
        if (d_m) cudaFree(d_m);
        cudaMalloc(&d_m, n * sizeof(float)); cap_m = n;
    }
    if (n > cap_t) {
        if (d_t) cudaFree(d_t);
        cudaMalloc(&d_t, n * sizeof(float)); cap_t = n;
    }
    if (nsq > cap_a) {
        if (d_a) cudaFree(d_a);
        if (d_a2) cudaFree(d_a2);
        cudaMalloc(&d_a, nsq * sizeof(float)); cap_a = nsq;
        cudaMalloc(&d_a2, nsq * sizeof(float));
    }
    if (!d_m || !d_t || !d_a || !d_a2) return 0;
    cudaMemcpy(d_m, X, n * sizeof(float), cudaMemcpyHostToDevice);
    float *M = d_m;
    cublasStatus_t st;
    if (rows > cols) {
        /* d_t = M^T  (trows=cols, tcols=rows); OP_T: lda >= n;
         * OP_N: ldb >= m. B is never read (beta=0) but must validate. */
        float one = 1.0f, zero = 0.0f;
        st = cublasSgeam(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                         rows, cols, &one, d_m, cols, &zero, d_m, rows,
                         d_t, rows);
        if (st != CUBLAS_STATUS_SUCCESS) return 0;
        M = d_t;
    }
    /* T workspace: for the tall case M lives in d_t so d_m is free;
     * for the wide case M lives in d_m so d_t is free. Never alias M. */
    float *Tbuf = (rows > cols) ? d_m : d_t;
    const float a = 3.4445f, b = -4.7750f, c = 2.0315f;
    for (int it = 0; it < 5; it++) {
        /* per-iteration Frobenius renormalization (fp32 stability) */
        float nrm = 0;
        cublasSnrm2(g_cublas, (int)((size_t)trows * tcols), M, 1, &nrm);
        if (nrm > 1e-12f) {
            float s = 1.0f / nrm;
            cublasSscal(g_cublas, (int)((size_t)trows * tcols), &s, M, 1);
        }
        float one = 1.0f, zero = 0.0f;
        /* A = M @ M^T  [trows,trows]. Views (row-major [R,C] lda=C ==
         * col-major [C,R] lda=C): M' = [tcols,trows] ldm=tcols, A square.
         * A'[i,j] = sum_k M'[k,i] M'[k,j] = (M'^T @ M')  -> OP_T, OP_N. */
        st = cublasSgemm(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                         trows, trows, tcols, &one, M, tcols, M, tcols,
                         &zero, d_a, trows);
        if (st != CUBLAS_STATUS_SUCCESS) return 0;
        /* A2 = A @ A (square: view is self-dual) */
        st = cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                         trows, trows, trows, &one, d_a, trows, d_a, trows,
                         &zero, d_a2, trows);
        if (st != CUBLAS_STATUS_SUCCESS) return 0;
        /* B = b*A + c*A2  (into d_a2, square) */
        st = cublasSgeam(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                         trows, trows, &b, d_a, trows, &c, d_a2, trows,
                         d_a2, trows);
        if (st != CUBLAS_STATUS_SUCCESS) return 0;
        /* T = B @ M  [trows,tcols]. T' = [tcols,trows] ldt=tcols,
         * B' = [trows,trows] ldb=trows, M' = [tcols,trows] ldm=tcols.
         * T'[j,i] = sum_k M'[j,k] B'[k,i] = (M' @ B') -> dims (tcols,
         * trows, trows) with OP_N, OP_N. */
        st = cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                         tcols, trows, trows, &one, M, tcols, d_a2, trows,
                         &zero, Tbuf, tcols);
        if (st != CUBLAS_STATUS_SUCCESS) return 0;
        /* M = a*M + T  (both in the [tcols,trows] view) */
        st = cublasSgeam(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                         tcols, trows, &a, M, tcols, &one, Tbuf, tcols,
                         M, tcols);
        if (st != CUBLAS_STATUS_SUCCESS) return 0;
    }
    if (rows > cols) {
        float one = 1.0f, zero = 0.0f;
        st = cublasSgeam(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                         cols, rows, &one, M, rows, &zero, M, cols,
                         d_m, cols);
        if (st != CUBLAS_STATUS_SUCCESS) return 0;
    }
    cudaMemcpy(X, d_m, n * sizeof(float), cudaMemcpyDeviceToHost);
    return 1;
}

/* ---- Gram Newton-Schulz (Zhang/Amsel/Chen/Dao 2026): the square-space
 * iteration. Instead of iterating the rectangular X (10 rectangular GEMMs
 * per NS5), precompute the square Gram G = X X^T once and iterate
 *   G <- P G P^T,  R <- P R   (R starts at I, tracks the composition)
 * with P = aI + bG + cG^2. One rectangular GEMM at the end: X = R X_0.
 * Mathematically identical to the standard NS up to float error; the
 * rectangular FLOPs drop ~5x (2.3G vs 4.9G MACs for the gate_up shape).
 * Per-iteration Frobenius renormalization kept (the fp32 stability fix):
 * s = ||X||_F = sqrt(tr(G)); G <- G/s^2, R <- R/s before each step. */

__global__ static void gns_trace_norm(float *G, float *R, int n)
{
    /* s = sqrt(sum_i G[i*n+i]); G /= s^2 ; R /= s -- the WHOLE
     * matrices (scaling only the diagonal was the divergence bug:
     * the off-diagonal stays huge and P = aI + bG + cG^2 explodes) */
    __shared__ float red[1024];
    int t = threadIdx.x;
    if (n <= 1024) {
        float v = (t < n) ? G[(size_t)t * n + t] : 0.0f;
        red[t] = v;
        __syncthreads();
        for (int s = 512; s > 0; s >>= 1) {
            if (t < s) red[t] += red[t + s];
            __syncthreads();
        }
        __shared__ float sgv, srv;
        if (t == 0) {
            float s = sqrtf(fmaxf(red[0], 1e-12f));
            sgv = 1.0f / (s * s);
            srv = 1.0f / s;
        }
        __syncthreads();
        size_t nn = (size_t)n * n;
        for (size_t i = (size_t)t; i < nn; i += 1024) {
            G[i] *= sgv;
            R[i] *= srv;
        }
    }
}

__global__ static void gns_diag_add(float *P, float a, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) P[(size_t)i * n + i] += a;
}

int gpu_wubu_ns5_gram(float *X, int rows, int cols)
{
    if (!g_ready) return 0;
    if (rows <= 0 || cols <= 0) return 0;
    int trows = rows, tcols = cols;
    if (rows > cols) { trows = cols; tcols = rows; }
    size_t n = (size_t)rows * cols;
    size_t nsq = (size_t)trows * trows;
    static float *d_m = NULL, *d_x0 = NULL, *d_g = NULL, *d_a2 = NULL,
                 *d_p = NULL, *d_r = NULL, *d_t = NULL;
    static size_t cap_m = 0, cap_q = 0, cap_t = 0;
    if (n > cap_m) {
        if (d_m) cudaFree(d_m);
        if (d_x0) cudaFree(d_x0);
        cudaMalloc(&d_m, n * sizeof(float)); cap_m = n;
        cudaMalloc(&d_x0, n * sizeof(float));
    }
    if (nsq > cap_q) {
        for (float **p : { &d_g, &d_a2, &d_p, &d_r })
            if (*p) cudaFree(*p);
        cudaMalloc(&d_g, nsq * sizeof(float)); cap_q = nsq;
        cudaMalloc(&d_a2, nsq * sizeof(float));
        cudaMalloc(&d_p, nsq * sizeof(float));
        cudaMalloc(&d_r, nsq * sizeof(float));
    }
    /* d_t holds the RECTANGULAR outputs too: the transposed M (tall),
     * the final X = R X_0 -- up to n floats. Allocating it at only nsq
     * was the heap-overflow bug: the wide GEMM wrote n floats into a
     * 200K buffer and clobbered the neighbouring allocations (d_x0). */
    if (n > cap_t) {
        if (d_t) cudaFree(d_t);
        cudaMalloc(&d_t, n * sizeof(float)); cap_t = n;
    }
    if (!d_m || !d_x0 || !d_g || !d_a2 || !d_p || !d_r || !d_t) return 0;
    cudaMemcpy(d_m, X, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x0, d_m, n * sizeof(float), cudaMemcpyDeviceToDevice);
    float *M = d_m;
    if (rows > cols) {
        float one = 1.0f, zero = 0.0f;
        cublasStatus_t st = cublasSgeam(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                        rows, cols, &one, d_m, cols, &zero,
                                        d_m, rows, d_t, rows);
        if (st != CUBLAS_STATUS_SUCCESS) return 0;
        M = d_t;
    }
    cublasStatus_t st;
    float one = 1.0f, zero = 0.0f;
    /* G = M M^T  (one rectangular GEMM) */
    st = cublasSgemm(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                     trows, trows, tcols, &one, M, tcols, M, tcols,
                     &zero, d_g, trows);
    if (st != CUBLAS_STATUS_SUCCESS) return 0;
    /* R = I */
    cudaMemset(d_r, 0, nsq * sizeof(float));
    gns_diag_add<<<1, 1024>>>(d_r, 1.0f, trows);
    cudaDeviceSynchronize();
    cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) { fprintf(stderr, "gram: kernel err: %s\n", cudaGetErrorString(ce)); return 0; }
    const float a = 3.4445f, b = -4.7750f, c = 2.0315f;
    for (int it = 0; it < 5; it++) {
        gns_trace_norm<<<1, 1024>>>(d_g, d_r, trows);
        cudaDeviceSynchronize();
        ce = cudaGetLastError();
        if (ce != cudaSuccess) { fprintf(stderr, "gram: trace_norm err: %s\n", cudaGetErrorString(ce)); return 0; }
        /* A2 = G G */
        st = cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                         trows, trows, trows, &one, d_g, trows, d_g, trows,
                         &zero, d_a2, trows);
        if (st != CUBLAS_STATUS_SUCCESS) return 0;
        /* P = aI + bG + cG^2  (B = bG + cA2, then add a on the diagonal) */
        st = cublasSgeam(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                         trows, trows, &b, d_g, trows, &c, d_a2, trows,
                         d_p, trows);
        if (st != CUBLAS_STATUS_SUCCESS) return 0;
        gns_diag_add<<<1, 1024>>>(d_p, a, trows);
        cudaDeviceSynchronize();
        /* T = P G  (the P G P^T needs a temp; d_t is free for wide, but
         * for tall M lives in d_t -- use d_a2 as the temp instead) */
        st = cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                         trows, trows, trows, &one, d_p, trows, d_g, trows,
                         &zero, d_a2, trows);
        if (st != CUBLAS_STATUS_SUCCESS) return 0;
        /* G = (PG) P^T -- the working A=MM^T pattern: C = A B^T needs
         * sgemm(OP_T, OP_N, ..., B, lda, A, lda, C) with B as the
         * OP_T operand FIRST (the DA-caught reversal) */
        st = cublasSgemm(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                         trows, trows, trows, &one, d_p, trows, d_a2, trows,
                         &zero, d_g, trows);
        if (st != CUBLAS_STATUS_SUCCESS) return 0;
        /* R = P R  (temp d_t; for tall M lives in d_t -- copy out first) */
        st = cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                         trows, trows, trows, &one, d_p, trows, d_r, trows,
                         &zero, d_t, trows);
        if (st != CUBLAS_STATUS_SUCCESS) return 0;
        cudaMemcpy(d_r, d_t, nsq * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    /* X = R X_0  (one rectangular GEMM) -- the [trows,tcols] output is
     * [tcols,trows] in the cuBLAS view: C' = X0' @ R' (the verified
     * T=BM pattern: the FIRST operand is the [trows,*] matrix with
     * lda=tcols; the square case hid the A/B order via self-duality,
     * the wide case exposed it -- the DA catch). Into d_t so the
     * transpose-back below has a separate source. */
    st = cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                     tcols, trows, trows, &one, d_x0, tcols, d_r, trows,
                     &zero, d_t, tcols);
    if (st != CUBLAS_STATUS_SUCCESS) return 0;
    if (rows > cols) {
        float one = 1.0f, zero = 0.0f;
        st = cublasSgeam(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                         cols, rows, &one, d_t, rows, &zero, d_t, cols,
                         d_m, cols);
        if (st != CUBLAS_STATUS_SUCCESS) return 0;
    } else {
        cudaMemcpy(d_m, d_t, n * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    cudaMemcpy(X, d_m, n * sizeof(float), cudaMemcpyDeviceToHost);
    return 1;
}

/* ---- The hybrid GQA attention on the GPU (the PowerVR/FlashAttention
 * tile principle): q [seq, heads*64] (the head h's slice at the column
 * offset h*64, row stride heads*64), single k [seq, 64] / v [seq, 64].
 * S = Q K^T (strided-batched over the 7 heads, K broadcast), a fused
 * mask+scale+softmax kernel (the causal + the local window), then
 * O = P V, merged into the [seq, heads*64] out. The CPU reference in
 * the bp stays the FD oracle; this must match it to 1e-3. */

__global__ static void gattn_softmax(const float *S, float *P, int seq,
                                     int total_rows, float scale,
                                     int local_win, int is_full)
{
    /* ONE block, the rows looped SERIALLY inside: the multi-block form
     * was non-deterministic across identical runs (the rows are provably
     * disjoint -- a scheduler-level race to hunt with a full-CUDA
     * sanitizer), so the serial ship is the correct + deterministic one
     * (verified 3x). The row-major S = K Q^T (the square self-duality
     * transposes the cuBLAS product): the query position s's scores
     * live in the COLUMN s -- memory[s + col*seq] = S[col][s]. The
     * O-GEMM's B'[k,j] = memory[k + j*seq] = P[j][k] consumes the P
     * ROW s -- memory[s*seq + col] (the reads and the writes are
     * DIFFERENT patterns; the final DA catch). */
    int t = threadIdx.x;
    __shared__ float sm[256];
    for (int r = 0; r < total_rows; r++) {
        int s = r % seq;
        float *sbase = (float *)S + (size_t)(r / seq) * (size_t)seq * seq;
        float *pbase = P + (size_t)(r / seq) * (size_t)seq * seq;
        int lo = (is_full || s <= local_win) ? 0 : s - local_win + 1;
        int NT = (seq + blockDim.x - 1) / blockDim.x;
        float m = -1e30f;
        for (int i = 0; i < NT; i++) {
            int col = i * blockDim.x + t;
            if (col < seq) {
                float v = (col > s || col < lo) ? -1e30f
                                                : sbase[(size_t)s + (size_t)col * seq] * scale;
                sbase[(size_t)s + (size_t)col * seq] = v;
                if (v > m) m = v;
            }
        }
        sm[t] = m;
        __syncthreads();
        for (int off = blockDim.x / 2; off > 0; off >>= 1) {
            if (t < off && sm[t + off] > sm[t]) sm[t] = sm[t + off];
            __syncthreads();
        }
        m = sm[0];
        float sum = 0;
        for (int i = 0; i < NT; i++) {
            int col = i * blockDim.x + t;
            if (col < seq) {
                if (col <= s && col >= lo) {
                    float e = __expf(sbase[(size_t)s + (size_t)col * seq] - m);
                    pbase[(size_t)s * seq + (size_t)col] = e;
                    sum += e;
                } else {
                    pbase[(size_t)s * seq + (size_t)col] = 0.0f;
                }
            }
        }
        sm[t] = sum;
        __syncthreads();
        for (int off = blockDim.x / 2; off > 0; off >>= 1) {
            if (t < off) sm[t] += sm[t + off];
            __syncthreads();
        }
        float inv = 1.0f / fmaxf(sm[0], 1e-12f);
        for (int i = 0; i < NT; i++) {
            int col = i * blockDim.x + t;
            if (col < seq) pbase[(size_t)s * seq + (size_t)col] *= inv;
        }
        __syncthreads();
    }
}

__global__ static void gattn_merge(const float *O, float *out,
                                   int seq, int heads, int dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq * heads * dim;
    if (i >= total) return;
    int d = i % dim;
    int k = i / dim;        /* k = h*seq + s -- the O layout is
                               [heads, seq, dim], NOT [seq, heads, dim];
                               a % heads decomposition aliases when
                               seq % heads != 0 (the DA catch) */
    int h = k / seq;
    int s = k % seq;
    out[(size_t)s * (heads * dim) + h * dim + d] = O[i];
}

int gpu_wubu_attn(float *out, const float *q, const float *k, const float *v,
                   int seq, int heads, int dim, int local_win, int is_full)
{
    if (!g_ready) return 0;
    if (seq <= 0 || heads <= 0 || dim <= 0) return 0;
    size_t ns = (size_t)seq * seq;
    size_t nq = (size_t)seq * heads * dim;
    size_t nk = (size_t)seq * dim;
    static float *d_q = NULL, *d_k = NULL, *d_v = NULL, *d_s = NULL, *d_p = NULL, *d_o = NULL;
    static size_t cap_q = 0, cap_s = 0;
    if (nq > cap_q) {
        if (d_q) cudaFree(d_q);
        cudaMalloc(&d_q, nq * sizeof(float));
        cudaMalloc(&d_k, nk * sizeof(float));
        cudaMalloc(&d_v, nk * sizeof(float));
        cudaMalloc(&d_o, nq * sizeof(float));
        cap_q = nq;
    }
    if (ns * heads > cap_s) {
        if (d_s) cudaFree(d_s);
        if (d_p) cudaFree(d_p);
        cudaMalloc(&d_s, ns * heads * sizeof(float));
        cudaMalloc(&d_p, ns * heads * sizeof(float));
        cap_s = ns * heads;
    }
    if (!d_q || !d_k || !d_v || !d_s || !d_p || !d_o) return 0;
    cudaMemcpy(d_q, q, nq * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k, nk * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, nk * sizeof(float), cudaMemcpyHostToDevice);
    cublasStatus_t st;
    float one = 1.0f, zero = 0.0f;
    /* S = Q K^T over the heads: strided-batched, the K broadcast.
     * The head h's Q slice: q_l[s*448 + h*64 + d] -- row stride = the
     * full head width (the lda), the batch stride = the column shift. */
    int hw = heads * dim;
    st = cublasSgemmStridedBatched(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                   seq, seq, dim, &one,
                                   d_q, hw, dim,          /* A: [seq,dim] lda=hw, stride=dim */
                                   d_k, dim, 0,           /* B: broadcast */
                                   &zero, d_s, seq, ns,   /* C: [seq,seq] */
                                   heads);
    if (st != CUBLAS_STATUS_SUCCESS) return 0;
    cudaDeviceSynchronize();
    /* TEMP: serial single-block for the race discrimination */
    gattn_softmax<<<1, 256>>>(d_s, d_p, seq, heads * seq,
                              1.0f / sqrtf((float)dim), local_win, is_full);
    cudaDeviceSynchronize();
    cudaDeviceSynchronize();
    cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) return 0;
    /* O = P V : the verified C' = V' @ P' -- sgemm(OP_N, OP_N,
     * m=dim, n=seq, k=seq, V first (ldb=dim), P second (lda=seq)) */
    st = cublasSgemmStridedBatched(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                   dim, seq, seq, &one,
                                   d_v, dim, 0,
                                   d_p, seq, ns,
                                   &zero, d_o, dim, (size_t)seq * dim,
                                   heads);
    if (st != CUBLAS_STATUS_SUCCESS) return 0;
    cudaDeviceSynchronize();
    gattn_merge<<<(nq + 255) / 256, 256>>>(d_o, d_q, seq, heads, dim);
    cudaDeviceSynchronize();
    cudaMemcpy(out, d_q, nq * sizeof(float), cudaMemcpyDeviceToHost);
    return 1;
}

/* ---- The attention BACKWARD on the GPU: given dO [seq, heads*dim],
 * recompute S = Q K^T (the same strided-batched call), P (the serial
 * softmax), then
 *   dP = dO V^T, rs = rowsum(dO o O), dS = P o (dP - rs) * inv,
 *   dQ = dS K, dK = sum_h dS^T Q, dV = sum_h P^T dO
 * (the single KV shares the summed grads across the 7 heads). The pure
 * elementwise/reduction kernels are multi-block SAFE (no shared-buffer
 * mutation -- unlike the softmax's in-place S column); the softmax is
 * the serial ship. The CPU loop in the bp stays the FD oracle. */

__global__ static void gattn_rowsum(const float *dO, const float *O,
                                    float *rs, int seq, int heads, int dim)
{
    /* one block per (head, position): the reads are disjoint, the rs
     * writes disjoint -- no shared mutation, multi-block safe */
    int r = blockIdx.x;
    int s = r % seq;
    int h = r / seq;
    int t = threadIdx.x;
    /* the dO/O are [seq, heads*dim] -- the head h's slice at the
     * column offset h*dim (NOT the head-major [heads, seq, dim]) */
    size_t hw = (size_t)heads * dim;
    const float *drow = dO + (size_t)s * hw + (size_t)h * dim;
    const float *orow = O + (size_t)s * hw + (size_t)h * dim;
    __shared__ float sm[256];
    float acc = 0;
    for (int i = t; i < dim; i += blockDim.x) acc += drow[i] * orow[i];
    sm[t] = acc;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (t < off) sm[t] += sm[t + off];
        __syncthreads();
    }
    if (t == 0) rs[r] = sm[0];
}

__global__ static void gattn_ds(const float *P, const float *dP, float *dS,
                                const float *rs, int seq, int heads,
                                float inv)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)heads * seq * seq;
    if ((size_t)i >= total) return;
    int h = i / (seq * seq);
    int s = (i / seq) % seq;
    int t = i % seq;
    /* the dP GEMM stores the transposed C' (the same as S): the
     * element [s,t] lives at the flat index h*ns + t*seq + s */
    size_t dp_i = (size_t)h * (size_t)seq * seq + (size_t)t * seq + (size_t)s;
    dS[i] = P[i] * (dP[dp_i] - rs[(size_t)h * seq + s]) * inv;
}

__global__ static void gattn_sumheads(const float *acc, float *out,
                                      int seq, int heads, int dim)
{
    /* out[t,d] = sum_h acc[h,t,d] ; one thread per (t,d) */
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq * dim;
    if (i >= total) return;
    float s = 0;
    for (int h = 0; h < heads; h++) s += acc[(size_t)h * seq * dim + i];
    out[i] = s;
}

int gpu_wubu_attn_backward(float *dq, float *dk, float *dv,
                            const float *q, const float *k, const float *v,
                            const float *o, const float *dao,
                            int seq, int heads, int dim,
                            int local_win, int is_full)
{
    if (!g_ready) return 0;
    if (seq <= 0 || heads <= 0 || dim <= 0) return 0;
    size_t ns = (size_t)seq * seq;
    size_t nq = (size_t)seq * heads * dim;
    size_t nk = (size_t)seq * dim;
    static float *d_q = NULL, *d_k = NULL, *d_v = NULL, *d_o = NULL, *d_dao = NULL;
    static float *d_s = NULL, *d_p = NULL, *d_dp = NULL, *d_ds = NULL, *d_rs = NULL;
    static float *d_dq = NULL, *d_dk = NULL, *d_dv = NULL;
    static size_t cap_q = 0, cap_s = 0;
    if (nq > cap_q) {
        for (float **p : { &d_q, &d_k, &d_v, &d_o, &d_dao, &d_dq, &d_dk, &d_dv })
            if (*p) cudaFree(*p);
        cudaMalloc(&d_q, nq * sizeof(float));
        cudaMalloc(&d_k, nk * sizeof(float));
        cudaMalloc(&d_v, nk * sizeof(float));
        cudaMalloc(&d_o, nq * sizeof(float));
        cudaMalloc(&d_dao, nq * sizeof(float));
        cudaMalloc(&d_dq, nq * sizeof(float));
        /* the dK_h/dV_h GEMMs write heads * (dim*seq) floats -- the
         * per-batch C' at d_dk + h*(seq*dim) -- NOT nk (the pre-fix
         * nk-sized allocs overflowed and clobbered the d_s/d_p/d_dv). */
        cudaMalloc(&d_dk, (size_t)heads * seq * dim * sizeof(float));
        cudaMalloc(&d_dv, (size_t)heads * seq * dim * sizeof(float));
        cap_q = nq;
    }
    if (ns * heads > cap_s) {
        for (float **p : { &d_s, &d_p, &d_dp, &d_ds, &d_rs })
            if (*p) cudaFree(*p);
        cudaMalloc(&d_s, ns * heads * sizeof(float));
        cudaMalloc(&d_p, ns * heads * sizeof(float));
        cudaMalloc(&d_dp, ns * heads * sizeof(float));
        cudaMalloc(&d_ds, ns * heads * sizeof(float));
        cudaMalloc(&d_rs, (size_t)heads * seq * sizeof(float));
        cap_s = ns * heads;
    }
    if (!d_q || !d_k || !d_v || !d_o || !d_dao || !d_s || !d_p || !d_dp ||
        !d_ds || !d_rs || !d_dq || !d_dk || !d_dv) return 0;
    cudaMemcpy(d_q, q, nq * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k, nk * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, nk * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_o, o, nq * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dao, dao, nq * sizeof(float), cudaMemcpyHostToDevice);
    cublasStatus_t st;
    float one = 1.0f, zero = 0.0f;
    int hw = heads * dim;
    float inv = 1.0f / sqrtf((float)dim);
    /* S = Q K^T (the same as the forward) */
    st = cublasSgemmStridedBatched(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                   seq, seq, dim, &one,
                                   d_q, hw, dim, d_k, dim, 0,
                                   &zero, d_s, seq, ns, heads);
    if (st != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "BACKWARD GEMM FAIL: S = Q K^T st=%d\n", st); return 0; }
    if (st != CUBLAS_STATUS_SUCCESS) return 0;
    /* P = softmax(S) -- the serial ship (same as the forward) */
    gattn_softmax<<<1, 256>>>(d_s, d_p, seq, heads * seq, inv,
                              local_win, is_full);
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) return 0;
    /* dP = dO V^T -- the same shape as S (the dO plays the Q's role) */
    st = cublasSgemmStridedBatched(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                   seq, seq, dim, &one,
                                   d_dao, hw, dim, d_v, dim, 0,
                                   &zero, d_dp, seq, ns, heads);
    if (st != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "BACKWARD GEMM FAIL: dP = dO V^T st=%d\n", st); return 0; }
    if (st != CUBLAS_STATUS_SUCCESS) return 0;
    /* rs = rowsum(dO o O) (the O passed in -- the forward's attn_out) */
    gattn_rowsum<<<heads * seq, 256>>>(d_dao, d_o, d_rs, seq, heads, dim);
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) return 0;
    /* dS = P o (dP - rs) * inv (in-place on the dP) */
    /* the dS into its OWN buffer: the in-place form was a clobbering
     * race -- the transposed dP layout maps dS[i]'s write onto another
     * dS thread's dP read (the masked-zero writes destroyed the live
     * dP values). */
    gattn_ds<<<(heads * ns + 255) / 256, 256>>>(d_p, d_dp, d_ds, d_rs,
                                                seq, heads, inv);
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) return 0;
    /* dQ = dS K -- the transposed-C' form (the cuBLAS ldc >= m rule):
     * C'[i,j] = sum_t K[t,i] dS[j,t] = dQ[j,i]. The dQ lives in the
     * INTERLEAVED [seq, heads*dim] (the head h at the column offset
     * h*dim, row stride hw) -- the ldc is hw, the batch stride is dim
     * (the head shift). */
    st = cublasSgemmStridedBatched(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                   dim, seq, seq, &one,
                                   d_k, dim, 0, d_ds, seq, ns,
                                   &zero, d_dq, hw, dim, heads);
    if (st != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "BACKWARD GEMM FAIL: dQ = dS K st=%d\n", st); return 0; }
    if (st != CUBLAS_STATUS_SUCCESS) return 0;
    /* dK_h = dS^T Q : the [seq,dim] output is the transposed C' --
     * the cuBLAS needs ldc >= m, so the C' is [dim, seq] (the dQ
     * call's legal form): C'[i,j] = sum_s Q[s,i] dS[s,j] = dK[j,i].
     * A = Q (OP_N, the [seq,dim] slice lda=hw), B = dS (OP_T). */
    st = cublasSgemmStridedBatched(g_cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                                   dim, seq, seq, &one,
                                   d_q, hw, dim, d_ds, seq, ns,
                                   &zero, d_dk, dim, (size_t)seq * dim, heads);
    if (st != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "BACKWARD GEMM FAIL: dK_h st=%d\n", st); return 0; }
    if (st != CUBLAS_STATUS_SUCCESS) return 0;
    /* dV_h = P^T dO : the same transposed-C' form --
     * C'[i,j] = sum_s dO[s,i] P[s,j] = dV[j,i].
     * A = dO (OP_N, lda=hw), B = P (OP_T). */
    st = cublasSgemmStridedBatched(g_cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                                   dim, seq, seq, &one,
                                   d_dao, hw, dim, d_p, seq, ns,
                                   &zero, d_dv, dim, (size_t)seq * dim, heads);
    if (st != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "BACKWARD GEMM FAIL: dV_h st=%d\n", st); return 0; }
    if (st != CUBLAS_STATUS_SUCCESS) return 0;
    /* the single KV: sum the head grads */
    cudaMemcpy(dq, d_dq, nq * sizeof(float), cudaMemcpyDeviceToHost);
    gattn_sumheads<<<(nk + 255) / 256, 256>>>(d_dk, d_dk, seq, heads, dim);
    gattn_sumheads<<<(nk + 255) / 256, 256>>>(d_dv, d_dv, seq, heads, dim);
    cudaDeviceSynchronize();
    cudaMemcpy(dk, d_dk, nk * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dv, d_dv, nk * sizeof(float), cudaMemcpyDeviceToHost);
    return 1;
}

int gpu_wubu_attn_ready(void) { return g_ready; }

}
