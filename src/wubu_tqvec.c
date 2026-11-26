/* wubu_tqvec.c - the TurboQuant vector quantiser (arXiv 2504.19874), distinct from wubu_turboquant.c.
 *
 * NOTE ON NAMING: src/wubu_turboquant.c already exists and is a KV-cache TILE MANAGER (allocation, LRU eviction,
 * frame planning) whose header comment mentions TurboQuant/Q2_0/Q4_0. It does not implement the algorithm from the
 * paper at all. This file is the actual data-oblivious vector quantiser, hence the separate name; do not merge them.
 *
 * Everything here was decided by measurement (docs/POINCARE-TURBOQUANT-RETRIEVAL.md):
 *
 *   - the rotation is BLOCK-DIAGONAL 8x8, not d x d. Measured identical MSE at every bit-width (0.1113 vs 0.1111 at
 *     b=2, d=32) at O(d) apply cost, and it regenerates from one 64-bit seed, so nothing has to be stored;
 *   - the codebook is a Lloyd-Max table for the rotated-coordinate density (1-t^2)^((d-3)/2), a function of (d, b)
 *     only - computed once at init, never fitted to data;
 *   - codes are bit-packed at b bits per coordinate.
 *
 * Measured distortion of this construction (tools/turboquant_probe.c, 4000 trials, d=32):
 *     D_mse = 0.3529 / 0.1111 / 0.0318 / 0.0087 for b = 1 / 2 / 3 / 4
 *     bound = 0.6802 / 0.1700 / 0.0425 / 0.0106
 *   i.e. 1.41x - 2.20x of the information-theoretic floor, inside the paper's guaranteed 2.7x.
 *
 * Retrieval caveat, measured (tools/turboquant_recall.c): scoring a corpus with wubu_tqvec_score alone gives
 * recall@10 of 0.23-0.62. Taking the top-100 by that score and re-ranking them exactly gives 0.94-1.00. The re-rank is
 * part of the design, not an optimisation.
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "wubu_tqvec.h"

#define TQV_BLOCK 8
#define TQV_GRID  8192

struct wubu_tqvec {
    int      d, bits, ncent, nblk;
    uint64_t seed;
    double  *rot;      /* nblk * 64: 8x8 orthogonal blocks */
    double  *cent;     /* 2^bits centroids */
    double  *scratch;  /* d doubles for the rotated query */
};

static uint64_t tqv_next(uint64_t *s)
{
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static double tqv_urand(uint64_t *s) { return (double) (tqv_next(s) >> 11) / 9007199254740992.0; }
static double tqv_nrand(uint64_t *s)
{
    double u = tqv_urand(s), v = tqv_urand(s);
    if (u < 1e-12) u = 1e-12;
    return sqrt(-2.0 * log(u)) * cos(6.283185307179586 * v);
}

static void tqv_build_codebook(struct wubu_tqvec *c)
{
    const int n = c->ncent;
    const double a = (double) (c->d - 3) / 2.0;
    double *t = malloc(sizeof(double) * TQV_GRID);
    double *w = malloc(sizeof(double) * TQV_GRID);
    if (!t || !w) { free(t); free(w); return; }
    double tot = 0.0;
    for (int i = 0; i < TQV_GRID; ++i) {
        t[i] = -1.0 + 2.0 * (i + 0.5) / TQV_GRID;
        w[i] = pow(1.0 - t[i] * t[i], a);
        tot += w[i];
    }
    for (int i = 0; i < TQV_GRID; ++i) w[i] /= tot;

    double acc = 0.0; int j = 0;
    for (int i = 0; i < TQV_GRID && j < n; ++i) {
        acc += w[i];
        if (acc >= (double) (j + 1) / (double) n - 1e-15) c->cent[j++] = t[i];
    }
    for (; j < n; ++j) c->cent[j] = t[TQV_GRID - 1];

    for (int it = 0; it < 120; ++it) {
        double bnd[260], sum[260], ws[260];
        for (int k = 1; k < n; ++k) bnd[k] = 0.5 * (c->cent[k - 1] + c->cent[k]);
        bnd[0] = -2.0; bnd[n] = 2.0;
        memset(sum, 0, sizeof sum); memset(ws, 0, sizeof ws);
        int seg = 0;
        for (int i = 0; i < TQV_GRID; ++i) {
            while (seg < n - 1 && t[i] > bnd[seg + 1]) ++seg;
            sum[seg] += t[i] * w[i];
            ws[seg] += w[i];
        }
        double dl = 0.0;
        for (int k = 0; k < n; ++k) {
            if (ws[k] <= 1e-18) continue;
            const double v = sum[k] / ws[k];
            dl += fabs(v - c->cent[k]);
            c->cent[k] = v;
        }
        if (dl < 1e-12) break;
    }
    free(t); free(w);
}

wubu_tqvec *wubu_tqvec_init(int d, int bits, uint64_t seed)
{
    if (d <= 0 || d % TQV_BLOCK != 0 || bits < 1 || bits > 8) return NULL;
    struct wubu_tqvec *c = calloc(1, sizeof *c);
    if (!c) return NULL;
    c->d = d; c->bits = bits; c->ncent = 1 << bits; c->nblk = d / TQV_BLOCK; c->seed = seed;
    c->rot = malloc(sizeof(double) * (size_t) c->nblk * TQV_BLOCK * TQV_BLOCK);
    c->cent = malloc(sizeof(double) * (size_t) c->ncent);
    c->scratch = calloc((size_t) d, sizeof(double));
    if (!c->rot || !c->cent || !c->scratch) { wubu_tqvec_free(c); return NULL; }

    uint64_t s = seed;
    for (int b = 0; b < c->nblk; ++b) {
        double *B = c->rot + (size_t) b * TQV_BLOCK * TQV_BLOCK;
        for (int i = 0; i < TQV_BLOCK; ++i)
            for (int j = 0; j < TQV_BLOCK; ++j) B[i * TQV_BLOCK + j] = tqv_nrand(&s);
        for (int i = 0; i < TQV_BLOCK; ++i) {            /* orthonormalise each 8x8 block */
            for (int k = 0; k < i; ++k) {
                double dot = 0.0;
                for (int j = 0; j < TQV_BLOCK; ++j) dot += B[i * TQV_BLOCK + j] * B[k * TQV_BLOCK + j];
                for (int j = 0; j < TQV_BLOCK; ++j) B[i * TQV_BLOCK + j] -= dot * B[k * TQV_BLOCK + j];
            }
            double nr = 0.0;
            for (int j = 0; j < TQV_BLOCK; ++j) nr += B[i * TQV_BLOCK + j] * B[i * TQV_BLOCK + j];
            nr = sqrt(nr);
            for (int j = 0; j < TQV_BLOCK; ++j) B[i * TQV_BLOCK + j] /= nr;
        }
    }
    tqv_build_codebook(c);
    return c;
}

void wubu_tqvec_free(wubu_tqvec *c)
{
    if (!c) return;
    free(c->rot); free(c->cent); free(c->scratch); free(c);
}

int    wubu_tqvec_dim(const wubu_tqvec *c)  { return c ? c->d : 0; }
int    wubu_tqvec_bits(const wubu_tqvec *c) { return c ? c->bits : 0; }
size_t wubu_tqvec_code_bytes(const wubu_tqvec *c)
{
    return c ? ((size_t) c->d * (size_t) c->bits + 7u) / 8u : 0;
}

static void tqv_rotate(const wubu_tqvec *c, const float *x, double *out)
{
    for (int b = 0; b < c->nblk; ++b) {
        const double *B = c->rot + (size_t) b * TQV_BLOCK * TQV_BLOCK;
        const int o = b * TQV_BLOCK;
        for (int i = 0; i < TQV_BLOCK; ++i) {
            double s = 0.0;
            for (int j = 0; j < TQV_BLOCK; ++j) s += B[i * TQV_BLOCK + j] * (double) x[o + j];
            out[o + i] = s;
        }
    }
}

static int tqv_nearest(const wubu_tqvec *c, double v)
{
    int best = 0; double bd = fabs(v - c->cent[0]);
    for (int k = 1; k < c->ncent; ++k) {
        const double dd = fabs(v - c->cent[k]);
        if (dd < bd) { bd = dd; best = k; }
    }
    return best;
}

static void tqv_put(unsigned char *code, int i, int bits, unsigned v)
{
    const size_t bit = (size_t) i * (size_t) bits;
    const size_t by  = bit >> 3;
    const int    of  = (int) (bit & 7u);
    code[by] |= (unsigned char) ((v << of) & 0xffu);
    if (of + bits > 8)  code[by + 1] |= (unsigned char) (v >> (8 - of));
    if (of + bits > 16) code[by + 2] |= (unsigned char) (v >> (16 - of));
}
static unsigned tqv_get(const unsigned char *code, int i, int bits)
{
    const size_t bit = (size_t) i * (size_t) bits;
    const size_t by  = bit >> 3;
    const int    of  = (int) (bit & 7u);
    unsigned v = (unsigned) code[by] >> of;
    if (of + bits > 8)  v |= (unsigned) code[by + 1] << (8 - of);
    if (of + bits > 16) v |= (unsigned) code[by + 2] << (16 - of);
    return v & ((1u << bits) - 1u);
}

void wubu_tqvec_encode(const wubu_tqvec *c, const float *x, void *code)
{
    if (!c || !x || !code) return;
    unsigned char *out = (unsigned char *) code;
    memset(out, 0, wubu_tqvec_code_bytes(c));
    tqv_rotate(c, x, c->scratch);
    for (int i = 0; i < c->d; ++i) tqv_put(out, i, c->bits, (unsigned) tqv_nearest(c, c->scratch[i]));
}

/* Hot path for a corpus scan: the score is a sum of q_z[r] * centroid[code_r] in the ROTATED frame, so the stored code
 * is never expanded back to d floats. */
float wubu_tqvec_score(const wubu_tqvec *c, const float *q, const void *code)
{
    if (!c || !q || !code) return 0.0f;
    const unsigned char *in = (const unsigned char *) code;
    tqv_rotate(c, q, c->scratch);
    double s = 0.0;
    for (int i = 0; i < c->d; ++i) s += c->scratch[i] * c->cent[tqv_get(in, i, c->bits)];
    return (float) s;
}

void wubu_tqvec_decode(const wubu_tqvec *c, const void *code, float *out)
{
    if (!c || !code || !out) return;
    const unsigned char *in = (const unsigned char *) code;
    double z[256];
    if (c->d > 256) return;                    /* keep it stack-only; d is 16-128 in this design */
    for (int i = 0; i < c->d; ++i) z[i] = c->cent[tqv_get(in, i, c->bits)];
    for (int b = 0; b < c->nblk; ++b) {        /* inverse of an orthogonal block is its transpose */
        const double *B = c->rot + (size_t) b * TQV_BLOCK * TQV_BLOCK;
        const int o = b * TQV_BLOCK;
        for (int j = 0; j < TQV_BLOCK; ++j) {
            double s = 0.0;
            for (int i = 0; i < TQV_BLOCK; ++i) s += B[i * TQV_BLOCK + j] * z[o + i];
            out[o + j] = (float) s;
        }
    }
}
