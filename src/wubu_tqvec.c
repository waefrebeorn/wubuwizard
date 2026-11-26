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
#define TQV_GRID  65536

struct wubu_tqvec {
    int      d, bits, ncent, nblk;
    uint64_t seed;
    signed char *sign; /* d random +/-1: the whole transform is a seed plus these signs */
    double  *cent;     /* 2^bits centroids */
    double  *scratch;  /* d doubles for the rotated query */
    double  *dscr;     /* d doubles for decode: decode ran once per row READ, and a malloc failure there
                        * returned silently leaving the caller's buffer uninitialised */
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
    const double sq = sqrt((double) c->d);          /* coordinate scale of a unit vector in R^d is 1/sqrt(d) */
    const double a  = (double) (c->d - 3) / 2.0;
    /* In u = t*sqrt(d) the coordinate has std 1 and its mass sits in |u| < ~4; the support reaches sqrt(d) but
     * with vanishing density. Spreading the grid to sqrt(d) put ~3 samples per bin at 256 centroids, so empty bins
     * kept their init positions and became bad codepoints - which is why 8 bits decoded worse than 6. */
    const double hi = 6.0;
    double *t = malloc(sizeof(double) * TQV_GRID);
    double *w = malloc(sizeof(double) * TQV_GRID);
    if (!t || !w) { free(t); free(w); return; }
    double tot = 0.0;
    for (int i = 0; i < TQV_GRID; ++i) {
        t[i] = -hi + 2.0 * hi * (i + 0.5) / TQV_GRID;
        const double r = t[i] / sq;
        w[i] = (r * r < 1.0) ? pow(1.0 - r * r, a) : 0.0;   /* outside the support pow() returns NaN */
        tot += w[i];
    }
    for (int i = 0; i < TQV_GRID; ++i) w[i] /= tot;

    double acc = 0.0; int j = 0;
    for (int i = 0; i < TQV_GRID && j < n; ++i) {
        acc += w[i];
        if (acc >= (double) (j + 1) / (double) n - 1e-15) c->cent[j++] = t[i] / sq;
    }
    for (; j < n; ++j) c->cent[j] = t[TQV_GRID - 1] / sq;

    for (int it = 0; it < 120; ++it) {
        double bnd[260], sum[260], ws[260];
        for (int k = 1; k < n; ++k) bnd[k] = 0.5 * (c->cent[k - 1] + c->cent[k]) * sq;
        bnd[0] = -2.0 * sq; bnd[n] = 2.0 * sq;
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
            const double v = sum[k] / ws[k] / sq;
            dl += fabs(v - c->cent[k]);
            c->cent[k] = v;
        }
        if (dl < 1e-14) break;
    }
    free(t); free(w);
}

wubu_tqvec *wubu_tqvec_init(int d, int bits, uint64_t seed)
{
    if (d <= 0 || bits < 1 || bits > 8) return NULL;
    if ((d & (d - 1)) != 0) return NULL;   /* Hadamard needs a power of two (32, 128, 1024 are) */
    struct wubu_tqvec *c = calloc(1, sizeof *c);
    if (!c) return NULL;
    c->d = d; c->bits = bits; c->ncent = 1 << bits; c->nblk = d / TQV_BLOCK; c->seed = seed;
    c->sign = malloc((size_t) c->d);
    c->cent = malloc(sizeof(double) * (size_t) c->ncent);
    c->scratch = calloc((size_t) d, sizeof(double));
    c->dscr = malloc((size_t) d * sizeof(double));
    if (!c->sign || !c->cent || !c->scratch || !c->dscr) { wubu_tqvec_free(c); return NULL; }

    uint64_t s = seed;
    for (int i = 0; i < c->d; ++i) c->sign[i] = (tqv_next(&s) & 1u) ? 1 : -1;
    tqv_build_codebook(c);
    return c;
}

void wubu_tqvec_free(wubu_tqvec *c)
{
    if (!c) return;
    free(c->sign); free(c->cent); free(c->scratch); free(c->dscr); free(c);
}

int    wubu_tqvec_dim(const wubu_tqvec *c)  { return c ? c->d : 0; }
int    wubu_tqvec_bits(const wubu_tqvec *c) { return c ? c->bits : 0; }
size_t wubu_tqvec_code_bytes(const wubu_tqvec *c)
{
    return c ? ((size_t) c->d * (size_t) c->bits + 7u) / 8u : 0;
}

static void tqv_fwht(double *a, int n)
{
    for (int h = 1; h < n; h <<= 1)
        for (int i = 0; i < n; i += h << 1)
            for (int j = i; j < i + h; ++j) {
                const double x = a[j], y = a[j + h];
                a[j] = x + y;
                a[j + h] = x - y;
            }
}

/* R = (1/sqrt(d)) H S : random signs then Hadamard. Orthogonal (R^T R = I) and it mixes EVERY coordinate with every
 * other in O(d log d), which the earlier 8x8 block-diagonal version did not: block rotation left structured rows
 * concentrated inside a few blocks, so the rotated coordinates were not i.i.d. Beta and real KV rows decoded with
 * 0.603 error on a norm-24.87 row against 4e-3 on a random unit vector. Random unit vectors are already spread, which
 * is why the block version measured identical in the synthetic probe and then failed on real data. */
static void tqv_rotate(const wubu_tqvec *c, const float *x, double *out)
{
    const double sc = 1.0 / sqrt((double) c->d);
    for (int i = 0; i < c->d; ++i) out[i] = (double) x[i] * (double) c->sign[i];
    tqv_fwht(out, c->d);
    for (int i = 0; i < c->d; ++i) out[i] *= sc;
}

/* inverse: R^T z = (1/sqrt(d)) S H z, H symmetric */
static void tqv_unrotate(const wubu_tqvec *c, const double *z, float *out)
{
    const double sc = 1.0 / sqrt((double) c->d);
    double *t = c->scratch;   /* distinct from dscr, which holds z */
    for (int i = 0; i < c->d; ++i) t[i] = z[i];
    tqv_fwht(t, c->d);
    for (int i = 0; i < c->d; ++i) out[i] = (float) (t[i] * (double) c->sign[i] * sc);
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
    double *z = c->dscr;
    if (!z) return;                            /* was a double z[256] with "if (d > 256) return" - which silently
                                                * left the output untouched for the KV dim (1024) */
    for (int i = 0; i < c->d; ++i) z[i] = c->cent[tqv_get(in, i, c->bits)];
    tqv_unrotate(c, z, out);

}
