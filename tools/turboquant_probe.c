/* turboquant_probe.c - does the TurboQuant bound actually hold?
 *
 * The storage claim in docs/POINCARE-TURBOQUANT-RETRIEVAL.md rests entirely on one inequality from arXiv 2504.19874:
 *
 *      D_mse <= (sqrt(3)*pi/2) * 4^-b        for unit vectors, any dimension
 *      D_prod <= sqrt(3)*pi^2 * ||y||^2 / (d * 4^b)
 *
 * A bound is a testable prediction, not a hope. This builds the real pipeline and measures against it:
 *
 *   1. rotate a unit vector by a fixed random orthogonal matrix (data-oblivious, seeded);
 *   2. after rotation each coordinate has density f(t) ~ (1-t^2)^((d-3)/2) on [-1,1] - the Beta law the paper exploits,
 *      derived here rather than assumed, by integrating it numerically;
 *   3. build a Lloyd-Max codebook for that density at b bits (1-D k-means, the table is a constant of d and b only);
 *   4. quantise, dequantise, measure E||x - x_hat||^2 and the inner-product error.
 *
 * If the measured error is at or below 2.72/4^b the whole design is sound and the bits-per-item table is real. If it is
 * above, something in the construction is wrong and the storage claim is dead until it is found.
 *
 * Build: gcc -O2 -std=c11 -static -o tq.exe tools/turboquant_probe.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define GRID 65536          /* integration grid over [-1,1] for the Lloyd-Max pass */

static uint64_t g_s = 0x9E3779B97F4A7C15ULL;
static double urand(void)
{
    g_s ^= g_s >> 12; g_s ^= g_s << 25; g_s ^= g_s >> 27;
    return (double) ((g_s * 2685821657736338717ULL) >> 11) / 9007199254740992.0;
}
static double nrand(void)
{
    double u = urand(), v = urand();
    if (u < 1e-12) u = 1e-12;
    return sqrt(-2.0 * log(u)) * cos(6.283185307179586 * v);
}

/* density of one coordinate of a uniform unit vector in R^d: f(t) = C * (1-t^2)^((d-3)/2) on [-1,1] */
static double coord_density(double t, int d)
{
    const double a = (double) (d - 3) / 2.0;
    if (t <= -1.0 || t >= 1.0) return 0.0;
    return pow(1.0 - t * t, a);
}

/* Lloyd-Max on the discrete grid measure. Returns the 2^b centroids. */
static void lloyd_max(int d, int bits, double *cent)
{
    const int n = 1 << bits;
    double *t = malloc(sizeof(double) * GRID);
    double *w = malloc(sizeof(double) * GRID);
    double total = 0.0;
    for (int i = 0; i < GRID; ++i) {
        t[i] = -1.0 + 2.0 * (i + 0.5) / GRID;
        w[i] = coord_density(t[i], d);
        total += w[i];
    }
    for (int i = 0; i < GRID; ++i) w[i] /= total;

    /* init by equal-mass quantiles */
    double acc = 0.0; int j = 0;
    for (int i = 0; i < GRID && j < n; ++i) {
        acc += w[i];
        if (acc >= (double) (j + 1) / (double) n - 1e-15) { cent[j] = t[i]; ++j; }
    }
    for (; j < n; ++j) cent[j] = t[GRID - 1];

    for (int it = 0; it < 200; ++it) {
        double bnd[65];
        for (int k = 1; k < n; ++k) bnd[k] = 0.5 * (cent[k - 1] + cent[k]);
        bnd[0] = -2.0; bnd[n] = 2.0;
        double sum[65], wsum[65];
        memset(sum, 0, sizeof sum); memset(wsum, 0, sizeof wsum);
        int seg = 0;
        for (int i = 0; i < GRID; ++i) {
            while (seg < n - 1 && t[i] > bnd[seg + 1]) ++seg;
            sum[seg] += t[i] * w[i];
            wsum[seg] += w[i];
        }
        double delta = 0.0;
        for (int k = 0; k < n; ++k) {
            if (wsum[k] <= 1e-18) continue;
            const double c = sum[k] / wsum[k];
            delta += fabs(c - cent[k]);
            cent[k] = c;
        }
        if (delta < 1e-12) break;
    }
    free(t); free(w);
}

static int nearest(const double *cent, int n, double v)
{
    int best = 0; double bd = fabs(v - cent[0]);
    for (int k = 1; k < n; ++k) { const double dd = fabs(v - cent[k]); if (dd < bd) { bd = dd; best = k; } }
    return best;
}

int main(int argc, char **argv)
{
    const int d = argc > 1 ? atoi(argv[1]) : 32;
    const int trials = argc > 2 ? atoi(argv[2]) : 4000;

    /* fixed random orthogonal matrix by Gram-Schmidt on a Gaussian matrix (data-oblivious) */
    double *R = malloc(sizeof(double) * d * d);
    for (int i = 0; i < d * d; ++i) R[i] = nrand();
    for (int i = 0; i < d; ++i) {
        for (int k = 0; k < i; ++k) {
            double dot = 0.0;
            for (int j = 0; j < d; ++j) dot += R[i * d + j] * R[k * d + j];
            for (int j = 0; j < d; ++j) R[i * d + j] -= dot * R[k * d + j];
        }
        double nrm = 0.0;
        for (int j = 0; j < d; ++j) nrm += R[i * d + j] * R[i * d + j];
        nrm = sqrt(nrm);
        for (int j = 0; j < d; ++j) R[i * d + j] /= nrm;
    }

    double *x = malloc(sizeof(double) * d), *y = malloc(sizeof(double) * d);
    double *z = malloc(sizeof(double) * d), *xh = malloc(sizeof(double) * d);

    printf("d=%d, %d trials, rotation = fixed random orthogonal (seeded)\n\n", d, trials);
    printf("%-4s %-12s %-12s %-10s %-12s %-10s\n", "bits", "meas D_mse", "bound D_mse", "verdict", "meas IP err", "bound IP");

    for (int bits = 1; bits <= 4; ++bits) {
        const int n = 1 << bits;
        double *cent = malloc(sizeof(double) * (size_t) n);
        lloyd_max(d, bits, cent);

        double mse = 0.0, iperr = 0.0;
        for (int tr = 0; tr < trials; ++tr) {
            /* random unit x and a random unit y to score against */
            double nx = 0.0, ny = 0.0;
            for (int j = 0; j < d; ++j) { x[j] = nrand(); y[j] = nrand(); nx += x[j] * x[j]; ny += y[j] * y[j]; }
            nx = sqrt(nx); ny = sqrt(ny);
            for (int j = 0; j < d; ++j) { x[j] /= nx; y[j] /= ny; }

            /* rotate: z = R x */
            for (int i = 0; i < d; ++i) {
                double s = 0.0;
                for (int j = 0; j < d; ++j) s += R[i * d + j] * x[j];
                z[i] = s;
            }
            /* quantise each rotated coordinate, reconstruct in the rotated frame, rotate back by R^T */
            for (int i = 0; i < d; ++i) z[i] = cent[nearest(cent, n, z[i])];
            for (int j = 0; j < d; ++j) {
                double s = 0.0;
                for (int i = 0; i < d; ++i) s += R[i * d + j] * z[i];   /* R^T x */
                xh[j] = s;
            }
            double e = 0.0;
            for (int j = 0; j < d; ++j) { const double dd = x[j] - xh[j]; e += dd * dd; }
            mse += e;

            double ip = 0.0, iph = 0.0;
            for (int j = 0; j < d; ++j) { ip += y[j] * x[j]; iph += y[j] * xh[j]; }
            iperr += (ip - iph) * (ip - iph);
        }
        mse /= trials; iperr /= trials;

        const double bound = sqrt(3.0) * M_PI / 2.0 / (double) (1 << (2 * bits));
        const double ipb = sqrt(3.0) * M_PI * M_PI / ((double) d * (double) (1 << (2 * bits)));
        printf("%-4d %-12.4f %-12.4f %-10s %-12.5f %-10.5f\n", bits, mse, bound,
               mse <= bound ? "PASS" : "EXCEEDS", sqrt(iperr), sqrt(ipb));
        free(cent);
    }

    free(R); free(x); free(y); free(z); free(xh);
    return 0;
}
