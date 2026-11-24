/* wubu_bench_stats.c -- EG-106 benchmark statistics. Strict C11. */
#include "wubu_bench_stats.h"

#include <math.h>

static int valid(const double *samples, int n)
{
    return samples && n >= 1 && n <= WUBU_BENCH_MAX_SAMPLES;
}

double wubu_bench_mean(const double *samples, int n)
{
    double sum = 0.0;
    if (!valid(samples, n)) return -1.0;
    for (int i = 0; i < n; ++i) sum += samples[i];
    return sum / n;
}

double wubu_bench_sd(const double *samples, int n)
{
    double mean, sum = 0.0;
    if (!valid(samples, n) || n < 2) return -1.0;
    mean = wubu_bench_mean(samples, n);
    for (int i = 0; i < n; ++i) {
        double d = samples[i] - mean;
        sum += d * d;
    }
    return sqrt(sum / (n - 1));
}

double wubu_bench_cv(const double *samples, int n)
{
    double mean = wubu_bench_mean(samples, n);
    double sd = wubu_bench_sd(samples, n);
    if (sd < 0.0 || !(mean > 0.0)) return -1.0;
    return sd / mean;
}

int wubu_bench_regressed(const double *baseline, const double *candidate, int n)
{
    double mb, mc, sb, sc, se, margin;
    if (!valid(baseline, n) || !valid(candidate, n) || n < 2) return -1;
    mb = wubu_bench_mean(baseline, n);
    mc = wubu_bench_mean(candidate, n);
    sb = wubu_bench_sd(baseline, n);
    sc = wubu_bench_sd(candidate, n);
    if (mb < 0.0 || mc < 0.0 || sb < 0.0 || sc < 0.0) return -1;
    se = sqrt((sb * sb + sc * sc) / (double)n);
    margin = 2.0 * se;
    return mc < mb - margin ? 1 : 0;
}
