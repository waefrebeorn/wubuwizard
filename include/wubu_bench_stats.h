#ifndef WUBU_BENCH_STATS_H
#define WUBU_BENCH_STATS_H

/*
 * wubu_bench_stats.h -- EG-106: benchmark statistics and decision rule.
 * Mean, sample standard deviation, coefficient of variation and a conservative
 * two-standard-error regression test. Used by the reproducible benchmark
 * workflow documented in docs/BENCHMARK-WORKFLOW.md; operates on measured
 * tok/s samples, not synthetic byte counts.
 */

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_BENCH_MAX_SAMPLES 128

double wubu_bench_mean(const double *samples, int n);
double wubu_bench_sd(const double *samples, int n);
double wubu_bench_cv(const double *samples, int n);

/* 1 when candidate mean is more than 2 combined standard errors below the
 * baseline; 0 when it is not a regression; -1 on invalid input. */
int wubu_bench_regressed(const double *baseline, const double *candidate, int n);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_BENCH_STATS_H */
