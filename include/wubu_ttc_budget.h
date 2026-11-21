#ifndef WUBU_TTC_BUDGET_H
#define WUBU_TTC_BUDGET_H

/* wubu_ttc_budget.h -- test-time-compute budget pad (LP-14). C11, opaque,
 * status-as-value, no third-party deps.
 *
 * Accounts measured wall time (ms) + KV bytes + candidate count against caps,
 * and stops early on a score plateau. try() returns 1 while more work is
 * allowed, 0 once any budget boundary or the plateau stop is hit. */

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_ttc_budget wubu_ttc_budget_t;

wubu_ttc_budget_t *wubu_ttc_budget_create(long long max_wall_ms,
                                          size_t max_kv_bytes,
                                          int max_candidates,
                                          int plateau_window,
                                          double min_gain);
void wubu_ttc_budget_free(wubu_ttc_budget_t *b);

/* Start a fresh accounting window (records the wall clock). */
int wubu_ttc_budget_begin(wubu_ttc_budget_t *b);

/* Account a candidate and its score. Returns 1 if more work is allowed,
 * 0 if the wall / kv / count / plateau budget is exhausted. */
int wubu_ttc_budget_try(wubu_ttc_budget_t *b, double score);

int wubu_ttc_budget_add_kv(wubu_ttc_budget_t *b, size_t kv_bytes);
int wubu_ttc_budget_add_wall_ms(wubu_ttc_budget_t *b, long long ms);

size_t wubu_ttc_budget_kv_bytes(const wubu_ttc_budget_t *b);
long long wubu_ttc_budget_elapsed_ms(const wubu_ttc_budget_t *b);
int wubu_ttc_budget_candidates(const wubu_ttc_budget_t *b);
int wubu_ttc_budget_plateaued(const wubu_ttc_budget_t *b);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_TTC_BUDGET_H */
