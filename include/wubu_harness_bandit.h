#ifndef WUBU_HARNESS_BANDIT_H
#define WUBU_HARNESS_BANDIT_H

/* wubu_harness_bandit.h -- repair-arm bandit pad (LP-04). C11, opaque,
 * status-as-value, no third-party deps.
 *
 * Arms are candidate revisions scored on severity / fixability / breadth /
 * side_effect. Selection explores under a cap then exploits the best value.
 * A revision whose side_effect exceeds the safety limit is rejected. */

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_harness_bandit wubu_harness_bandit_t;

#define WUBU_HARNESS_BANDIT_MAX_ARMS    32
#define WUBU_HARNESS_BANDIT_SAFETY_LIMIT 0.50

wubu_harness_bandit_t *wubu_harness_bandit_create(int exploration_cap);
void wubu_harness_bandit_free(wubu_harness_bandit_t *b);

/* Add an arm; scores are clamped to [0,1]. Returns arm index or -1. */
int wubu_harness_bandit_add(wubu_harness_bandit_t *b, double severity,
                            double fixability, double breadth, double side_effect);

/* Computed value = severity + fixability + breadth - side_effect. */
double wubu_harness_bandit_value(const wubu_harness_bandit_t *b, int arm);

/* Revise an arm's scores. Returns 0 accepted, -1 rejected as unsafe
 * (side_effect above the safety limit; scores left untouched). */
int wubu_harness_bandit_revise(wubu_harness_bandit_t *b, int arm,
                               double severity, double fixability,
                               double breadth, double side_effect);

/* Select an arm index under the exploration cap. The first `exploration_cap`
 * calls explore round-robin; thereafter greedy (highest reward-nudged value).
 * Returns -1 when empty/invalid. */
int wubu_harness_bandit_select(wubu_harness_bandit_t *b);

int wubu_harness_bandit_reward(wubu_harness_bandit_t *b, int arm, int reward);
int wubu_harness_bandit_count(const wubu_harness_bandit_t *b);
int wubu_harness_bandit_explorations(const wubu_harness_bandit_t *b);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_HARNESS_BANDIT_H */
