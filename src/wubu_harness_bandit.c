/* wubu_harness_bandit.c -- repair-arm bandit pad (LP-04).
 * C11, opaque struct, status-as-value, no third-party deps. */
#include <stdlib.h>
#include "wubu_harness_bandit.h"

struct hb_arm {
    double severity;
    double fixability;
    double breadth;
    double side_effect;
    int pulls;
    int rewards;
};

struct wubu_harness_bandit {
    int exploration_cap;
    int explore_used;
    int next_explore;
    int n;
    struct hb_arm arms[WUBU_HARNESS_BANDIT_MAX_ARMS];
};

static double clamp01(double x)
{
    return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x);
}

static double value_of(const struct hb_arm *a)
{
    return a->severity + a->fixability + a->breadth - a->side_effect;
}

wubu_harness_bandit_t *wubu_harness_bandit_create(int exploration_cap)
{
    if (exploration_cap < 0) return NULL;
    wubu_harness_bandit_t *b =
        (wubu_harness_bandit_t *)calloc(1, sizeof(*b));
    if (!b) return NULL;
    b->exploration_cap = exploration_cap;
    return b;
}

void wubu_harness_bandit_free(wubu_harness_bandit_t *b) { free(b); }

int wubu_harness_bandit_add(wubu_harness_bandit_t *b, double severity,
                            double fixability, double breadth, double side_effect)
{
    if (!b || b->n >= WUBU_HARNESS_BANDIT_MAX_ARMS) return -1;
    struct hb_arm *a = &b->arms[b->n];
    a->severity = clamp01(severity);
    a->fixability = clamp01(fixability);
    a->breadth = clamp01(breadth);
    a->side_effect = clamp01(side_effect);
    return b->n++;
}

double wubu_harness_bandit_value(const wubu_harness_bandit_t *b, int arm)
{
    if (!b || arm < 0 || arm >= b->n) return -1.0;
    return value_of(&b->arms[arm]);
}

int wubu_harness_bandit_revise(wubu_harness_bandit_t *b, int arm,
                               double severity, double fixability,
                               double breadth, double side_effect)
{
    if (!b || arm < 0 || arm >= b->n) return -1;
    if (side_effect > WUBU_HARNESS_BANDIT_SAFETY_LIMIT) return -1; /* unsafe */
    b->arms[arm].severity = clamp01(severity);
    b->arms[arm].fixability = clamp01(fixability);
    b->arms[arm].breadth = clamp01(breadth);
    b->arms[arm].side_effect = clamp01(side_effect);
    return 0;
}

int wubu_harness_bandit_select(wubu_harness_bandit_t *b)
{
    if (!b || b->n == 0) return -1;

    if (b->explore_used < b->exploration_cap) {
        int idx = b->next_explore % b->n;
        b->next_explore++;
        b->explore_used++;
        return idx;
    }

    /* Greedy: highest reward-nudged value, ties -> lowest index. */
    int best = 0;
    double best_v = value_of(&b->arms[0]) + 0.1 * (double)b->arms[0].rewards;
    for (int i = 1; i < b->n; i++) {
        double v = value_of(&b->arms[i]) + 0.1 * (double)b->arms[i].rewards;
        if (v > best_v) { best_v = v; best = i; }
    }
    return best;
}

int wubu_harness_bandit_reward(wubu_harness_bandit_t *b, int arm, int reward)
{
    if (!b || arm < 0 || arm >= b->n) return -1;
    b->arms[arm].pulls++;
    if (reward) b->arms[arm].rewards++;
    return 0;
}

int wubu_harness_bandit_count(const wubu_harness_bandit_t *b)
{
    return b ? b->n : -1;
}

int wubu_harness_bandit_explorations(const wubu_harness_bandit_t *b)
{
    return b ? b->explore_used : -1;
}
