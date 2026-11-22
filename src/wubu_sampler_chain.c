/*
 * wubu_sampler_chain.c -- EG-12 composable sampler chain. Pure C11.
 */
#include "wubu_sampler_chain.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct { wubu_sampler_stage_t stage; double param; } chain_stage_t;

struct wubu_sampler_chain {
    chain_stage_t *stages;
    int n, cap;
};

wubu_sampler_chain_t *wubu_sampler_chain_create(int max_stages) {
    wubu_sampler_chain_t *c;
    if (max_stages <= 0) return NULL;
    c = (wubu_sampler_chain_t *)calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->stages = (chain_stage_t *)calloc((size_t)max_stages,
                                        sizeof(*c->stages));
    if (!c->stages) { free(c); return NULL; }
    c->cap = max_stages;
    return c;
}

void wubu_sampler_chain_free(wubu_sampler_chain_t *c) {
    if (!c) return;
    free(c->stages);
    free(c);
}

int wubu_sampler_chain_stage_count(const wubu_sampler_chain_t *c) {
    return c ? c->n : -1;
}

int wubu_sampler_chain_add(wubu_sampler_chain_t *c,
                            wubu_sampler_stage_t stage, double param) {
    if (!c || c->n >= c->cap || param <= 0.0) return -1;
    switch (stage) {
        case WUBU_SAMPLE_REPETITION:
        case WUBU_SAMPLE_TEMPERATURE:
        case WUBU_SAMPLE_TOP_K:
        case WUBU_SAMPLE_TOP_P:
        case WUBU_SAMPLE_MIN_P:
        case WUBU_SAMPLE_TYPICAL_P: break;
        default: return -1;
    }
    c->stages[c->n].stage = stage;
    c->stages[c->n].param = param;
    c->n++;
    return 0;
}

/* Descending logit order, ties by ascending index. */
static void ranked(const float *v, const unsigned char *mask, int n, int *idx) {
    int i, j;
    (void)mask;
    for (i = 0; i < n; i++) idx[i] = i;
    for (i = 1; i < n; i++) {
        int key = idx[i];
        j = i - 1;
        while (j >= 0 &&
               (v[idx[j]] < v[key] || (v[idx[j]] == v[key] && idx[j] > key))) {
            idx[j + 1] = idx[j];
            j--;
        }
        idx[j + 1] = key;
    }
}

static double max_logit(const float *v, const unsigned char *mask, int n) {
    double m = -HUGE_VAL;
    for (int i = 0; i < n; i++)
        if (mask[i] && (double)v[i] > m) m = v[i];
    return m;
}

int wubu_sampler_chain_apply(const wubu_sampler_chain_t *c,
                              float *logits, int n,
                              const int *prev_tokens, int n_prev,
                              unsigned char *mask_out) {
    if (!c || !logits || !mask_out || n <= 0 || (n_prev > 0 && !prev_tokens))
        return -1;
    for (int i = 0; i < n; i++) mask_out[i] = 1;
    int *order = (int *)malloc((size_t)n * sizeof(int));
    if (!order) return -1;

    for (int s = 0; s < c->n; s++) {
        chain_stage_t st = c->stages[s];
        if (st.stage == WUBU_SAMPLE_REPETITION) {
            for (int k = 0; k < n_prev; k++) {
                int t = prev_tokens[k];
                if (t >= 0 && t < n && logits[t] > 0.0f)
                    logits[t] = (float)(logits[t] / st.param);
            }
        } else if (st.stage == WUBU_SAMPLE_TEMPERATURE) {
            for (int i = 0; i < n; i++) logits[i] = (float)(logits[i] / st.param);
        } else if (st.stage == WUBU_SAMPLE_TOP_K) {
            ranked(logits, mask_out, n, order);
            int k = (int)st.param, counted = 0;
            if (k < 1) k = 1;
            for (int i = 0; i < n; i++) {
                int id = order[i];
                if (!mask_out[id]) continue;
                counted++;
                if (counted > k) mask_out[id] = 0;
            }
        } else if (st.stage == WUBU_SAMPLE_TOP_P) {
            ranked(logits, mask_out, n, order);
            double m = max_logit(logits, mask_out, n), sum = 0.0;
            for (int i = 0; i < n; i++)
                if (mask_out[order[i]]) sum += exp((double)logits[order[i]] - m);
            double acc = 0.0;
            for (int i = 0; i < n; i++) {
                if (!mask_out[order[i]]) continue;
                acc += exp((double)logits[order[i]] - m) / sum;
                if (acc > st.param) mask_out[order[i]] = 0;
            }
        } else if (st.stage == WUBU_SAMPLE_MIN_P) {
            double m = max_logit(logits, mask_out, n);
            double floor_logit = m + log(st.param);
            for (int i = 0; i < n; i++)
                if (mask_out[i] && (double)logits[i] < floor_logit) mask_out[i] = 0;
        } else if (st.stage == WUBU_SAMPLE_TYPICAL_P) {
            ranked(logits, mask_out, n, order);
            double m = max_logit(logits, mask_out, n), sum = 0.0;
            for (int i = 0; i < n; i++)
                if (mask_out[order[i]]) sum += exp((double)logits[order[i]] - m);
            double entropy = 0.0;
            for (int i = 0; i < n; i++) {
                if (!mask_out[order[i]]) continue;
                double p = exp((double)logits[order[i]] - m) / sum;
                entropy -= p * log(p);
            }
            for (int i = 0; i < n; i++) {
                if (!mask_out[i]) continue;
                double p = exp((double)logits[i] - m) / sum;
                if (-log(p) > entropy + st.param) mask_out[i] = 0;
            }
        }
    }

    /* Never leave the distribution empty: keep the highest allowed logit. */
    int any = 0, best = -1;
    for (int i = 0; i < n; i++)
        if (mask_out[i]) { any = 1; if (best < 0 || logits[i] > logits[best]) best = i; }
    if (!any && best >= 0) mask_out[best] = 1;
    int allowed = 0;
    for (int i = 0; i < n; i++) allowed += mask_out[i] ? 1 : 0;
    free(order);
    return allowed;
}

int wubu_sampler_chain_pick(const wubu_sampler_chain_t *c,
                            const float *logits, const unsigned char *mask,
                            int n, uint64_t seed) {
    (void)c;
    if (!logits || !mask || n <= 0) return -1;
    int best = -1;
    for (int i = 0; i < n; i++)
        if (mask[i] && (best < 0 || logits[i] > logits[best])) best = i;
    if (best < 0 || seed == 0) return best;
    double acc = 0.0, sum = 0.0, m = -HUGE_VAL;
    for (int i = 0; i < n; i++)
        if (mask[i] && (double)logits[i] > m) m = logits[i];
    for (int i = 0; i < n; i++)
        if (mask[i]) sum += exp((double)logits[i] - m);
    uint64_t x = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    double r = (double)(x >> 11) / 9007199254740992.0;
    for (int i = 0; i < n; i++) {
        if (!mask[i]) continue;
        acc += exp((double)logits[i] - m) / sum;
        if (r <= acc) return i;
    }
    return best;
}
