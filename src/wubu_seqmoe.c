/*
 * wubu_seqmoe.c -- EG-52 SeqMoE residency gate. Pure C11.
 */
#include "wubu_seqmoe.h"

#include <stdlib.h>
#include <string.h>

int wubu_seqmoe_init(wubu_seqmoe_t *m, int experts, int capacity) {
    size_t n;
    if (!m || experts <= 1 || experts > 64 || capacity <= 0 ||
        capacity >= experts) return -1;
    n = (size_t)experts * (size_t)experts * (size_t)experts;
    m->trans = (unsigned long long *)calloc(n, sizeof(unsigned long long));
    m->freq = (unsigned long long *)calloc((size_t)experts,
                                           sizeof(unsigned long long));
    if (!m->trans || !m->freq) {
        wubu_seqmoe_free(m);
        return -1;
    }
    m->experts = experts;
    m->capacity = capacity;
    return 0;
}

int wubu_seqmoe_free(wubu_seqmoe_t *m) {
    if (!m) return -1;
    free(m->trans);
    free(m->freq);
    m->trans = NULL;
    m->freq = NULL;
    m->experts = 0;
    m->capacity = 0;
    return 0;
}

int wubu_seqmoe_observe(wubu_seqmoe_t *m, const int *trace, int n) {
    if (!m || !m->trans || !trace || n < WUBU_SEQMOE_ORDER + 1) return -1;
    for (int i = 0; i < n; i++)
        if (trace[i] < 0 || trace[i] >= m->experts) return -1;
    for (int i = WUBU_SEQMOE_ORDER; i < n; i++) {
        int a = trace[i - 2], b = trace[i - 1], c = trace[i];
        size_t k = ((size_t)a * m->experts + (size_t)b) * m->experts + (size_t)c;
        m->trans[k]++;
        m->freq[c]++;
    }
    return 0;
}

int wubu_seqmoe_plan(const wubu_seqmoe_t *m, int prev2, int prev1, int *out) {
    unsigned long long weight[64], selw[64];
    int pick[64], npick = 0;
    if (!m || !m->trans || !out || prev2 < 0 || prev2 >= m->experts ||
        prev1 < 0 || prev1 >= m->experts) return -1;
    for (int e = 0; e < m->experts; e++) {
        size_t k = ((size_t)prev2 * m->experts + (size_t)prev1) * m->experts + (size_t)e;
        weight[e] = m->trans[k];
    }
    for (int e = 0; e < m->experts; e++) {
        int j;
        if (weight[e] == 0) continue;
        if (npick == m->capacity && weight[e] <= selw[pick[npick - 1]]) continue;
        if (npick < m->capacity) npick++;
        j = npick - 1;
        while (j > 0 && (weight[e] > selw[pick[j - 1]] ||
               (weight[e] == selw[pick[j - 1]] && e < pick[j - 1]))) {
            selw[pick[j]] = selw[pick[j - 1]];
            pick[j] = pick[j - 1];
            j--;
        }
        selw[e] = weight[e];
        pick[j] = e;
    }
    /* Deadline sequencing: fill empty slots with globally frequent experts. */
    for (int e = 0; e < m->experts && npick < m->capacity; e++) {
        int seen = 0;
        for (int k = 0; k < npick; k++) if (pick[k] == e) seen = 1;
        if (seen || m->freq[e] == 0) continue;
        pick[npick++] = e;
    }
    for (int i = 0; i < npick; i++) out[i] = pick[i];
    return npick;
}

int wubu_seqmoe_hits(const wubu_seqmoe_t *m, const int *trace, int n,
                     int prev2, int prev1) {
    int next[64], hits = 0, nr = 0, p2 = prev2, p1 = prev1;
    if (!m || !m->trans || !trace || n < 1 || prev2 < 0 || prev2 >= m->experts ||
        prev1 < 0 || prev1 >= m->experts) return -1;
    nr = wubu_seqmoe_plan(m, p2, p1, next);
    if (nr <= 0) return -1;
    for (int i = 0; i < n; i++) {
        int hit = 0;
        for (int k = 0; k < nr; k++) if (next[k] == trace[i]) hit = 1;
        if (hit) hits++;
        p2 = p1;
        p1 = trace[i];
        if (i + 1 < n) {
            nr = wubu_seqmoe_plan(m, p2, p1, next);
            if (nr <= 0) return -1;
        }
    }
    return hits;
}

int wubu_lru_hits(int experts, int capacity, const int *trace, int n) {
    int slots[64], age[64], hits = 0, used = 0;
    if (experts <= 0 || capacity <= 0 || capacity > 64 ||
        !trace || n <= 0) return -1;
    for (int i = 0; i < 64; i++) age[i] = -1;
    for (int i = 0; i < n; i++) {
        int e = trace[i], found = -1;
        for (int k = 0; k < used; k++) if (slots[k] == e) { found = k; break; }
        if (found >= 0) {
            hits++;
            age[found] = i;
        } else if (used < capacity) {
            slots[used] = e;
            age[used++] = i;
        } else {
            int victim = 0;
            for (int k = 1; k < used; k++)
                if (age[k] < age[victim]) victim = k;
            slots[victim] = e;
            age[victim] = i;
        }
    }
    return hits;
}

int wubu_static_hits(int experts, int capacity, const int *trace, int n) {
    unsigned long long freq[64];
    int top[64], hits = 0;
    if (experts <= 0 || capacity <= 0 || capacity > experts ||
        !trace || n <= 0) return -1;
    for (int i = 0; i < experts; i++) freq[i] = 0;
    for (int i = 0; i < n; i++) {
        if (trace[i] < 0 || trace[i] >= experts) return -1;
        freq[trace[i]]++;
    }
    for (int i = 0; i < experts; i++) top[i] = i;
    for (int i = 0; i < experts; i++)
        for (int j = i + 1; j < experts; j++)
            if (freq[top[j]] > freq[top[i]]) {
                int t = top[i]; top[i] = top[j]; top[j] = t;
            }
    for (int i = 0; i < n; i++)
        for (int k = 0; k < capacity; k++)
            if (trace[i] == top[k]) { hits++; break; }
    return hits;
}
