/*
 * wubu_frspec.c -- EG-57 frequency-ranked draft head. Pure C11.
 */
#include "wubu_frspec.h"

int wubu_frspec_select(const unsigned long long *freq, int vocab, int k,
                       int *out) {
    if (!freq || !out || vocab <= 0 || k <= 0 || k > vocab) return -1;
    for (int i = 0; i < k; i++) out[i] = i;
    for (int j = k; j < vocab; j++) {
        int worst = 0;
        for (int i = 1; i < k; i++)
            if (freq[out[i]] < freq[out[worst]] ||
                (freq[out[i]] == freq[out[worst]] && out[i] > out[worst]))
                worst = i;
        if (freq[j] > freq[out[worst]] ||
            (freq[j] == freq[out[worst]] && j < out[worst]))
            out[worst] = j;
    }
    for (int i = 0; i < k; i++)
        for (int j = i + 1; j < k; j++)
            if (freq[out[j]] > freq[out[i]] ||
                (freq[out[j]] == freq[out[i]] && out[j] < out[i])) {
                int t = out[i]; out[i] = out[j]; out[j] = t;
            }
    return 0;
}

double wubu_frspec_coverage(const float *target, int vocab,
                            const int *subset, int k) {
    double sum = 0.0;
    if (!target || !subset || vocab <= 0 || k <= 0 || k > vocab) return -1.0;
    for (int i = 0; i < k; i++) {
        int t = subset[i];
        if (t < 0 || t >= vocab) return -1.0;
        sum += target[t];
    }
    return sum;
}
