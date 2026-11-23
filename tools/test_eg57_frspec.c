/* test_eg57_frspec.c -- EG-57 gate: draft head memory cut vs mass coverage. */
#include <stdio.h>

#include "wubu_frspec.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define V 64
#define K 8

int main(void) {
    unsigned long long freq[V];
    float target[V];
    int hot[K], cold[K], i;
    double cover, baseline, cut;

    for (i = 0; i < V; i++) freq[i] = (unsigned long long)(10000 / (i + 1));
    {
        unsigned long long sum = 0;
        for (i = 0; i < V; i++) sum += freq[i];
        for (i = 0; i < V; i++) target[i] = (float)freq[i] / (float)sum;
    }
    CHECK(wubu_frspec_select(freq, V, K, hot) == 0);
    for (i = 0; i < K; i++) CHECK(hot[i] == i);
    for (i = 0; i < K; i++) cold[i] = V - K + i;

    cover = wubu_frspec_coverage(target, V, hot, K);
    baseline = wubu_frspec_coverage(target, V, cold, K);
    cut = 1.0 - (double)K / (double)V;
    printf("EG-57 head memory cut=%.4f coverage=%.5f tail_baseline=%.5f\n",
           cut, cover, baseline);
    CHECK(cut > 0.8);
    CHECK(cover > 0.35);
    CHECK(cover > baseline + 0.3);

    CHECK(wubu_frspec_select(freq, V, V + 1, hot) == -1);
    CHECK(wubu_frspec_coverage(target, V, hot, 0) == -1.0);
    {
        int bad[K];
        for (i = 0; i < K; i++) bad[i] = i == 0 ? V : 0;
        CHECK(wubu_frspec_coverage(target, V, bad, K) == -1.0);
    }

    if (failures) {
        fprintf(stderr, "test_eg57_frspec: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg57_frspec: PASS\n");
    return 0;
}
