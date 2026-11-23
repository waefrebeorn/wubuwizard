/*
 * wubu_invalid_kv.c -- EG-40 invalidated-KV draft verification. Pure C11.
 */
#include "wubu_invalid_kv.h"

int wubu_invalid_kv_verify(const int *draft, const int *fresh, int n,
                           float threshold, int *accepted,
                           float *accept_rate, int *preempt) {
    int common = 0;
    float rate;
    if (!draft || !fresh || n <= 0 || !accepted || !accept_rate ||
        !(threshold >= 0.0f && threshold <= 1.0f)) return -1;

    while (common < n && draft[common] == fresh[common]) common++;
    rate = (float)common / (float)n;
    *accept_rate = rate;
    if (preempt) *preempt = rate < threshold ? 1 : 0;
    *accepted = rate < threshold ? 0 : common;
    return 0;
}
