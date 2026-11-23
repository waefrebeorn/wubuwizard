/* wubu_relaxed_verify.c -- EG-25d relaxed verification gate. Strict C11. */
#include "wubu_relaxed_verify.h"

int wubu_relaxed_accept(double p_draft, double p_target, double delta)
{
    if (!(p_draft > 0.0) || !(p_target >= 0.0) || !(delta >= 0.0) ||
        delta > 1.0) return -1;
    return p_target >= p_draft * (1.0 - delta) ? 1 : 0;
}

int wubu_relaxed_verify_chain(const double *p_draft, const double *p_target,
                              int n, double delta, int *flags)
{
    int rejected = 0;
    if (!p_draft || !p_target || !flags || n < 1 || delta < 0.0 ||
        delta > 1.0) return -1;
    for (int i = 0; i < n; ++i) {
        if (!rejected && wubu_relaxed_accept(p_draft[i], p_target[i], delta) == 0)
            rejected = 1;
        flags[i] = rejected ? 0 : 1;
    }
    return 0;
}

int wubu_relaxed_accepted(const int *flags, int n)
{
    int count = 0;
    if (!flags || n < 0) return -1;
    for (int i = 0; i < n; ++i) count += flags[i] ? 1 : 0;
    return count;
}

int wubu_relaxed_extra(const int *exact, const int *relaxed, int n)
{
    int extra = 0;
    if (!exact || !relaxed || n < 0) return -1;
    for (int i = 0; i < n; ++i)
        if (relaxed[i] && !exact[i]) ++extra;
    return extra;
}
