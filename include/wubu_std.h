/*
 * wubu_std.h -- the self-contained C11 compatibility surface.
 *
 * The user's directive (2026-08-04): WuBu compliance means WE define
 * the feature surface, not glibc's _GNU_SOURCE macro. The endgame is
 * our own compiler (wubuos HolyC) where WE define everything. Until
 * then, this header provides the tiny set of helpers WITHOUT needing
 * _GNU_SOURCE:
 *
 *   - M_PI / M_PI_2 (a constant — just define it, no feature macro)
 *   - wubu_strdup (4 lines, no GNU dependency)
 *   - wubu_fmaxf etc. if ever needed
 *
 * Include THIS instead of reaching for _GNU_SOURCE.
 */
#ifndef WUBU_STD_H
#define WUBU_STD_H

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif
#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923132169163975144
#endif
#ifndef M_E
#define M_E 2.71828182845904523536028747135266250
#endif

/* strdup without _GNU_SOURCE (POSIX strdup is behind feature macros;
 * this is 4 lines and dependency-free). */
static inline char *wubu_strdup(const char *s)
{
    if (!s) return NULL;
    size_t n = strlen(s) + 1;
    char *c = (char *)malloc(n);
    if (c) memcpy(c, s, n);
    return c;
}

#endif /* WUBU_STD_H */
