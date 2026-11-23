/*
 * wubu_blockverify.c -- EG-61 block verification. Pure C11.
 */
#include "wubu_blockverify.h"

int wubu_block_verify(const float *p, const float *q, int vocab,
                      const int *draft, int n, const float *uniforms,
                      int *out, int *accepted) {
    int acc = 0, emitted = 0;
    if (!p || !q || !draft || !uniforms || !out || !accepted ||
        vocab <= 0 || vocab > 64 || n <= 0) return -1;
    for (int i = 0; i < n; i++) {
        int t = draft[i];
        float a;
        if (t < 0 || t >= vocab) return -1;
        a = q[t] > 0.0f && p[t] / q[t] < 1.0f ? p[t] / q[t] : 1.0f;
        if (uniforms[2 * i] < a) {
            out[emitted++] = t;
            acc++;
            continue;
        }
        /* Distribution-exact correction: pointwise (p - q)+ residual. */
        {
            float r[64], z = 0.0f, cdf = 0.0f;
            for (int v = 0; v < vocab; v++) {
                r[v] = p[v] > q[v] ? p[v] - q[v] : 0.0f;
                z += r[v];
            }
            if (z <= 0.0f) return -1;
            for (int v = 0; v < vocab; v++) {
                cdf += r[v] / z;
                if (uniforms[2 * i + 1] < cdf) {
                    out[emitted++] = v;
                    *accepted = acc;
                    return emitted;
                }
            }
            out[emitted++] = vocab - 1;
            *accepted = acc;
            return emitted;
        }
    }
    *accepted = acc;
    return emitted;
}
