/*
 * wubu_sq_agreement.c -- EG-101 SQ acceptance gate. Pure C11.
 */
#include "wubu_sq_agreement.h"

#include <math.h>
#include <string.h>

int wubu_sq_evaluate(const float *ref, const float *cand,
                     int n_rows, int vocab,
                     float min_agreement, float max_kl,
                     wubu_sq_report_t *out) {
    int agrees = 0, exact = 1;
    double kl_sum = 0.0;
    if (!ref || !cand || !out || n_rows <= 0 || vocab <= 0 ||
        !(min_agreement >= 0.0f && min_agreement <= 1.0f) ||
        max_kl < 0.0f) return -1;

    for (int r = 0; r < n_rows; r++) {
        const float *a = ref + (size_t)r * vocab;
        const float *b = cand + (size_t)r * vocab;
        float sa = 0.0f, sb = 0.0f, mx = -INFINITY;
        int ia = 0, ib = 0;
        for (int v = 0; v < vocab; v++) {
            if (a[v] > mx) mx = a[v];
            if (memcmp(&a[v], &b[v], sizeof(float)) != 0) exact = 0;
        }
        /* Normalize both rows stably. */
        {
            float mb = -INFINITY;
            for (int v = 0; v < vocab; v++) if (b[v] > mb) mb = b[v];
            sb = 0.0f;
            for (int v = 0; v < vocab; v++) sb += expf(b[v] - mb);
            sa = 0.0f;
            for (int v = 0; v < vocab; v++) sa += expf(a[v] - mx);
            for (int v = 0; v < vocab; v++) {
                float pa = expf(a[v] - mx) / sa;
                float pb = expf(b[v] - mb) / sb;
                if (pa > 0.0f) kl_sum += (double)(pa * logf(pa / (pb + 1e-12f)));
            }
            /* argmax on raw rows */
            ia = 0; ib = 0;
            for (int v = 1; v < vocab; v++) {
                if (a[v] > a[ia]) ia = v;
                if (b[v] > b[ib]) ib = v;
            }
            if (ia == ib) agrees++;
        }
    }
    out->top1_agreement = (float)agrees / (float)n_rows;
    out->kl = (float)(kl_sum / (double)n_rows);
    out->bit_exact = exact;
    out->accepted = out->bit_exact ||
        (out->top1_agreement >= min_agreement && out->kl <= max_kl);
    return 0;
}
