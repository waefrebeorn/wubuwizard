/*
 * wubu_der.c -- Dark Experience Replay (BB07). C11.
 */
#include "wubu_der.h"
#include <string.h>
#include <math.h>
#include "wubu_activations.h"

int wubu_der_push(wubu_der_buffer_t *b, const float *teacher_logits, int ndim)
{
    if (!b || !teacher_logits || ndim <= 0 || ndim > WUBU_DER_DIMS) return -1;
    memcpy(b->logits[b->head], teacher_logits, (size_t)ndim * sizeof(float));
    if (!b->used[b->head]) b->count++;
    b->used[b->head] = 1;
    b->head = (b->head + 1) % WUBU_DER_BUFSZ;
    return 0;
}

/* softmax: use wubu_activations.h */

float wubu_der_loss(const wubu_der_buffer_t *b, const float *student_logits,
                    int ndim, float temperature)
{
    if (!b || !student_logits || ndim <= 0 || ndim > WUBU_DER_DIMS) return 0;
    if (b->count == 0) return 0;
    float t[WUBU_DER_DIMS], s[WUBU_DER_DIMS];
    double loss = 0;
    int n = 0;
    for (int i = 0; i < WUBU_DER_BUFSZ; i++) {
        if (!b->used[i]) continue;
        softmax_t(b->logits[i], ndim, temperature, t);
        softmax_t(student_logits, ndim, temperature, s);
        for (int d = 0; d < ndim; d++) {
            if (t[d] > 1e-12f) {
                float lt = logf(t[d]);
                float ls = logf(s[d] > 1e-12f ? s[d] : 1e-12f);
                loss -= t[d] * ls;      /* cross-entropy */
                loss += t[d] * lt;      /* minus the teacher entropy -> KL */
            }
        }
        n++;
    }
    return (float)(loss / (n > 0 ? n : 1));
}

float wubu_der_total(float ce, float der, float alpha)
{
    if (alpha < 0) alpha = 0;
    return ce + alpha * der;
}
