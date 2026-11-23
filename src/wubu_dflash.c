/*
 * wubu_dflash.c -- EG-56 block-diffusion draft gate. Pure C11.
 */
#include "wubu_dflash.h"

#include <math.h>

int wubu_dflash_block(const float *h_target, const float *h_draft,
                      int n_tokens, int dim, int layers, float inject,
                      const float *uniforms, float *probs) {
    int accepted = 0;
    if (!h_target || !h_draft || !uniforms || !probs || n_tokens <= 0 ||
        dim <= 0 || layers <= 0 || !(inject >= 0.0f)) return -1;
    for (int t = 0; t < n_tokens; t++) {
        const float *ht = h_target + (size_t)t * dim;
        const float *hd = h_draft + (size_t)t * dim;
        double diff2 = 0.0, tn = 0.0;
        float p;
        for (int d = 0; d < dim; d++) {
            float fused = hd[d] + inject * ht[d] * (float)layers;
            double e = (double)(fused - ht[d]);
            diff2 += e * e;
            tn += (double)ht[d] * ht[d];
        }
        /* Acceptance falls with the normalized hidden-state mismatch. */
        p = (float)exp(-sqrt(diff2) / (sqrt(tn) + 1e-6f));
        if (p > 1.0f) p = 1.0f;
        probs[t] = p;
        if (uniforms[t] < p) accepted++;
    }
    return accepted;
}
