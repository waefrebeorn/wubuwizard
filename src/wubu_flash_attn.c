#include "wubu_flash_attn.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

static float softmax_scale(int head_dim)
{
    return 1.0f / sqrtf((float)head_dim);
}

void wubu_flash_attn_forward(const float *q, const float *k, const float *v,
                              float *out, int nq, int nkv, int heads,
                              int dim, int causal)
{
    const float scale = softmax_scale(dim);
    const int stride_q = heads * dim;
    const int stride_k = stride_q;
    const int stride_v = stride_q;
    const int stride_o = stride_q;

    memset(out, 0, (size_t)nq * heads * dim * sizeof(float));

    for (int h = 0; h < heads; h++) {
        for (int i = 0; i < nq; i++) {
            float *oi = out + i * stride_o + h * dim;
            const float *qi = q + i * stride_q + h * dim;
            int attend_hi = causal ? i + 1 : nkv;

            float running_max = -1e30f;
            float running_sum = 0.0f;
            float *running_o = (float *)calloc(dim, sizeof(float));
            if (!running_o) continue;

            for (int kv_start = 0; kv_start < nkv; kv_start += 256) {
                int block_hi = kv_start + 256;
                if (block_hi > nkv) block_hi = nkv;
                int lo = kv_start;
                int hi = block_hi;
                if (lo >= attend_hi) break;
                if (hi > attend_hi) hi = attend_hi;
                if (lo >= hi) continue;

                float local_max = -1e30f;
                for (int j = lo; j < hi; j++) {
                    const float *kj = k + j * stride_k + h * dim;
                    float d = 0.0f;
                    for (int d2 = 0; d2 < dim; d2++) d += qi[d2] * kj[d2];
                    d *= scale;
                    if (d > local_max) local_max = d;
                }

                float new_max = (local_max > running_max) ? local_max : running_max;
                int first_block = (running_sum < 1e-30f);

                if (!first_block) {
                    float corr = expf(running_max - new_max);
                    for (int d2 = 0; d2 < dim; d2++) running_o[d2] *= corr;
                    running_sum *= corr;
                }
                running_max = new_max;

                float row_sum = 0.0f;
                for (int j = lo; j < hi; j++) {
                    const float *kj = k + j * stride_k + h * dim;
                    const float *vj = v + j * stride_v + h * dim;
                    float d = 0.0f;
                    for (int d2 = 0; d2 < dim; d2++) d += qi[d2] * kj[d2];
                    d = expf(d * scale - running_max);
                    row_sum += d;
                    for (int d2 = 0; d2 < dim; d2++) running_o[d2] += d * vj[d2];
                }
                running_sum += row_sum;
            }

            float inv = 1.0f / running_sum;
            for (int d2 = 0; d2 < dim; d2++) oi[d2] = running_o[d2] * inv;
            free(running_o);
        }
    }
}

void wubu_flash_attn_forward_f16(const float *q, const float *k, const float *v,
                                  uint16_t *out, int nq, int nkv, int heads,
                                  int dim, int causal)
{
    int total = nq * heads * dim;
    float *tmp = (float *)malloc(total * sizeof(float));
    if (!tmp) return;

    wubu_flash_attn_forward(q, k, v, tmp, nq, nkv, heads, dim, causal);

    for (int i = 0; i < total; i++) {
        float f = tmp[i];
        uint32_t u;
        memcpy(&u, &f, sizeof(u));
        int sign = (u >> 31) & 1;
        int exp = ((u >> 23) & 0xFF) - 127;
        int mant = u & 0x7FFFFF;
        if (exp > 15) exp = 15;
        else if (exp < -10) exp = 0;
        out[i] = (uint16_t)((sign << 15) | (exp << 10) | (mant >> 13));
    }

    free(tmp);
}
