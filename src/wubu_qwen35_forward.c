/*
 * wubu_qwen35_forward.c — the QWEN3.5 HYBRID FORWARD (see header).
 * The gated-attn branch (unfused) + the GDN branch (fused qk+v + z).
 */
#include "wubu_qwen35_forward.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>

/* the rope: partial rotary (0.25) + theta 1e7 (the 0.8B config) */
static void apply_rope(float *qk, int n_heads, int head_dim, int seq,
                       float rope_theta, float partial)
{
    int rot = (int)(partial * (float)head_dim);
    for (int h = 0; h < n_heads; h++) {
        for (int s = 0; s < seq; s++) {
            for (int d = 0; d < rot / 2; d++) {
                float ang = (float)s / powf(rope_theta, 2.0f * d / (float)head_dim);
                float co = cosf(ang), si = sinf(ang);
                int idx = ((h * seq + s) * head_dim) + d;
                int idx2 = ((h * seq + s) * head_dim) + (rot / 2 + d);
                float a = qk[idx], b = qk[idx2];
                qk[idx] = a * co - b * si;
                qk[idx2] = a * si + b * co;
            }
        }
    }
}

/* the GQA softmax attention: q [q_heads*seq, hd], k/v [kv_heads*seq,
 * hd]; q heads group over the kv heads. out [q_heads*seq, hd]. */
static void gqa_softmax(const float *q, const float *k, const float *v,
                        int seq, int q_heads, int kv_heads, int hd,
                        float *out)
{
    float *scores = (float *)malloc((size_t)seq * sizeof(float));
    if (!scores) return;
    for (int qh = 0; qh < q_heads; qh++) {
        int kh = qh % kv_heads;
        for (int s = 0; s < seq; s++) {
            float mx = -1e30f;
            for (int sp = 0; sp <= s; sp++) {   /* causal */
                float dot = 0;
                for (int d = 0; d < hd; d++)
                    dot += q[((size_t)qh * seq + s) * hd + d] *
                           k[((size_t)kh * seq + sp) * hd + d];
                scores[sp] = dot / sqrtf((float)hd);
                if (scores[sp] > mx) mx = scores[sp];
            }
            float sum = 0;
            for (int sp = 0; sp <= s; sp++) {
                scores[sp] = expf(scores[sp] - mx);
                sum += scores[sp];
            }
            float *o = &out[((size_t)qh * seq + s) * hd];
            memset(o, 0, (size_t)hd * sizeof(float));
            for (int sp = 0; sp <= s; sp++) {
                float w = scores[sp] / sum;
                for (int d = 0; d < hd; d++)
                    o[d] += w * v[((size_t)kh * seq + sp) * hd + d];
            }
        }
    }
    free(scores);
}

int wubu_q35_gated_attn_forward(const wubu_q35_cfg_t *cfg,
                                const wubu_q35_split_t *split,
                                const wubu_q35_weights_t *w,
                                const float *x, int seq, float *out_a)
{
    if (!cfg || !split || !w || !x || !out_a || seq <= 0) return -1;
    int d = (int)cfg->hidden_size;
    int q_heads = (int)cfg->num_attention_heads;
    int kv_heads = (int)cfg->num_key_value_heads;
    int hd = (int)cfg->head_dim;
    int d_out = q_heads * hd;

    /* the UNFUSED projections: attn_q (q_and_gate), attn_k, attn_v */
    float *qg = (float *)malloc((size_t)seq * (size_t)(2 * d_out) * sizeof(float));
    float *k = (float *)malloc((size_t)kv_heads * (size_t)seq * (size_t)hd * sizeof(float));
    float *v = (float *)malloc((size_t)kv_heads * (size_t)seq * (size_t)hd * sizeof(float));
    float *q = (float *)malloc((size_t)q_heads * (size_t)seq * (size_t)hd * sizeof(float));
    if (!qg || !k || !v || !q) { free(qg); free(k); free(v); free(q); return -1; }
    for (int s = 0; s < seq; s++) {
        for (int o = 0; o < 2 * d_out; o++) {
            float acc = 0;
            for (int i = 0; i < d; i++)
                acc += x[(size_t)s * d + i] * w->attn_q[(size_t)i * (2 * d_out) + o];
            qg[(size_t)s * (2 * d_out) + o] = acc;
        }
        for (int kh = 0; kh < kv_heads; kh++) {
            for (int dd = 0; dd < hd; dd++) {
                float ak = 0, av = 0;
                for (int i = 0; i < d; i++) {
                    ak += x[(size_t)s * d + i] *
                          w->attn_k[(size_t)i * (kv_heads * hd) + kh * hd + dd];
                    av += x[(size_t)s * d + i] *
                          w->attn_v[(size_t)i * (kv_heads * hd) + kh * hd + dd];
                }
                k[((size_t)kh * seq + s) * hd + dd] = ak;
                v[((size_t)kh * seq + s) * hd + dd] = av;
            }
        }
    }

    /* the QK RMS norms (the SHARED per-head norm vectors) + the rope */
    for (int qh = 0; qh < q_heads; qh++) {
        for (int s = 0; s < seq; s++) {
            float *dst = &q[((size_t)qh * seq + s) * hd];
            float ss = 0;
            for (int dd = 0; dd < hd; dd++) {
                dst[dd] = qg[((size_t)s * (2 * d_out)) + (size_t)qh * hd + dd];
                ss += dst[dd] * dst[dd];
            }
            ss = 1.0f / sqrtf(ss / (float)hd + 1e-6f);
            for (int dd = 0; dd < hd; dd++)
                dst[dd] *= ss * w->q_norm[dd];   /* shared [hd] */
        }
    }
    for (int kh = 0; kh < kv_heads; kh++) {
        for (int s = 0; s < seq; s++) {
            float *dst = &k[((size_t)kh * seq + s) * hd];
            float ss = 0;
            for (int dd = 0; dd < hd; dd++) ss += dst[dd] * dst[dd];
            ss = 1.0f / sqrtf(ss / (float)hd + 1e-6f);
            for (int dd = 0; dd < hd; dd++)
                dst[dd] *= ss * w->k_norm[dd];
        }
    }
    apply_rope(q, q_heads, hd, seq, 1e7f, 0.25f);
    apply_rope(k, kv_heads, hd, seq, 1e7f, 0.25f);

    /* the GQA softmax -> the per-head output */
    float *attn = (float *)malloc((size_t)q_heads * (size_t)seq * (size_t)hd * sizeof(float));
    if (!attn) { free(qg); free(k); free(v); free(q); return -1; }
    gqa_softmax(q, k, v, seq, q_heads, kv_heads, hd, attn);

    /* the gated-attn: attn * sigmoid(gate) — the gate is the second
     * half of the attn_q chunk */
    for (int qh = 0; qh < q_heads; qh++) {
        for (int s = 0; s < seq; s++) {
            for (int dd = 0; dd < hd; dd++) {
                float g = qg[((size_t)s * (2 * d_out)) + (size_t)qh * hd + d_out + dd];
                attn[((size_t)qh * seq + s) * hd + dd] *= 1.0f / (1.0f + expf(-g));
            }
        }
    }

    /* W_o [d_out, d] -> out_a [seq, d] */
    memset(out_a, 0, (size_t)seq * (size_t)d * sizeof(float));
    for (int s = 0; s < seq; s++) {
        for (int o = 0; o < d; o++) {
            float acc = 0;
            for (int qh = 0; qh < q_heads; qh++)
                for (int dd = 0; dd < hd; dd++)
                    acc += attn[((size_t)qh * seq + s) * hd + dd] *
                           w->attn_output[((size_t)qh * hd + dd) * d + o];
            out_a[(size_t)s * d + o] = acc;
        }
    }

    free(qg); free(k); free(v); free(q); free(attn);
    return 0;
}

int wubu_q35_layer_forward(const wubu_q35_cfg_t *cfg,
                           const wubu_q35_split_t *split,
                           const wubu_q35_weights_t *w,
                           wubu_q35_layer_kind_t kind,
                           const float *x, int seq, float *out,
                           float *gdn_state)
{
    if (!cfg || !split || !w || !x || !out || seq <= 0) return -1;
    int d = (int)cfg->hidden_size;
    float *xn = (float *)malloc((size_t)seq * (size_t)d * sizeof(float));
    if (!xn) return -1;
    for (int s = 0; s < seq; s++) {
        float ss = 0;
        for (int i = 0; i < d; i++) ss += x[(size_t)s * d + i] * x[(size_t)s * d + i];
        ss = 1.0f / sqrtf(ss / (float)d + 1e-6f);
        for (int i = 0; i < d; i++)
            xn[(size_t)s * d + i] = x[(size_t)s * d + i] * ss * w->attn_norm[i];
    }
    if (kind == WUBU_Q35_LAYER_GATED_ATTN) {
        float *out_a = (float *)malloc((size_t)seq * (size_t)d * sizeof(float));
        if (!out_a) { free(xn); return -1; }
        wubu_q35_gated_attn_forward(cfg, split, w, xn, seq, out_a);
        for (int s = 0; s < seq; s++) {
            float ss = 0;
            for (int i = 0; i < d; i++) ss += out_a[(size_t)s * d + i] * out_a[(size_t)s * d + i];
            ss = 1.0f / sqrtf(ss / (float)d + 1e-6f);
            for (int i = 0; i < d; i++)
                out[(size_t)s * d + i] = x[(size_t)s * d + i] +
                    out_a[(size_t)s * d + i] * ss * w->post_norm[i];
        }
        free(out_a);
    } else {
        /* the GDN branch: the fused qk+v + the z gate; the full delta
         * rule + conv state is the ssm module's (here the stateless
         * projection path — the shapes + the z gate + the ssm_out) */
        int qk_total = (int)split->gdn_qkv_total;
        float *qkv = (float *)malloc((size_t)seq * (size_t)qk_total * sizeof(float));
        float *z = (float *)malloc((size_t)seq * (size_t)split->gdn_z_len * sizeof(float));
        if (!qkv || !z) { free(qkv); free(z); free(xn); return -1; }
        for (int s = 0; s < seq; s++) {
            for (int o = 0; o < qk_total; o++) {
                float acc = 0;
                for (int i = 0; i < d; i++)
                    acc += xn[(size_t)s * d + i] * w->ssm_qkv[(size_t)i * qk_total + o];
                qkv[(size_t)s * qk_total + o] = acc;
            }
            for (int o = 0; o < (int)split->gdn_z_len; o++) {
                float acc = 0;
                for (int i = 0; i < d; i++)
                    acc += xn[(size_t)s * d + i] * w->ssm_gate[(size_t)i * split->gdn_z_len + o];
                z[(size_t)s * split->gdn_z_len + o] = acc;
            }
        }
        /* the stateless placeholder: the z-gated average + the ssm_out
         * projection (the shape proof; the recurrence is the ssm path) */
        for (int s = 0; s < seq; s++) {
            float zsum = 0;
            for (int o = 0; o < (int)split->gdn_z_len; o++)
                zsum += z[(size_t)s * split->gdn_z_len + o];
            float zact = 1.0f / (1.0f + expf(-zsum / (float)split->gdn_z_len));
            for (int i = 0; i < d; i++) {
                float acc = 0;
                for (int o = 0; o < (int)split->gdn_out_rows; o++)
                    acc += qkv[(size_t)s * qk_total + (qk_total - (int)split->gdn_v_len) + (o % (int)split->gdn_v_len)] *
                           w->ssm_out[(size_t)o * d + i];
                out[(size_t)s * d + i] = x[(size_t)s * d + i] + zact * acc;
            }
        }
        free(qkv); free(z);
        (void)gdn_state;
    }
    free(xn);
    return 0;
}
