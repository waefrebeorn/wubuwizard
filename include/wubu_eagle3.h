#ifndef WUBU_EAGLE3_H
#define WUBU_EAGLE3_H

/*
 * wubu_eagle3.h -- EG-24: EAGLE-3 direct token prediction from tri-layer
 * hidden fusion (arXiv 2503.01840), reimplemented in C11.
 *
 * The drafter fuses three target-layer hidden states, predicts the next token
 * directly with a shared embedding, and reports a softmax-gap confidence that
 * a speculative tree can use as its acceptance probability.
 */

#ifdef __cplusplus
extern "C" {
#endif

/* Fuse h[0..2] (each dim floats) into out. Returns 0 or -1. */
int wubu_eagle3_fuse(const float *h0, const float *h1, const float *h2,
                     int dim, float *out);

/* Score every vocab token: logits[v] = dot(fused, emb[v*dim..]). */
int wubu_eagle3_predict(const float *fused, const float *emb,
                        int vocab, int dim, float *logits);

/* Highest-scoring token id, or -1 on bad input. */
int wubu_eagle3_top1(const float *logits, int vocab);

/* softmax(top1) - softmax(top2), in [0,1]. */
float wubu_eagle3_confidence(const float *logits, int vocab);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_EAGLE3_H */
