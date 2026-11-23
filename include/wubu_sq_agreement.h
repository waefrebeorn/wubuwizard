#ifndef WUBU_SQ_AGREEMENT_H
#define WUBU_SQ_AGREEMENT_H

/*
 * wubu_sq_agreement.h -- EG-101: acceptance contract for a quantized build
 * (Strata SQ-10/15/20 lineage). Compares reference and candidate logit rows by
 * top-1 agreement and KL divergence, with an exact-bit shortcut. Pure C11.
 */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float top1_agreement;   /* fraction of rows with the same argmax */
    float kl;               /* mean KL(reference || candidate), nats */
    int bit_exact;          /* all floats bitwise identical */
    int accepted;
} wubu_sq_report_t;

/* ref/cand are [n_rows, vocab] rows of normalized probabilities or logits;
 * KL is computed after each row is normalized. Returns 0 or -1. */
int wubu_sq_evaluate(const float *ref, const float *cand,
                     int n_rows, int vocab,
                     float min_agreement, float max_kl,
                     wubu_sq_report_t *out);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_SQ_AGREEMENT_H */
