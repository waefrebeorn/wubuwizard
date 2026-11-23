#ifndef WUBU_SPECTR_SELECT_H
#define WUBU_SPECTR_SELECT_H

/*
 * wubu_spectr_select.h -- EG-25e: SpecTr candidate mass selection primitive.
 * A k-token draft set is chosen to maximise target probability mass covered,
 * which minimises the untransported mass that verification must reject. The
 * gate brute-forces small distributions to prove the choice optimal. This is
 * a 1-D transport-cost policy gate, not a learned drafter or a speed claim.
 */

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_SPECTR_MAX_V 256

int wubu_spectr_select(const double *p_target, int n, int k, int *out);

/* Untransported target mass: 1 - sum of selected probabilities. */
double wubu_spectr_transport_cost(const double *p_target,
                                  const int *selected, int k);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_SPECTR_SELECT_H */
