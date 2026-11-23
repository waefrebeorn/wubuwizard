#ifndef WUBU_LOGITS_PROC_H
#define WUBU_LOGITS_PROC_H

/*
 * wubu_logits_proc.h -- EG-34: composable logits processors.
 * typical-p masks tokens outside the cumulative typical-probability mass;
 * a Sotoudeh-Gerard style green-list watermark adds a keyed bias to a token
 * subset before sampling. Both are pure logits transforms, no model change.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_LOGITS_MAX 256

typedef struct wubu_logits_proc wubu_logits_proc_t;

/* typical_p in (0,1], delta >= 0. NULL on bad input. */
wubu_logits_proc_t *wubu_logits_proc_create(float typical_p, float delta,
                                            uint64_t seed);
void wubu_logits_proc_free(wubu_logits_proc_t *p);

/* Apply typical-p masking then the keyed watermark bias in place; returns the
 * number of tokens kept by typical-p. -1 on bad input. */
int wubu_logits_proc_apply(wubu_logits_proc_t *p, float *logits, int n, int pos);

/* Green-list membership for (seed, position, token id). -1 on bad input. */
int wubu_logits_proc_is_green(const wubu_logits_proc_t *p, int pos, int token);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_LOGITS_PROC_H */
