#ifndef WUBU_TOKEN_RECYCLE_H
#define WUBU_TOKEN_RECYCLE_H

/*
 * wubu_token_recycle.h -- EG-25b: token recycling from verified chains.
 * After a verify pass, the accepted token chain is kept keyed by its anchor
 * (first token). On the next decode step a matching anchor recycles the
 * following tokens as a draft chain. No trained weights or timing claims.
 */

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_RECYCLE_MAX_SLOTS 64
#define WUBU_RECYCLE_MAX_LEN   16

typedef struct wubu_token_recycle wubu_token_recycle_t;

wubu_token_recycle_t *wubu_token_recycle_create(int slots, int max_len);
void wubu_token_recycle_free(wubu_token_recycle_t *p);

/* Record [anchor, following...]; n must be >= 2. 0/-1. */
int wubu_token_recycle_record(wubu_token_recycle_t *p,
                              const int *tokens, int n);

/* Copy the newest chain following `last_token`, truncated to cap.
 * Returns the copied count; 0 when no anchor matches. */
int wubu_token_recycle_fetch(const wubu_token_recycle_t *p,
                             int last_token, int *out, int cap);

int wubu_token_recycle_count(const wubu_token_recycle_t *p);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_TOKEN_RECYCLE_H */
