#ifndef WUBU_KV_ROLLBACK_H
#define WUBU_KV_ROLLBACK_H

/*
 * wubu_kv_rollback.h -- EG-93: bounded-checkpoint sliding-KV prefix rollback.
 * rec M29: keep at most two checkpoints (ring + valid + ring index + logits).
 * On a mismatched prompt restore the newest checkpoint at position <= common
 * and replay the suffix with PREFILL arithmetic only. Decode KV must never be
 * staged: its arithmetic differs from batched prefill and changes logits.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_KV_MAX_CP       2
#define WUBU_KV_MAX_WINDOW   64
#define WUBU_KV_MAX_HISTORY  256

typedef struct wubu_kv_rollback wubu_kv_rollback_t;

/* window in [1,64]. NULL on invalid input. */
wubu_kv_rollback_t *wubu_kv_rollback_create(int window);
void wubu_kv_rollback_free(wubu_kv_rollback_t *p);

/* Append tokens with prefill ring arithmetic. 0/-1. */
int wubu_kv_rollback_prefill(wubu_kv_rollback_t *p, const int *tokens, int n);

/* Snapshot current ring state; retain only the two newest checkpoints. */
int wubu_kv_rollback_checkpoint(wubu_kv_rollback_t *p);

int wubu_kv_rollback_checkpoint_count(const wubu_kv_rollback_t *p);
int wubu_kv_rollback_position(const wubu_kv_rollback_t *p);

/* FNV-1a over ring[0..window), valid count and ring index. */
uint64_t wubu_kv_rollback_ring_hash(const wubu_kv_rollback_t *p);

/* Roll back to the newest checkpoint whose position <= the common prefix of
 * the cached token history and new_tokens, restore its ring, discard later
 * checkpoints and replay the suffix with prefill arithmetic only.
 * Returns the selected checkpoint position, -1 if none is eligible. */
int wubu_kv_rollback_to_prompt(wubu_kv_rollback_t *p,
                               const int *new_tokens, int n);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_KV_ROLLBACK_H */
