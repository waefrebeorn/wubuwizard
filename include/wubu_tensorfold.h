#ifndef WUBU_TENSORFOLD_H
#define WUBU_TENSORFOLD_H

/*
 * wubu_tensorfold.h -- EG-66: disk prefix snapshot with a serial-equivalence
 * gate (TensorFold lineage), reimplemented in C11. This ships only the
 * snapshot round-trip mechanism; adaptive caps, kv8 and AOT kernel packs
 * remain separate TODOs under the same gap row.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Write a prefix snapshot: magic/version/count/dim header, token ids, KV rows
 * and an FNV-1a digest over the payload. Returns 0 or -1. */
int wubu_tensorfold_snapshot(const char *path, const int *tokens, int n_tokens,
                             const float *kv, int dim);

/* Restore a snapshot written by wubu_tensorfold_snapshot. Validates the
 * digest and header against the caller's expected n_tokens/dim. */
int wubu_tensorfold_restore(const char *path, int *tokens, int n_tokens,
                            float *kv, int dim);

/* Deterministic serial KV recomputation used by the equivalence gate. */
void wubu_tensorfold_serial_kv(const int *tokens, int n_tokens, int dim,
                               float *kv);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_TENSORFOLD_H */
