/* wubu_kv_audit.h — immune audit hooks over the KV-FS seams (AN08)
 *
 * The KV cache is a file system; files move through a lifecycle that can
 * be audited for *consistency* — the detect + isolate steps of the 5+1
 * recovery (detect -> isolate -> rollback -> diagnose -> heal). This
 * module wraps the operational seams (LOAD, SAVE, FORWARD, MUTATE) with a
 * rule-checking ledger, per-path so an anomaly is attributable:
 *
 *   MUTATE_BEFORE_LOAD — wrote a path that was never loaded (spurious write)
 *   SAVE_WITHOUT_MUTATE — persisted a path that never changed (spurious save)
 *   DIRTY_LOAD       — read a path with unsaved mutations (data-loss risk)
 *
 * A per-path FNV-1a rolling checksum over the op sequence flags tampering
 * or out-of-band edits (detect); the path + offending op ID is the isolate.
 * heal() resets a path's dirty/suspect state after a guided recovery.
 *
 * C11, opaque structs, no third-party deps.
 * Design: docs/wubu1-hive-mind-plan.md §1 Track B, module #10 (AN22);
 * research/060 INDEX AN08 (immune system).
 * WaefreBeorn Umbrella License v3.0
 */
#ifndef WUBU_KV_AUDIT_H
#define WUBU_KV_AUDIT_H

#include <stddef.h>
#include <stdint.h>
#include "wubu_kv_embedding.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Operation seams being audited. */
typedef enum {
    WUBU_KV_AUDIT_LOAD    = 1,
    WUBU_KV_AUDIT_SAVE    = 2,
    WUBU_KV_AUDIT_FORWARD = 3,
    WUBU_KV_AUDIT_MUTATE  = 4,
} wubu_kv_audit_op_t;

/* Anomaly codes returned by wubu_kv_audit_record. */
typedef enum {
    WUBU_KV_AUDIT_OK                  = 0,
    WUBU_KV_AUDIT_MUTATE_BEFORE_LOAD  = 1,
    WUBU_KV_AUDIT_SAVE_WITHOUT_MUTATE = 2,
    WUBU_KV_AUDIT_DIRTY_LOAD          = 3,
} wubu_kv_audit_code_t;

/* Which rules are active (default: all on). */
typedef struct {
    int detect_mutate_before_load;   /* MUTATE on a never-loaded path */
    int detect_save_without_mutate;  /* SAVE on an unchanged path */
    int detect_dirty_load;           /* LOAD on a path with unsaved mutations */
} wubu_kv_audit_cfg_t;

/* Opaque handle. */
typedef struct wubu_kv_audit wubu_kv_audit_t;

wubu_kv_audit_cfg_t wubu_kv_audit_default_cfg(void);

/* Create an audit ledger over an embedding bridge. cfg may be NULL for
 * defaults. Returns NULL on failure. Does NOT own kv. */
wubu_kv_audit_t *wubu_kv_audit_create(wubu_kv_embedding_t *kv,
                                      const wubu_kv_audit_cfg_t *cfg);

/* Record an operation on a KV path. Returns WUBU_KV_AUDIT_OK if the op
 * is consistent, or an anomaly code naming the violated rule. Every call
 * still advances the per-path ledger + checksum; the code only reports. */
wubu_kv_audit_code_t wubu_kv_audit_record(wubu_kv_audit_t *a,
                                          wubu_kv_audit_op_t op,
                                          const char *path);

/* Total anomalies detected since creation. */
unsigned long wubu_kv_audit_anomalies(const wubu_kv_audit_t *a);

/* Checksum of the LAST anomaly's path (isolate). 0 if none yet. */
uint32_t wubu_kv_audit_last_path_checksum(const wubu_kv_audit_t *a);

/* Per-path rolling checksum of the op sequence (tamper detect). */
uint32_t wubu_kv_audit_checksum(const wubu_kv_audit_t *a, const char *path);

/* HEAL: after a guided recovery, reset a path's dirty/suspect state and
 * re-baseline its checksum. Returns 0 on success, -1 if the path is
 * not in the ledger. */
int wubu_kv_audit_heal(wubu_kv_audit_t *a, const char *path);

/* Free the audit ledger. Does NOT free kv. */
void wubu_kv_audit_free(wubu_kv_audit_t *a);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_KV_AUDIT_H */
