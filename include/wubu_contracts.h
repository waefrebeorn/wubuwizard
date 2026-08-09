/*
 * wubu_contracts.h -- FORMAL FLOOR EXPANSION INTO RUNTIME CONTRACTS
 * (roadmap #5). C11. Self-modification must be bounded by something
 * stronger than "loss went down on the last batch."
 *
 * The Lean-backed fitness floor (the amoeba's loss_tol + prover)
 * becomes a set of cheap C11 RUNTIME CONTRACTS attached to mutation
 * validation:
 *   - ball closure (the hyperbolic ball stays inside its radius)
 *   - exp/log identity-class properties (the foldmath invariants)
 *   - routing capacity constraints (the expert load stays bounded)
 *   - quant error bounds (the oracles already verify these)
 *
 * A mutation that improves empirical loss but VIOLATES a contract is
 * REJECTED even if the Lean prover was only used at design time.
 * Contracts are versioned HIVE META-CELLS: the enforced set can
 * itself evolve under the same accept/rollback discipline (only
 * expand, never silently weaken without an explicit archive entry).
 */
#ifndef WUBU_CONTRACTS_H
#define WUBU_CONTRACTS_H

#include <stdint.h>
#include <stddef.h>

#include "wubu_hive.h"

/* the contract kinds (each is a cheap runtime check) */
typedef enum {
    WUBU_CT_BALL = 0,        /* ball closure: ||x|| <= R + eps */
    WUBU_CT_EXP = 1,         /* exp identity: exp(a+b) ~ exp(a)*exp(b) */
    WUBU_CT_ROUTE = 2,       /* routing capacity: expert load <= cap */
    WUBU_CT_QUANT = 3,       /* quant error bound: |dq(x)-x| <= eps */
    WUBU_CT_FINITE = 4       /* the everything-finite guard (no NaN/Inf) */
} wubu_contract_kind_t;

/* one contract (a versioned hive meta-cell) */
typedef struct {
    uint32_t   version;      /* the contract version (only grows) */
    wubu_contract_kind_t kind;
    float      bound;        /* the tolerance */
    int        enabled;      /* 0 = suspended (only via archive entry) */
    uint64_t   batch;        /* when it was installed */
} wubu_contract_t;

/* the contract registry state */
typedef struct {
    wubu_hive_t *tissue;
    wubu_contract_t list[8];  /* the enforced set (bounded) */
    int  n;
    uint32_t next_version;
    uint64_t n_checks, n_violations;
} wubu_contracts_t;

/* CT1: init with the default contract set (all enabled). */
int wubu_contracts_init(wubu_contracts_t *ct, wubu_hive_t *tissue);

/* CT2: run the contract checks against a mutation's probes. The
 * probes are kind-ordered (the check for kind i reads probes[i]).
 * Returns the number of VIOLATIONS (0 = the mutation is contract-clean).
 * The probe for WUBU_CT_FINITE is the pointer-count of the checked
 * buffer's elements. */
int wubu_contracts_check(wubu_contracts_t *ct, const float *probes);

/* CT3: the finite guard: scan a buffer for NaN/Inf. Returns 1 when
 * the buffer is finite (the cheap every-step guard). */
int wubu_contracts_finite(const float *buf, size_t n);

/* CT4: expand the enforced set (a new contract version — only grows).
 * Writes the contract as a hive meta-cell. Returns the version. */
uint32_t wubu_contracts_add(wubu_contracts_t *ct, wubu_contract_kind_t kind,
                            float bound, uint64_t batch);

/* CT5: the stats. */
void wubu_contracts_stats(const wubu_contracts_t *ct, char *buf, size_t cap);

#endif
